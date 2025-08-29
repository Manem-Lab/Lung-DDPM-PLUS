#-*- coding:utf-8 -*-

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

warnings.filterwarnings("ignore", category=UserWarning)

# ================================
# Utility Functions
# ================================

def exists(x):
    """Check if value is not None"""
    return x is not None

def default(val, d):
    """Return val if it exists, otherwise return d (or d() if d is a function)"""
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    """Create infinite iterator from dataloader"""
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    """Split number into groups of divisor size"""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    """Handle backward pass with optional mixed precision"""
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# ================================
# Helper Classes
# ================================

class EMA():
    """Exponential Moving Average for model parameters"""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        """Update moving average model with current model parameters"""
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """Calculate exponential moving average"""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# ================================
# Diffusion Utilities
# ================================

def extract(a, t, x_shape):
    """Extract values from tensor a at timesteps t"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    """Generate noise tensor of given shape"""
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values in diffusion process
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# ================================
# Gaussian Diffusion Model
# ================================

class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion Process for 3D medical image generation"""
    
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels=1,
        timesteps=1000,
        loss_type='l1',
        betas=None,
        with_pairwised=False,
        apply_bce=False,
        lambda_bce=0.0
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce

        # Initialize beta schedule
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        # Calculate alpha values and cumulative products
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Register buffers for diffusion parameters
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        # Log calculation clipped because posterior variance is 0 at beginning of diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        """Calculate mean and variance for forward diffusion process"""
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        """Predict original image from noisy image and predicted noise"""
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        """Calculate posterior distribution q(x_{t-1} | x_t, x_0)"""
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c=None):
        """Calculate mean and variance for reverse diffusion process"""
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x, c], 1), t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    def q_sample(self, x_start, t, noise=None, c=None):
        """Sample from forward diffusion process q(x_t | x_0)"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        """Calculate training loss"""
        b, c, h, w, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False):
        """Single reverse diffusion step"""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None):
        """Full reverse diffusion sampling loop"""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        start_time = time.time()
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition_tensors=condition_tensors)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Sampling time: {elapsed_time:.2f} seconds")
        return img

    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None):
        """Generate samples using reverse diffusion"""
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors=condition_tensors)

    @torch.no_grad()
    def sample_dpm_solver(self, x_start, batch_size=2, condition_tensors=None, mix_from=250):
        """Generate samples using DPM solver (faster sampling)"""
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels

        return self.p_sample_loop_dpm_solver(x_start, (batch_size, channels, depth_size, image_size, image_size),
                                           condition_tensors=condition_tensors, mix_from=250)

    def p_sample_loop_dpm_solver(self, x_start, shape, condition_tensors=None, mix_from=250):
        """DPM solver sampling implementation"""
        device = self.betas.device
        img = torch.randn(shape, device=device)

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(cosine_beta_schedule(250)))
        model_fn = model_wrapper(
            self.denoise_fn,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                                correcting_x0_fn="dynamic_thresholding", condition=condition_tensors)
        start_time = time.time()
        img, cal = dpm_solver.sample(
            img.to(dtype=torch.float),
            x_start=x_start,
            steps=50,
            order=3,
            skip_type="time_uniform",
            method="multistep",
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Sampling time: {elapsed_time:.2f} seconds")
        return img

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        """Forward pass for training"""
        b, c, d, h, w, device, img_size, depth_size = *x.shape, x.device, self.image_size, self.depth_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)

# ================================
# Trainer Class
# ================================

class Trainer(object):
    """Training orchestrator for diffusion model"""
    
    def __init__(
        self,
        dataset_name,
        diffusion_model,
        dataset,
        start_steps=0,
        ema_decay=0.995,
        image_size=128,
        depth_size=128,
        train_batch_size=2,
        train_lr=2e-6,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        deploy='local',
        with_pairwised=False
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model = diffusion_model
        
        # Initialize EMA
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        # Training parameters
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every

        # Dataset and optimizer setup
        self.ds = dataset
        self.dl = cycle(data.DataLoader(
            self.ds, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=0 if deploy == 'local' else 4, 
            pin_memory=True
        ))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size

        # Training state
        self.step = start_steps
        self.start_steps = start_steps

        # Mixed precision setup
        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.deploy = deploy

        # Logging setup
        self.log_dir = self.create_log_dir()
        self.results_folder = self.log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.reset_parameters()

    def create_log_dir(self):
        """Create timestamped logging directory"""
        now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
        log_dir = os.path.join("./runs/Lung-DDPM+", f'Lung-DDPM+_fold_{self.dataset_name}_{now}')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        """Reset EMA model parameters to match current model"""
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """Update EMA model parameters"""
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        """Save model checkpoint"""
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))

    def load(self, milestone):
        """Load model checkpoint"""
        data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        """Main training loop"""
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        pbar = tqdm(initial=self.step, total=self.train_num_steps, unit='step')

        while self.step <= self.train_num_steps:
            accumulated_loss = []
            
            # Gradient accumulation loop
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                input_tensors = data['input'].cuda()
                target_tensors = data['target'].cuda()
                loss = self.model(target_tensors, condition_tensors=input_tensors)
                loss = loss.sum() / self.batch_size
                print(f'Training loss {i}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss.append(loss.item())

            # Log training metrics
            average_loss = np.mean(accumulated_loss)
            end_time = time.time()
            self.writer.add_scalar("training_loss", average_loss, self.step, self.train_num_steps)

            # Optimizer step
            self.opt.step()
            self.opt.zero_grad()

            # Update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Save and sample
            if self.step != self.start_steps and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(1, self.batch_size)

                # Generate samples
                all_images_list = list(map(
                    lambda n: self.ema_model.sample(batch_size=n, condition_tensors=self.ds.sample_conditions(batch_size=n)), 
                    batches
                ))
                all_images = torch.cat(all_images_list, dim=0)

                # Save as NIfTI
                all_images = all_images.transpose(4, 2)
                sampleImage = all_images.cpu().numpy()
                sampleImage = sampleImage.reshape([self.image_size, self.image_size, self.depth_size])
                nifti_img = nib.Nifti1Image(sampleImage, affine=np.eye(4))
                nib.save(nifti_img, os.path.join(self.results_folder, f'sample-{milestone}.nii.gz'))

                self.save(milestone)

            self.step += 1
            pbar.update(1)

        print('Training completed')
        end_time = time.time()
        execution_time = (end_time - start_time) / 3600
        
        # Log hyperparameters and final metrics
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "image_size": self.image_size,
                "depth_size": self.depth_size,
                "execution_time (hour)": execution_time
            },
            {"last_loss": average_loss}
        )
        self.writer.close()


