# -*- coding:utf-8 -*-

import argparse
import os
import torch
from torchvision.transforms import Compose, Lambda

from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import LIDC_IDRI_dataset


def setup_gpu():
    """Configure GPU settings for training."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Lung DDPM+ diffusion model")
    
    # Data paths
    parser.add_argument('--ct_path', type=str, default='data/LIDC-IDRI/CT',
                       help='Path to CT image directory')
    parser.add_argument('--mask_path', type=str, default='data/LIDC-IDRI/CT',
                       help='Path to mask image directory')
    
    # Training parameters
    parser.add_argument('--train_lr', type=float, default=1e-5,
                       help='Learning rate for training')
    parser.add_argument('--batchsize', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100000,
                       help='Number of training epochs')
    parser.add_argument('--timesteps', type=int, default=250,
                       help='Number of diffusion timesteps')
    parser.add_argument('--save_and_sample_every', type=int, default=1000,
                       help='Save model and generate samples every N steps')
    
    # Resume training parameters
    parser.add_argument('-r', '--resume_weight', type=str, default="",
                       help='Path to resume weights from checkpoint')
    parser.add_argument('--start_steps', type=int, default=0,
                       help='Starting step number for resumed training')
    
    args = parser.parse_args()
    args.input_size = 64
    args.depth_size = 64
    args.num_channels = 64
    args.num_res_blocks = 1
    args.num_class_labels = 2
    
    return args


def create_transforms():
    """Create data transforms for input and target images."""
    # Transform for target CT images (single channel output)
    target_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),        # Convert to tensor
        Lambda(lambda t: (t * 2) - 1),                   # Normalize to [-1, 1]
        Lambda(lambda t: t.unsqueeze(0)),                # Add channel dimension
        Lambda(lambda t: t.transpose(3, 1)),             # Rearrange dimensions
    ])

    # Transform for input mask images (multi-channel input)
    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),        # Convert to tensor
        Lambda(lambda t: (t * 2) - 1),                   # Normalize to [-1, 1]
        Lambda(lambda t: t.permute(3, 0, 1, 2)),         # Permute dimensions
        Lambda(lambda t: t.transpose(3, 1)),             # Rearrange dimensions
    ])
    
    return input_transform, target_transform


def create_dataset(args, input_transform, target_transform):
    """Create and return the training dataset."""
    dataset = LIDC_IDRI_dataset(
        input_folder=args.mask_path,
        target_folder=args.ct_path,
        input_size=args.input_size,
        depth_size=args.depth_size,
        input_channel=args.num_class_labels,
        transform=input_transform,
        target_transform=target_transform,
        full_channel_mask=True,
        num_class_labels=args.num_class_labels
    )
    
    print(f'{len(dataset)} samples loaded.')
    return dataset


def create_diffusion_model(args):
    """Create and return the diffusion model."""
    in_channels = args.num_class_labels
    out_channels = 1

    channel_mult = (1, 2, 3, 4)
    # Create UNet model
    model = create_model(
        args.input_size, 
        args.num_channels, 
        args.num_res_blocks, 
        in_channels=in_channels, 
        out_channels=out_channels, 
        channel_mult=channel_mult
    ).cuda()

    # Create Gaussian diffusion wrapper
    diffusion = GaussianDiffusion(
        model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type='l1',
        channels=out_channels
    ).cuda()
    
    return diffusion


def load_checkpoint(diffusion, resume_weight, start_steps):
    """Load model checkpoint if resume weight is provided."""
    if len(resume_weight) > 0:
        weight = torch.load(resume_weight, map_location='cuda')
        diffusion.load_state_dict(weight['ema'])
        print(f"Pretrained model loaded. Resume training from {start_steps}th steps!")


def main():
    """Main training function."""
    # Setup environment and parse arguments
    setup_gpu()
    args = parse_arguments()
    
    # Create data transforms
    input_transform, target_transform = create_transforms()
    
    # Create dataset
    dataset = create_dataset(args, input_transform, target_transform)
    
    # Create diffusion model
    diffusion = create_diffusion_model(args)
    
    # Load checkpoint if resuming training
    load_checkpoint(diffusion, args.resume_weight, args.start_steps)
    
    # Create trainer and start training
    trainer = Trainer(
        dataset_name='LIDC-IDRI',
        diffusion_model=diffusion,
        dataset=dataset,
        start_steps=args.start_steps,
        depth_size=args.depth_size,
        train_batch_size=args.batchsize,
        train_lr=args.train_lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        save_and_sample_every=args.save_and_sample_every,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()