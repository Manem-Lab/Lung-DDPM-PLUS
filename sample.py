# -*- coding:utf-8 -*-

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import imageio
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from torchvision.transforms import Compose, Lambda
import monai.transforms as transforms

# Import custom modules
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.unet import create_model
from utils.dtypes import LabelEnum_Plus

# CUDA configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ====================== ARGUMENT PARSING ======================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Lung DDPM+ Sampling Script")
    parser.add_argument('--task', choices=['2D', '3D'], help='Choose the mode of operation')
    parser.add_argument('--ctfolder', type=str, default="")
    parser.add_argument('--maskfolder', type=str, default="")
    parser.add_argument('--samplefolder', type=str, default="exports/")
    parser.add_argument('--samplemaskfolder', type=str, default="exports/")
    parser.add_argument('--checkpointfolder', type=str, default="exports/")
    parser.add_argument('--visfolder', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--depth_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_class_labels', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('--mix_from', type=int, default=250)
    parser.add_argument('--bboxjson', type=str, default="")
    parser.add_argument('--sample_num', type=int, default=10)
    return parser.parse_args()

# ====================== UTILITY FUNCTIONS ======================
def count_nodule(segmentation_map):
    """
    Count the number of distinct tumors in a 3D segmentation map.
    
    Parameters:
    -----------
    segmentation_map : numpy.ndarray
        3D numpy array where tumor voxels are labeled as 2 and background as 0
    
    Returns:
    --------
    int
        Number of distinct tumors in the segmentation map
    """
    # Create a binary mask of tumor voxels
    tumor_mask = (segmentation_map == 2).astype(int)
    
    # Use connected component labeling to identify distinct tumor regions
    labeled_tumors, num_tumors = ndimage.label(tumor_mask)
    
    return num_tumors

def crop_and_resize_nodule(ct_image, segmentation_map, bbox, crop_depth=64, target_size=[224, 224]):
    """
    Crop and resize nodule region from CT image based on bounding box.
    
    Parameters:
    -----------
    ct_image : numpy.ndarray
        3D CT image array
    segmentation_map : numpy.ndarray
        3D segmentation map
    bbox : tuple
        Bounding box coordinates (min_w, max_w, min_h, max_h, min_d, max_d)
    crop_depth : int
        Depth of the cropped volume
    target_size : list
        Target size for resizing [height, width]
    
    Returns:
    --------
    tuple
        Resized CT image, resized segmentation map, and depth range
    """
    # Ensure the input arrays have the same shape
    assert ct_image.shape == segmentation_map.shape, "CT image and segmentation map must have the same shape"
    
    # Find depth range of the nodule
    min_depth = bbox[4]
    max_depth = bbox[5]
    
    # Ensure crop depth doesn't exceed nodule depth range
    crop_depth = min(crop_depth, max_depth - min_depth + 1)
    
    # Randomly select start depth within nodule range
    if max_depth - min_depth + 1 > crop_depth:
        start_depth = np.random.randint(min_depth, max_depth - crop_depth + 2)
    else:
        start_depth = min_depth
    
    # Crop the volume
    end_depth = start_depth + crop_depth
    cropped_ct = ct_image[:, :, start_depth:end_depth]
    cropped_seg = segmentation_map[:, :, start_depth:end_depth]
    
    # Resize using MONAI transforms
    target_size.append(crop_depth)
    cropped_ct = np.expand_dims(cropped_ct, axis=0)
    cropped_seg = np.expand_dims(cropped_seg, axis=0)
    resize_transform = transforms.Resize(spatial_size=target_size, mode='nearest')
    
    # Apply resize to CT image and segmentation map
    resized_ct = resize_transform(cropped_ct)
    resized_seg = resize_transform(cropped_seg)
    
    return resized_ct, resized_seg, (start_depth, end_depth)

def crop_nodule_centered_cube(ct_image, segmentation_map, cube_size=64):
    """
    Crop a cube centered on a randomly selected nodule.
    
    Parameters:
    -----------
    ct_image : numpy.ndarray
        3D CT image
    segmentation_map : numpy.ndarray
        3D segmentation map
    cube_size : int
        Size of the cube to crop
    
    Returns:
    --------
    tuple
        Cropped CT image, cropped segmentation map, and bounding box coordinates
    """
    # Ensure the input arrays have the same shape
    assert ct_image.shape == segmentation_map.shape, "CT image and segmentation map must have the same shape"
    
    # Find the coordinates of all nodule voxels
    nodule_coords = np.argwhere(segmentation_map == 1)
    
    if len(nodule_coords) == 0:
        raise ValueError("No nodules found in the segmentation map")
    
    # Randomly select one nodule if there are multiple
    selected_nodule = nodule_coords[np.random.choice(len(nodule_coords))]
    
    # Get image dimensions
    w, h, d = ct_image.shape
    
    # Calculate the boundaries of the cube
    half_size = cube_size // 2
    
    # Ensure the cube doesn't cross image boundaries
    start_w = max(0, min(w - cube_size, selected_nodule[0] - half_size))
    start_h = max(0, min(h - cube_size, selected_nodule[1] - half_size))
    start_d = max(0, min(d - cube_size, selected_nodule[2] - half_size))
    
    end_w = start_w + cube_size
    end_h = start_h + cube_size
    end_d = start_d + cube_size
    
    # Crop the cube
    cropped_ct = ct_image[start_w:end_w, start_h:end_h, start_d:end_d]
    cropped_seg = segmentation_map[start_w:end_w, start_h:end_h, start_d:end_d]
    
    # Create bbox coordinates
    bbox = (start_w, end_w, start_h, end_h, start_d, end_d)
    
    return cropped_ct, cropped_seg, bbox

def insert_processed_crop(original_image, processed_crop, bbox):
    """
    Insert a processed crop back into the original image.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original image
    processed_crop : numpy.ndarray
        Processed crop to insert
    bbox : tuple
        Bounding box coordinates
    
    Returns:
    --------
    numpy.ndarray
        Image with inserted crop
    """
    # Unpack the bbox coordinates
    start_w, end_w, start_h, end_h, start_d, end_d = bbox
    
    # Create a copy of the original image to avoid modifying the input
    result_image = np.copy(original_image)
    
    # Insert the processed crop back into the original image
    result_image[start_w:end_w, start_h:end_h, start_d:end_d] = processed_crop
    
    return result_image

def save_gif(sampleImage, vis_path, file_name, type):
    """
    Save 3D image as animated GIF.
    
    Parameters:
    -----------
    sampleImage : numpy.ndarray
        3D image to save
    vis_path : str
        Path to save directory
    file_name : str
        Base filename
    type : str
        Type of image ('ct', 'sample', 'mask')
    """
    gif_frames = []
    slice_3d = sampleImage
    slice_3d = ((slice_3d - slice_3d.min()) / (slice_3d.max() - slice_3d.min()) * 255).astype(np.uint8)
    
    # Special processing for mask images
    if type == 'mask':
        slice_3d[slice_3d < 10] = 0
        slice_3d[slice_3d > 240] = 255
        slice_3d[(slice_3d <= 240) & (slice_3d >= 10)] = 127
    
    # Create GIF frames
    for i in range(slice_3d.shape[2]):
        slice_2d = slice_3d[:, :, i]
        # Convert to RGB
        slice_2d = np.stack([slice_2d, slice_2d, slice_2d], axis=0)
        slice_2d = np.transpose(slice_2d, (1, 2, 0))
        gif_frames.append(slice_2d)
    
    # Save GIF
    gif_path = os.path.join(vis_path, f'{file_name}_{type}.gif')
    imageio.mimsave(gif_path, gif_frames, format='GIF', fps=15)

def label2masks(masked_img):
    """
    Convert label image to one-hot encoded masks.
    
    Parameters:
    -----------
    masked_img : numpy.ndarray
        Labeled segmentation image
    
    Returns:
    --------
    numpy.ndarray
        One-hot encoded masks
    """
    result_img = np.zeros(masked_img.shape + (1,)) 
    result_img[masked_img == LabelEnum_Plus.Nodule.value, 0] = 1
    return result_img

def find_case(bboxes, patient_id, fold_num, sample_idx):
    """
    Find bounding box for specific case.
    
    Parameters:
    -----------
    bboxes : list
        List of bounding box dictionaries
    patient_id : str
        Patient ID
    fold_num : int
        Fold number
    sample_idx : int
        Sample index
    
    Returns:
    --------
    list
        Bounding box coordinates
    """
    found_bbox = next((item['bounding_box'] for item in bboxes 
                      if item['patient_id'] == patient_id 
                      and item['fold'] == str(fold_num).zfill(2) 
                      and item['sample_idx'] == str(sample_idx).zfill(2)), None)
    
    bbox = [found_bbox['width']['start'], found_bbox['width']['end'],
            found_bbox['height']['start'], found_bbox['height']['end'],
            found_bbox['depth']['start'], found_bbox['depth']['end']]
    return bbox

# ====================== DATA TRANSFORMS ======================
def create_transforms():
    """Create data transformation pipelines"""
    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(4, 2))
    ])
    
    ct_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(3, 1)),
    ])
    
    return input_transform, ct_transform

# ====================== MAIN SAMPLING FUNCTION ======================
def main():
    """Main sampling function"""
    args = parse_arguments()
    
    # Extract parameters
    ctfolder = args.ctfolder
    maskfolder = args.maskfolder
    samplefolder = args.samplefolder
    samplemaskfolder = args.samplemaskfolder
    visfolder = args.visfolder
    input_size = args.input_size
    depth_size = args.depth_size
    batchsize = args.batchsize
    num_channels = args.num_channels
    num_res_blocks = args.num_res_blocks
    num_samples = args.num_samples
    in_channels = args.num_class_labels
    mix_from = args.mix_from
    out_channels = 1
    
    # Create output directories
    os.makedirs(samplefolder, exist_ok=True)
    os.makedirs(samplemaskfolder, exist_ok=True)
    if visfolder:
        os.makedirs(visfolder, exist_ok=True)
    
    # Load data
    ct_list = sorted(glob.glob(f"{ctfolder}/*.nii.gz"))
    
    with open(args.bboxjson, 'r') as f:
        bboxes = json.load(f)
    
    # Create transforms
    input_transform, ct_transform = create_transforms()
    
    # Load model
    model = create_model(input_size, num_channels, num_res_blocks, 
                        in_channels=in_channels, out_channels=out_channels).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=args.timesteps,
        loss_type='L1',
    ).cuda()
    
    diffusion.load_state_dict(torch.load(os.path.join(args.checkpointfolder, f'best_model.pt'))['ema'])
    print(f"Lung-DDPM+ model loaded!")
    
    # Get training data for current fold
    new_ct_list = []
    for inputfile in ct_list:
        patient_id = inputfile.split('/')[-1].split('.')[0]
        new_ct_list.append(os.path.join(ctfolder, patient_id + '.nii.gz'))
    
    scaler = MinMaxScaler()

    for fold_num in range(5):
        # Process each CT image
        for k, ct_path in enumerate(pbar := tqdm(new_ct_list)):
            pbar.set_description(f"Processing Fold {fold_num}")
            
            # Extract patient ID
            ct_name = ct_path.split('/')[-1]
            patient_id = ct_name.split('.')[0]

            print(f"Processing CT for patient: {patient_id}")

            # Load CT image
            ct = nib.load(ct_path)
            ctImg = ct.get_fdata()
            ct_arr = np.transpose(ctImg, (1, 2, 0))
            
            # Process multiple samples for each patient
            for sample_idx in range(args.sample_num):
                print(f'Processing image {sample_idx+1}/{args.sample_num}.')
                
                # Get bounding box and load mask
                bbox = find_case(bboxes, patient_id, fold_num, sample_idx)
                file_name = f'{str(fold_num).zfill(2)}_{str(sample_idx).zfill(2)}_{patient_id}'
                mask_path = os.path.join(maskfolder, file_name + '.nii.gz')
                
                ref = nib.load(mask_path)
                refImg = ref.get_fdata()
                refImg = np.transpose(refImg, (1, 2, 0))
                
                # Crop based on task type
                if args.task == '2D':
                    input_img_0 = np.array(refImg[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
                else:
                    input_img_0 = np.array(refImg)
                
                target_img_0 = np.array(ct_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
                
                # Preprocess images
                input_img_0[input_img_0 == 1] = 0
                input_img_0[input_img_0 == 2] = 1
                target_img_1, input_img_1, bbox_seg = crop_nodule_centered_cube(target_img_0, input_img_0, input_size)
                
                # Normalize CT values
                scaler.fit(target_img_1.reshape(-1, target_img_1.shape[-1]))
                target_img_1 = scaler.transform(target_img_1.reshape(-1, target_img_1.shape[-1])).reshape(target_img_1.shape)
                
                # Create masks and tensors
                mask = label2masks(input_img_1)
                input_tensor = input_transform(mask)
                ct_tensor = ct_transform(target_img_1)
                
                # Generate samples
                batches = num_to_groups(num_samples, batchsize)
                sample_count = 0
                counter = 0
                
                for i, bsize in enumerate(batches):
                    condition_tensors, counted_samples, ct_tensors = [], [], []
                    
                    for b in range(bsize):
                        ct_tensors.append(ct_tensor)
                        condition_tensors.append(input_tensor)
                        counted_samples.append(sample_count)
                        sample_count += 1
                    
                    condition_tensors = torch.cat(condition_tensors, 0).cuda()
                    ct_tensors = torch.cat(ct_tensors, 0).cuda()
                    ct_tensors = ct_tensors.unsqueeze(0)
                    
                    # Sample using diffusion model
                    all_images_list = list(map(lambda n: diffusion.sample_dpm_solver(
                        x_start=ct_tensors, batch_size=n,
                        condition_tensors=condition_tensors,
                        mix_from=mix_from), [bsize]))
                    
                    all_images = torch.cat(all_images_list, dim=0)
                    all_images = all_images.unsqueeze(1)
                    all_images = all_images.transpose(5, 3)
                    sampleImages = all_images.cpu()
                    
                    # Process and save samples
                    for b, c in enumerate(counted_samples):
                        counter = counter + 1
                        sampleImage = sampleImages[b][0]
                        sampleImage = sampleImage.numpy()
                        sampleImage = np.squeeze(sampleImage)
                        
                        # Denormalize
                        sampleImage = (sampleImage + 1) / 2
                        sampleImage = scaler.inverse_transform(
                            sampleImage.reshape(-1, sampleImage.shape[-1])).reshape(sampleImage.shape)
                        
                        # Insert back into original image
                        sampleImage = insert_processed_crop(target_img_0, sampleImage, bbox_seg)
                        
                        if args.task == '2D':
                            sampleImage = insert_processed_crop(ct_arr, sampleImage, bbox)
                            sampleImage_0 = np.copy(sampleImage)
                            
                            # Ensure valid crop with nodules
                            while True:
                                sampleImage, maskImage, depth_range = (
                                    crop_and_resize_nodule(sampleImage_0, refImg, bbox, 16, [512, 512]))
                                if np.unique(maskImage).shape[0] == 3:
                                    sampleImage = np.squeeze(sampleImage)
                                    maskImage = np.squeeze(maskImage)
                                    break
                            # Save visualization if requested
                            if visfolder:
                                save_gif(sampleImage, visfolder, file_name, 'sample')
                                save_gif(maskImage, visfolder, file_name, 'mask')
                                save_gif(ct_arr[:, :, depth_range[0]:depth_range[1]], visfolder, file_name, 'ct')
                        else:
                            if visfolder:
                                save_gif(sampleImage, visfolder, file_name, 'sample')
                                save_gif(refImg, visfolder, file_name, 'mask')
                                save_gif(target_img_0, visfolder, file_name, 'ct')
                        
                        # Save NIfTI files
                        sampleImage = np.transpose(sampleImage, (2, 0, 1))
                        nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
                        nib.save(nifti_img, os.path.join(samplefolder, file_name + '.nii.gz'))
                
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
