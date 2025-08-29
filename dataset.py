# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import torchio as tio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import monai.transforms as transforms
from utils.dtypes import LabelEnum, LabelEnum_Plus


class LIDC_IDRI_dataset(Dataset):
    """
    Dataset class for LIDC-IDRI lung nodule dataset.
    Handles loading, preprocessing, and augmentation of CT images and segmentation masks.
    """
    
    def __init__(self,
                 input_folder: str,
                 target_folder: str,
                 input_size: int,
                 depth_size: int,
                 input_channel: int = 3,
                 num_class_labels: int = 3,
                 transform=None,
                 target_transform=None,
                 full_channel_mask=False,
                 combine_output=False):
        """
        Initialize the dataset.
        
        Args:
            input_folder: Path to input segmentation masks
            target_folder: Path to target CT images
            input_size: Target size for height/width dimensions
            depth_size: Target size for depth dimension
            input_channel: Number of input channels
            num_class_labels: Number of classes in segmentation (2 or 3)
            transform: Optional transform for input images
            target_transform: Optional transform for target images
            full_channel_mask: Whether to use full channel mask encoding
            combine_output: Whether to combine input and target in output
        """
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.num_class_labels = num_class_labels
        self.input_channel = input_channel
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
        
        # Initialize preprocessing components
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        
        self.pair_files = self._pair_file()

    def _pair_file(self):
        """
        Create pairs of input and target files by matching filenames between folders.
        
        Returns:
            List of tuples containing (input_file_path, target_file_path)
        """
        pairs = []
        
        # Get all .nii.gz files from input folder
        input_files = [f for f in os.listdir(self.input_folder) if f.endswith('.nii.gz')]
        
        for input_filename in input_files:
            input_file = os.path.join(self.input_folder, input_filename)
            target_file = os.path.join(self.target_folder, input_filename)
            
            # Check if corresponding target file exists
            if os.path.exists(target_file):
                pairs.append((input_file, target_file))
        
        return pairs

    def crop_nodule_centered_cube(self, ct_image, segmentation_map, cube_size=64):
        """
        Crop a cube centered around a randomly selected nodule.
        
        Args:
            ct_image: CT image array
            segmentation_map: Segmentation mask array
            cube_size: Size of the cube to crop
            
        Returns:
            Tuple of (cropped_ct, cropped_seg)
        """
        assert ct_image.shape == segmentation_map.shape, "CT image and segmentation map must have the same shape"

        # Find nodule coordinates based on number of classes
        if self.num_class_labels == 2:
            nodule_coords = np.argwhere(segmentation_map == 1)
        else:
            nodule_coords = np.argwhere(segmentation_map == 2)

        if len(nodule_coords) == 0:
            raise ValueError("No nodules found in the segmentation map")

        # Randomly select one nodule if there are multiple
        selected_nodule = nodule_coords[np.random.choice(len(nodule_coords))]
        w, h, d = ct_image.shape
        half_size = cube_size // 2

        # Calculate cube boundaries ensuring they don't exceed image bounds
        start_w = max(0, min(w - cube_size, selected_nodule[0] - half_size))
        start_h = max(0, min(h - cube_size, selected_nodule[1] - half_size))
        start_d = max(0, min(d - cube_size, selected_nodule[2] - half_size))

        end_w = start_w + cube_size
        end_h = start_h + cube_size
        end_d = start_d + cube_size

        # Crop the cube
        cropped_ct = ct_image[start_w:end_w, start_h:end_h, start_d:end_d]
        cropped_seg = segmentation_map[start_w:end_w, start_h:end_h, start_d:end_d]

        return cropped_ct, cropped_seg

    def crop_and_resize_nodule(self, ct_image, segmentation_map, crop_depth=64, target_size=[224, 224]):
        """
        Crop around nodule region and resize to target dimensions.
        
        Args:
            ct_image: CT image array
            segmentation_map: Segmentation mask array
            crop_depth: Target depth for cropping
            target_size: Target size for height/width dimensions
            
        Returns:
            Tuple of (resized_ct, resized_seg, depth_range)
        """
        assert ct_image.shape == segmentation_map.shape, "CT image and segmentation map must have the same shape"

        # Find nodule positions (class 2)
        nodule_positions = np.where(segmentation_map == 2)
        if len(nodule_positions[0]) == 0:
            raise ValueError("No nodule found in segmentation map")

        # Calculate nodule depth range
        min_depth = np.min(nodule_positions[2])
        max_depth = np.max(nodule_positions[2])
        nodule_depth = max_depth - min_depth + 1

        # Determine cropping boundaries
        if nodule_depth < crop_depth:
            # Expand around nodule if it's smaller than crop_depth
            extra_space = crop_depth - nodule_depth
            space_before = extra_space // 2
            space_after = extra_space - space_before

            start_depth = min_depth - space_before
            end_depth = max_depth + space_after + 1

            # Adjust for image boundaries
            if start_depth < 0:
                start_depth = 0
                end_depth = min(ct_image.shape[2], crop_depth)
            elif end_depth > ct_image.shape[2]:
                end_depth = ct_image.shape[2]
                start_depth = max(0, end_depth - crop_depth)
        else:
            # Center crop around nodule if it's larger than crop_depth
            center_depth = (min_depth + max_depth) // 2
            half_crop = crop_depth // 2
            start_depth = center_depth - half_crop
            end_depth = start_depth + crop_depth

            # Adjust for image boundaries
            if start_depth < 0:
                start_depth = 0
                end_depth = crop_depth
            elif end_depth > ct_image.shape[2]:
                end_depth = ct_image.shape[2]
                start_depth = end_depth - crop_depth

        # Crop the volume
        cropped_ct = ct_image[:, :, start_depth:end_depth]
        cropped_seg = segmentation_map[:, :, start_depth:end_depth]

        # Resize to target dimensions
        target_size.append(crop_depth)
        cropped_ct = np.expand_dims(cropped_ct, axis=0)
        cropped_seg = np.expand_dims(cropped_seg, axis=0)
        
        resize_transform = transforms.Resize(spatial_size=target_size, mode='nearest')
        resized_ct = resize_transform(cropped_ct)
        resized_seg = resize_transform(cropped_seg)

        return resized_ct, resized_seg, (start_depth, end_depth)

    def label2masks(self, masked_img):
        """
        Convert segmentation labels to multi-channel mask format.
        
        Args:
            masked_img: Input segmentation mask
            
        Returns:
            Multi-channel mask array
        """
        result_img = np.zeros(masked_img.shape + (self.input_channel - 1,))
        
        if self.num_class_labels == 2:
            result_img[masked_img == LabelEnum_Plus.Nodule.value, 0] = 1
        elif self.num_class_labels == 3:
            result_img[masked_img == LabelEnum.LUNG.value, 0] = 1
            result_img[masked_img == LabelEnum.Nodule.value, 1] = 1
            
        return result_img

    def read_image(self, file_path, pass_scaler=False):
        """
        Read and preprocess image from file.
        
        Args:
            file_path: Path to the image file
            pass_scaler: Whether to skip MinMax scaling
            
        Returns:
            Preprocessed image array
        """
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            # Apply MinMax scaling to normalize values to [0, 1]
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        return img

    def resize_img(self, img):
        """
        Resize 3D image to target dimensions.
        
        Args:
            img: Input 3D image
            
        Returns:
            Resized image
        """
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            resize_op = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(resize_op(img))[0]
        return img

    def resize_img_4d(self, input_img):
        """
        Resize 4D image (with channels) to target dimensions.
        
        Args:
            input_img: Input 4D image with channels
            
        Returns:
            Resized 4D image
        """
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, self.input_channel - 1))
        
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                resize_op = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(resize_op(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        """
        Sample random conditions/inputs for batch generation.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Batch tensor of input conditions
        """
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
                
        return torch.cat(input_tensors, 0).cuda()

    def plot(self, index, n_slice=30):
        """
        Plot a specific slice of input and target images for visualization.
        
        Args:
            index: Dataset index to plot
            n_slice: Slice number to display
        """
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.title('Input')
        
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.title('Target')
        
        plt.show()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.pair_files)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing 'input' and 'target' tensors, or combined tensor if combine_output=True
        """
        input_file, target_file = self.pair_files[index]
        
        # Load and transpose images (change from (D, H, W) to (H, W, D))
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        input_img = np.transpose(input_img, (1, 2, 0))
        
        target_img = self.read_image(target_file)
        target_img = np.transpose(target_img, (1, 2, 0))

        # Apply different cropping strategies based on input parameters
        if self.input_channel == 2 or self.input_size == 128:
            target_img, input_img = self.crop_nodule_centered_cube(target_img, input_img, self.input_size)
        elif self.input_size == 512:
            target_img, input_img, _ = self.crop_and_resize_nodule(target_img, input_img,
                                                                   self.depth_size,
                                                                   [self.input_size, self.input_size])
            target_img = torch.squeeze(target_img, 0)
            input_img = torch.squeeze(input_img, 0)

        # Convert labels to masks if using full channel mask
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        
        # Resize images if not already at target size
        if self.input_size != 512:
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            target_img = self.resize_img(target_img)

        # Apply transforms if specified
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        # Return combined output or separate input/target
        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input': input_img, 'target': target_img}


