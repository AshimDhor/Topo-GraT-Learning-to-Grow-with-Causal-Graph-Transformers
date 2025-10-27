# data/dataset.py

import torch
from torch.utils.data import Dataset
import openslide
import numpy as np
import cv2
from torchvision import transforms
from ..utils.helpers import generate_cgf_pseudo_truth

class WSIDataset(Dataset):
    """
    Custom PyTorch Dataset for handling Whole Slide Images (WSIs).
    This class is responsible for:
    1. Opening WSIs using openslide.
    2. Performing on-the-fly tissue detection.
    3. Tiling the WSI into patches.
    4. Providing patches and their corresponding ground truth masks and CGFs.
    """
    def __init__(self, image_paths, mask_paths, patch_size=512, level=0):
        """
        Args:
            image_paths (list): A list of file paths to the WSIs.
            mask_paths (list): A list of file paths to the ground truth segmentation masks.
            patch_size (int): The size of the square patches to extract.
            level (int): The magnification level to read from the WSI (0 is the highest).
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.level = level
        
        self.patch_coords = []
        self._prepare_patch_coords()

    def _prepare_patch_coords(self):
        """
        Pre-calculates the coordinates of all valid tissue patches from all WSIs.
        This is done once at the beginning to speed up data loading during training.
        """
        print("Preprocessing WSIs and extracting patch coordinates...")
        for idx, image_path in enumerate(self.image_paths):
            slide = openslide.OpenSlide(image_path)
            
            # Use a downsampled thumbnail for efficient tissue detection
            thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
            thumbnail_np = np.array(thumbnail)
            
            # Convert to HSV and use Otsu's threshold on the saturation channel
            hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
            _, s, _ = cv2.split(hsv)
            _, tissue_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Scale factor to map thumbnail coordinates back to level 0
            width_scale = slide.level_dimensions[0][0] / tissue_mask.shape[1]
            height_scale = slide.level_dimensions[0][1] / tissue_mask.shape[0]

            # Iterate over the WSI at the specified level to find valid patches
            width, height = slide.level_dimensions[self.level]
            for y in range(0, height, self.patch_size):
                for x in range(0, width, self.patch_size):
                    # Check if the center of the patch corresponds to a tissue region in the thumbnail
                    thumb_x = int((x + self.patch_size / 2) / width_scale)
                    thumb_y = int((y + self.patch_size / 2) / height_scale)
                    
                    if thumb_y < tissue_mask.shape[0] and thumb_x < tissue_mask.shape[1] and tissue_mask[thumb_y, thumb_x] > 0:
                        # This is a valid tissue patch, save its coordinates
                        self.patch_coords.append({
                            'slide_idx': idx,
                            'x': x,
                            'y': y
                        })
        print(f"Found {len(self.patch_coords)} valid tissue patches.")

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        """
        Retrieves a single patch and its corresponding ground truth.
        """
        coord = self.patch_coords[idx]
        slide_idx, x, y = coord['slide_idx'], coord['x'], coord['y']

        # --- Load Image Patch ---
        image_path = self.image_paths[slide_idx]
        slide = openslide.OpenSlide(image_path)
        image_patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size)).convert('RGB')
        image_patch = np.array(image_patch)

        # --- Load Mask Patch ---
        mask_path = self.mask_paths[slide_idx]
        mask_slide = openslide.OpenSlide(mask_path)
        # Masks are usually single channel, so we take the first channel
        mask_patch = mask_slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        mask_patch = np.array(mask_patch)[:, :, 0]

        # --- Generate CGF Pseudo-Truth ---
        # This is done on-the-fly for each patch
        cgf_true = generate_cgf_pseudo_truth(mask_patch)

        # --- Convert to Tensors and Apply Transforms ---
        # Note: In a full implementation, you would add stain normalization and data augmentation here.
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image_patch)
        mask_tensor = torch.from_numpy(mask_patch).long().unsqueeze(0) # (1, H, W)
        cgf_tensor = torch.from_numpy(cgf_true) # (2, H, W)

        # Determine the patch-level label for the instance selection loss
        patch_label = torch.tensor(1.0 if mask_tensor.sum() > 0 else 0.0, dtype=torch.float32)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'cgf': cgf_tensor,
            'patch_label': patch_label,
            'coords': (slide_idx, x, y) # For tracking/debugging
        }