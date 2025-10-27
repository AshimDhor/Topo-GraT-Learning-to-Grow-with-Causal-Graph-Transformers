# utils/helpers.py

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
import numpy as np

def generate_cgf_pseudo_truth(mask):
    """
    Generates the pseudo-ground-truth Causal Growth Field (CGF) from a binary segmentation mask.
    This function is a key part of the training process.

    Args:
        mask (np.ndarray): A 2D numpy array representing the binary ground truth mask (H, W).
                           1 indicates tumor, 0 indicates background.

    Returns:
        np.ndarray: A 2-channel numpy array representing the CGF vector field (2, H, W).
    """
    if mask.sum() == 0:
        # If there is no tumor, the CGF is a zero field.
        return np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)

    # 1. Core Identification using Medial Axis Transform
    # This finds the "skeleton" of the tumor shape.
    skeleton, dist_from_skeleton = medial_axis(mask, return_distance=True)

    # 2. Distance Transform Gradient (from boundary inwards)
    # The distance transform gives the distance of each pixel from the nearest background pixel.
    dist_from_boundary = distance_transform_edt(mask)
    
    # Compute the gradient of the distance map. The gradient points in the direction of
    # the steepest ascent, which is from the boundary towards the core.
    dy, dx = np.gradient(dist_from_boundary)

    # 3. Field Inversion
    # We invert the gradient field to create vectors that point from the core outwards.
    # This simulates a growth process.
    cgf_field = np.stack([-dx, -dy], axis=0)

    # Normalize the vectors to have a unit length (direction is more important than magnitude)
    norms = np.sqrt(cgf_field[0]**2 + cgf_field[1]**2)
    norms[norms == 0] = 1.0  # Avoid division by zero
    cgf_field = cgf_field / norms

    # Ensure the CGF is only defined within the tumor mask
    cgf_field[:, mask == 0] = 0

    return cgf_field.astype(np.float32)


def compute_uncertainty_map(seg_logits):
    """
    Computes a pixel-wise uncertainty map using Shannon entropy.

    Args:
        seg_logits (torch.Tensor): The raw logit outputs from the segmentation model (B, C, H, W).

    Returns:
        torch.Tensor: A single-channel uncertainty map (B, 1, H, W).
    """
    # Apply softmax to get probabilities
    probs = F.softmax(seg_logits, dim=1)
    
    # Shannon Entropy: H(p) = - sum(p * log(p))
    # We add a small epsilon to avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1, keepdim=True)
    
    return entropy