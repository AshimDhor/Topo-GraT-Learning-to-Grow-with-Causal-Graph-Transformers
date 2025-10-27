# train.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

# Import our custom modules
import config
from data.dataset import WSIDataset
from models.topo_grat import TopoGraT
from utils.losses import TopoGraTLoss

def main():
    """
    Main function to run the training process.
    """
    print("Starting Topo-GraT training process...")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths = sorted(glob.glob(os.path.join(config.DATA_DIR, "camelyon16/images/*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(config.DATA_DIR, "camelyon16/masks/*.tif")))

    train_dataset = WSIDataset(image_paths=image_paths, mask_paths=mask_paths, patch_size=config.PATCH_SIZE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_STAGE2, # Using Stage 2 batch size as we train the segmentation network
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
