# evaluate.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import argparse

# Import our custom modules
import config
from data.dataset import WSIDataset
from models.topo_grat import TopoGraT

def main(args):
    """
    Main function to run the evaluation process.
    """
    # --- 1. Setup and Configuration ---
    print("Starting Topo-GraT evaluation...")
    
    # Set up device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading for Test Set ---
    # This should point to your test data
    image_paths = sorted(glob.glob(os.path.join(config.DATA_DIR, "camelyon16_test/images/*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(config.DATA_DIR, "camelyon16_test/masks/*.tif")))

    test_dataset = WSIDataset(image_paths=image_paths, mask_paths=mask_paths, patch_size=config.PATCH_SIZE)
    
    # Use a batch size of 1 for evaluation to process one patch at a time for stitching
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )

    # --- 3. Load Trained Model ---
    print(f"Loading model checkpoint from: {args.checkpoint_path}")
    
    model = TopoGraT(in_channels=3, out_channels=2, base_c=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval() # Set the model to evaluation mode

    # --- 4. Evaluation Metrics ---
    # MONAI provides excellent, robust metric calculators
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    # --- 5. Inference and Stitching Loop ---
    # We need to reconstruct the full slide predictions to evaluate them.
    # We'll store patch predictions in a dictionary.
    slide_predictions = {}

    print("Running inference on test set patches...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch['image'].to(device)
            coords = batch['coords']
            
            slide_idx, x, y = coords[0].item(), coords[1].item(), coords[2].item()

            # Forward pass
            seg_logits, _ = model(images)
            
            # Get probabilities and the predicted class (argmax)
            probs = torch.softmax(seg_logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)

            # Store the prediction
            if slide_idx not in slide_predictions:
                # Get slide dimensions to create a full-size empty mask
                slide = openslide.OpenSlide(test_dataset.image_paths[slide_idx])
                width, height = slide.level_dimensions[test_dataset.level]
                slide_predictions[slide_idx] = {
                    'pred': np.zeros((height, width), dtype=np.uint8),
                    'true': np.zeros((height, width), dtype=np.uint8)
                }
            
            # Place the predicted patch into the full slide mask
            h, w = pred_mask.shape[1], pred_mask.shape[2]
            slide_predictions[slide_idx]['pred'][y:y+h, x:x+w] = pred_mask[0]
            
            # Also store the true mask for comparison
            true_mask_patch = batch['mask'].numpy().astype(np.uint8)
            slide_predictions[slide_idx]['true'][y:y+h, x:x+w] = true_mask_patch[0, 0]

    # --- 6. Calculate Metrics ---
    print("Stitching complete. Calculating final metrics...")
    
    all_preds = []
    all_trues = []
    for slide_idx in slide_predictions:
        pred_full = slide_predictions[slide_idx]['pred']
        true_full = slide_predictions[slide_idx]['true']
        
        # MONAI metrics expect (B, C, H, W) format, so we add batch and channel dims
        # and convert to one-hot format for the Dice metric.
        pred_tensor = torch.from_numpy(pred_full).unsqueeze(0).unsqueeze(0)
        true_tensor = torch.from_numpy(true_full).unsqueeze(0).unsqueeze(0)
        
        pred_one_hot = F.one_hot(pred_tensor.long(), num_classes=2).permute(0, 4, 1, 2, 3).squeeze(-1)
        true_one_hot = F.one_hot(true_tensor.long(), num_classes=2).permute(0, 4, 1, 2, 3).squeeze(-1)
        
        dice_metric(y_pred=pred_one_hot, y=true_one_hot)
        hd95_metric(y_pred=pred_one_hot, y=true_one_hot)

    # Aggregate the metrics
    final_dice = dice_metric.aggregate().item()
    final_hd95 = hd95_metric.aggregate().item()

    print("\n--- Evaluation Results ---")
    print(f"Mean Dice Score: {final_dice:.4f}")
    print(f"Mean 95% Hausdorff Distance: {final_hd95:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Topo-GraT model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the trained model .pth checkpoint file.")
    args = parser.parse_args()
    main(args)