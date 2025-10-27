# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

class TopoGraTLoss(nn.Module):
    """
    A composite, multi-task loss function for training Topo-GraT.
    This class combines segmentation, topology preservation, and instance selection losses.
    """
    def __init__(self, alpha=0.1, beta=0.05):
        """
        Initializes the Topo-GraT loss function.
        Args:
            alpha (float): The weight for the topology preservation loss (L_topo).
            beta (float): The weight for the instance selection loss (L_instance).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # 1. Segmentation Loss (L_seg)
        # A combination of Dice and Cross-Entropy, which is robust for medical imaging.
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.ce_loss = nn.CrossEntropyLoss()

        # 2. Instance Selection Loss (L_instance)
        # A standard Binary Cross-Entropy for the patch selection task in Stage 1.
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        Calculates the total loss.
        Args:
            outputs (dict): A dictionary containing the model's predictions.
                            Expected keys: 'seg_logits', 'cgf_pred', 'patch_probs'.
            targets (dict): A dictionary containing the ground truth.
                            Expected keys: 'seg_mask', 'cgf_true', 'patch_labels'.
        Returns:
            torch.Tensor: The total computed loss.
            dict: A dictionary containing the individual loss components for logging.
        """
        
        # --- 1. Segmentation Loss ---
        seg_logits = outputs['seg_logits']
        seg_mask = targets['seg_mask']
        loss_dice = self.dice_loss(seg_logits, seg_mask)
        loss_ce = self.ce_loss(seg_logits, seg_mask.squeeze(1).long())
        loss_seg = loss_dice + loss_ce

        # --- 2. Topology Preservation Loss (L_topo) ---
        cgf_pred = outputs['cgf_pred']
        cgf_true = targets['cgf_true']
        
        # We only compute the loss on the foreground pixels of the tumor mask
        # to ensure the model learns meaningful growth patterns.
        foreground_mask = (seg_mask > 0).float()
        
        # Cosine Similarity Loss: 1 - cos(theta)
        # F.normalize ensures the vectors are unit vectors before the dot product.
        cos_sim = F.cosine_similarity(cgf_pred, cgf_true, dim=1)
        
        # Apply the mask and average over the foreground pixels
        masked_cos_sim = cos_sim * foreground_mask.squeeze(1)
        
        # Ensure we don't divide by zero if there are no foreground pixels
        if foreground_mask.sum() > 0:
            loss_topo = 1 - (masked_cos_sim.sum() / foreground_mask.sum())
        else:
            loss_topo = torch.tensor(0.0, device=seg_logits.device)

        # --- 3. Instance Selection Loss (L_instance) ---
        patch_probs = outputs['patch_probs']
        patch_labels = targets['patch_labels']
        loss_instance = self.bce_loss(patch_probs, patch_labels)

        # --- 4. Total Composite Loss ---
        total_loss = loss_seg + (self.alpha * loss_topo) + (self.beta * loss_instance)

        # For logging and debugging
        loss_components = {
            'total_loss': total_loss.item(),
            'seg_loss': loss_seg.item(),
            'topo_loss': loss_topo.item(),
            'instance_loss': loss_instance.item()
        }

        return total_loss, loss_components