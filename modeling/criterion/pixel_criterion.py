# modeling/criterion/pixel_criterion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class PixelCriterion(nn.Module):
    """
    Simple criterion for pixel-wise segmentation loss (e.g., Cross Entropy).
    """
    def __init__(self, weight_dict, losses, ignore_index=255):
        """
        Args:
            weight_dict (dict): A dictionary containing weights for different loss types (e.g., {'loss_sem_seg_ce': 1.0}).
            losses (list): A list of loss types to compute (e.g., ['sem_seg_ce']).
            ignore_index (int): Specifies a target value that is ignored.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.ignore_index = ignore_index
        logger.info(f"Initialized PixelCriterion with losses: {self.losses}, weights: {self.weight_dict}")


    def loss_sem_seg_ce(self, outputs, targets, **kwargs):
        """
        Computes Cross Entropy loss for semantic segmentation.
        outputs: dict containing 'sem_seg_logits' (B, C, H, W)
        targets: dict containing 'masks' (B, H, W) - the ground truth masks
        """
        if "sem_seg_logits" not in outputs:
            logger.warning("PixelCriterion: 'sem_seg_logits' not found in outputs.")
            # Return zero loss on the correct device if possible, otherwise CPU
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return {'loss_sem_seg_ce': torch.tensor(0.0, device=device)} 

        if "masks" not in targets or targets["masks"] is None:
             logger.warning("PixelCriterion: 'masks' not found in targets or is None.")
             return {'loss_sem_seg_ce': torch.tensor(0.0, device=outputs["sem_seg_logits"].device)}

        logits = outputs["sem_seg_logits"]
        gt_masks = targets["masks"].long() # Ensure masks are LongTensor for CE

        # Check shapes
        if logits.shape[0] != gt_masks.shape[0]:
            logger.warning(f"Batch size mismatch! Logits: {logits.shape[0]}, Masks: {gt_masks.shape[0]}. Skipping loss.")
            return {'loss_sem_seg_ce': torch.tensor(0.0, device=logits.device)}
            
        # Handle spatial mismatch between logits and masks (e.g., due to different strides)
        if logits.shape[-2:] != gt_masks.shape[-2:]:
            logger.warning(f"Spatial shape mismatch! Logits: {logits.shape}, Masks: {gt_masks.shape}. Attempting target resize.")
            # Reshape gt_masks to match logits spatial dimensions. Input NCHW expected.
            gt_masks = F.interpolate(gt_masks.float(), size=logits.shape[2:], mode='nearest').long()

        # Ensure mask is on the same device as logits
        gt_masks = gt_masks.to(logits.device)
        
        # Squeeze the channel dimension (dim=1) from gt_masks for cross_entropy
        loss = F.cross_entropy(logits, gt_masks.squeeze(1), ignore_index=self.ignore_index, reduction='mean')
        losses = {"loss_sem_seg_ce": loss}
        return losses

    def forward(self, outputs, targets):
        """
        Calculates the specified losses.
        """
        losses = {}
        for loss_type in self.losses:
            # Construct method name (e.g., loss_sem_seg_ce)
            method_name = f"loss_{loss_type}"
            if hasattr(self, method_name):
                loss_func = getattr(self, method_name)
                # Pass only outputs and targets, ignore potential extra args in signature if not needed
                loss_result = loss_func(outputs, targets)
                if isinstance(loss_result, dict):
                     losses.update(loss_result)
                else:
                     logger.error(f"Loss function {method_name} did not return a dictionary.")
            else:
                logger.warning(f"Loss type '{loss_type}' (method {method_name}) not implemented in PixelCriterion.")

        # Apply weights
        weighted_losses = {}
        for k in losses.keys():
            if k in self.weight_dict:
                weighted_losses[k] = losses[k] * self.weight_dict[k]
            else:
                 logger.warning(f"Weight not found for loss '{k}' in weight_dict. Using weight 1.0.")
                 weighted_losses[k] = losses[k] # Keep the original loss if weight is missing
                 
        # Return the dict of weighted losses
        return weighted_losses 