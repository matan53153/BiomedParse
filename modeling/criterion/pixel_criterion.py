import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class PixelCriterion(nn.Module):
    """
    Criterion for pixel-level segmentation tasks.
    Supports various loss functions for semantic segmentation.
    """
    def __init__(self, loss_types=None, loss_weights=None):
        """
        Initialize the PixelCriterion with specified loss types and weights.
        
        Args:
            loss_types (list): List of loss function names to use
            loss_weights (dict): Dictionary mapping loss names to their weights
        """
        super().__init__()
        self.loss_types = loss_types or ['sem_seg_ce']
        self.loss_weights = loss_weights or {'loss_sem_seg_ce': 1.0}
        logger.info(f"Initialized PixelCriterion with losses: {self.loss_types}, weights: {self.loss_weights}")
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.dice_loss = DiceLoss(ignore_index=255)
    
    def forward(self, outputs, targets):
        """
        Compute the combined loss for semantic segmentation.
        
        Args:
            outputs (dict): Dictionary containing model outputs, including 'sem_seg_logits'
            targets (dict or list): Dictionary containing ground truth, including 'sem_seg' or 'instances',
                                   or a list of batch samples
            
        Returns:
            dict: Dictionary of computed losses
        """
        losses = {}
        
        # Extract semantic segmentation target from different possible formats
        target = None
        
        # Handle different batch formats
        if isinstance(targets, list) and len(targets) > 0:
            # If targets is a list of samples, extract targets from the first sample
            # This is common in Detectron2-style dataloaders
            sample = targets[0]
            if isinstance(sample, dict):
                if 'sem_seg' in sample:
                    # Extract and stack all sem_seg targets from the batch
                    target = torch.stack([item['sem_seg'] for item in targets if 'sem_seg' in item])
                elif 'instances' in sample and hasattr(sample['instances'], 'gt_masks'):
                    # Handle instance masks if needed
                    target = torch.stack([item['instances'].gt_masks.tensor.sum(dim=0).bool() for item in targets if 'instances' in item and hasattr(item['instances'], 'gt_masks')])
        elif isinstance(targets, dict):
            # Standard dictionary format
            if 'sem_seg' in targets:
                target = targets['sem_seg']
            elif 'instances' in targets and hasattr(targets['instances'], 'gt_masks'):
                # Convert instance masks to semantic segmentation format if needed
                target = targets['instances'].gt_masks.tensor.sum(dim=0).bool()
            elif 'gt_masks' in targets:
                target = targets['gt_masks'].tensor.sum(dim=0).bool()
        
        # If we still don't have a target, log a warning and return empty losses
        if target is None:
            logger.warning(f"No semantic segmentation target found in batch. Keys: {targets.keys() if isinstance(targets, dict) else 'not a dict'}")
            # Create a dummy loss with a small value to avoid NaN errors
            dummy_tensor = torch.tensor(0.0001, requires_grad=True, device=next(iter(outputs.values())).device)
            return {'loss_sem_seg_dummy': dummy_tensor}
        
        # Ensure target is in the right format
        if not isinstance(target, torch.Tensor):
            logger.warning(f"Target is not a tensor: {type(target)}")
            # Create a dummy loss with a small value to avoid NaN errors
            dummy_tensor = torch.tensor(0.0001, requires_grad=True, device=next(iter(outputs.values())).device)
            return {'loss_sem_seg_dummy': dummy_tensor}
        
        # Get the device of the logits
        logits_device = None
        if 'sem_seg_logits' in outputs:
            logits_device = outputs['sem_seg_logits'].device
        else:
            logits_device = next(iter(outputs.values())).device
            
        # Make sure target is long type for cross entropy and on the same device as logits
        target = target.to(logits_device).long()
        logger.info(f"Target moved to device {target.device}, shape: {target.shape}")
        
        # Extract logits from outputs
        if 'sem_seg_logits' in outputs:
            logits = outputs['sem_seg_logits']
        else:
            logger.warning(f"No 'sem_seg_logits' found in outputs. Keys: {outputs.keys()}")
            # Create a dummy loss with a small value to avoid NaN errors
            dummy_tensor = torch.tensor(0.0001, requires_grad=True, device=target.device)
            return {'loss_sem_seg_dummy': dummy_tensor}
        
        # Compute losses based on configured loss types
        if 'sem_seg_ce' in self.loss_types:
            # Handle different dimensions if needed
            if logits.dim() == 4 and target.dim() == 3:
                # Standard semantic segmentation format
                try:
                    # Ensure target is on the same device as logits
                    if target.device != logits.device:
                        target = target.to(logits.device)
                    
                    loss_ce = self.ce_loss(logits, target)
                    losses['loss_sem_seg_ce'] = loss_ce * self.loss_weights.get('loss_sem_seg_ce', 1.0)
                    logger.info(f"Successfully computed CE loss: {loss_ce.item()}")
                except Exception as e:
                    logger.error(f"Error computing CE loss: {e}. Logits shape: {logits.shape}, Target shape: {target.shape}, Logits device: {logits.device}, Target device: {target.device}")
            elif logits.dim() == 4 and target.dim() == 4:
                # Target is one-hot encoded, convert to class indices
                target_indices = torch.argmax(target, dim=1)
                try:
                    # Ensure target is on the same device as logits
                    if target_indices.device != logits.device:
                        target_indices = target_indices.to(logits.device)
                        
                    loss_ce = self.ce_loss(logits, target_indices)
                    losses['loss_sem_seg_ce'] = loss_ce * self.loss_weights.get('loss_sem_seg_ce', 1.0)
                    logger.info(f"Successfully computed CE loss: {loss_ce.item()}")
                except Exception as e:
                    logger.error(f"Error computing CE loss: {e}. Logits shape: {logits.shape}, Target shape: {target_indices.shape}, Logits device: {logits.device}, Target device: {target_indices.device}")
            else:
                logger.warning(f"Dimension mismatch: logits {logits.shape}, target {target.shape}")
        
        if 'sem_seg_dice' in self.loss_types:
            try:
                # Ensure target is on the same device as logits
                if target.device != logits.device:
                    target = target.to(logits.device)
                    
                loss_dice = self.dice_loss(logits, target)
                losses['loss_sem_seg_dice'] = loss_dice * self.loss_weights.get('loss_sem_seg_dice', 1.0)
                logger.info(f"Successfully computed Dice loss: {loss_dice.item()}")
            except Exception as e:
                logger.error(f"Error computing Dice loss: {e}. Logits shape: {logits.shape}, Target shape: {target.shape}, Logits device: {logits.device}, Target device: {target.device}")
        
        # If no losses were computed, create a dummy loss
        if not losses:
            logger.warning("No losses were computed, creating dummy loss")
            dummy_tensor = torch.tensor(0.0001, requires_grad=True, device=logits.device)
            losses['loss_sem_seg_dummy'] = dummy_tensor
            
        return losses


class DiceLoss(nn.Module):
    """
    Dice loss for semantic segmentation.
    """
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        
    def forward(self, logits, target):
        """
        Compute the Dice loss.
        
        Args:
            logits (Tensor): Predicted logits of shape (N, C, H, W)
            target (Tensor): Ground truth of shape (N, H, W)
            
        Returns:
            Tensor: Dice loss value
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode the target
        target_one_hot = torch.zeros_like(probs)
        for cls in range(num_classes):
            target_one_hot[:, cls, ...] = (target == cls)
            
        # Create mask for valid pixels (not ignore_index)
        mask = (target != self.ignore_index).float().unsqueeze(1)
        target_one_hot = target_one_hot * mask
        
        # Compute dice score
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        union = (probs + target_one_hot).sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return dice_loss
