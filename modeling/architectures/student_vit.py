# modeling/architectures/student_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import timm # Use timm for ViT models
import torchvision.models as models # <-- Add torchvision import

from ..utils import configurable
from .build import register_model
from ..BaseModel import BaseModel

logger = logging.getLogger(__name__)

class SimpleMLPHead(nn.Module):
    """
    Simple MLP head for ViT segmentation.
    Takes patch embeddings, applies MLP, reshapes, and upsamples.
    """
    def __init__(self, in_channels, num_classes, embed_dim=768, output_embed_dim=256):
        super().__init__()
        # Example MLP structure
        self.proj_1 = nn.Linear(in_channels, embed_dim)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.activation_1 = nn.GELU()
        self.proj_2 = nn.Linear(embed_dim, output_embed_dim)
        self.ln_2 = nn.LayerNorm(output_embed_dim)
        self.activation_2 = nn.GELU()
        self.proj_final = nn.Linear(output_embed_dim, num_classes)

        self.num_classes = num_classes

    def forward(self, features):
        # Features expected shape: [Batch, NumPatches + 1, InChannels] (incl. class token)
        # Remove class token if present
        if features.shape[1] > 0: # Check if there are tokens
             features = features[:, 1:, :] # Assume class token is first

        # Apply MLP layers
        x = self.proj_1(features)
        x = self.ln_1(x)
        x = self.activation_1(x)
        x = self.proj_2(x)
        x = self.ln_2(x)
        x = self.activation_2(x)
        x = self.proj_final(x) # [B, NumPatches, NumClasses]

        # Reshape to spatial format
        # Calculate H, W from NumPatches (assuming square patch grid)
        num_patches = x.shape[1]
        h_patches = w_patches = int(num_patches**0.5)
        if h_patches * w_patches != num_patches:
             raise ValueError("Cannot infer square patch grid from num_patches.")

        # [B, NumPatches, NumClasses] -> [B, NumClasses, NumPatches] -> [B, NumClasses, H_patches, W_patches]
        x = x.permute(0, 2, 1).reshape(-1, self.num_classes, h_patches, w_patches)

        return x


class StudentViTSegmentation(nn.Module):
    """
    Student model using a ViT backbone and a simple MLP head.
    Produces per-pixel segmentation logits.
    """
    @configurable
    def __init__(self, *, backbone: nn.Module, head: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes

        # Expected patch size (common for ViT-Base) - needed for upsampling factor
        self.patch_size = 16

    @classmethod
    def from_config(cls, cfg):
        student_cfg = cfg['MODEL']['STUDENT']
        backbone_name = student_cfg.get('BACKBONE_NAME', 'vit_base_patch16_224')
        pretrained = student_cfg.get('PRETRAINED', True)
        num_classes = student_cfg.get('NUM_CLASSES', 16)

        # Use torchvision to load ViT and utilize the pre-downloaded cache
        if backbone_name == 'vit_base_patch16_224': # Match the name we set in YAML
            vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vit_b_16(weights=vit_weights)
            # Remove the classification head from torchvision ViT
            backbone.heads.head = nn.Identity()
            embed_dim = backbone.hidden_dim # Get embed dim from torchvision ViT

            # Interpolate positional embeddings if image size differs
            cfg_image_size = cfg.get('BioMed', {}).get('INPUT', {}).get('IMAGE_SIZE', 224)
            model_image_size = backbone.image_size
            patch_size = backbone.patch_size
            if cfg_image_size != model_image_size:
                logger.info(f"Interpolating ViT positional embeddings from {model_image_size}x{model_image_size} to {cfg_image_size}x{cfg_image_size}")
                pos_embedding = backbone.encoder.pos_embedding
                pos_embedding_cls = pos_embedding[:, :1, :]
                pos_embedding_patches = pos_embedding[:, 1:, :]
                
                orig_num_patches_side = model_image_size // patch_size
                new_num_patches_side = cfg_image_size // patch_size
                
                pos_embedding_patches = pos_embedding_patches.reshape(1, orig_num_patches_side, orig_num_patches_side, embed_dim).permute(0, 3, 1, 2)
                pos_embedding_patches = F.interpolate(
                    pos_embedding_patches,
                    size=(new_num_patches_side, new_num_patches_side),
                    mode='bicubic',
                    align_corners=False,
                )
                pos_embedding_patches = pos_embedding_patches.permute(0, 2, 3, 1).flatten(1, 2)
                backbone.encoder.pos_embedding = nn.Parameter(torch.cat([pos_embedding_cls, pos_embedding_patches], dim=1))

        else:
            raise ValueError(f"Unsupported ViT backbone name for torchvision: {backbone_name}")
        
        # Assume a simple MLP head for ViT segmentation (modify if needed)
        # You might need a more sophisticated head like a UPerNet or SegFormer head
        # depending on how you want to get spatial resolution back.
        # For now, using a simple MLP that predicts per-patch.
        head = SimpleMLPHead(in_channels=embed_dim, num_classes=num_classes)

        return {
            "backbone": backbone,
            "head": head,
            "num_classes": num_classes,
        }

    def forward(self, batched_inputs, mode='default'):
        if isinstance(batched_inputs, list):
            # Use next(self.parameters()) to reliably get device for the whole module
            device = next(self.parameters()).device
            images = [x["image"].to(device) for x in batched_inputs]
            # TODO: Add ViT-specific normalization
            input_tensor = torch.stack(images, dim=0)
            input_tensor = input_tensor.float() # <-- Convert to float
        elif isinstance(batched_inputs, torch.Tensor):
            device = next(self.parameters()).device
            input_tensor = batched_inputs.to(device)
            input_tensor = input_tensor.float() # <-- Convert to float
        else:
            raise TypeError(f"Unsupported input type: {type(batched_inputs)}")

        input_shape = input_tensor.shape[-2:]

        # Forward through ViT backbone
        # Replicate necessary steps from torchvision's _process_input without the assertion
        # 1. Patch embedding
        x = self.backbone.conv_proj(input_tensor) # Shape: [B, HiddenDim, H_patches, W_patches]
        x = x.flatten(2).transpose(1, 2)          # Shape: [B, NumPatches, HiddenDim]
        
        # 2. Prepend class token
        # Shape: [1, 1, HiddenDim] -> [B, 1, HiddenDim]
        batch_class_token = self.backbone.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) # Shape: [B, 1 + NumPatches, HiddenDim]
        
        # 3. Add positional embedding and pass through encoder
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        x = self.backbone.encoder.layers(x)
        features = self.backbone.encoder.ln(x) 

        # Forward through segmentation head
        # Output shape [B, NumClasses, H_patches, W_patches]
        logits_low_res = self.head(features)

        # Upsample logits to original input size
        logits = F.interpolate(
            logits_low_res,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )

        # During training, return dictionary for criterion
        if self.training:
            return {"pred_logits": logits}
        else:
            # During evaluation, always return the raw logits tensor
            # Log a warning if the mode isn't one explicitly handled by SemSegEvaluator, but return tensor anyway.
            if mode not in ['sem_seg', 'default_eval']:
                 logger = logging.getLogger(__name__)
                 logger.warning(f"StudentViT model received unexpected evaluation mode: '{mode}'. Returning logits tensor directly.")
            return logits


@register_model
def get_student_vit_segmentation(cfg, **kwargs):
    """
    Factory function to build the StudentViTSegmentation model.
    Wraps the core model in BaseModel for consistency.
    """
    if 'STUDENT' not in cfg['MODEL']:
         logger.warning("MODEL.STUDENT config not found. Adding default student config for ViT.")
         cfg['MODEL']['STUDENT'] = {
             'BACKBONE_NAME': 'vit_base_patch16_224',
             'PRETRAINED': True,
             'NUM_CLASSES': 16
         }

    core_model = StudentViTSegmentation(cfg)
    model = BaseModel(cfg, core_model)
    return model 