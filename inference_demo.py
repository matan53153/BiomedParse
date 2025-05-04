#!/usr/bin/env python3
"""
Inference demo script for BiomedParse model
"""

# Import necessary libraries
import os
import torch
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# Import project modules
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

def main():
    # Reduce memory usage
    torch.cuda.empty_cache()
    
    # Set environment variables to limit resource usage
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Load configuration
    print("Loading configuration...")
    conf_files = "configs/biomedparse_inference.yaml"
    opt = load_opt_from_config_files([conf_files])
    opt = init_distributed(opt)
    
    # Limit DataLoader workers
    if 'DATALOADER' not in opt:
        opt['DATALOADER'] = {}
    opt['DATALOADER']['NUM_WORKERS'] = 1
    
    # Load model with careful memory management
    print("Loading model...")
    model_file = "pretrained/biomedparse_v1.pt"
    
    try:
        # Load model with reduced precision to save memory
        with torch.cuda.amp.autocast():
            model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
            
        # Pre-compute text embeddings
        print("Computing text embeddings...")
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"], is_eval=True
            )
        
        # Load and process image
        print("Loading image...")
        image_path = 'examples/Part_1_516_pathology_breast.png'
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            
            # Define text prompts
            text_prompts = ["tumor"]
            
            # Run inference
            print(f"Running inference with prompts: {text_prompts}")
            with torch.no_grad():
                masks = interactive_infer_image(model, image, text_prompts)
            
            # Display results
            print("Inference complete. Results:")
            for i, (prompt, mask) in enumerate(zip(text_prompts, masks)):
                print(f"Prompt: {prompt}, Mask shape: {mask.shape}, Positive pixels: {mask.sum()}")
                
            print("Done!")
        else:
            print(f"Image not found: {image_path}")
            print("Please provide a valid image path.")
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
