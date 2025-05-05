#!/usr/bin/env python
"""
Script to download ResNet18 pretrained weights for offline use.
This will download the weights to the default PyTorch cache location.
"""
import torch
import torchvision.models as models
import os

# Print the cache directory where models will be saved
cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
print(f"PyTorch will download models to: {cache_dir}")
print(f"Creating directory if it doesn't exist...")
os.makedirs(cache_dir, exist_ok=True)

# Download ResNet18 pretrained model
print("Downloading ResNet18 pretrained model...")
model = models.resnet18(pretrained=True)
print("ResNet18 model downloaded successfully!")

# Verify the file exists
resnet18_path = os.path.join(cache_dir, "resnet18-f37072fd.pth")
if os.path.exists(resnet18_path):
    print(f"Verified: {resnet18_path} exists")
    print(f"File size: {os.path.getsize(resnet18_path) / (1024*1024):.2f} MB")
else:
    print(f"WARNING: Expected file not found at {resnet18_path}")
    print("Available files in cache directory:")
    for file in os.listdir(cache_dir):
        print(f"  - {file}")
