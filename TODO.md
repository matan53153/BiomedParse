# BiomedParse Todo

## Environment Setup
- [x] Set up computing environment on della cluster
- [x] Install required libraries and dependencies (PyTorch, Hugging Face, etc.)
- [x] Configure GPU acceleration and memory allocation
- [x] Create project directory structure
- [x] Set up version control

## Data Acquisition and Preparation
- [ ] Download BiomedParse dataset from Hugging Face
- [ ] Download BiomedParse model from Hugging Face
- [ ] Create data loaders for the dataset
- [ ] Split data into training, validation, and test sets (if not already done)
- [ ] Verify data loading and preprocessing pipeline

## BiomedParse Inference
- [ ] Implement inference script for BiomedParse
- [ ] Run inference on test set to generate teacher logits
- [ ] Measure and document inference time and resource usage
- [ ] Store teacher model outputs for distillation

## Student Model Implementation
- [ ] Implement ViT-Base student architecture
- [ ] Implement MobileNet student architecture
- [ ] Verify student models can process input data correctly
- [ ] Confirm student models produce compatible output formats
- [ ] Benchmark initial student model sizes and resource requirements

## Distillation Implementation
- [ ] Implement KL divergence distillation loss
- [ ] Implement supervised loss calculation
- [ ] Create combined loss function with weighting parameter α
- [ ] Implement temperature scaling for softening probability distributions
- [ ] Set up optimization strategy (learning rate, scheduler, etc.)

## Training Pipeline
- [ ] Implement training loop with both distillation and supervised losses
- [ ] Add logging and checkpointing functionality
- [ ] Set up baseline training (supervised only, no distillation)
- [ ] Implement early stopping based on validation performance
- [ ] Create hyperparameter tuning process

## Training Execution
- [ ] Train ViT-Base with distillation
- [ ] Train ViT-Base with supervision only (baseline)
- [ ] Train MobileNet with distillation
- [ ] Train MobileNet with supervision only (baseline)
- [ ] Save model checkpoints for best performing configurations

## Evaluation
- [ ] Implement evaluation metrics (mIoU for segmentation)
- [ ] Implement evaluation metrics (mAP for detection)
- [ ] Implement evaluation metrics (accuracy and F1 for classification)
- [ ] Run evaluations on test set for all trained models
- [ ] Measure inference time and resource usage for all models

## Analysis
- [ ] Compare performance of student models to BiomedParse
- [ ] Compare performance across different imaging modalities
- [ ] Analyze performance vs. model size tradeoffs
- [ ] Evaluate impact of distillation vs. supervised-only training
- [ ] Create visualizations for results

## Optional Extensions (if time permits)
- [ ] Implement additional student architectures
- [ ] Experiment with DINO and iBOT distillation techniques
- [ ] Investigate distilling pre-trained components
- [ ] Test models on edge devices or simulated low-resource environments
- [ ] Explore different temperature settings in distillation

## Documentation and Publication
- [ ] Document codebase with clear comments and README
- [ ] Write up methodology and implementation details
- [ ] Create visualizations for performance comparisons
- [ ] Prepare final results for the paper
- [ ] Draft conclusions based on experimental results

