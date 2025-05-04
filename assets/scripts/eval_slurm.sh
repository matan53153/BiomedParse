#!/bin/bash
#SBATCH --job-name=biomedparse_eval    # Job name
#SBATCH --output=biomedparse_eval_%j.log   # Standard output and error log
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Total memory limit
#SBATCH --time=0:59:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --constraint=gpu80
#SBATCH --mail-type=BEGIN,END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=km4074@princeton.edu  # Where to send mail (adjust with your email)

# Print node information
echo "Job running on $(hostname)"
echo "Available GPUs:"
nvidia-smi

# Load required modules
module purge
module load anaconda3
module load openmpi/gcc/4.1.6

# Activate conda environment (adjust the path if needed)
source activate biomedparse

# Set environment variables
export DETECTRON2_DATASETS=biomedparse_datasets/
export DATASET=biomedparse_datasets/
export DATASET2=biomedparse_datasets/
export VLDATASET=biomedparse_datasets/
export PATH=$PATH:biomedparse_datasets/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:biomedparse_datasets/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here

# Run the evaluation
mpirun -n 1 python entry.py evaluate \
            --conf_files configs/biomed_seg_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 1 \
            FP16 True \
            WEIGHT True \
            STANDARD_TEXT_FOR_EVAL False \
            RESUME_FROM pretrained/biomedparse_v1.pt
