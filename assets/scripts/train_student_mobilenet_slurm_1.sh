#!/bin/bash
#SBATCH --job-name=biomedparse-train-mobilenet # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task (mpirun handles sub-processes)
#SBATCH --cpus-per-task=4            # Request 4 CPUs total for the node (4 per GPU)
#SBATCH --mem-per-cpu=16G            # Job memory request per CPU core (total 64GB)
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU total for the job
#SBATCH --constraint=gpu80           # Specify 80GB VRAM A100 GPUs (optional, if needed)
#SBATCH --time=00:59:00              # Adjusted time limit hrs:min:sec
#SBATCH --output=slurm_train_mobilenet_%j.log  # Standard output and error log (%j expands to jobId)
#SBATCH --mail-type=BEGIN,END,FAIL   # Send email notifications
#SBATCH --mail-user=km4074@princeton.edu # Use your NetID

# --- Environment Setup ---
module purge
module load anaconda3/2024.10 # Load Anaconda module (adjust version if needed)
module load openmpi/gcc/4.1.6     # Load correct OpenMPI module for mpirun
conda activate biomedparse   # ACTIVATE YOUR CONDA ENVIRONMENT HERE

# Set environment variables (OMPI vars are now in train.sh)
export DETECTRON2_DATASETS=biomedparse_datasets/
export DATASET=biomedparse_datasets/
export DATASET2=biomedparse_datasets/
export VLDATASET=biomedparse_datasets/
export PATH=$PATH:biomedparse_datasets/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:biomedparse_datasets/coco_caption/
export HF_HOME=/scratch/gpfs/km4074/.cache/huggingface/ # Set Hugging Face cache for Della
export HF_HUB_OFFLINE=1 # Force Hugging Face offline mode
export NLTK_DATA=/scratch/gpfs/km4074/nltk_data # Point NLTK to local data
export OMPI_ALLOW_RUN_AS_ROOT=1 # Might be needed by mpirun
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 # Might be needed by mpirun


# --- Change to Working Directory ---
cd /scratch/gpfs/km4074/BiomedParse

echo "Starting 1-GPU SUPERVISED training job for MobileNet Student..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Allocated GPUs (visible to mpirun): $CUDA_VISIBLE_DEVICES"

# --- Run Training Command Directly --- 
# Note: Using student_mobilenet config
CUDA_VISIBLE_DEVICES=0 python entry.py train \
            --conf_files configs/student_mobilenet.yaml \
            --overrides \
            RANDOM_SEED 2024 \
            BioMed.INPUT.IMAGE_SIZE 1024 \
            TRAIN.BATCH_SIZE_PER_GPU 56 \
            SOLVER.MAX_NUM_EPOCHS 20 \
            FP16 False # Disable FP16 to avoid gradient scaler issues
            # Add/remove other relevant overrides from original train.sh if needed

echo "Training finished."
echo "Date: $(date)" 