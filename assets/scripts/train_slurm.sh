#!/bin/bash
#SBATCH --job-name=biomedparse-train-4gpu-mpirun # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task (mpirun handles sub-processes)
#SBATCH --cpus-per-task=16           # Request 16 CPUs total for the node (4 per GPU)
#SBATCH --mem-per-cpu=32G            # Job memory request per CPU core (total 512GB)
#SBATCH --gres=gpu:a100:4            # Request 4 A100 GPUs total for the job
#SBATCH --constraint=gpu80           # Specify 80GB VRAM A100 GPUs
#SBATCH --time=00:59:00              # Increased time limit hrs:min:sec
#SBATCH --output=slurm_train_%j.log  # Standard output and error log (%j expands to jobId)
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

# --- Distributed Training Env Vars (Not needed for mpirun) ---
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "MASTER_PORT: $MASTER_PORT"

# --- Change to Working Directory ---
cd /scratch/gpfs/km4074/BiomedParse

echo "Starting mpirun-launched 4-GPU training job..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Allocated GPUs (visible to mpirun): $CUDA_VISIBLE_DEVICES"

# --- Run Training Script (which contains mpirun) ---
bash assets/scripts/train.sh

echo "Training finished."
echo "Date: $(date)" 