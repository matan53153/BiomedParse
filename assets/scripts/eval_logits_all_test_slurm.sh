#!/bin/bash
#SBATCH --job-name=biomedparse-eval-logits-all-test # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task (mpirun handles sub-processes)
#SBATCH --cpus-per-task=16           # Request 16 CPUs total for the node (4 per GPU)
#SBATCH --mem-per-cpu=32G            # Job memory request per CPU core (total 512GB)
#SBATCH --gres=gpu:a100:4            # Request 4 A100 GPUs
#SBATCH --constraint=gpu80           # Specify 80GB VRAM A100 GPUs (optional, if needed)
#SBATCH --time=02:00:00              # Time limit hrs:min:sec (adjust if needed for dataset size)
#SBATCH --output=slurm_eval_logits_all_test_%j.log # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL   # Send email notifications
#SBATCH --mail-user=km4074@princeton.edu # Use your NetID

# --- Environment Setup ---
module purge
module load anaconda3/2024.10 # Load Anaconda module (adjust version if needed)
module load openmpi/gcc/4.1.6     # Load OpenMPI module
conda activate biomedparse   # ACTIVATE YOUR CONDA ENVIRONMENT HERE

# Set environment variables (needed for model/data loading)
export DETECTRON2_DATASETS=biomedparse_datasets/
export DATASET=biomedparse_datasets/
export DATASET2=biomedparse_datasets/
export VLDATASET=biomedparse_datasets/
export PATH=$PATH:biomedparse_datasets/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:biomedparse_datasets/coco_caption/
export HF_HOME=/scratch/gpfs/km4074/.cache/huggingface/ # Set Hugging Face cache for Della
export HF_HUB_OFFLINE=1 # Force Hugging Face offline mode
export NLTK_DATA=/scratch/gpfs/km4074/nltk_data # Point NLTK to local data
export OMPI_ALLOW_RUN_AS_ROOT=1 # Needed by mpirun in container?
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 # Needed by mpirun in container?

# --- Change to Working Directory ---
cd /scratch/gpfs/km4074/BiomedParse

echo "Starting 4-GPU logit generation job for ALL TEST datasets..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Allocated GPUs (visible to mpirun): $CUDA_VISIBLE_DEVICES"

# --- Run Logit Generation Command Directly ---
# NOTE: Removed DATASETS.TEST override. Using TEST datasets from config.
# NOTE: LOGITS_SAVE_DIR is now the base directory, subdirs created per dataset.
mpirun -n 4 --oversubscribe python entry.py evaluate \
            --conf_files configs/biomed_seg_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 4 \
            FP16 True \
            WEIGHT True \
            STANDARD_TEXT_FOR_EVAL False \
            RESUME_FROM pretrained/biomedparse_v1.pt \
            EVAL.SAVE_LOGITS True \
            EVAL.LOGITS_SAVE_DIR teacher_logits # Optional: Set base directory

echo "Logit generation finished for all test datasets."
echo "Date: $(date)" 