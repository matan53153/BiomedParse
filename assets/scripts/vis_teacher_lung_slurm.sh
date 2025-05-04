#!/bin/bash
#SBATCH --job-name=biomedparse-vis-lung # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 # Reduced CPUs
#SBATCH --mem-per-cpu=16G # Reduced memory
#SBATCH --gres=gpu:a100:1 # Reduced to 1 GPU for evaluation
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00      # Reduced time limit
#SBATCH --output=slurm_vis_lung_teacher_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=km4074@princeton.edu # Use your NetID

# --- Environment Setup ---
module purge
module load anaconda3/2024.10
module load openmpi/gcc/4.1.6 # Still needed if entry.py uses MPI internally? Safer to keep.
conda activate biomedparse   # ACTIVATE YOUR CONDA ENVIRONMENT HERE

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

echo "Starting 1-GPU visualization generation job for TEACHER model on Lung Test..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# --- Run Evaluation Command ---
# Use mpirun -n 1 as the original script used it
# Use --conf_files and --overrides matching the eval_logits script style
mpirun -n 1 --oversubscribe python entry.py evaluate \
    --conf_files configs/biomed_seg_lang_v1.yaml \
    --overrides \
    RESUME True \
    WEIGHT True \
    MODEL.DECODER.HIDDEN_DIM 512 \
    MODEL.ENCODER.CONVS_DIM 512 \
    MODEL.ENCODER.MASK_DIM 512 \
    TEST.BATCH_SIZE_TOTAL 4 \
    FP16 True \
    WEIGHT True \
    STANDARD_TEXT_FOR_EVAL False \
    RESUME_FROM pretrained/biomedparse_v1.pt \
    EVAL_AT_START True \
    SAVE_DIR './output/teacher_lung_vis_eval' # Separate output dir

echo "Evaluation for visualization finished."
