#!/bin/bash
#SBATCH --job-name=resnet18_std     # updated job name
#SBATCH --output=logs/resnet18_std_train_%j.log # updated log file name
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --mem=64G                # memory per cpu-core (4G is default)
#SBATCH --constraint=gpu80           # Specify 80GB VRAM A100 GPUs (optional, if needed)
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=km4074@princeton.edu  # Where to send mail (adjust with your email)

echo "Starting 4-GPU RESNET18 STD training job..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Allocated GPUs (visible to mpirun): $CUDA_VISIBLE_DEVICES"

# Load necessary modules
module purge
module load anaconda3/2023.9
module load openmpi/gcc/4.1.6

# Activate Conda environment
conda activate biomedparse

# Define the config file for this training run
CONFIG_FILE="configs/student_resnet18.yaml"
export DETECTRON2_DATASETS=biomedparse_datasets/
export DATASET=biomedparse_datasets/
export DATASET2=biomedparse_datasets/
export VLDATASET=biomedparse_datasets/
export PATH=$PATH:biomedparse_datasets/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:biomedparse_datasets/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NLTK_DATA=/scratch/gpfs/km4074/nltk_data # Point NLTK to local data

# Run the training script
# Using mpirun for consistency, even with 1 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 4 --oversubscribe -x NLTK_DATA python entry.py train \
    --conf_files $CONFIG_FILE \
    --overrides \
    TRAIN.BATCH_SIZE_PER_GPU 8 \
    FP16 True \
    RANDOM_SEED 2024 \
    BioMed.INPUT.IMAGE_SIZE 1024 \
    SOLVER.MAX_NUM_EPOCHS 20 \
    SOLVER.BASE_LR 0.00001 \
    SOLVER.FIX_PARAM.backbone False \
    SOLVER.FIX_PARAM.lang_encoder False \
    SOLVER.FIX_PARAM.pixel_decoder False \
    MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 1.0 \
    MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 1.0 \
    MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 1.0 \
    MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
    MODEL.DECODER.SPATIAL.ENABLED True \
    MODEL.DECODER.GROUNDING.ENABLED True \
    LOADER.SAMPLE_PROB prop \
    BioMed.INPUT.RANDOM_ROTATE True \
    FIND_UNUSED_PARAMETERS True \
    ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
    MODEL.DECODER.SPATIAL.MAX_ITER 0 \
    ATTENTION_ARCH.QUERY_NUMBER 3 \
    STROKE_SAMPLER.MAX_CANDIDATE 10 \
    
echo "Training finished."
echo "Date: $(date)"
echo "Exit code: $?"
