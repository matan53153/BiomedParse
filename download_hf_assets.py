# download_hf_assets.py
import os
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the cache directory exists (should match HF_HOME in slurm script)
# Using os.environ.get ensures it uses the env var if set, otherwise defaults
default_cache = f"/scratch/gpfs/{os.environ.get('USER', 'km4074')}/.cache/huggingface/"
hf_cache_dir = os.environ.get("HF_HOME", default_cache)
logger.info(f"Using Hugging Face cache directory: {hf_cache_dir}")
os.makedirs(hf_cache_dir, exist_ok=True)

# Models identified from logs/code
models_to_download = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    # Add any other models your specific config might require
]

for model_name in models_to_download:
    logger.info(f"Attempting to download {model_name}...")
    try:
        # Download both tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=hf_cache_dir)
        logger.info(f"Successfully downloaded and cached {model_name}")
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}", exc_info=True)

# Verify the local CLIP model path exists (as it seems to be loaded locally in logs)
clip_path = "/scratch/gpfs/km4074/BiomedParse/pretrained_assets/openai__clip-vit-base-patch32"
logger.info(f"Checking for local CLIP model path: {clip_path}")
if not os.path.isdir(clip_path):
        logger.warning(f"Local path {clip_path} for CLIP model not found or not a directory. Please ensure it exists and contains the necessary files.")
else:
        logger.info(f"Found local path for CLIP model: {clip_path}")

logger.info("Hugging Face asset download script finished.")