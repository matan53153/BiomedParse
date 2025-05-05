# download_hf_assets.py
import os
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the cache directory exists (should match HF_HOME in slurm script)
# Using os.environ.get ensures it uses the env var if set, otherwise defaults
default_cache = f"/scratch/gpfs/{os.environ.get('USER', 'km4074')}/.cache/huggingface/"
hf_cache_dir = os.environ.get("HF_HOME", default_cache)
logger.info(f"Using Hugging Face cache directory: {hf_cache_dir}")
os.makedirs(hf_cache_dir, exist_ok=True)

# Models identified from logs/code to download to main cache
models_to_download = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    # Add any other models your specific config might require
]

for model_name in models_to_download:
    logger.info(f"Attempting to download {model_name} to {hf_cache_dir}...")
    try:
        # Download both tokenizer and model to main cache
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=hf_cache_dir)
        logger.info(f"Successfully downloaded and cached {model_name} to {hf_cache_dir}")
    except Exception as e:
        logger.error(f"Error downloading {model_name} to {hf_cache_dir}: {e}", exc_info=True)

# --- Download CLIP model and tokenizer to specific local path --- 
clip_assets_dir = "/scratch/gpfs/km4074/BiomedParse/pretrained_assets"
clip_model_id = "openai/clip-vit-base-patch32"
logger.info(f"Ensuring CLIP model and tokenizer ({clip_model_id}) are downloaded to {clip_assets_dir}...")
os.makedirs(clip_assets_dir, exist_ok=True)

try:
    logger.info(f"Downloading/verifying CLIP Tokenizer...")
    CLIPTokenizer.from_pretrained(clip_model_id, cache_dir=clip_assets_dir)
    logger.info(f"CLIP Tokenizer OK.")

    logger.info(f"Downloading/verifying CLIP Model...")
    CLIPModel.from_pretrained(clip_model_id, cache_dir=clip_assets_dir)
    logger.info(f"CLIP Model OK.")
    
    logger.info(f"Successfully ensured CLIP assets are present in {clip_assets_dir}")
except Exception as e:
    logger.error(f"Error ensuring CLIP assets in {clip_assets_dir}: {e}", exc_info=True)

logger.info("Hugging Face asset download script finished.")