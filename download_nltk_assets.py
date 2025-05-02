# download_nltk_assets.py
import nltk
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a download directory on scratch space
nltk_data_dir = f"/scratch/gpfs/{os.environ.get('USER', 'km4074')}/nltk_data"
logger.info(f"Using NLTK data directory: {nltk_data_dir}")
os.makedirs(nltk_data_dir, exist_ok=True)

# Assets identified from logs
nltk_assets = ['punkt', 'averaged_perceptron_tagger']

for asset in nltk_assets:
    logger.info(f"Attempting to download NLTK '{asset}' to {nltk_data_dir}...")
    try:
        nltk.download(asset, download_dir=nltk_data_dir)
        logger.info(f"Successfully downloaded '{asset}'")
    except Exception as e:
        logger.error(f"Error downloading NLTK '{asset}': {e}", exc_info=True)

# Verify download (optional)
try:
    logger.info(f"Verification: Files/folders in {nltk_data_dir}: {os.listdir(nltk_data_dir)}")
except Exception as e:
    logger.error(f"Could not list NLTK directory for verification: {e}")

logger.info("NLTK asset download script finished.")