import os
import argparse
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of Hugging Face identifiers for models and their tokenizers
MODEL_IDENTIFIERS = [
    "openai/clip-vit-base-patch32",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    # Add any other known model/tokenizer identifiers used in your configs here
]

def download_hf_assets(identifier, save_directory):
    """Downloads model and tokenizer for a given Hugging Face identifier."""
    logging.info(f"Attempting to download: {identifier}")
    # Create a specific subdirectory for this identifier
    # Replace '/' with '__' for directory naming if needed, or handle hierarchy
    # For simplicity here, we use the identifier directly if it's simple,
    # or replace '/' for path compatibility.
    safe_identifier_name = identifier.replace("/", "__") # e.g., openai__clip-vit-base-patch32
    model_save_path = os.path.join(save_directory, safe_identifier_name)

    try:
        # Download tokenizer
        logging.info(f"Downloading tokenizer for {identifier} to {model_save_path}...")
        tokenizer = AutoTokenizer.from_pretrained(identifier)
        tokenizer.save_pretrained(model_save_path)
        logging.info(f"Tokenizer for {identifier} saved successfully.")

        # Download model
        logging.info(f"Downloading model for {identifier} to {model_save_path}...")
        model = AutoModel.from_pretrained(identifier)
        model.save_pretrained(model_save_path)
        logging.info(f"Model for {identifier} saved successfully.")

    except Exception as e:
        logging.error(f"Failed to download assets for {identifier}. Error: {e}")
        # Optionally, re-raise the exception or handle it
        # raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face models and tokenizers for offline use.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./pretrained_assets",
        help="Directory where the downloaded assets will be saved.",
    )
    args = parser.parse_args()

    save_directory = args.save_dir
    os.makedirs(save_directory, exist_ok=True)
    logging.info(f"Ensured save directory exists: {save_directory}")

    for identifier in MODEL_IDENTIFIERS:
        download_hf_assets(identifier, save_directory)

    logging.info("Finished downloading all specified assets.")
    print(f"\nAssets downloaded to: {os.path.abspath(save_directory)}")
    print("Please transfer this directory to your offline environment.") 