# download_checkpoint.py
import os
import argparse
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
REPO_ID = "microsoft/BiomedParse"  # Repository ID from Hugging Face
FILENAME = "biomedparse_v1.pt"      # Filename to download
# --- End Configuration ---

def download_file_hf(repo_id, filename, save_dir):
    """Downloads a specific file from a Hugging Face repository."""
    local_path = os.path.join(save_dir, filename)
    logging.info(f"Attempting to download '{filename}' from repo '{repo_id}' to '{save_dir}'...")
    logging.info("Note: This repository may require you to be logged in via 'huggingface-cli login' and accept terms on the website.")

    try:
        # hf_hub_download will check the cache first, then download if needed.
        # It saves the file directly to the specified local_dir if local_dir_use_symlinks=False.
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=save_dir,
            local_dir_use_symlinks=False, # Ensures the actual file is copied/downloaded directly to save_dir
            # token=None, # Add your Hugging Face token if needed: token="hf_YOUR_TOKEN"
            # force_download=True, # Uncomment to force redownload, ignoring cache
        )

        # Verify the expected file exists at the target path
        expected_final_path = os.path.join(save_dir, filename)
        if os.path.exists(expected_final_path):
            if downloaded_path == expected_final_path:
                 logging.info(f"File '{filename}' downloaded/verified successfully at '{expected_final_path}'.")
            else:
                 # This case should be less likely with local_dir_use_symlinks=False
                 logging.warning(f"File downloaded to '{downloaded_path}' but expected at '{expected_final_path}'. Check setup.")
        else:
             logging.error(f"Download completed but the file '{filename}' was not found at the expected location '{expected_final_path}'. Check permissions or download path.")
             # If downloaded_path is different, maybe try to move it?
             # Consider: import shutil; shutil.move(downloaded_path, expected_final_path)

    except Exception as e:
        logging.error(f"Failed to download '{filename}' from '{repo_id}'. Error: {e}")
        logging.error("Please ensure you are logged in ('huggingface-cli login') and have accepted the terms for the model at https://huggingface.co/microsoft/BiomedParse")
        raise # Re-raise the exception

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Download {FILENAME} from {REPO_ID} on Hugging Face Hub.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/gpfs/km4074/BiomedParse/pretrained",
        help="Directory where the downloaded file will be saved.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=REPO_ID,
        help="Hugging Face repository ID.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=FILENAME,
        help="The specific filename to download.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(f"Ensuring save directory exists: {args.save_dir}")

    download_file_hf(args.repo_id, args.filename, args.save_dir)

    logging.info("Download process finished.")
    final_path = os.path.join(args.save_dir, args.filename)
    if os.path.exists(final_path):
        print(f"\nFile should be available at: {os.path.abspath(final_path)}")
        try:
            size_bytes = os.path.getsize(final_path)
            print(f"Size: {size_bytes} bytes ({size_bytes / (1024*1024):.2f} MB)")
            if size_bytes < 1024 * 1024: # Check if less than 1MB
                 print("\nWARNING: File size seems small for a model checkpoint. Please double-check the download.")
        except OSError as e:
            print(f"Could not get file size: {e}")
    else:
        print(f"\nError: File {args.filename} was not found in {args.save_dir} after download attempt.") 