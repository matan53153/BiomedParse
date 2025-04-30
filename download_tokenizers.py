from transformers import CLIPTokenizer

tokenizer_name = "openai/clip-vit-base-patch32"
local_save_path = "./clip-vit-base-patch32-tokenizer" # Choose a local path

# Download and save the tokenizer files
tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
tokenizer.save_pretrained(local_save_path)

print(f"Tokenizer files saved to: {local_save_path}")