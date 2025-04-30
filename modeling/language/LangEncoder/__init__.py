import os
from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .transformer import *
from .build import *

# Define the root directory where pretrained assets are stored locally
# IMPORTANT: Update this path if you transferred the assets elsewhere
PRETRAINED_ASSETS_ROOT = "/scratch/gpfs/km4074/BiomedParse/pretrained_assets"

def _get_local_path(identifier):
    """Constructs the local path for a given Hugging Face identifier."""
    safe_identifier_name = identifier.replace("/", "__")
    local_path = os.path.join(PRETRAINED_ASSETS_ROOT, safe_identifier_name)
    # Basic check, you might want more robust error handling
    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Pretrained asset directory not found for '{identifier}' at expected path: {local_path}. Please ensure assets are downloaded and transferred correctly.")
    return local_path

def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']
    hf_identifier = config_encoder.get('PRETRAINED_MODEL', None) # Assuming config might specify the model identifier

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unknown model: {model_name}')

    # If it's a type known to load from HF, use the local path
    # This assumes the encoder uses the same identifier as the tokenizer specified in MODEL.TEXT
    if config_encoder['TOKENIZER'] in ['clip', 'clip-fast']:
        # Default to clip identifier if not specified otherwise
        hf_identifier = config_encoder.get('PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32')
        local_path = _get_local_path(hf_identifier)
        # Pass the local path to the specific encoder builder if it accepts it
        # Note: The actual loading happens within the registered lang_encoders function,
        # which might need modification if it doesn't accept a path string.
        # Assuming it implicitly uses the identifier or needs modification:
        # For demonstration, we pass it, but lang_encoders(model_name) might need changes.
        kwargs['pretrained_model_name_or_path'] = local_path
        print(f"INFO: Building lang encoder {model_name} using local path: {local_path}")
        # Ensure the actual model loading within lang_encoders(model_name) uses this path
        # This might involve modifying the specific encoder class (e.g., CLIPEncoder)

    # Add similar logic if other encoder types need local loading

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)

def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Set to false often needed with local files
    tokenizer_type = config_encoder['TOKENIZER']
    
    local_path = None
    hf_identifier = None

    if tokenizer_type == 'clip':
        hf_identifier = config_encoder.get('PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32')
        local_path = _get_local_path(hf_identifier)
        print(f"INFO: Loading CLIPTokenizer from local path: {local_path}")
        tokenizer = CLIPTokenizer.from_pretrained(local_path)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    elif tokenizer_type == 'clip-fast':
        hf_identifier = config_encoder.get('PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32')
        local_path = _get_local_path(hf_identifier)
        print(f"INFO: Loading CLIPTokenizerFast from local path: {local_path}")
        tokenizer = CLIPTokenizerFast.from_pretrained(local_path, from_slow=True)
    elif tokenizer_type == 'biomed-clip':
        hf_identifier = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        local_path = _get_local_path(hf_identifier)
        print(f"INFO: Loading AutoTokenizer ({hf_identifier}) from local path: {local_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_path)
    else:
        # Handle arbitrary tokenizers specified in config
        hf_identifier = tokenizer_type # Assume the type IS the identifier
        try:
            local_path = _get_local_path(hf_identifier)
            print(f"INFO: Loading AutoTokenizer ({hf_identifier}) from local path: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_path)
        except FileNotFoundError:
             print(f"WARNING: Could not find local path for tokenizer '{hf_identifier}'. Attempting to load directly (may fail offline).")
             print(f"Ensure '{hf_identifier}' assets are downloaded to {PRETRAINED_ASSETS_ROOT}/{hf_identifier.replace('/', '__')} and transfered.")
             # Fallback or error? Forcing offline, we should error or make it clearer.
             # For now, let it attempt and likely fail if truly offline.
             tokenizer = AutoTokenizer.from_pretrained(hf_identifier) # This will likely fail if offline and not found locally

    return tokenizer