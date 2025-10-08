# ModelStatus

Automated health checks for HuggingFace models across different tasks.

## Scripts

- `txt_clsf.py` - Text classification models (transformers, sentence-transformers)
- `txt_gen.py` - Text generation models (transformers)
- `txt2img.py` - Text-to-image models (diffusers)
- `fet_ext.py` - Feature extraction models (transformers, sentence-transformers)

## Setup

1. Set your HuggingFace token in each script:
HF_TOKEN = "your_token_here"
2. Update database path if needed (default: `../ModelStatus/huggingface2.db`)


## Usage
Run any script directly.


## Each Script

- Tests models from the database in batches  
- Updates health status: `OK`, `FAIL`, `OOM`, `NOT_FOUND`, `GATED`, etc.  
- Cleans up cache after each model  
- Shows progress and final summary

## Database Columns

Scripts add these columns to the Models table:

- `health_status` - Current model status  
- `health_error` - Error message if failed  
- `last_checked` - Timestamp of last check
