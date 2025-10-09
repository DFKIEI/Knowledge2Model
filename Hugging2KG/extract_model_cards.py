import os
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
#os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"      # hide HF's own bars
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # hide symlink warning

import sqlite3
import time
from tqdm import tqdm
from huggingface_hub import ModelCard, login
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from huggingface_hub import logging
logging.set_verbosity_error()


# Get token from environment and authenticate
HF_TOKEN = "Use your HF_TOKEN"

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Authenticated with HuggingFace")
else:
    print("No HF_TOKEN found - only public models will be accessible")

# Connect to the SQLite database
conn = sqlite3.connect('./Hugging2KG/huggingface2.db')
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
cursor = conn.cursor()

# Check if model_card column exists before adding it 
cursor.execute("PRAGMA table_info(Models);")
columns = [col[1] for col in cursor.fetchall()]
if "model_card" not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN model_card TEXT;")
    print("Added model_card column to the Models table.")
    conn.commit()

# Get all models that do not have a model card
cursor.execute('SELECT model_id, model_name FROM Models WHERE model_card is NULL or model_card = ""')
rows = cursor.fetchall()

print(f"Found {len(rows)} models without a model card.")

# Function to fetch model_card from Hugging Face
def fetch_model_card(model_name):
    """
    Fetch the README.md content for a model from Hugging Face.
    """
    try:
        # Attempt to load the model card
        card = ModelCard.load(model_name, ignore_metadata_errors=True, token=HF_TOKEN if HF_TOKEN else None)
        return card.content  # Returns the full README.md text
        
    except RepositoryNotFoundError:
        # Model doesn't exist or is private
        return None
        
    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            # Rate limited
            raise e  # Reraise to handle in main loop
        elif e.response.status_code == 403:
            # model is private
            return None
        elif e.response.status_code == 404:
            # No README file
            return None
        else:
            tqdm.write(f"Error fetching model card for {model_name}: {e.response.status_code}")
            return None
            
    except Exception as e:
        tqdm.write(f"Unexpected error for model {model_name}: {e}")
        return None

# Process each model without a model card
successful = 0
failed = 0
access_denied = 0  # Count of access denied errors
rate_limit_wait = 1  # Start with 1 second wait after rate limit

# Create progress bar with better formatting
pbar = tqdm(
    total=len(rows),
    desc="Fetching model cards",
    ncols=100,
    dynamic_ncols=True,
    mininterval=0.1,   # refresh faster
    miniters=1,
    ascii=True,
    leave=True
)

for idx, (model_id, model_name) in enumerate(rows, start=1):
    try:
        # Fetch the model card
        model_card = fetch_model_card(model_name)

        if model_card:
            cursor.execute("UPDATE Models SET model_card = ? WHERE model_id = ?", (model_card, model_id))
            successful += 1
            rate_limit_wait = max(1, rate_limit_wait * 0.9)
        else:
            failed += 1



        # Commit every 500 records
        if (idx % 500) == 0:
            conn.commit()
            tqdm.write(f"Progress: {idx}/{len(rows)} - Successful: {successful}, Failed: {failed}")

        time.sleep(0.1)  

    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            ra = e.response.headers.get("Retry-After")
            wait = int(ra) if ra and ra.isdigit() else rate_limit_wait
            time.sleep(wait)
            rate_limit_wait = min(60, rate_limit_wait * 2)

            try:
                model_card = fetch_model_card(model_name)
                if model_card:
                    cursor.execute("UPDATE Models SET model_card = ? WHERE model_id = ?", (model_card, model_id))
                    successful += 1
                else:
                    failed += 1
            except:
                failed += 1
        elif e.response.status_code == 403:
            access_denied += 1
        else:
            failed += 1

    except Exception as e:
        tqdm.write(f"Error processing {model_name}: {e}")
        failed += 1

    finally:
        # Always advance the bar exactly once per row
        pbar.update(1)


# Final commit
conn.commit()

print(f"\nCompleted!")
print(f"Successfully fetched: {successful} model cards")
print(f"Failed/Missing: {failed} model cards")
print(f"Access denied: {access_denied} model cards")
# Statistics about the database
cursor.execute("SELECT COUNT(*) FROM Models WHERE model_card IS NOT NULL")
total_with_cards = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM Models")
total_models = cursor.fetchone()[0]

print(f"\nDatabase statistics:")
print(f"Total models: {total_models}")
print(f"Models with cards: {total_with_cards}")

conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
conn.close()