import sqlite3
from collections import Counter
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed  # for parallel processing

MAX_WORDS = 15000 # Maximum number of words to process in the model
BATCH_SIZE = 300  # commit every 300 updates
MAX_WORKERS = 3   # 3 in-flight LM calls

def clean_model_card(text):
    """ Remove images (Markdown and HTML) from the model card """
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove Markdown images
    text = re.sub(r'<img[^>]+>', '', text)  # Remove HTML images
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'https?://[^\s]{40,}', '[URL]', text)  # Remove very long URLs
    return text

PROMPT = """You are a structured data extractor. Parse ML model specifications and output ONLY in format: tag:value, tag:value

CRITICAL: Output ONLY the comma-separated tags. Do NOT add explanations, notes, or sentences after the tags.

Extract model fundamentals with these tag types:
task_type: The primary task (e.g., classification, generation)
architecture: Model architecture (e.g., transformer, bert, gpt)
base_model: Original model if fine-tuned
parameters: Number of parameters (e.g., 7B, 110M)
dataset: Training datasets used
modality: Input/output types (e.g., text→labels, image→text)
sequence_length: Maximum context length
quantization: Model quantization details
metric: Format as metric:name|dataset|score (e.g., metric:accuracy|squad|92.4)

RULES:
- Skip licenses, languages, generic metadata
- If no technical info found, output: none
- NO explanations, notes, or additional sentences
- STOP after outputting the tags

WRONG - DO NOT DO THIS:
- "The model uses transformer architecture with 124M parameters..."
- "Based on the model card, I can extract: architecture:transformer"
- "Here are the tags: architecture:transformer, parameters:unknown"
- "Also note that there is no mention of..."
"However, some metrics like..."

IMPORTANT: Output ONLY comma-separated tags in the exact format shown below. Do NOT add explanations, sentences, or extra text.
EXAMPLES:
architecture:transformer, parameters:124M, dataset:webtext, modality:text, sequence_length:1024, quantization:fp16, metric:accuracy|lambada|55.4, metric:ppl|wikitext2|18.3

Input model card:"""

# Connect to the SQLite database
conn = sqlite3.connect('.//huggingface2.db', check_same_thread=False)  # NEW: allow threads
cursor = conn.cursor()



# Check if the column exists before adding it
cursor.execute("PRAGMA table_info(Models);")
columns = [col[1] for col in cursor.fetchall()]
if "model_card_tags" not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN model_card_tags TEXT;")
if "metrics" not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN metrics TEXT;")

cursor.execute('SELECT model_id, model_card FROM Models WHERE downloads > 15 AND model_card_tags IS NULL AND model_card IS NOT NULL')
rows = cursor.fetchall()

# Connect to LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to generate tags
def generate_tags(input_text):
    try:
        response = client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": input_text}]
        )
        if not response.choices or not hasattr(response.choices[0].message, 'content'):
            print("Error: No valid response received.")
            return ""
        tags = response.choices[0].message.content.strip()
        return tags
    except Exception as e:
        print(f"Error parsing response: {e}")
        return ""

# Post-process the raw output
def clean_tags(raw_output: str):
    """
    Keep only proper key:value pairs.
    Drop any extra sentences like 'However...' or 'Also note...'.
    """
    allowed = {
        "task_type", "architecture", "base_model", "parameters",
        "dataset", "modality", "sequence_length", "quantization", "metric"
    }
    tags, metrics = [], []
    for part in raw_output.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.lower().strip()
        value = value.strip()
        if key not in allowed:
            continue
        if key == "metric":
            metrics.append(f"metric:{value}")
        else:
            tags.append(f"{key}:{value}")
    return tags, metrics

toLargeModelCards = []

# a tiny worker that does ONLY CPU/network work, no DB writes inside threads
def _worker(model_id, model_card):
    if model_card is None:
        return (model_id, None, None, False)

    model_card = clean_model_card(model_card)
    too_large = False
    if len(model_card.split()) > MAX_WORDS:
        model_card = ' '.join(model_card.split()[:MAX_WORDS])
        too_large = True

    generated_tags = generate_tags(PROMPT + f' Hugging Face model card: {model_card}')
    if not generated_tags:
        return (model_id, None, None, too_large)

    tags, metrics = clean_tags(generated_tags)
    tags_csv = ', '.join(tags) if tags else None
    metrics_csv = ', '.join(metrics) if metrics else None
    return (model_id, tags_csv, metrics_csv, too_large)

# Process with small thread pool + batch commits
toGo = len(rows)
print(f"Total to process: {toGo}")

updates_batch = []  
extracted = 0 
done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(_worker, mid, mcard) for (mid, mcard) in rows] 
    print(f"Submitted {len(futures)} jobs; waiting for first results...", flush=True)

    for fut in as_completed(futures):
        done += 1
        model_id, tags_csv, metrics_csv, was_large = fut.result()

        if was_large:
            toLargeModelCards.append(model_id)

        # Only queue updates if at least one column has content
        if tags_csv is not None or metrics_csv is not None:
            updates_batch.append((tags_csv, metrics_csv, model_id))
            extracted += 1 

        # Commit in batches
        if len(updates_batch) >= BATCH_SIZE:
            cursor.executemany(
                "UPDATE Models SET model_card_tags = ?, metrics = ? WHERE model_id = ?",
                updates_batch
            )
            conn.commit()
            updates_batch.clear()
            print(f"Committed {done}/{toGo}")

        if done % 10 == 0:
            print(f"Progress: {done}/{toGo}")

# Flush any remaining updates
if updates_batch: 
    cursor.executemany(
        "UPDATE Models SET model_card_tags = ?, metrics = ? WHERE model_id = ?",
        updates_batch
    )
    conn.commit()
    updates_batch.clear()

print(f"\nExtracted: {extracted} / {toGo}")

# Print truncated models summary
print(f"\nTruncated models: {len(toLargeModelCards)} out of {toGo}")
if toLargeModelCards: print(f"First 10: {toLargeModelCards[:10]}")

conn.close()
