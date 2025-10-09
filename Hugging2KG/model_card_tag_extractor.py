import sqlite3
from collections import Counter
from openai import OpenAI
import re

MAX_WORDS = 15000 # Maximum number of words to process in the model 

def clean_model_card(text):
    """ Remove images (Markdown and HTML) from the model card """
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove Markdown images
    text = re.sub(r'<img[^>]+>', '', text)  # Remove HTML images
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'https?://[^\s]{40,}', '[URL]', text)  # Remove very long URLs
    return text

# PROMPT = """Extract technical and functional tags from this Hugging Face model card. Focus on these key areas:

# Extract model fundamentals with these tag types:
# task_type: The primary task (e.g., classification, generation)
# architecture: Model architecture (e.g., transformer, bert, gpt)
# base_model: Original model if fine-tuned
# parameters: Number of parameters (e.g., 7B, 110M)
# dataset: Training datasets used
# modality: Input/output types (e.g., text→labels, image→text)
# sequence_length: Maximum context length
# quantization: Model quantization details

# For performance metrics, combine as:
# metric: Format as metric:name|dataset|score
# Examples: metric:accuracy|squad2|92.4, metric:rouge1|cnn|41.2, metric:bleu4|wmt14|44.3
# Only include metrics belonging to the model itself, not comparison metrics.

# For technical requirements:
# speed: Training/inference speed metrics
# memory: Memory requirements
# hardwar_needs: Specific hardware needs


# Output format: 
# - Comma-separated values using standard ML terminology
# - Skip generic metadata like licenses, languages, or repository info
# - Only include tags where information is present in the model card
# - Silently skip any tag types where no information is available
# - Output 'none' only if NO relevant technical information exists for ANY of the requested tag types

# Example output:
# architecture:transformer, parameters:124M, dataset:webtext, modality:text, sequence_length:1024, quantization:fp16, metric:accuracy|lambada|55.4, metric:ppl|wikitext2|18.3

# Input model card:"""


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
conn = sqlite3.connect('.//huggingface2.db')
cursor = conn.cursor()

# Check if the 'model_card_tags' column exists before adding it
cursor.execute("PRAGMA table_info(Models);")
columns = [col[1] for col in cursor.fetchall()]
if "model_card_tags" not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN model_card_tags TEXT;")
if "metrics" not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN metrics TEXT;")

cursor.execute('SELECT model_id, model_card FROM Models WHERE downloads > 15 AND model_card_tags IS NULL AND model_card IS NOT NULL')
# cursor.execute('SELECT tags FROM Models WHERE downloads > 15 AND model_card_tags IS NULL ')
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

        # Ensure response is valid
        if not response.choices or not hasattr(response.choices[0].message, 'content'):
            print("Error: No valid response received.")
            return ""

        # Extract the content directly as it's in plain text format
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
            continue  # skip anything without "key:value"

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


# tag_counter = Counter()
toLargeModelCards = []

# Process model cards and update database
toGo= len(rows)
for idx, (model_id, model_card) in enumerate(rows):
    try:
        print(f'{idx}/{toGo}')
        print(f"Processing model {model_id}")
        if model_card is None:
            print('No Model Card')
            continue

        # Remove images
        model_card = clean_model_card(model_card)

        if len(model_card.split()) > MAX_WORDS:
            print(f"⚠️ Warning: Input exceeds {MAX_WORDS} words! -> trimmed to {MAX_WORDS} words.")
            model_card = ' '.join(model_card.split()[:MAX_WORDS])
            toLargeModelCards.append(model_id)

        generated_tags = generate_tags(PROMPT + f' Hugging Face model card: {model_card}')
        if generated_tags:
            print('update tags')
            tags, metrics = clean_tags(generated_tags)

            cursor.execute("UPDATE Models SET model_card_tags = ? WHERE model_id = ?", (', '.join(tags) or None, model_id))
            cursor.execute("UPDATE Models SET metrics = ? WHERE model_id = ?", (', '.join(metrics) or None, model_id))
            conn.commit()
    except Exception as e:
        print(f"Skipping model {model_id} due to error: {e}")
        continue

# Commit changes and close the database connection
conn.commit()

# Print truncated models summary
print(f"\nTruncated models: {len(toLargeModelCards)} out of {toGo}")
if toLargeModelCards: print(f"First 10: {toLargeModelCards[:10]}")

conn.close()
