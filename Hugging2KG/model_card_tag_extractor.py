import sqlite3
from collections import Counter
from openai import OpenAI
import re

MAX_TOKENS = 15000  # Your LM Studio token limit
def clean_model_card(text):
    """ Remove images (Markdown and HTML) from the model card """
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove Markdown images
    text = re.sub(r'<img[^>]+>', '', text)  # Remove HTML images
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

PROMPT = """Extract technical and functional tags from this Hugging Face model card. Focus on these key areas:

Extract model fundamentals with these tag types:
task_type: The primary task (e.g., classification, generation)
architecture: Model architecture
base_model: Original model if fine-tuned
parameters: Number of parameters
dataset: Training datasets used
modality: Input/output types
sequence_length: Maximum context length
quantization: Model quantization details

For performance metrics, combine as:
metric: Format as metric:name|dataset|score
Examples: metric:accuracy|squad2|92.4, metric:rouge1|cnn|41.2, metric:bleu4|wmt14|44.3
Only include metrics belonging to the model itself, not comparison metrics.

For technical requirements:
speed: Training/inference speed metrics
memory: Memory requirements
hardwar_needs: Specific hardware needs


Output format: 
- Comma-separated values using standard ML terminology
- Skip generic metadata like licenses, languages, or repository info
- Only include tags where information is present in the model card
- Silently skip any tag types where no information is available
- Output 'none' only if NO relevant technical information exists for ANY of the requested tag types

Example output:
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
            model="qwen2.5-coder-32b-instruct",
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

        if len(model_card.split()) > MAX_TOKENS:
            print(f"⚠️ Warning: Input exceeds {MAX_TOKENS} tokens! -> trimmed to {MAX_TOKENS} words.")
            model_card = ' '.join(model_card.split()[:MAX_TOKENS])
            toLargeModelCards.append(model_id)

        generated_tags = generate_tags(PROMPT + f' Hugging Face model card: {model_card}')
        if generated_tags:
            print('update tags')
            splited_tags = generated_tags.split(',')
            metrics = [m for m in splited_tags if m.strip().startswith('metric:')]
            tags = [t for t in splited_tags if not t.strip().startswith('metric:')]

            cursor.execute("UPDATE Models SET model_card_tags = ? WHERE model_id = ?", (', '.join(tags), model_id))
            cursor.execute("UPDATE Models SET metrics = ? WHERE model_id = ?", (', '.join(metrics), model_id))
            conn.commit()
    except Exception as e:
        print(f"Skipping model {model_id} due to error: {e}")
        continue

# Commit changes and close the database connection
conn.commit()
conn.close()
