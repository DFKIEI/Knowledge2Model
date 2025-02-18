import sqlite3
import re
from tqdm import tqdm
import json

conn = sqlite3.connect('.//huggingface2.db')
cursor = conn.cursor()


cursor.execute('SELECT model_id, model_card_tags, metrics FROM Models WHERE model_card_tags IS NOT NULL')
rows = cursor.fetchall()

# RDF Pattren that matches the one defined in the Tag generating Prompt
valid_tag_pattern = re.compile(
    r'^(task_type|architecture|base_model|parameters|dataset|modality|sequence_length|quantization|speed|memory|hardware_needs|metric):')


def filter_tags(tag_list):
    # Remove tags with 'none' or line breaks (typical errors of the prompt), and invalid tags
    valid_tags = [tag.strip() for tag in tag_list if
                  'none' not in tag.lower() and '\n' not in tag and valid_tag_pattern.match(tag.strip())]
    return ', '.join(valid_tags)


for model_id, tags, metrics in tqdm(rows, desc="tags", unit="model"):
    try:
        tags_list = tags.split(',') if tags else []
        metrics_list = metrics.split(',') if metrics else []

        filtered_tags = filter_tags(tags_list)
        filtered_metrics = filter_tags(metrics_list)

        filtered_tags = filtered_tags if filtered_tags else None
        filtered_metrics = filtered_metrics if filtered_metrics else None

        cursor.execute("UPDATE Models SET model_card_tags = ?, metrics = ? WHERE model_id = ?",
                       (filtered_tags, filtered_metrics, model_id))

    except Exception as e:
        print(f"Error processing model {model_id}: {e}")



conn.commit()
conn.close()
