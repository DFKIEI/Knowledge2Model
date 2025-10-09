import sqlite3
import re
from tqdm import tqdm
import json

conn = sqlite3.connect('./Hugging2KG/huggingface2.db')
cursor = conn.cursor()


cursor.execute('SELECT model_id, model_card_tags, metrics FROM Models WHERE model_card_tags IS NOT NULL')
rows = cursor.fetchall()

# RDF Pattren that matches the one defined in the Tag generating Prompt
valid_tag_pattern = re.compile(
    r'^(task_type|architecture|base_model|parameters|dataset|modality|sequence_length|quantization|speed|memory|hardware_needs|metric):')


def filter_tags(tag_list):
    valid_tags = []
    for tag in tag_list:
        # Replace line breaks with commas
        cleaned_tag = tag.strip().replace('\n', ', ').replace('\r', ', ')
        
        # Split into individual tags
        individual_tags = cleaned_tag.split(',')
        
        # Check each tag individually
        for individual_tag in individual_tags:
            individual_tag = individual_tag.strip()
            
            # Check each tag separately
            if (individual_tag and 
                'none' not in individual_tag.lower() and 
                valid_tag_pattern.match(individual_tag)):
                valid_tags.append(individual_tag)
    
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
