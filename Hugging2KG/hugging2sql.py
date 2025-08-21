import sqlite3
import requests
import json
from huggingface_hub import HfApi
from tqdm import tqdm

# Initialize the API
api = HfApi()

# Get pipeline tags
response = requests.get(
    "https://huggingface.co/api/models-tags-by-type",
    params={},
    headers={}
)
tags = json.loads(response.content)["pipeline_tag"]
print(f"Found {len(tags)} pipeline tags")

# Create database
conn = sqlite3.connect('./Hugging2KG/huggingface2.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS Models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    problem TEXT,
    tags TEXT,
    coverTag TEXT,
    library TEXT,
    downloads INTEGER,
    likes INTEGER,
    lastModified TEXT
)
''')
conn.commit()

total_inserted = 0

# Process each tag
for tag in tqdm(tags, desc="Processing pipeline tags"):
    problem = tag["id"]
    coverTag = tag["subType"]
    
    print(f"\nProcessing: {problem}")
    
    try:
        # Get ALL models - no limit parameter
        models = api.list_models(
            task=problem,
            sort="downloads",
            direction=-1,
            full=True  # Get full metadata
        )
        
        # Process models directly from generator
        model_count = 0
        for model in models:
            try:
                # Extract model information
                model_id = getattr(model, 'modelId', None)
                model_name = getattr(model, 'modelId', None)
                tags_list = getattr(model, 'tags', []) or []
                tags_json = json.dumps(tags_list, ensure_ascii=False)

                
                # Get library
                library = getattr(model, 'library_name', None)
                if not library and tags_list:
                    lib_tags = [
                        t.split('library:')[1]
                        for t in tags_list
                        if isinstance(t, str) and t.startswith('library:')
                    ]
                    library = lib_tags[0] if lib_tags else None
                
                downloads = getattr(model, 'downloads', 0) or 0
                likes = getattr(model, 'likes', 0) or 0
                modified = str(getattr(model, 'lastModified', "")) or None
                # ADD DOWNLOAD FILTER HERE
                if downloads <= 15:
                    continue  # Skip models with 15 or fewer downloads
                
                # Ensure all required fields are present
                if not all([model_id, model_name, tags_list, library, downloads, likes, modified]):
                    continue

                cursor.execute('''
                    INSERT OR IGNORE INTO Models (model_id, model_name, problem, tags, coverTag, library, downloads, likes, lastModified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (model_id, model_name, problem, tags_json, coverTag, library, downloads, likes, modified))
                
                total_inserted += cursor.rowcount
                model_count += 1
                
            except Exception as e:
                print(f"Error inserting model {model_id}: {e}")
                continue
        
        print(f"Found {model_count} models for {problem}")
                
    except Exception as e:
        print(f"Error fetching models for {problem}: {e}")
        continue

    # Commit after each tag group
    conn.commit()
    
    # Show progress
    cursor.execute('SELECT COUNT(*) FROM Models')
    print(f"Total models in database: {cursor.fetchone()[0]}")

print(f"\nFinished! Total new models inserted: {total_inserted}")
conn.close()