import sqlite3
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

# Database connection
conn = sqlite3.connect('./Hugging2KG/huggingface2.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM Models")
rows = cursor.fetchall()

# Extract relevant data with labels
texts = []
metadata = []  # Store category information

for row in rows:
    model_name = row[1]
    problem = row[2]
    tags = row[3]
    library = row[5]
    metrics = row[11]

    if model_name:
        texts.append(model_name)
        metadata.append(("Model", model_name))

    if problem:
        texts.append(problem)
        metadata.append(("Problem", problem))

    if tags:
        texts.append(tags)
        metadata.append(("Tags", tags))

    if library:
        texts.append(library)
        metadata.append(("Library", library))

    if metrics:
        for metric_str in metrics.split(','):
            texts.append(metric_str.strip())
            metadata.append(("Metric", metric_str.strip()))

# Save texts and metadata
np.save('model_texts.npy', np.array(texts))
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)

# Load texts from .npy
texts = np.load('model_texts.npy', allow_pickle=True)

# Create sentence embeddings
sentence_model = SentenceTransformer('BAAI/bge-large-en')

# Use tqdm for progress bar
embeddings = np.array([
    sentence_model.encode(text, convert_to_tensor=True).cpu().detach().numpy()
    for text in tqdm(texts, desc="Encoding embeddings", unit="sentence")
])

# Save embeddings
np.save('model_embeddings.npy', embeddings)

# Load embeddings
embeddings = np.load('model_embeddings.npy')

# Create Annoy index
dimension = embeddings.shape[1]
print(dimension)
annoy_index = AnnoyIndex(dimension, 'angular')

# Add items to Annoy index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

# Build and save the index
annoy_index.build(10)
annoy_index.save('models_index.ann')

# Load metadata
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Reload index and texts
annoy_index = AnnoyIndex(dimension, 'angular')
annoy_index.load('models_index.ann')
texts = np.load('model_texts.npy', allow_pickle=True)

# Search function with optional category filter
def search_semantic(query, top_k=5, category=None):
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)

    results = []
    for idx in nearest_neighbors:
        text = texts[idx]
        cat, original_text = metadata[idx]  # Retrieve metadata
        
        # Filter by category if specified
        if category is None or cat.lower() == category.lower():
            results.append({"category": cat, "text": original_text})

    return results

# Example searches
print("\n🔍 General Search:")
query = "classification models with high accuracy"
results = search_semantic(query)
for res in results:
    print(f"{res['category']}: {res['text']}")

print("\n🔍 Category-Specific Search (Problem):")
query = "text classification"
results = search_semantic(query, category="Problem")
for res in results:
    print(f"{res['category']}: {res['text']}")

print("\n🔍 Category-Specific Search (Model):")
query = "Llama-3.2-1B-imdb"
results = search_semantic(query, category="Model")
for res in results:
    print(f"{res['category']}: {res['text']}")