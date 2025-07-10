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
    model_id = row[0]
    model_name = row[1] or ""
    problem = row[2] or ""
    tags = row[3] or ""
    library = row[5] or ""
    metrics = row[11] or ""

    combined = " | ".join(filter(None, [
        f"Name: {model_name}",
        f"Problem: {problem}",
        f"Tags: {tags}",
        f"Library: {library}",
        f"Metrics: {metrics}"
    ]))

    texts.append(combined)
    metadata.append({"id": model_id, "name": model_name})

# Save texts and metadata
np.save('embedding/model_texts.npy', np.array(texts))
with open('embedding/model_metadata.json', 'w') as f:
    json.dump(metadata, f)

# Load texts from .npy
texts = np.load('embedding/model_texts.npy', allow_pickle=True)

# Create sentence embeddings
sentence_model = SentenceTransformer('BAAI/bge-large-en')

# Use tqdm for progress bar
embeddings = np.array([
    sentence_model.encode(text, convert_to_tensor=True).cpu().detach().numpy()
    for text in tqdm(texts, desc="Encoding embeddings", unit="sentence")
])

# Save embeddings
np.save('embedding/model_embeddings.npy', embeddings)

# Load embeddings
embeddings = np.load('embedding/model_embeddings.npy')

# Create Annoy index
dimension = embeddings.shape[1]
print("Dimension for annoy is: ", dimension)    # debug
annoy_index = AnnoyIndex(dimension, 'angular')

# Add items to Annoy index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

# Build and save the index
annoy_index.build(50)
annoy_index.save('embedding/models_index.ann')

# Load metadata
with open('embedding/model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Reload index and texts
annoy_index = AnnoyIndex(dimension, 'angular')
annoy_index.load('embedding/models_index.ann')
texts = np.load('embedding/model_texts.npy', allow_pickle=True)

# Search function with optional category filter
def search_semantic(query, top_k=100):
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)

    results = []
    for idx in nearest_neighbors:
        results.append(metadata[idx])

    return results

# Example searches
print("\n🔍 General Search:")
query = "classification models with high accuracy"
results = search_semantic(query)
for res in results:
    print(f"{res['id']}: {res['name']}")

print("\n🔍 Category-Specific Search (Problem):")
query = "text classification"
results = search_semantic(query)
for res in results:
    print(f"{res['id']}: {res['name']}")

print("\n🔍 Category-Specific Search (Model):")
query = "Llama-3.2-1B-imdb"
results = search_semantic(query)
for res in results:
    print(f"{res['id']}: {res['name']}")
