import sqlite3
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Reload index and texts
annoy_index = AnnoyIndex(1024, 'angular')
annoy_index.load('models_index.ann')
texts = np.load('model_texts.npy', allow_pickle=True)

sentence_model = SentenceTransformer('BAAI/bge-large-en')

def search_semantic(query, top_k=30):
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)

    results = []
    for idx in nearest_neighbors:
        results.append(metadata[idx])

    return results

# Example searches
print("\n🔍 General Search:")
query = "video segmentation"
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
