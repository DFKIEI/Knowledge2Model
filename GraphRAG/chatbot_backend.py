import sqlite3
import json
from urllib import response
import numpy as np
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from neo4j import GraphDatabase
from openai import OpenAI
# from ollama import Client

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load precomputed embeddings and metadata
with open('./GraphRAG/embedding/model_metadata.json', 'r') as f:
    metadata = json.load(f)

annoy_index = AnnoyIndex(1024, 'angular')
annoy_index.load('./GraphRAG/embedding/models_index.ann')
texts = np.load('./GraphRAG/embedding/model_texts.npy', allow_pickle=True)

# Load sentence transformer for semantic search
sentence_model = SentenceTransformer('BAAI/bge-large-en', device='cpu')

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

MODEL = "mistral:7b-instruct-q4_0" 

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# LLM Client (e.g., OpenAI or Local LM Studio)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lmstudio")
# client = Client(host='http://localhost:11434')

# In-memory conversation history
# conversation_history = []

# def chatcompletion(model, messages, classify=False):
#     # global conversation_history  # Ensure we modify the global variable

#     # # Filter out 'system' messages
#     # conversation_history = [message for message in conversation_history if message['role'] != 'system']

#     # if not classify:
#     #     conversation_history.extend(messages)  # Extend history with new messages
#     #     completion = client.chat(
#     #         model=model,
#     #         messages=conversation_history,
#     #     )
#     # else:
#     completion = client.chat(
#         model=model,
#         messages=messages,  # Use only new messages for classification
#     )

#     return completion['message']['content'].strip()

def chatcompletion(model, messages, classify=False):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    # openai v1 returns an object → extract text
    return response.choices[0].message.content.strip()

# class LMStudioClient:

def search_semantic(query, top_k=100): # Change top_k
    """Performs semantic search on the Annoy index."""
    print("Search Semantic")
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)

    results = []
    for idx in nearest_neighbors:
        results.append(metadata[idx])
    print(results)
    return results

def generate_cypher_query(semantic_results):
    """Generates a Cypher query based on retrieved metadata."""
    print("generate Cypher")
    if not semantic_results:
        return None

    relevant_models = [res["name"] for res in semantic_results]

    cypher_query = f"""
    WITH {relevant_models} AS model_names
    MATCH (m:Model)
      WHERE m.name IN model_names
    OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
    OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
    OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
    OPTIONAL MATCH (m)-[:HAS_COVER_TAG]->(ct:CoverTag)
    RETURN
      m.name         AS name,
      m.id           AS modelId,
      m.downloads    AS downloads,
      m.likes        AS likes,
      m.lastModified AS lastModified,
      m.health_status AS healthStatus,
      m.last_checked AS lastChecked,
      m.health_error AS healthError,
      p.name         AS problem,
      l.name         AS library,
      collect(DISTINCT t.name)  AS tags,
      collect(DISTINCT ct.name) AS coverTags
    ORDER BY m.downloads DESC
    LIMIT 20
    """
    return cypher_query

def execute_cypher_query(cypher_query):
    """Executes the Cypher query on the Neo4j database."""
    print("Execute cypher")
    with neo4j_driver.session() as session:
        results = session.run(cypher_query)
        data = [dict(record) for record in results]
        print(data)
        return data

# Focussed prompting for answer generation
def generate_natural_answer(knowledge, user_question):
    """Generates a natural language response using the LLM."""
    print("generate natural answer")
    print(knowledge)
    
    final_prompt = f"""
You are a technical ML model search assistant. Answer based ONLY on the provided data.

DATA:
{knowledge}

QUESTION: {user_question}

TASK: List the top 5-10 most relevant models (sorted by downloads) with these details:
- Model name
- Library
- Problem type
- Input and output modalities (extract from tags: "modality:input→output")
- Health status (OK/FAIL/OOM/UNKNOWN)
- Last checked date
- Health error (only if status is not OK)
- Downloads

FORMAT: Use numbered list (1, 2, 3) with bullet points for details. No asterisks. No markdown bold.

EXAMPLE:
1. model-name-here
   • Library: transformers
   • Problem: text-classification
   • Modality: Text → Label
   • Health: OK (checked: 2025-01-15)
   • Downloads: 7.2M

Extract information exactly as provided. Do not invent or assume data.
"""
    messages = [
        {"role": "system", "content": "You are a factual technical assistant. Output structured data only. Use numbered lists with bullet points. No asterisks or markdown formatting."},
        {"role": "user", "content": final_prompt}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content.strip()

# def classify_input(user_input):
#     """Classifies user input into 'conversation' or 'hybrid' (for search)."""
#     print("classify input")
#     prompt = f"""
#     Classify the user input into:
#     - "conversation" if it is general chat without needing search, use it only if there is no search keywords for about the ML landscape present.
#     - "hybrid" if the input requires very specific data that is only available through retrieving such information from external sources. Such question should not allow any space for discussion. The sources contain content of the ML landscape with details on Models, Problems, characteristics and details.


#     User Input: "{user_input}"
#     Return only the category.
#     """

#     messages = [
#         {"role": "system", "content": "You are an expert classifier."},
#         {"role": "user", "content": prompt}
#     ]

#     classification = chatcompletion(model=MODEL, messages=messages, classify=True)
#     return classification.strip().replace('"', '').lower()

# Removed classification step
def answer_question(user_question):
    """Handles all questions using the full RAG pipeline."""
    print("Executing RAG pipeline...")
    
    # Always execute full RAG pipeline
    semantic_results = search_semantic(user_question)
    cypher_query = generate_cypher_query(semantic_results)

    if cypher_query:
        graph_results = execute_cypher_query(cypher_query)
        knowledge = semantic_results + graph_results
    else:
        knowledge = semantic_results

    return generate_natural_answer(knowledge, user_question)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("message", "")
    cleaned_input = user_question.replace("\n", " ").replace("\r", " ")
    response = answer_question(cleaned_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
