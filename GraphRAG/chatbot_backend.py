import sqlite3
import json
import numpy as np
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from neo4j import GraphDatabase
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load precomputed embeddings and metadata
with open('./model_metadata.json', 'r') as f:
    metadata = json.load(f)

annoy_index = AnnoyIndex(1024, 'angular')
annoy_index.load('./models_index.ann')
texts = np.load('./model_texts.npy', allow_pickle=True)

# Load sentence transformer for semantic search
sentence_model = SentenceTransformer('BAAI/bge-large-en')

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

MODEL = "lmstudio-community/gemma-3-27B-it-qat-GGUF"

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# LLM Client (e.g., OpenAI or Local LM Studio)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# In-memory conversation history
conversation_history = []

def chatcompletion(model, messages, classify=False):
    global conversation_history  # Ensure we modify the global variable

    # Filter out 'system' messages
    conversation_history = [message for message in conversation_history if message['role'] != 'system']

    if not classify:
        conversation_history.extend(messages)  # Extend history with new messages
        completion = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            temperature=0.5,
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # Use only new messages for classification
            temperature=0.1,
        )

    return completion.choices[0].message.content.strip()
    
# class LMStudioClient:

def search_semantic(query, top_k=10, category=None):
    """Performs semantic search on the Annoy index."""
    print("Search Semantic")
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)

    results = []
    for idx in nearest_neighbors:
        text = texts[idx]
        cat, original_text = metadata[idx]

        if category is None or cat.lower() == category.lower():
            results.append({"category": cat, "text": original_text})
    print(results)
    return results

def generate_cypher_query(semantic_results):
    """Generates a Cypher query based on retrieved metadata."""
    print("generate Cypher")
    if not semantic_results:
        return None

    relevant_terms = [res["text"] for res in semantic_results]
    term_filters = ' OR '.join([f'm.name CONTAINS "{term}"' for term in relevant_terms])

    cypher_query = f"""
    MATCH (m:Model)
    WHERE {term_filters}
    RETURN m.name, m.problem, m.library, m.downloads, m.likes
    LIMIT 10
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

def generate_natural_answer(knowledge, user_question):
    """Generates a natural language response using the LLM."""
    print("generate natural answer")
    print(knowledge)
    final_prompt = f"""
    Based on the retrieved knowledge:
    {knowledge}

    Answer the following question in natural language: {user_question}
    """
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant providing structured responses."},
        {"role": "user", "content": final_prompt}
    ]

    return chatcompletion(model=MODEL, messages=messages)

def classify_input(user_input):
    """Classifies user input into 'conversation' or 'hybrid' (for search)."""
    print("classify input")
    prompt = f"""
    Classify the user input into:
    - "conversation" if it is general chat without needing search, use it only if there is no search keywords for about the ML landscape present.
    - "hybrid" if the input requires very specific data that is only available through retrieving such information from external sources. Such question should not allow any space for discussion. The sources contain content of the ML landscape with details on Models, Problems, characteristics and details.


    User Input: "{user_input}"
    Return only the category.
    """

    messages = [
        {"role": "system", "content": "You are an expert classifier."},
        {"role": "user", "content": prompt}
    ]

    classification = chatcompletion(model=MODEL, messages=messages, classify=True)
    return classification.strip().lower()

def answer_question(user_question):
    """Handles user questions by either chatting or retrieving hybrid search results."""
    category = classify_input(user_question)
    print(f"Classified as: {category}")

    if category == "conversation":
        return chatcompletion(
            model=MODEL,
            messages=[{"role": "system", "content": "You are a friendly and smart assistant for the ML landscape search. Please give short answers."},
                      {"role": "user", "content": user_question}]
        )

    if category == "hybrid":
        # TBD category
        semantic_results = search_semantic(user_question)
        cypher_query = generate_cypher_query(semantic_results)
        
        if cypher_query:
            graph_results = execute_cypher_query(cypher_query)
            knowledge = semantic_results + graph_results
        else:
            knowledge = semantic_results

        return generate_natural_answer(knowledge, user_question)

    return "I'm not sure how to classify that. Could you rephrase?"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("message", "")
    cleaned_input = user_question.replace("\n", " ").replace("\r", " ")
    response = answer_question(cleaned_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)