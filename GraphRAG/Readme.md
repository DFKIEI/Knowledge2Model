db2annoy.py STAGE 1 preprocessing. Generate the model_texts.npy, model_embeddings.npy and model_index.ann
db2neo4j.py: Convert SQL Knowledge Graph to Neo4j, uses similiar approach as db2rdf, with different URI representation

chatbot_backend.py: Flask backend, implementing the whole pipeline of 2 stage GRAG (with conversation history, not sure if necessary for the framework) (uses the LLM through LMStudio server, but can be implemented the same way with other LLm services)
chatbot_frontend.py: HTML to have the chat as test

semantic_search.py: File to test the Stage 1 search on the trained embedding space. THis is the prestep to GRAG, only the indexing search without true search on the Neo4j database

neo4j.dump: database to import

EXPORT Neo4j file:
neo4j-admin database dump neo4j --to-path="..."

Import Neo4j file:
neo4j-admin database load neo4j --from-path=".../neo4j.dump" --overwrite-destination=true


local Neo4j credentials (if required):
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"