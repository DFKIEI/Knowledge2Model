## README

This repository contains scripts and tools for processing, indexing, and querying a SQL-based knowledge graph, plus a simple chat interface. Below is an overview of each component and instructions for exporting/importing the Neo4j database.

---

## Scripts

### Pipeline Flow

#### **Stage 1: Data Preparation**
1. **`db2annoy.py`** - Preprocessing
   - Reads models from SQLite database
   - Generates semantic embeddings using sentence transformers
   - Builds and saves Annoy index for fast similarity search
   - **Outputs**: `model_metadata.json`, `model_embeddings.npy`, `models_index.ann`

#### **Stage 2: Graph Database Setup**
2. **`db2neo4j.py`** - SQL → Neo4j Conversion
   - Transfers model data from SQLite to Neo4j graph database
   - Creates nodes (Model, Problem, Library, Tag, Metric, HealthStatus)


#### **Stage 3: Backend Service**
3. **`chatbot_backend.py`** - Flask Backend (RAG Pipeline)
   - Implements two-stage retrieval:
     1. Semantic search using Annoy index (fast similarity matching)
     2. Graph queries using Neo4j (relationship-based filtering)
   - Connects to LLM (LMStudio/Ollama) for natural language generation

#### **Stage 4: User Interface**
4. **`chatbot_frontend.html`** - Web Interface
   - Minimal chat UI for interacting with the backend
   - Sends queries to Flask backend and displays responses

---

### Testing & Debugging Tools

- **`semantic_search.py`** - Standalone Search Tester
  - Tests the Annoy index independently without running the full pipeline
  - Useful for debugging semantic search without Neo4j or LLM

---

## Workflow Summary

**Setup (run once or when data changes):**
```bash
1. python db2annoy.py      # Build embeddings
2. python db2neo4j.py      # Populate Neo4j
```

## Neo4j Database Dump & Restore
### Export (Dump)

```bash
sudo neo4j-admin dump system \
  --to-path=<path>/Knowledge2Model/GraphRAG/backup_neo4j

sudo neo4j-admin dump neo4j \
  --to-path=<path>/Knowledge2Model/GraphRAG/backup_neo4j
```

### Import (Load)

```bash
sudo neo4j-admin database load system \
  --from-path=<path>/Knowledge2Model/GraphRAG/backup_neo4j \
  --overwrite-destination=true

sudo neo4j-admin database load neo4j \
  --from-path=<path>/Knowledge2Model/GraphRAG/backup_neo4j \
  --overwrite-destination=true

sudo chown -R neo4j:neo4j /var/lib/neo4j/data
```

### Neo4j Credentials

Set environment variables or update your .env file:

```bash
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="12345678"
```
