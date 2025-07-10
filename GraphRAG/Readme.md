## README

This repository contains scripts and tools for processing, indexing, and querying a SQL-based knowledge graph, plus a simple chat interface. Below is an overview of each component and instructions for exporting/importing the Neo4j database.

---

## Scripts

### 1. `db2annoy.py`
**Stage 1 Preprocessing**
Generate the following files for semantic search:
- `model_metadata.json`
- `model_index.ann`

### 2. `db2neo4j.py`
**Stage 2 Neo4j**
SQL → Neo4j Conversion

Convert the SQL knowledge graph into a Neo4j database using URI-style labels (similar to db2rdf).

### 3. `semantic_search.py`
Stage 1 Search Test

Load the Annoy index and perform nearest-neighbor queries on the embedding space (pre-step to GRAG; no Neo4j interaction).

### 4. `chatbot_backend.py`
Flask Backend

Implements the full two-stage GRAG pipeline (semantic search + graph queries), with optional conversation history. Connects to an LLM via LMStudio or any HTTP-based service.

### 5. `chatbot_frontend.py`
HTML Frontend

Minimal chat interface for testing the backend.

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
