# Changelog

## [Unreleased]

### Chat Backend (`chatbot_backend.py`)
- Removed the `classify_input` function that differentiated between conversation and hybrid modes.
- The model now always executes the full RAG pipeline to generate a direct response for queries.
- Updated chatbot to include modality details in model recommendations.
- **Old Approach:** Hybrid chatbot that could do both casual conversation and technical search.
- **New Approach:** Specialized ML recommendation system that treats every query as a technical search request.

### Hugging2SQL (`hugging2sql.py`)
**Purpose:** This script crawls the Hugging Face API to collect machine learning models and stores it in a SQLite database for knowledge graph construction.
- Switched to official Hugging Face Hub API (replaced direct HTTP requests with the library).
- Implemented data quality filtering → only models with more than 15 downloads are included.
- Database expansion: model repository increased from ~25,000 to ~32,000 models.

### Model Status Check Script
**Purpose:** This script validates the usability of a model present in the database by testing if it can be loaded and used properly. 
- Created a script that downloads, loads, and executes test predictions on each model to verify usability.
- Enhanced database schema with new columns: `health_status`, `last_checked`, and `health_error`.
- `health_error` keeps records of error messages for debugging.
- Current progress: Model validation testing is ongoing across the entire model library; database will be updated as validation completes.

### Model Card Extraction Script
- Now uses Hugging Face authentication to access both private and public models.

### Semantic Search (`semantic_search.py`)
- (No changes recorded yet)

### DB → Annoy (`db2annoy.py`)
- (No changes recorded yet)

### DB → Neo4j (`db2neo4j.py`)
- (No changes recorded yet)

### Chatbot Frontend (`chatbot_frontend.html`)
- (No changes recorded yet)
