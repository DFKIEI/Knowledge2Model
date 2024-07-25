### Hugging2KG

## Pipeline

1. Run `hugging2sql.py` to crawl the Huggingface from their API and save it to SQLite database
2. Run `tagExtractor.py`to preprocess the stringified list of tags to useful format + filter unimportant (underrepresent tags)
3. Run `sql2graph.py`to generate the final KG. It is saved in `Graphs/graph.ttl`


