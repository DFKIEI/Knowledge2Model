from neo4j import GraphDatabase
import sqlite3
import json
import ast
from tqdm import tqdm  # Import tqdm

# Connect to Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Connect to SQLite
conn = sqlite3.connect('../GraphRAG/huggingface2.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM Models")
rows = cursor.fetchall()

# Load mappings
with open("../Hugging2KG/modality_mapping.json", "r") as f:
    modality_mapping = json.load(f)

with open("../Hugging2KG/metric_mapping.json", "r") as f:
    metric_mapping = json.load(f)

def sanitize_string(value):
    if isinstance(value, str):
        # Attempt to replace invalid characters with a placeholder or ignore them
        return value.encode('utf-8', 'ignore').decode('utf-8')
    return value  # Return as is if it's not a string

# Define function to insert data
def insert_model(tx, model_id, model_name, problem, coverTag, library, downloads, likes, lastModified, metrics, model_card_tags, health_status, last_checked, health_error):
    # Sanitize all string parameters before passing them to the query
    model_name = sanitize_string(model_name)
    problem = sanitize_string(problem)
    coverTag = sanitize_string(coverTag)
    library = sanitize_string(library)
    health_status = sanitize_string(health_status)
    health_error = sanitize_string(health_error)
    last_checked = sanitize_string(last_checked)
    model_card_tags = [sanitize_string(tag) for tag in model_card_tags]  # Handle list of tags
    metrics = [{
        'name': sanitize_string(metric['name']),
        'dataset': sanitize_string(metric['dataset']),
        'score': sanitize_string(metric['score'])
    } for metric in metrics]  # Handle list of metrics
    
    query = """
    MERGE (m:Model {id: $model_id})
    SET m.name = $model_name, m.downloads = $downloads, m.likes = $likes, m.lastModified = $lastModified,
        m.health_status = $health_status, m.last_checked = $last_checked, m.health_error = $health_error

    MERGE (p:Problem {name: $problem})
    MERGE (m)-[:HAS_PROBLEM]->(p)

    MERGE (c:CoverTag {name: $coverTag})
    MERGE (m)-[:HAS_COVER_TAG]->(c)

    MERGE (l:Library {name: $library})
    MERGE (m)-[:USES_LIBRARY]->(l)
    
    MERGE (h:HealthStatus {status: $health_status})
    SET h.last_checked = $last_checked, h.error_message = $health_error
    MERGE (m)-[:HAS_HEALTH_STATUS]->(h)

    FOREACH (tag IN $model_card_tags |
        MERGE (t:Tag {name: tag})
        MERGE (m)-[:HAS_TAG]->(t)
    )

    FOREACH (metric IN $metrics |
        MERGE (metricNode:Metric {name: metric.name})
        MERGE (datasetNode:Dataset {name: metric.dataset})
        MERGE (scoreNode:Score {value: metric.score})
        MERGE (m)-[:EVALUATED_BY]->(metricNode)
        MERGE (metricNode)-[:ON_DATASET]->(datasetNode)
        MERGE (metricNode)-[:HAS_SCORE]->(scoreNode)
    )
    """
    tx.run(query, model_id=model_id, model_name=model_name, downloads=downloads, likes=likes, lastModified=lastModified,
           problem=problem, coverTag=coverTag, library=library, model_card_tags=model_card_tags, metrics=metrics,
           health_status=health_status, last_checked=last_checked, health_error=health_error)
    
# Insert data into Neo4j with tqdm progress bar
with driver.session() as session:
    total_rows = len(rows)

    # Initialize tqdm progress bar
    for idx, row in tqdm(enumerate(rows), total=total_rows, desc="Inserting models", unit="row"):
        model_id, model_name, problem, tags, coverTag, library, downloads, likes, lastModified, model_card, model_card_tags, metrics, health_status, last_checked, health_error = row

        # # Parse tags
        # try:
        #     tags_list = ast.literal_eval(tags) if isinstance(tags, str) else []
        # except:
        #     tags_list = []
        # Parse model_card_tags
        # Handle health-related fields with defaults
        health_status = health_status or "UNKNOWN"
        last_checked = last_checked or ""
        health_error = health_error or ""

        # Parse model_card_tags
        
        if isinstance(model_card_tags, str) and model_card_tags.strip():
            model_card_tags_list = [tag.strip() for tag in model_card_tags.split(',') if tag.strip()]
        else:
            model_card_tags_list = ["untagged"]

        # Parse metrics
        metric_list = []
        if metrics:
            for metric_str in metrics.split(','):
                try:
                    parts = metric_str.strip().split('|')
                    if len(parts) == 3:
                        metric_name = parts[0].split(":")[1] if "metric:" in parts[0] else parts[0]
                        dataset = parts[1]
                        score = parts[2]

                        # Normalize metric name
                        for standardized_name, aliases in metric_mapping.items():
                            if metric_name.lower() in [alias.lower() for alias in aliases]:
                                metric_name = standardized_name
                                break
                        metric_list.append({"name": metric_name, "dataset": dataset, "score": score})
                except:
                    continue

        # Insert into Neo4j
        session.execute_write(insert_model, model_id, model_name, problem, coverTag, library, downloads, likes, 
                            lastModified, metric_list, model_card_tags_list, health_status, last_checked, health_error)

driver.close()
conn.close()