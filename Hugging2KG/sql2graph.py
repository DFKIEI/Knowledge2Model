import sqlite3
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import XSD
import ast
import json
from urllib.parse import quote

# Namespaces
CONN = Namespace("http://example.org/conn/")
MODEL = Namespace("http://example.org/model/")
PROBLEM = Namespace("http://example.org/problem/")
TAG = Namespace("http://example.org/tag/")
LIBRARY = Namespace("http://example.org/library/")
METRIC = Namespace("http://example.org/metric/")
TECH = Namespace("http://example.org/tech/")
MODALITY = Namespace("http://example.org/modality/")

# Extract Tags from Stringified List
allowed_tags = set()
with open('./topTags.txt', 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip()
        if tag:
            allowed_tags.add(tag)

# Load Modality Mapping from JSON
with open("modality_mapping.json", "r") as f:
    modality_mapping = json.load(f)

with open("metric_mapping.json", "r") as f:
    metric_mapping = json.load(f)

problem_nodes = {}
modality_nodes = {}

conn = sqlite3.connect('.//huggingface2.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM Models")
rows = cursor.fetchall()

g = Graph()

# Namespaces
g.bind("conn", CONN)
g.bind("model", MODEL)
g.bind("problem", PROBLEM)
g.bind("tag", TAG)
g.bind("library", LIBRARY)
g.bind("metric", METRIC)  # Bind the metric namespace
g.bind("tech", TECH)
g.bind("modality", MODALITY)

print("Moving SQL to Graph")
for idx_count, row in enumerate(rows):
    # if idx_count>500:
    # break
    model_id = row[0]
    model_name = row[1]
    problem = row[2]
    tags = row[3]
    coverTag = row[4]
    library = row[5]
    downloads = row[6]
    likes = row[7]
    lastModified = row[8]
    model_card = row[9]
    model_card_tags = row[10]
    metrics = row[11]

    """    # Literals for string based KG
    model_node = Literal(model_name)
    problem_node =PROBLEM[problem]  #  Literal(problem)
    coverTag_node = Literal(coverTag)
    library_node = Literal(library)"""

    # Use URIRef for nodes (IRIs)
    model_node = CONN[model_name]  # Or a more specific IRI if needed
    problem_node = PROBLEM[problem]
    coverTag_node = TAG[coverTag]  # Use IRI for cover tag
    library_node = LIBRARY[quote(library)]

    # Add Types
    g.add((model_node, RDF.type, CONN.Model))
    g.add((problem_node, RDF.type, CONN.Problem))
    g.add((coverTag_node, RDF.type, CONN.CoverTag))
    g.add((library_node, RDF.type, CONN.Library))

    # Add Model Details
    g.add((model_node, CONN.model_name, Literal(model_name, datatype=XSD.string)))
    g.add((model_node, CONN.model_id, Literal(model_id, datatype=XSD.string)))
    g.add((model_node, CONN.downloads, Literal(downloads, datatype=XSD.integer)))
    g.add((model_node, CONN.likes, Literal(likes, datatype=XSD.integer)))
    g.add((model_node, CONN.lastModified, Literal(lastModified, datatype=XSD.dateTime)))

    # Problem and Modality
    if problem in modality_mapping:
        if problem not in problem_nodes:
            problem_node = PROBLEM[problem]
            g.add((problem_node, RDF.type, CONN.Problem))
            problem_nodes[problem] = problem_node
        else:
            problem_node = problem_nodes[problem]

        g.add((model_node, CONN.hasProblem, problem_node))

        input_modality = modality_mapping[problem]["input"]
        output_modality = modality_mapping[problem]["output"]

        if input_modality not in modality_nodes:
            input_node = MODALITY[input_modality]
            g.add((input_node, RDF.type, MODALITY.Modality))
            modality_nodes[input_modality] = input_node
        else:
            input_node = modality_nodes[input_modality]

        if output_modality not in modality_nodes:
            output_node = MODALITY[output_modality]
            g.add((output_node, RDF.type, MODALITY.Modality))
            modality_nodes[output_modality] = output_node
        else:
            output_node = modality_nodes[output_modality]

        g.add((problem_node, MODALITY.hasInput, input_node))
        g.add((problem_node, MODALITY.hasOutput, output_node))
        g.add((problem_node, CONN.hasCoverTag, coverTag_node))

        # Add metrics
        if metrics:
            for metric_str in metrics.split(','):
                metric_str = metric_str.strip()
                if metric_str:
                    try:
                        parts = metric_str.split('|')
                        if len(parts) == 3 and parts[0].startswith("metric:"):  # check if metric has the correct format
                            metric_name = parts[0].split(":")[1]
                            if len(metric_name) == 1:
                                continue
                            # Metric Normalization
                            normalized_metric_name = None
                            for standardized_name, aliases in metric_mapping.items():  # use metric_mapping_metrics here
                                if metric_name.lower() in [alias.lower() for alias in aliases]:
                                    normalized_metric_name = standardized_name
                                    break
                            if normalized_metric_name:
                                metric_name = normalized_metric_name

                            dataset = parts[1]
                            score = parts[2]

                            metric_literal = Literal(metric_name, datatype=XSD.string)
                            dataset_literal = Literal(dataset, datatype=XSD.string)
                            score_literal = Literal(score, datatype=XSD.string)

                            g.add((model_node, METRIC.hasMetric, metric_literal))

                            # NEW DIRECT LINKS FROM MODALITY TO METRIC
                            g.add((input_node, MODALITY.hasRelatedMetric, metric_literal))
                            g.add((output_node, MODALITY.hasRelatedMetric, metric_literal))

                            g.add((metric_literal, RDF.type, METRIC.Metric))
                            g.add((metric_literal, METRIC.onDataset, dataset_literal))
                            g.add((metric_literal, METRIC.hasScore, score_literal))

                        else:
                            print(f"Invalid metric format: {metric_str}")

                    except Exception as e:
                        print(f"Error processing metric '{metric_str}': {e}")

    # Relationships
    # g.add((model_node, CONN.hasProblem, problem_node))
    g.add((model_node, CONN.hasCoverTag, coverTag_node))
    g.add((model_node, CONN.usesLibrary, library_node))

    try:
        tags_list = ast.literal_eval(tags)
        if isinstance(tags_list, list):
            for tag in tags_list:
                tag = tag.strip()
                if tag in allowed_tags:
                    tag_literal = Literal(tag, datatype=XSD.string)
                    g.add((model_node, CONN.hasTag, tag_literal))
                    g.add((tag_literal, RDF.type, CONN.Tag))
        else:
            print(f"Tags for model {model_name} are not in a list format")
    except Exception as e:
        print(f"Error parsing tags for model {model_name}: {e}")

    # Add model_card_tags
    if model_card_tags:
        for tag_str in model_card_tags.split(','):
            tag_str = tag_str.strip()
            if tag_str:
                try:
                    parts = tag_str.split(':')
                    if len(parts) == 2:
                        edge_name = parts[0].strip()
                        node_value = parts[1].strip()
                        edge_uri = None
                        if edge_name == "metric":
                            continue  # skip
                        elif edge_name in ["task_type", "architecture", "base_model", "parameters", "dataset",
                                           "modality", "sequence_length", "quantization"]:
                            edge_uri = CONN[edge_name]  # Use CONN namespace for model fundamentals
                        elif edge_name in ["speed", "memory", "hardware_needs"]:
                            edge_uri = TECH[edge_name]  # Use TECH namespace for technical requirements
                        else:
                            print(f"Unknown edge type: {edge_name}")
                            continue

                        if edge_uri:
                            node_literal = Literal(node_value, datatype=XSD.string)
                            g.add((model_node, edge_uri, node_literal))
                            # Add type for the node based on edge type or some other logic
                            if edge_name in ["task_type", "architecture", "base_model", "parameters", "dataset",
                                             "modality", "sequence_length", "quantization"]:
                                g.add((node_literal, RDF.type, CONN[edge_name.capitalize()]))
                            elif edge_name in ["speed", "memory", "hardware_needs"]:
                                g.add((node_literal, RDF.type, TECH[edge_name.capitalize()]))

                    else:
                        print(f"Invalid tag format: {tag_str}")

                except Exception as e:
                    print(f"Error processing tag '{tag_str}': {e}")

# TO FILE
print("Saving to Graph")
g.serialize("./test_graph.ttl", format="turtle")

conn.close()
