import sqlite3
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import XSD
import ast

# Namespaces
CONN = Namespace("http://example.org/conn/")
MODEL = Namespace("http://example.org/model/")
PROBLEM = Namespace("http://example.org/problem/")
TAG = Namespace("http://example.org/tag/")
LIBRARY = Namespace("http://example.org/library/")

# Extract Tags from Stringified List
allowed_tags = set()
with open('./Hugging2KG/topTags.txt', 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip()
        if tag: 
            allowed_tags.add(tag)

conn = sqlite3.connect('./Hugging2KG/huggingface.db')
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

print("Moving SQL to Graph")
for row in rows:
    model_id = row[0]
    model_name = row[1]
    problem = row[2]
    tags = row[3]
    coverTag = row[4]
    library = row[5]
    downloads = row[6]
    likes = row[7]
    lastModified = row[8]

    # Literals for string based KG
    model_node = Literal(model_name)
    problem_node = Literal(problem)
    coverTag_node = Literal(coverTag)
    library_node = Literal(library)

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

    # Relationships
    g.add((model_node, CONN.hasProblem, problem_node))
    g.add((model_node, CONN.hasCoverTag, coverTag_node))
    g.add((model_node, CONN.usesLibrary, library_node))
    g.add((problem_node, CONN.hasCoverTag, coverTag_node))

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

# TO FILE
print("Saving to Graph")
g.serialize("./Graphs/graph.ttl", format="turtle")

conn.close()
