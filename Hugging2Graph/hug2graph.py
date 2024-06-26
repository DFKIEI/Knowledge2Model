import os
import json
from huggingface_hub import HfApi, ModelFilter, ModelCard
from rdflib import Graph, URIRef, Literal

# Create an instance of the Hugging Face API
api = HfApi()

# Define the KG namespace and graph
kg_namespace = "https://huggingface.co/kg/"
graph = Graph(namespace_manager=kg_namespace)

# Define the model collections to fetch
model_collections = ["transformers"]#, "bert", "roberta"]

# Create a dictionary to store the models
models = {}

# maybe we need to create a root architectures
import requests
response = requests.get(
  "https://huggingface.co/api/models",
  params={"limit":5,"full":"True","config":"True"},
  headers={}
)
print(response)

# Fetch the models from Hugging Face Hub
models_filter = ModelFilter(task="image-classification")
#models_filter = ModelFilter(id="yolov8m-painting-classification")
#print(list(api.list_models(filter=models_filter)))
for model in list(api.list_models(filter=models_filter)):
    model_id = model.modelId
    downloads = model.downloads
    print(model)
    if ~model.gated & downloads>0:
        print(model_id)
        card = ModelCard.load(model_id, ignore_metadata_errors=True)
        #model_description = card.data
        #print(card)
        #print(api.models(model_id))
        # Create a URI for the model
        model_uri = URIRef(kg_namespace + "model/" + model_id)
        
        # Add the model to the graph
        #graph.add((model_uri, URIRef("http://schema.org/name"), Literal(model_id)))
        #graph.add((model_uri, URIRef("http://schema.org/description"), Literal(str(card))))
        
        # Fetch the model's metadata
        model_metadata = api.model_info(model_id)
        print(model_metadata)
        # is the model derived from another architecture?

        # Add the metadata to the graph 
        # not yet working
#        for key, value in model_metadata.items():
#            if key != "id" and key != "name":
#                graph.add((model_uri, URIRef(kg_namespace + "property/" + key), Literal(value)))
#        print(model_id)

        # Add the model to the dictionary
#        models[model_id] = {
#            "uri": str(model_uri),
#            "model_id": model_id,
#            "description": str(card),
#            "metadata": model_metadata
#        }

# Save the graph to a file
#graph.serialize("huggingface_kg.ttl", format="turtle")

# Save the models dictionary to a JSON file
#with open("models.json", "w") as f:
#    json.dump(models, f, indent=4)