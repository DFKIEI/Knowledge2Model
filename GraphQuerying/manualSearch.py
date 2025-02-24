from rdflib import Graph, Namespace, RDF, Literal
from rdflib.namespace import XSD


def load_graph(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")
    return g


def get_cover_tags(graph):
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?coverTag
    WHERE {
      ?coverTag a conn:CoverTag .
    }
    """
    results = graph.query(query)
    cover_tags = [row[0] for row in results]
    return cover_tags


def get_problems(graph):
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?problem
    WHERE {
      ?problem a conn:Problem .
    }
    """
    results = graph.query(query)
    problems = [row[0] for row in results]
    return problems


def get_problems_for_cover_tag(graph, cover_tag_text):
    # cover_tag_literal = Literal(cover_tag_literal_text, datatype=XSD.string)
    TAG = Namespace("http://example.org/tag/")
    cover_tag_iri = TAG[cover_tag_text]

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX problem: <http://example.org/problem/>
    PREFIX tag: <http://example.org/tag/>
    SELECT ?problem
    WHERE {
      ?problem a conn:Problem .
      ?problem conn:hasCoverTag ?coverTag .
      FILTER (?coverTag = ?cover_tag_iri)
    }
    """

    results = graph.query(query, initBindings={'cover_tag_iri': cover_tag_iri})
    problems = [row[0] for row in results]
    return problems


def get_all_metrics(graph):
    query = """
    PREFIX conn: <http://example.org/conn/>  # Add conn prefix
    PREFIX metric: <http://example.org/metric/>
    SELECT DISTINCT ?metric 
    WHERE {
      ?metric a metric:Metric.
    }
    """
    results = graph.query(query)
    metrics = {str(metric[0]) for metric in results}
    return metrics



def get_models_with_higher_score(graph, metric_name, dataset, score_threshold):
    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX metric: <http://example.org/metric/>
    SELECT ?model
    WHERE {
        ?model a conn:Model .
        ?metric a metric:Metric .
        ?model metric:hasMetric ?metric .
        ?metric metric:metricName ?metricName .
        ?metric metric:onDataset ?dataset .
        ?metric metric:hasScore ?score .
        FILTER (xsd:float(?score) > ?score_threshold)  
    }
    """

    # convert score to literal
    score_threshold_literal = Literal(score_threshold, datatype=XSD.float)

    results = graph.query(
        query,
        initBindings={
            'metricName': Literal(metric_name),
            'dataset': Literal(dataset),
            'score_threshold': score_threshold_literal
        }
    )

    models = [str(row[0]) for row in results]
    return models


def get_models_with_max_size(graph, max_parameters=None):
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?model
    WHERE {
        ?model a conn:Model .
        ?model conn:parameters ?parameters .
        FILTER (xsd:integer(?parameters) >= ?min_parameters)
        """

    if max_parameters is not None:
        query += "FILTER (xsd:integer(?parameters) <= ?max_parameters)"

    query += "}"  # Close the WHERE clause

    max_parameters_literal = Literal(max_parameters, datatype=XSD.integer)

    results = graph.query(query, initBindings={'max_parameters': max_parameters_literal})
    models = [str(row[0]) for row in results]
    return models


def get_models_for_problem(graph, problem_literal_text):
    problem_literal = Literal(problem_literal_text, datatype=XSD.string)

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX model: <http://example.org/model/>
    SELECT ?model ?downloads
    WHERE {
      ?model a conn:Model .
      ?model conn:hasProblem ?problem .
      ?model conn:downloads ?downloads .
      FILTER (?problem = ?problem_literal)
    }
    ORDER BY DESC(?downloads)
    """

    results = graph.query(query, initBindings={'problem_literal': problem_literal})
    models = [(row[0], row[1]) for row in results]
    return models


def get_model_details(graph, model_name):
    model_literal = Literal(model_name, datatype=XSD.string)

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX model: <http://example.org/model/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?model ?id ?name ?problem ?coverTag ?library ?downloads ?likes ?lastModified
    WHERE {
      ?model a conn:Model .
      ?model conn:model_name ?name .
      ?model conn:model_id ?id .
      ?model conn:hasProblem ?problem .
      ?model conn:hasCoverTag ?coverTag .
      ?model conn:usesLibrary ?library .
      ?model conn:downloads ?downloads .
      ?model conn:likes ?likes .
      ?model conn:lastModified ?lastModified .
      FILTER (?name = ?model_literal)
    }
    """

    results = graph.query(query, initBindings={'model_literal': model_literal})
    details = {}
    for row in results:
        details = {
            'model_uri': row[0],
            'id': row[1],
            'name': row[2],
            'problem': row[3],
            'coverTag': row[4],
            'library': row[5],
            'downloads': row[6],
            'likes': row[7],
            'lastModified': row[8]
        }
    return details


def print_results(literals, label):
    print(f"List of available {label}:")
    for literal in literals:
        print(literal)
    print()


def print_models(models):
    print("Models ordered by downloads:")
    for model, downloads in models:
        print(f"Model: {model}, Downloads: {downloads}")
    print()


def print_model_details(details):
    print("Model details:")
    for key, value in details.items():
        print(f"{key}: {value}")
    print()


def find_problem_by_modalities(graph, input_modality, output_modality):
    query = f"""
    PREFIX modality: <http://example.org/modality/>
    PREFIX conn: <http://example.org/conn/>
    
    SELECT DISTINCT ?problem
    WHERE {{
        ?problem a conn:Problem ;
                 modality:hasInput "{input_modality}"^^xsd:string ;
                 modality:hasOutput "{output_modality}"^^xsd:string .
    }}
    """
    results = graph.query(query)

    problems = [str(row[0]) for row in results]
    return problems

def find_models_by_problem(graph, problem_uri):
    query = f"""
    PREFIX conn: <http://example.org/conn/>
    
    SELECT DISTINCT ?model
    WHERE {{
        ?model a conn:Model ;
               conn:hasProblem "{problem_uri}"^^xsd:string .
    }}
    """
    results = graph.query(query)

    models = [str(row[0]) for row in results]
    return models

def find_metrics_by_model(graph, model_name):
    query = f"""
    PREFIX metric: <http://example.org/metric/>
    PREFIX conn: <http://example.org/conn/>

    SELECT DISTINCT ?metric
    WHERE {{
        ?model a conn:Model ;
               conn:model_name "{model_name}"^^xsd:string ;
               metric:hasMetric ?metric .
    }}
    """
    results = graph.query(query)


    metrics = [str(row[0]) for row in results]
    return metrics

def search_metrics_by_modalities(graph, input_modality, output_modality):
    # Multi Stage Search
    problems = find_problem_by_modalities(graph, input_modality, output_modality)
    
    metrics_for_all_problems = {}

    for problem in problems:
        models = find_models_by_problem(graph, problem)
        models_with_metrics = {}
        
        for model in models:
            metrics = find_metrics_by_model(graph, model)
            models_with_metrics[model] = metrics
        
        metrics_for_all_problems[problem] = models_with_metrics

    return metrics_for_all_problems


def print_metrics_for_problem(metrics_for_all_problems):
    for problem, metrics_list in metrics_for_all_problems.items():
        print(f"Problem: {problem}")
        unique_metrics = set()  
        for model, metrics in metrics_list.items():
            for metric in metrics:
                unique_metrics.add(metric)
        
        for metric in unique_metrics:
            print(f"  Metric: {metric}")

if __name__ == "__main__":
    graph = load_graph("./Graphs/graph_v2.ttl")

    # Get Modality (Cover Tags)
    cover_tags = get_cover_tags(graph)
    print_results(cover_tags, "cover tags")

    # Get Problem List 
    problems = get_problems(graph)
    print_results(problems, "problems")

    cover_tag = "audio"
    problems = get_problems_for_cover_tag(graph, cover_tag)
    print_results(problems, "problems connected to the cover tag")

    # Get Model by Problem to solve
    problem_literal_text = "image-to-text"
    models = get_models_for_problem(graph, problem_literal_text)
    print_models(models)

    # Get Model Details (Example)
    model_name = "OleehyO/TexTeller"
    model_details = get_model_details(graph, model_name)
    print_model_details(model_details)

    # Get all Metrics supported (LONG PRINT)
    # metrics = get_all_metrics(graph)
    # print_results(metrics, "metrics supported")

    # Get Metrics for Modality
    input_modality = "Image"
    output_modality = "Label"
   
    metrics = search_metrics_by_modalities(graph, input_modality, output_modality)

    print_metrics_for_problem(metrics)


