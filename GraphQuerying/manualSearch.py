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


def get_metrics_for_modality_slow(graph, input_modality, output_modality):
    """
    input_modality: The input modality (e.g., "Text", "Image").
    output_modality: The output modality (e.g., "Label", "Text").
    """

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX modality: <http://example.org/modality/>
    PREFIX metric: <http://example.org/metric/>

    SELECT DISTINCT ?metric
WHERE {
    ?problem a conn:Problem .
    ?problem modality:hasInput ?input .
    ?problem modality:hasOutput ?output .
    ?model a conn:Model .
    ?model conn:hasProblem ?problem .
    ?model metric:hasMetric ?metric .
    ?metric a metric:Metric.

    FILTER(?input = ?inputMod && ?output = ?outputMod)
    }
    """

    input_uri = Namespace("http://example.org/modality/")[input_modality]
    output_uri = Namespace("http://example.org/modality/")[output_modality]

    results = graph.query(query, initBindings={
        'inputMod': input_uri,
        'outputMod': output_uri
    })

    metrics = {str(row) for row in results}
    return metrics


def get_metrics_for_modality(graph, input_modality, output_modality):
    """
    input_modality: The input modality (e.g., "Text", "Image").
    output_modality: The output modality (e.g., "Label", "Text").
    """

    query = """
    PREFIX modality: <http://example.org/modality/>
    PREFIX metric: <http://example.org/metric/>

    SELECT DISTINCT ?metric
    WHERE {
      { # Metrics related to input modality
        ?inputModalityNode modality:hasRelatedMetric ?metric .
        ?metric a metric:Metric .
        FILTER (?inputModalityNode = ?inputMod)
      } UNION {
        ?outputModalityNode modality:hasRelatedMetric ?metric .
        ?metric a metric:Metric .
        FILTER (?outputModalityNode = ?outputMod)
      }
    }
    """

    input_uri = Namespace("http://example.org/modality/")[input_modality]
    output_uri = Namespace("http://example.org/modality/")[output_modality]

    results = graph.query(query, initBindings={
        'inputMod': input_uri,
        'outputMod': output_uri
    })

    metrics = {str(row[0]) for row in results}
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

    # convert scoree to liiterl
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


if __name__ == "__main__":
    graph = load_graph("./Hugging2KG/test_graph.ttl")

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

    # metrics =get_all_metrics(graph)
    # print_results(metrics, "metrics supported")

    # Get Metrics for Modality
    input_modality = "Image"
    output_modality = "Label"
    metrics = get_metrics_for_modality(graph, input_modality, output_modality)
    print_results(metrics, "Metrics for Image to Label modality")
