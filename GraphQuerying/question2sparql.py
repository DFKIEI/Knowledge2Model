from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import rdflib

g = rdflib.Graph()
g.parse("./Graphs/graph_old.ttl")
ml_goal_query = """
    PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
    SELECT ?o
    WHERE { ?s sc:mlgoal ?o } 
    """

def construct_schema(graph):
    query_classes = """
    SELECT DISTINCT ?class WHERE {
        ?s a ?class.
    }
    """

    query_properties = """
    SELECT DISTINCT ?property WHERE {
        ?s ?property ?o.
    }
    """
    classes = graph.query(query_classes)
    class_list = []
    for row in classes:
        class_list.append(str(row['class']))

    # Execute the query for properties
    properties = graph.query(query_properties)

    property_list = []
    for row in properties:
        property_list.append(str(row['property']))

    namespaces_list = []
    prefix_list = []
    for prefix, namespace in graph.namespaces():
        namespaces_list.append(str(namespace))
        prefix_list.append(str(prefix))

    used_ns_list = []

    updated_class_list = []
    for element in class_list:
        for ns in namespaces_list:
            if ns in element:
                updated = class_list[class_list.index(element)].replace(ns, prefix_list[namespaces_list.index(ns)]+":")
                used_ns_list.append(namespaces_list.index(ns))
                updated_class_list.append(updated)

    updated_prop_list = []
    for element in property_list:
        for ns in namespaces_list:
            if ns in element:
                update = property_list[property_list.index(element)].replace(ns, prefix_list[namespaces_list.index(ns)]+":")
                used_ns_list.append(namespaces_list.index(ns))
                updated_prop_list.append(update)

    used_ns_list = list(set(used_ns_list))

    cleaned_ns_str = ""
    for index, e in enumerate(namespaces_list):
        if index in used_ns_list:
            cleaned_ns_str += '@prefix ' + prefix_list[index] + ": " + namespaces_list[index] + " .\n"

    final_schema = "Prefixes:\n" + cleaned_ns_str + "\nClasses:\n"
    for c in updated_class_list:
        final_schema += c + "\n"
    final_schema += "\nProperties:\n"
    for p in updated_prop_list:
        final_schema += p + "\n"

    return final_schema, prefix_list, namespaces_list


clean_schema, prefix_list, namespaces_list = construct_schema(g)
print(prefix_list)
print(namespaces_list)


def query_graph(query, graph):
    q_res = graph.query(query)
    res_set = set()
    for row in q_res:
        obj = None
        if not row["o"] is None:
            obj = row["o"].rsplit("/", 1)[1]
        res_set.add(obj)

    return list(res_set)


def query_all_graphs(query, graph, prefixes, namespaces, show_prefix=True):
    q_result = graph.query(query)
    result_str = ""
    for row in q_result:
        print(row)
        result_str += str(row)
    res = result_str.replace("(rdflib.term.URIRef(", "").replace("'", "").replace("),)", ",")
    res_list = res.split(",")
    final_res = []
    for res in res_list:
        for name in namespaces:
            if name in res:
                if show_prefix:
                    clean_name = res.replace(name, prefixes[namespaces.index(name)]+":")
                else:
                    clean_name = res.replace(name, "")
                final_res.append(clean_name)
                break

    return list(set(final_res))


def result_parser(input_str):
    clean = input_str.content.replace("`", "")

    return clean.replace("sparql", "")


# ------------------------------------


sparql_tool_description = """This tool is used to generate a SPARQL SELECT statement for querying a graph database.
For instance, to find all email addresses of John Doe, the following query in backticks would be suitable:
```
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {{
    ?person foaf:name "John Doe" .
        ?person foaf:mbox ?email .
    }}
```
If you want to find all pairs of people where person a knows person b you could write a query like this:
'''
SELECT DISTINCT ?aname ?bname
WHERE {
    ?a foaf:knows ?b .
    ?a foaf:name ?aname .
    ?b foaf:name ?bname .
}'''
"""

prompt = """ You are a SPARQL query generator agent. You job is to use the tools available to you to construct 
and execute viable search queries on a provided knowledge graph. 
When creating a SPARQL query make sure to only the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Include all necessary prefixes.
Schema:
{schema}
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.

The question is:
{input}"""


query_generator_prompt = """ You are a SPARQL query generator. You job is to to construct viable search queries. 
When creating a SPARQL query make sure to only use the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Include all necessary prefixes.
Schema:
{schema}
Examples:
{examples}
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.

The question is:
{input}"""

sparql_examples = [
    {'question': 'how can I find all email addresses of John Doe?',
     'answer': '''PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {{
    ?person foaf:name "John Doe" .
    ?person foaf:mbox ?email .
    }}'''
     },
    {
        'question': 'What are the machine learning goals in the KG?',
        'answer': '''PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
SELECT ?o
 WHERE {{
  ?s sc:mlgoal ?o 
  }}
'''
    }]

prefix_prompt = """ You are a SPARQL query generator. You job is to to construct viable SPARQL queries. 
When creating a SPARQL query make sure to only use the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Include all necessary prefixes.
Schema:
{schema}""".format(schema=clean_schema)

suffix_template = """Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.
Question:{input}
Answer:"""
llm = ChatOpenAI(model="gpt-4o", temperature=0)

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\nAnswer:{answer}"
)
prefix_template = PromptTemplate(
    input_variables=['schema'], template=prefix_prompt
)

few_shot_prompt = FewShotPromptTemplate(
    prefix=prefix_prompt,
    examples=sparql_examples,
    example_prompt=example_prompt,
    suffix=suffix_template,
    input_variables=["input", 'schema'],
)

chain = few_shot_prompt | llm | result_parser

#result = chain.invoke({'input': 'What are the machine learning goals in the KG?'})
#print(result)
#q_res = g.query(result)
#for row in q_res:
#    print(row)
a = """PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
SELECT ?o
WHERE {
  ?s sc:mlgoal ?o 
}"""
g_result = query_all_graphs(a, g, prefix_list, namespaces_list)  # -> doesn't work if o not select operator
print(g_result)
