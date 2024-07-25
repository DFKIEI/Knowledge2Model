def get_mlgoals(graph_path):
    ###########################################################
    ### get typs of machinlearning goals: ###
    ###########################################################
    import rdflib
    g = rdflib.Graph()
    g.parse(graph_path)
    ml_goal_query = """
    PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
    SELECT ?o
    WHERE { ?s sc:mlgoal ?o } 
    """

    qres = g.query(ml_goal_query)
    resSet = set()
    for row in qres:
        obj = None
        if not row["o"] == None:
            obj = row["o"].rsplit("/", 1)[1]
        resSet.add(obj)

    return list(resSet)


def get_models(mlgoal, graph_path):
    ###########################################################
    ### get models with correwct machinelearning goal: ###
    ###########################################################
    print(mlgoal)
    from rdflib import Graph

    # Load your RDF graph
    g = Graph()
    g.parse(graph_path, format="turtle")

    # Query to find all models
    query = """
    PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?model
    WHERE {
        {
            ?model rdf:type sc:Model .
            ?model sc:mlgoal """ + "sc:" + mlgoal + """
            FILTER NOT EXISTS {
                ?otherModel rdf:type ?model .
            }
        }
        UNION
        {
            ?subModel rdf:type sc:Model .
            ?model rdf:type ?subModel  .
            ?model sc:mlgoal """ + "sc:" + mlgoal + """
            FILTER NOT EXISTS {
                ?otherSubclass rdf:type ?model  .
            }
        }
    }
    """
    res = [row.model.rsplit("/", 1)[1] for row in g.query(query)]
    return res


def get_model_info(suggested_models, graph_path):
    ###########################################################
    ### get info about model ###
    ###########################################################
    import rdflib
    g = rdflib.Graph()
    g.parse(graph_path)
    model_info_dict = {}
    for model in suggested_models:
        query_m = """
        PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
        SELECT ?s ?p ?o
        WHERE { sc:%s ?p ?o}
        """ % (model)
        qres = g.query(query_m)
        resDict = []
        for row in qres:
            obj = {}
            for entry in ["p", "o"]:
                if not row[entry] == None:
                    obj[entry] = row[entry].rsplit("/", 1)[1]
                else:
                    obj[entry] = None
            resDict.append(obj)
        model_in = {}
        for dicts in resDict:
            key, value = dicts['p'], dicts['o']
            if key in model_in:
                if isinstance(model_in[key], list):
                    model_in[key].append(value)
                else:
                    model_in[key] = [model_in[key], value]
            else:
                model_in[key] = value
        model_info_dict[model] = model_in
    return model_info_dict
