import os
import logging
import numpy as np
from openai import OpenAI
import model_selection
from rdfCode import get_mlgoals, get_models, get_model_info
from ModelCodebase import model
import InputCode as ic

api_key = None


def create_openai_prompt(question, content):
    """Create a prompt for the OpenAI API."""
    return f"Given the following question: \"{question}\". {content}"


def get_openai_response(client, model_version, prompt):
    """Get a response from the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in getting response from OpenAI: {e}")
        return None


def write_to_file(file_name, content):
    """Write content to a file."""
    try:
        with open(file_name, 'w+') as f:
            f.write(content)
        logging.info('File written successfully.')
    except Exception as e:
        logging.error(f"Error writing to file: {e}")


def main(question):
    if api_key is None:
        raise Exception('Chat gpt Api key is required')
    graph_path = './Graph/graph_old.ttl'
    client = OpenAI(
        api_key=api_key)  # API key can be set here or through environment variable

    # First OpenAI Prompt
    mlgoals = get_mlgoals(graph_path)
    goals = ', '.join(mlgoals)
    prompt1 = f"Which of the following machine learning Goals can be used to solve this problem (or part of it): {goals}. Answer with only  one of the Machine learning goals and with nothing else"
    prompt1 = create_openai_prompt(question, prompt1)
    mlgoal = get_openai_response(client, "gpt-3.5-turbo", prompt1)
    # mlgoal= 'ObjectDetection'
    # mlgoal = 'SentimentAnalysis'
    if mlgoal:
        # Model selection and information retrieval
        suggested_models = get_models(mlgoal, graph_path)
        model_info = get_model_info(suggested_models, graph_path)
        model_names = list(model_info.keys())

        models = [f"{m},  Numebr of parameters: {edges['hasParameters']}" for m, edges in model_info.items()]
        parameters = [edges['hasParameters'] for m, edges in model_info.items()]

        sort_order = np.array(parameters).argsort()[::-1]
        models = [models[i] for i in sort_order]
        model_names = [model_names[i] for i in sort_order]

        chosen_model = model_names[0] if len(models) == 1 else model_names[
            RDF_GPT_TOOL.model_selection.select_model(models)]

        # Prepare input code based on input type
        input_type = model_info[chosen_model]['input']

        input_code = ic.input_code(input_type)

        # Generate model code and keywords
        model_code, keywords = model(chosen_model)
        keywords_str = ': '.join(keywords)
        full_code = input_code + model_code

        # Second OpenAI Prompt
        prompt2 = f".A python implementation of:{chosen_model} has already benn been implemented. with the wolloing code: {full_code}. And given this information {keywords_str}. Compleate the code so that it for fills the desired question (goal). You should only output the python code that is appended to the already existing code and your answer is not allowed to include anything else. Not evan where to place the code. Just the pure python code as plane text."
        prompt2 = create_openai_prompt(question, prompt2)
        endcode = get_openai_response(client, "gpt-4", prompt2)

        if endcode:
            full_code += endcode
            file_path = os.path.join(os.getcwd(), 'generated_code.py')
            write_to_file(file_path, full_code)
        else:
            logging.error("Failed to generate end code.")
    else:
        logging.error("Failed to determine ML goal.")


def get_possibel_models(question):
    graph_path = './Graph/graph_old.ttl'
    client = OpenAI(api_key=api_key)  # API key can be set here or through environment variable

    # First OpenAI Prompt
    mlgoals = get_mlgoals(graph_path)
    goals = ', '.join(mlgoals)
    prompt1 = f"Which of the following machine learning Goals can be used to solve this problem (or part of it): {goals}. Answer with only  one of the Machine learning goals and with nothing else"
    prompt1 = create_openai_prompt(question, prompt1)
    # mlgoal = get_openai_response(client, "gpt-3.5-turbo", prompt1)
    mlgoal = get_openai_response(client, "gpt-4", prompt1)
    # mlgoal= 'ObjectDetection'
    # mlgoal = 'SentimentAnalysis'
    if mlgoal:
        # Model selection and information retrieval
        suggested_models = get_models(mlgoal, graph_path)
        model_info = get_model_info(suggested_models, graph_path)
        model_names = list(model_info.keys())

        models = [f"{m},  Numebr of parameters: {edges['hasParameters']}" for m, edges in model_info.items()]
        parameters = [int(edges['hasParameters']) for m, edges in model_info.items()]

        sort_order = np.array(parameters).argsort()[::-1]
        models = [models[i] for i in sort_order]
        model_names = [model_names[i] for i in sort_order]
        return models, model_names, model_info


def generate_code(chosen_model, question, model_info):
    client = OpenAI(api_key=api_key)  # API key can be set here or through environment variable

    # Prepare input code based on input type
    input_type = model_info[chosen_model]['input']

    input_code = ic.input_code(input_type)

    # Generate model code and keywords
    model_code, keywords = model(chosen_model)
    keywords_str = ': '.join(keywords)
    full_code = input_code + model_code

    # Second OpenAI Prompt
    prompt2 = f".A python implementation of:{chosen_model} has already benn been implemented. with the wolloing code: {full_code}. And given this information {keywords_str}. Compleate the code so that it for fills the desired question (goal). You should only output the python code that is appended to the already existing code and your answer is not allowed to include anything else. Not evan where to place the code. Just the pure python code as plane text."
    prompt2 = create_openai_prompt(question, prompt2)
    endcode = get_openai_response(client, "gpt-4", prompt2)

    if endcode:
        full_code += endcode
        file_path = os.path.join(os.getcwd(), 'generated_code.py')
        write_to_file(file_path, full_code)


if __name__ == '__main__':
    question = "How many people are in the image?"
    # question = "Which part of the image is just Background?"
    # question = "What is happening in the image?"
    # question = "Who is the tallest person in the image? Can you visualize it for me ?"
    # question = "Is this statement positive or negative?"
    main(question)
