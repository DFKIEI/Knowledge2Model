# Copyright 2023 SustainML Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SustainML ML Model Provider Node Implementation."""

from sustainml_py.nodes.MLModelNode import MLModelNode

# Manage signaling
import signal
import threading
import time

# Whether to go on spinning or interrupt
running = False


# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    MLModelNode.terminate()
    global running
    running = False


def get_openai_response(client, model_version, ml_model_metadata, prompt):
    """Get a response from the OpenAI API."""
    prompt = f"Given the following Information: \"{ml_model_metadata}\". {prompt}"
    try:
        response = client.chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in getting response from OpenAI: {e}")
        return None


# User Callback implementation
# Inputs: ml_model_metadata, app_requirements, hw_constraints, ml_model_baseline, hw_baseline, carbonfootprint_baseline
# Outputs: node_status, ml_model
def task_callback(ml_model_metadata,
                  app_requirements,
                  hw_constraints,
                  ml_model_baseline,
                  hw_baseline,
                  carbonfootprint_baseline,
                  node_status,
                  ml_model):
    # Callback implementation here
    api_key = None

    if api_key is None:
        raise Exception('Chat gpt Api key is required')

    graph_path = './Graph/graph_old.ttl'
    from openai import OpenAI
    client = OpenAI(api_key=api_key)  # API key can be set here or through environment variable

    from RDF_GPT_TOOL.rdfCode import get_mlgoals, get_models, get_model_info
    # First OpenAI Prompt
    # Retereve Possible Ml Goals from graph
    mlgoals = get_mlgoals(graph_path)
    goals = ', '.join(mlgoals)

    # Select MLGoal Using GPT
    prompt = f"Which of the following machine learning Goals can be used to solve this problem (or part of it): {goals}. Answer with only  one of the Machine learning goals and with nothing else"
    mlgoal = get_openai_response(client, "gpt-3.5-turbo", ml_model_metadata, prompt)

    if mlgoal:
        # Model selection and information retrieval
        suggested_models = get_models(mlgoal, graph_path)
        model_info = get_model_info(suggested_models, graph_path)
        model_names = list(model_info.keys())

        # Random Model is selected here. In the Final code there should be some sort of selection to choose between Possible Models
        chosen_model = model_names[0]

        # Generate model code and keywords
        from RDF_GPT_TOOL.ModelONNXCodebase import model
        onnx_path = model(chosen_model)
        ml_model.model(onnx_path)
    else:
        raise Exception("Failed to determine ML goal.")


# Main workflow routine
def run():
    node = MLModelNode(callback=task_callback)
    global running
    running = True
    node.spin()


# Call main in program execution
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    """Python does not process signals async if
    the main thread is blocked (spin()) so, tun
    user work flow in another thread """
    runner = threading.Thread(target=run)
    runner.start()

    while running:
        time.sleep(1)

    runner.join()
