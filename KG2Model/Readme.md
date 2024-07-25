# KG2Model README

### Question-Answering Module

This module leverages OpenAI's GPT models to answer questions based on the RDF graph data. It selects appropriate models from the graph to fulfill the user's query. The module can be used in two ways:

1. **GUI Mode:** Run `gui.py` for a graphical interface.
2. **Command Line Mode:** Direct execution of the `main.py` method.

### Using the Question-Answering Module

#### GUI Mode

Run `gui.py` to start the graphical interface. Enter your question in the provided field, and the system will process it using the RDF graph and GPT models.

#### Command Line Mode

Execute the main method directly with your question as an argument. The code snippet for this is:

```python
if __name__ == '__main__':
    question = "<your_question_here>"
    main(question)
```
    
#### API Key
An API key from OpenAI is required for the question-answering module. Set your API key in the code or as an environment variable.

## Additional Technical Requirements

Currently, models like BERT and DistilBERT are not supported for conversion into ONNX format. To run ml_model_provider_node.py, ModelONNXCodebase.py, rdfCode.py, and CustomGraph.ttl are required.
