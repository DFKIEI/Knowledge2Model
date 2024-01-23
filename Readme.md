# RDF_GPT_Tool README

## Overview

RDF_GPT_Tool is a  Python-based toolkit for interacting with RDF (Resource Description Framework) data and utilizing OpenAI's GPT models. It includes two primary components: `RDF_Tool.py` for graph operations and a GPT-powered question-answering module.

### RDF_Tool.py

`RDF_Tool.py` is a graphical tool for managing RDF data. It allows users to perform various operations on RDF graphs in a visual manner. Key features include:

- **Node Management:** Add, delete, and modify nodes within the RDF graph.
- **Graph Visualization:** Interactive display of the RDF graph.
- **File Input:** Takes `.ttl` (Turtle) files as input for RDF data.

### Question-Answering Module

This module leverages OpenAI's GPT models to answer questions based on the RDF graph data. It selects appropriate models from the graph to fulfill the user's query. The module can be used in two ways:

1. **GUI Mode:** Run `gui.py` for a graphical interface.
2. **Command Line Mode:** Direct execution of the main method.

## Installation

1. Ensure Python is installed on your system.
2. Clone the repository or download the source code.
3. Install required dependencies (if any are listed).

## Usage

### Running RDF_Tool.py

To use the graph tool, simply start `RDF_Tool.py` and load your `.ttl` file. The GUI will provide options for node management and visualization.

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
