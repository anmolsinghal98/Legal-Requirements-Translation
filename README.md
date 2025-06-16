# Legal Requirements Translation from Law

## Summary of Artifact

This repository provides a framework for translating legal requirements into an executable Python code representation using GPT-4o. The representation preserves the structural and semantic metadata relationships in legal text. The artifact includes:

1. **Translation Methodology**
   - A GPT-4o-based approach for converting legal text into structured code
   - Custom prompt engineering for legal requirement translation
   - Integration with a domain-specific class structure for legal requirements

2. **Core Components**
   - Structured representation of legal requirements using custom Python classes
   - Serialization utilities for data conversion
   - Testing framework for validation and evaluation

3. **Evaluation Framework**
   - Structural testing for code validation
   - Semantic testing for requirement accuracy
   - Compilation testing for code quality
   - Performance metrics calculation (pass@k, attribute-level scores)

4. **Test Suite**
   - Test cases from multiple US states
   - development dataset for model development
   - Comprehensive evaluation metrics

This artifact enables researchers and practitioners to:
- Convert legal requirements into a structured Python code representation
- Validate translated code through multiple testing methods
- Evaluate translation quality using various metrics

This repository contains code and tools for translating legal requirements into a structured Python code representation and evaluating the translations through various testing methods.

## Description of Artifact

### Core Code Files (`/code`)

#### Core Classes and Utilities
- `class_structure.py`: Defines the core data structures for representing legal requirements:
  - `Section`: Represents a bullet point in legal text
  - `Expression`: Represents text snippets within sections
  - `Statement`: Represents legal statements spanning multiple sections
  - `Information`, `Definition`, `Rule`, `Exemption`: Specialized statement types
  - `Reference`: Represents references to other parts of the legal text

- `serialize.py`: Provides serialization utilities for converting between Python objects and structured formats.

- `test_statements.py`: Contains test cases and validation logic for legal statements, including:
  - Information description testing
  - Definition term and meaning testing
  - Rule entity and type testing
  - Exemption description testing
  - Reference relationship testing

### Method
- `Code-with-Demo-with-Class.ipynb`: The main implementation file that contains:
  - GPT-4 prompts and configurations for legal requirement translation
  - Code generation pipeline for converting legal text to structured format
  - Implementation of the translation methodology
  - Demonstration selection 
  - Integration of the class structure with the prompt

### Testing Notebooks
- `Code-Gen-Structural-Testing.ipynb`: Implements structural testing for generated code:
  - Section number validation
  - Expression comparison
  - Reference validation

- `Code-Gen-Semantic-Testing.ipynb`: Implements semantic testing:
  - Tests 16 different semantic aspects
  - Calculates true positives, false positives, and false negatives
  - Computes accuracy, recall, and precision metrics

- `Code-Gen-Compliation-Testing.ipynb`: Tests code compilation and execution:
  - Validates code syntax
  - Tests code execution
  - Checks for runtime errors

### Evaluation Notebooks

- `Compute-pass-at-k.ipynb`: Implements pass@k evaluation:
  - Calculates pass rates across multiple attempts
  - Consolidates best results
  - Generates distribution statistics
  - Analyzes attribute-level scores

### Test Files (`/test files`)
Contains CSV files with test cases from different states:
- `OR.csv`: Oregon test cases
- `MS.csv`: Mississippi test cases
- `VA.csv`: Virginia test cases
- `VT.csv`: Vermont test cases
- `UT.csv`: Utah test cases
- `WI.csv`: Wisconsin test cases

### Data
- `development-set.csv`: Contains the development dataset for the translation model

### Intermediate Results
- Contains the sample output on `MS.csv` in the test set. 

## System Requirements

### Hardware Requirements
- CPU: 2+ cores
- RAM: 8GB minimum (16GB recommended)
- Storage: 1GB free space

### Software Requirements
- Operating System: Linux, macOS, or Windows
- Python 3.8 or higher
- Git

### Required Programs and Libraries
- Jupyter Notebook or JupyterLab
- OpenAI API access (for GPT-4o)
- Python packages (specified in requirements.txt)

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/anmolsinghal98/Legal-Requirements-Translation.git
cd Legal-Requirements-Translation
```

### 2. Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Configure OpenAI API
1. Create a `.env` file in the root directory:
```bash
touch .env
```

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 5. Start Jupyter Notebook
```bash
jupyter notebook
```

### Obtaining Intermediate Results within 60 minutes
Executing the pipeline across all the sentences in the development and test set may take longer than 60 minutes. This is because the OpenAI API response time can depend on several external factors such as downtime, response latency, and request overload. Consequently, we also provide `intermediate-results` by running the pipeline across a single test file `MS.csv`. 


## Usage Instructions

1. Start with `Code-with-Demo-with-Class.ipynb` to understand the translation methodology:
   - Review the prompt engineering approach
   - Understand the code generation pipeline
   - Study example translations
   - Learn best practices for legal requirement translation
   - Execute the notebook by following the instructions provided in comments

2. Use the testing notebooks to evaluate translations:
   - Compilation testing for validating compilation errors 
   - Structural testing for running structural tests
   - Semantic testing for running semantic tests

3. Use the pass@k evaluation notebook to compute metrics:
   - Pass@k for checking how many test examples pass all the test cases in k passes
   - Calculate the attribute-level accuracy, precision and recall scores 

## Dependencies

Key dependencies include:
- `numpy` and `pandas` for data manipulation
- `jupyter` and `notebook` for running notebooks
- `pytest` for testing
- `openai` and `langchain` for LLM integration
- `scikit-learn` for metrics calculation


## Steps to Reproduce

1. For reproducing the results on the development set, perform five-fold cross-validation by following the steps in the `Code-with-Demo-with-Class.ipynb` notebook. Generate model outputs for all the five folds and compute the average scores by executing the testing and evaluation notebooks. 
2. For reproducing the results on the test set, follow the steps in the `Code-with-Demo-with-Class.ipynb` notebook. Generate model outputs for each test file and compute the average scores by executing the testing and evaluation notebooks. 
3. A potential threat to reproducibility is that GPT-4o is a closed-source model by OpenAI. The scores reported in the paper may be subject to minor changes due to repeated updates to the GPT-4o model. Nevertheless, the key findings reported in the paper are likely to remain valid.
4. A potential issue that may lead to errors in testing and evaluation can be the presence of unwanted spaces and characters at the beginning and end of model output, which can affect code compilability. We have largely mitigated this issue by including the necessary post-processing steps. However, in rare occurrences, we cannot rule out the need for minor updates to post-processing that may be necessary to the output. To support these efforts, we have pointed out the lines in the code that may require changes in the comments. 
5. Generating the output for all the folds in the development set and the test set may take longer than 60 minutes. Therefore, we have provided intermediate results by executing the code on a single test file `MS.csv`. The notebooks also contain the output after executing the same test file. 

## Author Information

1. Anmol Singhal - Carnegie Mellon University (Email ID: anmolsinghal@cmu.edu)
2. Travis Breaux - Carnegie Mellon University (Email ID: tdbreaux@andrew.cmu.edu)

## Artifact Location

The artifact can be found at: https://doi.org/10.5281/zenodo.15670090

## How to cite

If you use this work in your research, please cite:

### APA

A. Singhal, T.D. Breaux (2025). "Legal Requirements Translation from Law." 33rd IEEE International Requirements Engineering Conference. 

## License

See the [LICENSE](LICENSE) file for details.