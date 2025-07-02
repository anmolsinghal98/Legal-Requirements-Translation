# Legal Requirements Translation from Law

![License](https://img.shields.io/github/license/anmolsinghal98/Legal-Requirements-Translation)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenAI GPT-4o](https://img.shields.io/badge/LLM-GPT--4o-lightgrey)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15794182.svg)](https://doi.org/10.5281/zenodo.15794182)

## 📝 Summary of the Artifact

Ensuring software compliance with legal regulations is especially challenging for small organizations. Extracting legal requirements from lengthy, complex texts demands significant legal expertise.

Prior automated approaches often overlook the **interdependencies** between legal metadata attributes. Yet legal meaning frequently arises from these relationships—for example, how obligations, conditions, and exceptions interact across clauses and sections.

This artifact provides a **framework for translating legal requirements into executable Python code** using GPT-4o. Our representation preserves **structural and semantic relationships**, enabling downstream **compliance verification**.

### ✨ Key Features

- **Translation Methodology**
  - GPT-4o-based code generation pipeline
  - Custom prompt engineering for legal text
  - Integration with a domain-specific class structure

- **Core Components**
  - Python classes representing legal constructs
  - Serialization and data conversion utilities
  - Test framework for validation

- **Evaluation Framework**
  - Structural, semantic, and compilation testing
  - Metrics: `pass@k`, attribute-level precision/recall

- **Test Suite**
  - Ground truth from six U.S. state laws
  - Development and evaluation datasets
  - Precomputed intermediate results

---

## 📂 Artifact Description

### 🔧 Code Directory (`/code`)

#### Core Classes
- `class_structure.py`: Defines the core data structures for representing legal requirements:
  - `Section`: Represents a bullet point in legal text
  - `Expression`: Represents text snippets within sections
  - `Statement`: Represents legal statements spanning multiple sections
  - `Information`, `Definition`, `Rule`, `Exemption`: Specialized statement types
  - `Reference`: Represents references to other parts of the legal text

- `serialize.py`: Serialization utilities for converting Python objects to structured formats.

- `test_statements.py`: Tests and validations for metadata types and attributes.

#### Main Translation Notebook
- `Code-with-Demo-with-Class.ipynb`: The main implementation file that contains:
  - GPT-4 prompts and configurations for legal requirement translation
  - Code generation pipeline for converting legal text to structured format
  - Implementation of the translation methodology
  - Demonstration selection 
  - Integration of the class structure with the prompt

#### Testing Notebooks
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

#### Evaluation Notebook
- `Compute-pass-at-k.ipynb`: Calculates `pass@k` and attribute-level metrics.

---

## 🧪 Test Files (`/test files`)

These CSV files contain legal paragraphs and their Python translations from six U.S. state data breach laws:

- `OR.csv` – Oregon  
- `MS.csv` – Mississippi  
- `VA.csv` – Virginia  
- `VT.csv` – Vermont  
- `UT.csv` – Utah  
- `WI.csv` – Wisconsin  

> Although tailored for U.S. state laws, our class structure is domain-agnostic and supports adaptation to other legal domains.

---

## 📊 Data

- `development-set.csv`: Dataset used for model development.
- `intermediate-results/`: Sample outputs from `MS.csv`.

---

## ⚙️ System Requirements

**Hardware**:  
- CPU: ≥ 2 cores  
- RAM: 8GB minimum (16GB recommended)  
- Disk: 1GB free space  

**Software**:  
- OS: Linux/macOS/Windows  
- Python: 3.8 or higher  
- Tools: Git, Jupyter

**Python Dependencies** (in `requirements.txt`):  
- `numpy`, `pandas`, `pytest`, `openai`, `langchain`, `scikit-learn`  
- Optional: `transformers`, `torch` (for HuggingFace LLMs)

---

## 🔧 Installation Instructions

### 1. Clone the Repository
  ```bash
   git clone https://github.com/anmolsinghal98/Legal-Requirements-Translation.git
   cd Legal-Requirements-Translation
  ```

### 2. Create and Activate Virtual Environment
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

Instructions to create an OpenAI API Key can be found here: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key.

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 5. Start Jupyter Notebook
```bash
jupyter notebook
```

### ⏱ Obtaining Intermediate Results within 60 minutes
Executing the pipeline across all the sentences in the development and test set may take longer than 60 minutes. This is because the OpenAI API response time can depend on several external factors such as downtime, response latency, and request overload. Consequently, we also provide `intermediate-results` by running the pipeline across a single test file `MS.csv`. 


## 🚀 Usage Instructions

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

3. Use `Compute-pass-at-k.ipynb` to compute metrics:
   - Pass@k for checking how many test examples pass all the test cases in k passes
   - Calculate the attribute-level accuracy, precision and recall scores 

## Dependencies

Key dependencies include:
- `numpy` and `pandas` for data manipulation
- `jupyter` and `notebook` for running notebooks
- `pytest` for testing
- `openai` and `langchain` for LLM integration
- `scikit-learn` for metrics calculation

Optional dependencies for running code on HuggingFace Language Models:
- `transformers`
- `torch`

## Steps to Reproduce

1. For reproducing the results on the development set, perform five-fold cross-validation by following the steps in the `Code-with-Demo-with-Class.ipynb` notebook. Generate model outputs for all the five folds and compute the average scores by executing the testing and evaluation notebooks. 
2. For reproducing the results on the test set, follow the steps in the `Code-with-Demo-with-Class.ipynb` notebook. Generate model outputs for each test file and compute the average scores by executing the testing and evaluation notebooks. 
3. A potential threat to reproducibility is that GPT-4o is a closed-source model by OpenAI. The scores reported in the paper may be subject to minor changes due to repeated updates to the GPT-4o model. Nevertheless, the key findings reported in the paper are likely to remain valid.
4. A potential issue that may lead to errors in testing and evaluation can be the presence of unwanted spaces and characters at the beginning and end of model output, which can affect code compilability. We have largely mitigated this issue by including the necessary post-processing steps. However, in rare occurrences, we cannot rule out the need for minor updates to post-processing that may be necessary to the output. To support these efforts, we have pointed out the lines in the code that may require changes in the comments. 
5. Generating the output for all the folds in the development set and the test set may take longer than 60 minutes. Therefore, we have provided intermediate results by executing the code on a single test file `MS.csv`. The notebooks also contain the output after executing the same test file. 

## 🔄 Using Language Models Other Than GPT-4o

The prompts provided in the `code` directory are model-agnostic, i.e., they can be used with any language model. The notebook provides the code setup to substitute `openai` models with any model hosted on `huggingface`. However, please note that the results reported in the paper using GPT-4o can differ significantly from the results obtained using a different model. The difference in model performance can be attributed to multiple factors that include the number of model parameters, training dataset size and quality, instruction tuning, and other training-specific details.

## 👥 Author Information

1. Anmol Singhal - Carnegie Mellon University (Email ID: anmolsinghal@cmu.edu)
2. Travis Breaux - Carnegie Mellon University (Email ID: tdbreaux@andrew.cmu.edu)

## 🌍 Artifact Location

The artifact can be found at: https://doi.org/10.5281/zenodo.15794182

## 📚 How to cite

If you use this work in your research, please cite:

### APA

A. Singhal, T.D. Breaux (2025). "Legal Requirements Translation from Law." 33rd IEEE International Requirements Engineering Conference. 

You can also use the citation file provided in `CITATION.cff`.

## 📄 License

See the [LICENSE](LICENSE) file for details.