# CARA Pipeline: Candidate Classification Engine

## Architecture Overview
This solution implements a deterministic, semantic-similarity matching engine to classify healthcare candidates. Rather than relying on brittle keyword-matching (heuristics) or a generative LLM wrapper (which risks hallucination and violates the assessment constraints), this pipeline uses vector embeddings.

The system uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert candidate profiles and the Curavolv rubric into dense vector representations. It then calculates the cosine similarity to objectively score and rank the best-fit career tracks and degrees based on holistic semantic alignment.

## How to Run
1. Create a virtual environment: `python -m venv venv`
2. Activate the environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the classification pipeline: `python src/main.py`

*Note: The generated output is saved to `outputs/final_recommendations.json`.*

## AI/LLM Usage Declaration
In accordance with the assessment guidelines, no generative LLM was used to build the core classification logic, write the final output, or evaluate the candidates. 

An LLM was utilized solely as a developer productivity tool for:
1. **Data Formatting:** Converting the unstructured text from the prompt email into the structured `candidates.json` and `rubric.json` files.
2. **Boilerplate Architecture:** Brainstorming the project directory structure and generating the `requirements.txt`.
The core mathematical logic (cosine similarity) and algorithmic design remain strictly my own original implementation.