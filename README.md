# Simple RAG Pipeline

This project is a beginner-friendly tutorial project for building a Retrieval Augmented Generation (RAG) system. It demonstrates how to index documents, retrieve relevant content, generate AI-powered responses, and evaluate results—all through a command line interface (CLI).

![rag-image](./rag-design-basic.png)

## Overview

The RAG Framework lets you:

- **Index Documents:** Process and break documents (e.g., PDFs) into smaller, manageable chunks.
- **Store & Retrieve Information:** Save document embeddings in a vector database (using LanceDB) and search using similarity.
-- **Generate Responses:** Use a local model (via Crawl4AI or a local `sentence-transformers` model) to provide concise answers based on the retrieved context.
- **Evaluate Responses:** Compare the generated response against expected answers and view the reasoning behind the evaluation.

## Architecture

- **Pipeline (src/rag_pipeline.py):**  
  Orchestrates the process using:

  - **Datastore:** Manages embeddings and vector storage.
  - **Indexer:** Processes documents and creates data chunks. Two versions are available—a basic PDF indexer and one using the Docling package.
  - **Retriever:** Searches the datastore to pull relevant document segments.
  - **ResponseGenerator:** Generates answers by calling the AI service.
  - **Evaluator:** Compares the AI responses to expected answers and explains the outcome.

- **Interfaces (interface/):**  
  Abstract base classes define contracts for all components (e.g., BaseDatastore, BaseIndexer, BaseRetriever, BaseResponseGenerator, and BaseEvaluator), making it easy to extend or swap implementations.

## Installation

#### Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

We use a local Crawl4AI server or local models for LLM and embeddings. See `./scripts/crawl4ai_postinstall.sh` to set up Crawl4AI locally. If you prefer local embeddings only, install `sentence-transformers`.

If you previously used OpenAI, remove any `OPENAI_API_KEY` environment variables — this project no longer depends on OpenAI.
```sh
set -x CO_API_KEY "xxx"
```

## Crawl4AI Post-install (optional)

This project can integrate with a local Crawl4AI installation, but the post-install helper is disabled by default because it performs OS-specific setup (browsers, model downloads) and may not be appropriate in every environment.

To run the post-install helper once (idempotent), set the environment variable `RUN_CRAWL4AI_POSTINSTALL=1` and then run the CLI or the script manually:

```bash
# enable and run via the CLI
RUN_CRAWL4AI_POSTINSTALL=1 PYTHONPATH=src python main.py run

# or run the helper directly
RUN_CRAWL4AI_POSTINSTALL=1 ./scripts/crawl4ai_postinstall.sh
```

If you do not wish to use Crawl4AI, skip this step — the pipeline will still work with local `sentence-transformers` for embeddings.

## Usage

The CLI provides several commands to interact with the RAG pipeline. By default, they will use the source/eval paths specified in `main.py`, but there are flags to override them.

```python
DEFAULT_SOURCE_PATH = "sample_data/source/"
DEFAULT_EVAL_PATH = "sample_data/eval/sample_questions.json"
```

#### Run the Full Pipeline

This command resets the datastore, indexes documents, and evaluates the model.

```bash
python main.py run
```

#### Reset the Database

Clears the vector database.

```bash
python main.py reset
```

#### Add Documents

Index and embed documents. You can specify a file or directory path.

```bash
python main.py add -p "sample_data/source/"
```

#### Query the Database

Search for information using a query string.

```bash
python main.py query "What is the opening year of The Lagoon Breeze Hotel?"
```

#### Evaluate the Model

Use a JSON file (with question/answer pairs) to evaluate the response quality.

```bash
python main.py evaluate -f "sample_data/eval/sample_questions.json"
```
