import glob
import json
import os
import sys
from typing import List
from pathlib import Path

# Ensure `src` is importable when running `python main.py` from project root.
repo_root = Path(__file__).resolve().parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pipeline import RAGPipeline
from create_parser import create_parser


DEFAULT_SOURCE_PATH = "sample_data/source/"
DEFAULT_EVAL_PATH = "sample_data/eval/sample_questions.json"


def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    # Import impl components lazily to avoid import-time errors when optional
    # dependencies are not installed.
    from impl import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator

    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)


def main():
    parser = create_parser()  # Create the CLI parser
    args = parser.parse_args()
    pipeline = create_pipeline()

    # Process source paths and eval path
    source_path = args.path if args.path else DEFAULT_SOURCE_PATH
    eval_path = args.eval_file if args.eval_file else DEFAULT_EVAL_PATH
    document_paths = get_files_in_directory(source_path)

    # Execute commands
    if args.command in ["reset", "run"]:
        print("🗑️  Resetting the database...")
        pipeline.reset()

    if args.command in ["add", "run"]:
        print(f"🔍 Adding documents: {', '.join(document_paths)}")
        pipeline.add_documents(document_paths)

    if args.command in ["evaluate", "run"]:
        print(f"📊 Evaluating using questions from: {eval_path}")
        with open(eval_path, "r") as file:
            sample_questions = json.load(file)
        pipeline.evaluate(sample_questions)

    if args.command == "query":
        print(f"✨ Response: {pipeline.process_query(args.prompt)}")


def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    return glob.glob(os.path.join(source_path, "*"))


if __name__ == "__main__":
    main()
