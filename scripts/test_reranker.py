import sys
from pathlib import Path

# Ensure project root and src are on PYTHONPATH for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import importlib.util
from pathlib import Path

# Load retriever module directly to avoid importing src.impl.__init__
retriever_path = Path(__file__).resolve().parents[1] / "src" / "impl" / "retriever.py"
spec = importlib.util.spec_from_file_location("retriever", str(retriever_path))
retriever_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retriever_mod)
Retriever = retriever_mod.Retriever

from interface.base_datastore import BaseDatastore, DataItem


class DummyDatastore(BaseDatastore):
    def __init__(self):
        self.items = [
            DataItem(content="Document about cats.", source="a"),
            DataItem(content="Information on dogs and training.", source="b"),
            DataItem(content="Cats and their habits.", source="c"),
            DataItem(content="A deep dive into birds.", source="d"),
        ]

    def add_items(self, items):
        pass

    def get_vector(self, content: str):
        return [0.0] * 1536

    def search(self, query: str, top_k: int = 5):
        # simple naive match order for smoke test
        return [item.content for item in self.items][:top_k]


if __name__ == '__main__':
    ds = DummyDatastore()
    r = Retriever(ds)
    results = r.search("Tell me about cats", top_k=3)
    print("Results:")
    for r_ in results:
        print(" -", r_)
