from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever
from typing import List

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class Retriever(BaseRetriever):
    """Retriever that uses the datastore for candidate retrieval and a
    CrossEncoder for reranking.

    If `sentence-transformers` is not available, the datastore's initial
    scoring/order is returned unchanged as a graceful fallback.
    """

    def __init__(self, datastore: BaseDatastore, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.datastore = datastore
        self.reranker_model = reranker_model
        self._cross_encoder = None
        if CrossEncoder is not None:
            try:
                self._cross_encoder = CrossEncoder(self.reranker_model)
            except Exception as e:
                print(f"Warning: failed to load CrossEncoder '{self.reranker_model}': {e}")

    def search(self, query: str, top_k: int = 10) -> List[str]:
        # retrieve a larger candidate set then rerank
        search_results = self.datastore.search(query, top_k=top_k * 3)
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(self, query: str, search_results: List[str], top_k: int = 10) -> List[str]:
        # If CrossEncoder isn't available or failed to load, return candidates as-is
        if self._cross_encoder is None:
            print("⚠️ CrossEncoder not available — returning original search order")
            return search_results[:top_k]

        # Prepare pairs for CrossEncoder: (query, candidate)
        pairs = [(query, doc) for doc in search_results]
        try:
            scores = self._cross_encoder.predict(pairs)
        except Exception as e:
            print(f"Error during CrossEncoder.predict: {e}")
            return search_results[:top_k]

        # sort indices by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]
        print(f"✅ Reranked Indices (CrossEncoder): {top_indices}")
        return [search_results[i] for i in top_indices]
