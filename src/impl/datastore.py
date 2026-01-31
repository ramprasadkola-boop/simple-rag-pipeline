from typing import List
from interface.base_datastore import BaseDatastore, DataItem
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import logging
from util.http_client import get_session

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class Datastore(BaseDatastore):

    DB_PATH = "data/chroma"
    COLLECTION_NAME = "rag-collection"

    def __init__(self):
        self.vector_dimensions = 1536
        self.open_ai_client = None  # Removed OpenAI client initialization
        self.client = None
        self.collection = None
        self.embed_model = None
        # Crawl4AI HTTP base URL (optional) and enable flag
        self.crawl4ai_url = os.environ.get("CRAWL4AI_URL", "http://localhost:11235")
        self.crawl4ai_enabled = os.environ.get("CRAWL4AI_ENABLE", "0").lower() in ("1", "true", "yes")
        self.logger = logging.getLogger(__name__)
        # Use singleton HTTP session (1 connection max, shared across app)
        self._http = get_session()
        # try to load a local sentence-transformers model for embeddings
        if SentenceTransformer is not None:
            try:
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embed_model = None
        if chromadb is not None and Settings is not None:
            try:
                # Use a persisted DuckDB+Parquet backend if available, otherwise fallback to in-memory
                self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.DB_PATH))
            except Exception:
                try:
                    self.client = chromadb.Client()
                except Exception:
                    self.client = None

        self.collection = self._get_collection()

        # If a crawl4ai_custom_context.md file exists in repo root, preload it
        try:
            repo_root = Path(__file__).resolve().parents[1]
            custom_md = repo_root / "crawl4ai_custom_context.md"
            if custom_md.exists() and self.collection is not None:
                text = custom_md.read_text(encoding="utf-8")
                # split into chunks by double newlines for simple sections
                parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                items = [DataItem(content=part, source=str(custom_md)) for part in parts]
                if items:
                    # Only add if collection empty to avoid duplicates
                    try:
                        existing = self.collection.count()
                    except Exception:
                        existing = None
                    if not existing:
                        self.add_items(items)
        except Exception:
            pass

    def reset(self):
        if self.client is None:
            raise RuntimeError("chromadb client is not available; cannot reset datastore")

        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(name=self.COLLECTION_NAME)
        print(f"✅ Chroma collection reset/created: {self.COLLECTION_NAME} in {self.DB_PATH}")
        return self.collection

    def get_vector(self, content: str) -> List[float]:
        # Use local sentence-transformers model for embeddings when available
        if self.embed_model is not None:
            emb = self.embed_model.encode(content)
            # ensure list[float]
            return emb.tolist() if hasattr(emb, "tolist") else list(map(float, emb))

        # Otherwise, optionally request embeddings from a local Crawl4AI HTTP service.
        # The HTTP call is only attempted if `CRAWL4AI_ENABLE` is set (1/true/yes).
        # This avoids waiting on timeouts when the service is not running.
        if not getattr(self, "crawl4ai_enabled", False):
            self.logger.debug("Crawl4AI HTTP embeddings disabled (CRAWL4AI_ENABLE not set). Using fallback embedding.")
            return self._simple_hash_embedding(content)

        # Try common endpoints in order: /embeddings, /embed
        endpoints = ["/embeddings", "/embed"]
        headers = {"Content-Type": "application/json"}
        payload = {"input": content}

        for ep in endpoints:
            url = f"{self.crawl4ai_url.rstrip('/')}{ep}"
            try:
                resp = self._http.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                try:
                    data = resp.json()
                except ValueError:
                    # Not JSON; skip
                    continue

                # common keys
                for key in ("embedding", "embeddings", "vector"):
                    if key in data:
                        emb = data[key]
                        # if embeddings wrapped in list for batch
                        if isinstance(emb, list) and len(emb) and isinstance(emb[0], list):
                            emb = emb[0]
                        return [float(x) for x in emb]
                # if response is a list directly
                if isinstance(data, list) and len(data) and isinstance(data[0], (int, float)):
                    return [float(x) for x in data]
            except Exception as e:
                # Log at debug level and try next endpoint
                self.logger.debug("Crawl4AI endpoint %s failed: %s", url, e)
                continue

        # As a last-resort fallback produce a deterministic hash-based vector so
        # the system can still operate (not ideal for semantic search, but
        # useful for offline testing or limited environments).
        try:
            return self._simple_hash_embedding(content)
        except Exception:
            raise RuntimeError(
                "No local embedding model available and Crawl4AI embedding endpoints did not respond. "
                "Install 'sentence-transformers' or run './scripts/crawl4ai_postinstall.sh' and/or start a Crawl4AI server exposing /embeddings or /embed."
            )

    def _simple_hash_embedding(self, content: str) -> List[float]:
        """Deterministic, lightweight embedding fallback using token hashing.

        This creates a fixed-size vector by hashing tokens into buckets and
        accumulating term counts. It's not semantically meaningful but allows
        the pipeline to run without heavy dependencies.
        """
        import hashlib

        dim = self.vector_dimensions
        vec = [0.0] * dim
        # simple tokenization
        tokens = [t.lower() for t in content.split() if t.strip()]
        if not tokens:
            return vec

        for t in tokens:
            # stable hash to index
            h = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0

        # simple normalization
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def add_items(self, items: List[DataItem]) -> None:
        if self.client is None or self.collection is None:
            raise RuntimeError("chromadb client/collection not available; cannot add items")

        # Compute embeddings in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            embeddings = list(executor.map(lambda it: self.get_vector(it.content), items))

        ids = [str(uuid.uuid4()) for _ in items]
        documents = [it.content for it in items]
        metadatas = [{"source": it.source} for it in items]

        # Upsert into Chroma collection
        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        except Exception as e:
            # Some chromadb versions use 'upsert' instead of 'add'
            try:
                self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            except Exception:
                raise

    def search(self, query: str, top_k: int = 5) -> List[str]:
        vector = self.get_vector(query)
        if self.client is None or self.collection is None:
            raise RuntimeError("chromadb client/collection not available; cannot search")

        try:
            results = self.collection.query(query_embeddings=[vector], n_results=top_k, include=["documents", "metadatas", "distances"])
        except Exception as e:
            # Some older/newer versions may use different parameter names
            results = self.collection.query(query_embeddings=[vector], n_results=top_k, include=["documents"])

        # 'documents' is a list of lists (one per query)
        documents = results.get("documents", [[]])[0]
        
        # If no results from local collection, escalate to Crawl4AI directly
        if not documents:
            self.logger.info("No local results for query '%s'; escalating to Crawl4AI", query)
            crawl4ai_result = self._escalate_to_crawl4ai(query)
            if crawl4ai_result:
                # Return Crawl4AI result as a single-item list for consistency
                return [crawl4ai_result]
        
        return documents
    
    def _escalate_to_crawl4ai(self, query: str) -> str:
        """Escalate search to Crawl4AI when local collection is empty or has no results."""
        if not self.crawl4ai_enabled:
            self.logger.debug("Crawl4AI escalation disabled (CRAWL4AI_ENABLE not set)")
            return None
        
        try:
            # Try /search endpoint first
            search_url = f"{self.crawl4ai_url.rstrip('/')}/search"
            try:
                resp = self._http.get(search_url, params={"q": query}, timeout=10)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        if isinstance(data, dict):
                            for key in ("results", "content", "answer"):
                                if key in data and data[key]:
                                    result = data[key]
                                    if isinstance(result, list):
                                        return "\n".join(str(r) for r in result[:3])
                                    return str(result)
                        elif isinstance(data, list) and data:
                            return "\n".join(str(r) for r in data[:3])
                    except ValueError:
                        return resp.text
            except Exception as e:
                self.logger.debug("Crawl4AI /search endpoint failed: %s", e)
            
            # Fallback: try /ask endpoint
            ask_url = f"{self.crawl4ai_url.rstrip('/')}/ask"
            try:
                resp = self._http.get(
                    ask_url,
                    params={"q": f"Search for information about: {query}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        for key in ("answer", "text", "result", "response"):
                            if key in data and data[key]:
                                return str(data[key])
                        return str(data)
                    except ValueError:
                        return resp.text
            except Exception as e:
                self.logger.debug("Crawl4AI /ask escalation failed: %s", e)
        
        except Exception as e:
            self.logger.warning("Crawl4AI escalation error: %s", e)
        
        return None

    def _get_collection(self):
        if chromadb is None or self.client is None:
            return None

        try:
            # get_or_create_collection may not exist on all versions; prefer get_collection then create
            try:
                return self.client.get_collection(name=self.COLLECTION_NAME)
            except Exception:
                return self.client.create_collection(name=self.COLLECTION_NAME)
        except Exception as e:
            print(f"Error getting/creating Chroma collection: {e}")
            return self.reset()

    def _convert_item_to_entry(self, item: DataItem) -> dict:
        """Legacy helper retained for compatibility; not used by chromadb flow."""
        vector = self.get_vector(item.content)
        return {"vector": vector, "content": item.content, "source": item.source}
