import os
import logging
from typing import List, Any
from pathlib import Path
from interface.base_datastore import DataItem
from interface.base_indexer import BaseIndexer

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    _HAS_DOCLING = True
except Exception:
    _HAS_DOCLING = False


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Indexer(BaseIndexer):
    """Indexer with optional `docling` support and configurable ignore list.

    If `docling` is installed the original behavior is used. Otherwise a
    lightweight fallback splits plain text files into fixed-size chunks with
    overlap. Chunk size and overlap can be configured via environment vars:
    `INDEXER_CHUNK_SIZE` and `INDEXER_CHUNK_OVERLAP` (defaults: 500, 50).
    
    Files matching patterns in the ignore list are skipped during indexing.
    Configure via `INDEXER_IGNORE_PATTERNS` env var (comma-separated, e.g., "test,tmp,ignore").
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.chunk_size = int(chunk_size or os.environ.get("INDEXER_CHUNK_SIZE", 500))
        self.overlap = int(overlap or os.environ.get("INDEXER_CHUNK_OVERLAP", 50))
        
        # Parse ignore patterns from environment variable
        ignore_patterns_str = os.environ.get("INDEXER_IGNORE_PATTERNS", "test")
        self.ignore_patterns = [p.strip().lower() for p in ignore_patterns_str.split(",") if p.strip()]
        logger.info("Indexer ignore patterns: %s", self.ignore_patterns)

        if _HAS_DOCLING:
            logger.info("docling detected — using DocumentConverter + HybridChunker")
            try:
                self.converter = DocumentConverter()
                self.chunker = HybridChunker()
            except Exception as e:
                logger.warning("Failed to initialize docling components: %s — falling back", e)
                self.converter = None
                self.chunker = None
        else:
            logger.info("docling not available — using simple text chunker fallback")
            self.converter = None
            self.chunker = None

    def _should_ignore(self, document_path: str) -> bool:
        """Check if a document path matches any ignore pattern."""
        path_lower = Path(document_path).name.lower()
        for pattern in self.ignore_patterns:
            if pattern in path_lower:
                logger.debug("Ignoring file (matches pattern '%s'): %s", pattern, document_path)
                return True
        return False

    def index(self, document_paths: List[str]) -> List[DataItem]:
        items: List[DataItem] = []
        for document_path in document_paths:
            # Skip files matching ignore patterns
            if self._should_ignore(document_path):
                continue
            
            try:
                if self.converter and self.chunker:
                    # Use docling pipeline
                    result = self.converter.convert(document_path)
                    chunks = list(self.chunker.chunk(result.document))
                    items.extend(self._items_from_chunks_docling(chunks))
                else:
                    # Fallback: treat as plain text file and chunk
                    items.extend(self._index_plain_text(document_path))
            except Exception as e:
                logger.error("Failed to index '%s': %s", document_path, e)
        return items

    def _items_from_chunks_docling(self, chunks: List[Any]) -> List[DataItem]:
        # Using List[Any] avoids needing a direct docling import for types
        out: List[DataItem] = []
        for chunk in chunks:
            try:
                content = getattr(chunk, "text", str(chunk))
                source = "unknown"
                meta = getattr(chunk, "meta", None)
                if meta and getattr(meta, "doc_items", None):
                    try:
                        source = meta.doc_items[0].self_ref
                    except Exception:
                        source = str(getattr(meta, "doc_items", ""))
                out.append(DataItem(content=content, source=source))
            except Exception as e:
                logger.debug("Skipping malformed chunk: %s", e)
        return out

    def _index_plain_text(self, document_path: str) -> List[DataItem]:
        out: List[DataItem] = []
        try:
            with open(document_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            logger.error("Could not read file '%s': %s", document_path, e)
            return out

        # Simple whitespace tokenizer for chunking
        words = text.split()
        if not words:
            return out

        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            out.append(DataItem(content=chunk_text, source=document_path))
            i += max(1, self.chunk_size - self.overlap)

        logger.info("Indexed %d chunks from %s", len(out), document_path)
        return out
