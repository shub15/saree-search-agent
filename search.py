"""
search.py
---------
FAISS index wrapper for cosine-similarity saree retrieval.

The index is built once (scripts/build_index.py) and committed to the repo.
At runtime the app loads it from data/ with zero rebuild cost.

Index type: IndexFlatIP (exact inner-product search on L2-normalised vectors)
            ≡ exact cosine similarity — no quantisation loss for ~1 K vectors.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (relative to repo root; override via env vars for testing)
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("SAREE_DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "metadata.json")


# ---------------------------------------------------------------------------
# Data model for a single search result
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    rank: int
    filename: str
    similarity: float          # cosine similarity ∈ [0, 1]
    thumbnail_b64: str = ""    # base64-encoded JPEG thumbnail (for display)
    # Rich product fields from CSV
    name: str = ""
    sku: str = ""
    retail_price: str = ""
    discounted_price: str = ""
    stock: str = ""
    image_url: str = ""
    website_link: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "filename": self.filename,
            "similarity": round(self.similarity, 4),
            "thumbnail_b64": self.thumbnail_b64,
            "name": self.name,
            "sku": self.sku,
            "retail_price": self.retail_price,
            "discounted_price": self.discounted_price,
            "stock": self.stock,
            "image_url": self.image_url,
            "website_link": self.website_link,
        }


# ---------------------------------------------------------------------------
# Index manager
# ---------------------------------------------------------------------------
class SareeIndex:
    """
    Wraps a FAISS index + JSON metadata file.

    Usage:
        idx = SareeIndex.load()
        results = idx.search(query_vector, top_k=5)
    """

    def __init__(self, index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
        self._index = index
        self._meta = metadata  # list[dict], each entry keyed by "filename" etc.

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        index_path: str = INDEX_PATH,
        meta_path: str = META_PATH,
    ) -> "SareeIndex":
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run  python scripts/build_index.py  to build it first."
            )
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata file not found at '{meta_path}'. "
                "Run  python scripts/build_index.py  to build it first."
            )

        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.info(
            "Loaded FAISS index with %d vectors (dim=%d)", index.ntotal, index.d
        )
        return cls(index, metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        exclude_index: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Cosine similarity search.

        Parameters
        ----------
        query:
            L2-normalised float32 vector of shape (dim,) or (1, dim).
        top_k:
            Number of results to return (capped at index size).
        exclude_index:
            Optional index position to exclude (e.g. when the query image
            itself is in the index).

        Returns
        -------
        List of SearchResult, sorted by similarity descending.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)

        # Fetch a few extra in case we exclude the query itself
        k_fetch = min(top_k + 5, self._index.ntotal)
        scores, indices = self._index.search(query, k_fetch)

        results: List[SearchResult] = []
        seen = set()
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx in seen:
                continue
            if exclude_index is not None and idx == exclude_index:
                continue
            seen.add(idx)

            meta = self._meta[idx]
            # Raw inner product on L2-normed vectors ∈ [-1, 1].
            # Map to [0, 1] for a friendlier "similarity score".
            similarity = float((score + 1.0) / 2.0)

            results.append(
                SearchResult(
                    rank=rank,
                    filename=meta.get("filename", f"image_{idx}"),
                    similarity=similarity,
                    thumbnail_b64=meta.get("thumbnail_b64", ""),
                    name=meta.get("name", ""),
                    sku=meta.get("sku", ""),
                    retail_price=meta.get("retail_price", ""),
                    discounted_price=meta.get("discounted_price", ""),
                    stock=meta.get("stock", ""),
                    image_url=meta.get("image_url", ""),
                    website_link=meta.get("website_link", ""),
                )
            )
            rank += 1
            if rank > top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._index.d
