"""
RAG retriever: Top-K fraud pattern retrieval from Qdrant.

Given SHAP feature values or transaction context, retrieves the most relevant
fraud patterns from the vector store. These are then passed to the LLM for
natural language explanation generation.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

warnings.filterwarnings("ignore")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "fraud_patterns"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3

ROOT = Path(__file__).resolve().parents[2]
PATTERNS_DIR = ROOT / "data" / "fraud_patterns"


# ---------------------------------------------------------------------------
# In-memory fallback: used when Qdrant is unreachable (e.g. production ECS)
# ---------------------------------------------------------------------------

class _LocalFallback:
    """
    Pure-numpy nearest-neighbour retrieval over the bundled fraud pattern JSONs.
    Zero external dependencies beyond sentence-transformers (already required).
    Loaded once at retriever init; patterns are tiny (~50 docs).
    """

    def __init__(self, patterns_dir: Path, embedding_model: SentenceTransformer):
        self._model = embedding_model
        self._patterns: list[dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._load(patterns_dir)

    def _load(self, patterns_dir: Path) -> None:
        if not patterns_dir.exists():
            print(f"[LocalFallback] Patterns dir not found: {patterns_dir}")
            return
        for fp in sorted(patterns_dir.glob("*.json")):
            try:
                with open(fp) as f:
                    self._patterns.append(json.load(f))
            except Exception as e:
                print(f"[LocalFallback] Skipping {fp.name}: {e}")

        if not self._patterns:
            return

        texts = [f"{p['name']}. {p['description']}" for p in self._patterns]
        self._matrix = self._model.encode(texts, convert_to_numpy=True)
        # Pre-normalise rows for fast cosine via dot product
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True).clip(1e-9)
        self._matrix = self._matrix / norms
        print(f"[LocalFallback] {len(self._patterns)} patterns loaded in memory")

    def query(self, query_text: str, top_k: int) -> list[dict]:
        if self._matrix is None or len(self._patterns) == 0:
            return []
        q = self._model.encode(query_text, convert_to_numpy=True)
        q = q / max(np.linalg.norm(q), 1e-9)
        sims = self._matrix @ q
        indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in indices:
            p = self._patterns[idx]
            results.append({
                "pattern_id": p.get("id"),
                "name": p.get("name"),
                "description": p.get("description"),
                "fraud_rate_pct": p.get("fraud_rate_pct", 0),
                "ieee_cis_context": p.get("ieee_cis_context", ""),
                "typical_amount": p.get("typical_amount", "mixed"),
                "feature_signatures": p.get("feature_signatures", []),
                "shap_signature": p.get("shap_signature", {}),
                "similarity_score": float(sims[idx]),
            })
        return results


class FraudPatternRetriever:
    """Retrieve relevant fraud patterns from Qdrant based on SHAP features."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Initialize retriever.

        Args:
            qdrant_url: URL to Qdrant instance
            collection_name: Name of Qdrant collection
            top_k: Number of patterns to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._local = _LocalFallback(PATTERNS_DIR, self.embedding_model)
        self._available = False
        try:
            self.client = QdrantClient(url=qdrant_url, timeout=3)
            self.client.get_collections()  # probe connection
            self._available = True
            print(f"[Retriever] Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            self.client = None
            print(f"[Retriever] Qdrant unavailable ({e}); using in-memory retrieval")

    def retrieve_by_shap_values(
        self, shap_values: dict[str, float], top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Retrieve fraud patterns similar to SHAP feature drivers.

        Args:
            shap_values: Dict of {feature_name: shap_value} (can be positive or negative)
            top_k: Override default top_k

        Returns:
            List of retrieved pattern dicts with similarity scores
        """
        top_k = top_k or self.top_k

        # Create query text from top SHAP features
        # Use feature names as the retrieval signal
        feature_names = list(shap_values.keys())
        query_text = " ".join(feature_names)

        return self._query(query_text, top_k)

    def retrieve_by_transaction_context(
        self,
        transaction_summary: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve fraud patterns based on transaction summary text.

        Args:
            transaction_summary: Plain-text description of transaction (e.g.,
                "High-value international transaction from new device with email mismatch")
            top_k: Override default top_k

        Returns:
            List of retrieved pattern dicts with similarity scores
        """
        top_k = top_k or self.top_k
        return self._query(transaction_summary, top_k)

    def _query(self, query_text: str, top_k: int) -> list[dict]:
        """
        Execute similarity search. Uses Qdrant when available, local fallback otherwise.
        """
        if not self._available:
            return self._local.query(query_text, top_k)

        try:
            # Embed query
            query_embedding = self.embedding_model.encode(
                query_text,
                convert_to_numpy=True,
            ).tolist()

            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )

            # Format results
            patterns = []
            for scored_point in results:
                payload = scored_point.payload
                patterns.append({
                    "pattern_id": payload.get("pattern_id"),
                    "name": payload.get("name"),
                    "description": payload.get("description"),
                    "fraud_rate_pct": payload.get("fraud_rate_pct"),
                    "ieee_cis_context": payload.get("ieee_cis_context"),
                    "typical_amount": payload.get("typical_amount"),
                    "feature_signatures": payload.get("feature_signatures", []),
                    "shap_signature": payload.get("shap_signature", {}),
                    "similarity_score": scored_point.score,
                })

            return patterns

        except Exception as e:
            print(f"[Retriever] Error querying Qdrant: {e}")
            return []

    def get_pattern_by_id(self, pattern_id: str) -> Optional[dict]:
        """Retrieve a specific pattern by ID."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "pattern_id",
                            "match": {"value": pattern_id},
                        }
                    ]
                },
                limit=1,
            )

            if results[0]:
                payload = results[0][0].payload
                return {
                    "pattern_id": payload.get("pattern_id"),
                    "name": payload.get("name"),
                    "description": payload.get("description"),
                    "fraud_rate_pct": payload.get("fraud_rate_pct"),
                    "ieee_cis_context": payload.get("ieee_cis_context"),
                    "typical_amount": payload.get("typical_amount"),
                    "feature_signatures": payload.get("feature_signatures", []),
                    "shap_signature": payload.get("shap_signature", {}),
                }
            return None

        except Exception as e:
            print(f"[Retriever] Error retrieving pattern {pattern_id}: {e}")
            return None

    def get_all_patterns_summary(self) -> list[dict]:
        """Get summary of all patterns in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            total_count = info.points_count

            # Scroll through all points
            patterns = []
            offset = 0
            limit = 100

            while offset < total_count:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=offset,
                    limit=limit,
                )

                for point in results:
                    payload = point.payload
                    patterns.append({
                        "pattern_id": payload.get("pattern_id"),
                        "name": payload.get("name"),
                        "fraud_rate_pct": payload.get("fraud_rate_pct"),
                    })

                offset = next_offset
                if offset is None or offset == 0:
                    break

            return patterns

        except Exception as e:
            print(f"[Retriever] Error getting all patterns: {e}")
            return []


# Singleton instance for API
_retriever: Optional[FraudPatternRetriever] = None


def get_retriever(top_k: int = DEFAULT_TOP_K) -> FraudPatternRetriever:
    """Lazy-load and return singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = FraudPatternRetriever(top_k=top_k)
    return _retriever
