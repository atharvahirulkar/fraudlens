"""
Ingest fraud patterns into Qdrant vector store.

Loads ~50 JSON pattern documents from data/fraud_patterns/, embeds them
using all-MiniLM-L6-v2 (sentence-transformers), and indexes into Qdrant.

Run as script:
    python -m src.rag.ingest

Qdrant must be running:
    docker run -d -p 6333:6333 qdrant/qdrant
"""

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

warnings.filterwarnings("ignore")

# Configuration
ROOT = Path(__file__).resolve().parents[2]
PATTERNS_DIR = ROOT / "data" / "fraud_patterns"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "fraud_patterns"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# - Vector store management


class FramdsPatternStore:
    """Interface to Qdrant vector store for fraud patterns."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ):
        """Initialize connection to Qdrant."""
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Verify collection exists; create if needed
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            print(f"[Qdrant] Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"[Qdrant] Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Qdrant] ✓ Collection created")

    def ingest_patterns(self, patterns_dir: Path = PATTERNS_DIR) -> int:
        """
        Load all JSON patterns and index into Qdrant.

        Args:
            patterns_dir: Directory containing *.json pattern files

        Returns:
            Number of patterns indexed
        """
        if not patterns_dir.exists():
            raise FileNotFoundError(f"Patterns directory not found: {patterns_dir}")

        patterns_files = sorted(patterns_dir.glob("*.json"))
        if not patterns_files:
            raise ValueError(f"No JSON files found in {patterns_dir}")

        print(f"[Ingest] Found {len(patterns_files)} pattern files")

        points = []
        for idx, filepath in enumerate(patterns_files):
            try:
                with open(filepath, "r") as f:
                    pattern = json.load(f)

                # Extract text for embedding
                # Combine name + description for semantic richness
                text_to_embed = f"{pattern['name']}. {pattern['description']}"

                # Generate embedding
                embedding = self.embedding_model.encode(
                    text_to_embed,
                    convert_to_numpy=True,
                ).tolist()

                # Create point with metadata
                point = PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "pattern_id": pattern["id"],
                        "name": pattern["name"],
                        "description": pattern["description"],
                        "fraud_rate_pct": pattern.get("fraud_rate_pct", 0),
                        "ieee_cis_context": pattern.get("ieee_cis_context", ""),
                        "typical_amount": pattern.get("typical_amount", "mixed"),
                        "feature_signatures": pattern.get("feature_signatures", []),
                        "shap_signature": pattern.get("shap_signature", {}),
                        "source_file": filepath.name,
                    },
                )
                points.append(point)
                print(f"  ✓ {filepath.name}")

            except Exception as e:
                print(f"  ⚠ {filepath.name}: {e}")
                continue

        # Upsert all points to collection
        if points:
            print(f"\n[Ingest] Upserting {len(points)} vectors to Qdrant...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"[Ingest] ✓ All vectors indexed")

        return len(points)

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_config": info.config.params.vectors,
        }


# - CLI


def main():
    """Ingest all fraud patterns into Qdrant."""
    print("[Ingest] Starting fraud pattern ingestion...\n")

    try:
        store = FramdsPatternStore()
        n_ingested = store.ingest_patterns()

        info = store.get_collection_info()
        print(f"\n[Ingest] ✓ Complete!")
        print(f"  Collection: {info['name']}")
        print(f"  Total points: {info['points_count']}")
        print(f"  Vector dim: {info['vectors_config']['size']}")
        print(f"  Distance metric: {info['vectors_config']['distance']}")

    except Exception as e:
        print(f"\n[Ingest] ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
