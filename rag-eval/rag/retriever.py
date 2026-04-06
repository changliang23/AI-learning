from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from services.ollama_client import OllamaClient


@dataclass
class RetrievedChunk:
    text: str
    score: float


class QdrantRetriever:
    def __init__(self, collection_name: str = "rag_eval_docs") -> None:
        self.collection_name = collection_name
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = OllamaClient(model_name=os.getenv("OLLAMA_EMBED_MODEL", "qwen3:8b"))
        self.vector_size = self._detect_vector_size()
        self._ensure_collection()

    def _detect_vector_size(self) -> int:
        probe = self.embedder.embed("vector size probe")
        return len(probe)

    def _ensure_collection(self) -> None:
        existing = [item.name for item in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def rebuild_collection(self, documents: List[str]) -> None:
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        points = []
        for index, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id=index,
                    vector=self.embedder.embed(doc),
                    payload={"text": doc, "doc_id": str(uuid4())},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_vector = self.embedder.embed(query)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [RetrievedChunk(text=hit.payload["text"], score=float(hit.score)) for hit in hits]
