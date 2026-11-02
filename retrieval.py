import json
from pathlib import Path

import numpy as np

from embedding import Embedder
from schemas import Chunk, SearchHit


class Retriever:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("片段数量与向量行数不一致")
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, output_dir: Path) -> "Retriever":
        output_dir.mkdir(parents=True, exist_ok=True)
        vectors = embedder.encode([chunk.text for chunk in chunks])
        np.save(output_dir / "embeddings.npy", vectors)
        data = [chunk.model_dump() for chunk in chunks]
        (output_dir / "chunks.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return cls(chunks, vectors, embedder)

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        if not question.strip():
            raise ValueError("问题不能为空")
        k = min(max(1, top_k), len(self.chunks))
        query = self.embedder.encode([question])[0]
        scores = self.vectors @ query
        order = np.argsort(-scores)[:k]
        return [SearchHit(chunk=self.chunks[i],
                          score=float(scores[i])) for i in order]
