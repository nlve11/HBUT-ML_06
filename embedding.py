import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """固定在 CPU 上运行的小型多语言句向量模型。"""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("待编码文本不能为空")
        vectors = self.model.encode(
            texts, batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
