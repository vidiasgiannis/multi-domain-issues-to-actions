from sentence_transformers import SentenceTransformer
import numpy as np
class SBERTEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        X = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X
