"""
train_finetuned_knn.py
Train a kNN classifier using embeddings from the fine-tuned MiniLM model.
"""

import os
import joblib
import time
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier


def train_finetuned_knn(X_train, y_train, cfg):
    """
    Train kNN using embeddings from the fine-tuned MiniLM model.
    Returns the trained classifier, embeddings, and timing info.
    """
    start = time.time()

    # --- Load fine-tuned model ---
    model_path = cfg["fine_tuning"]["output_dir"]
    model = SentenceTransformer(model_path)

    # --- Generate embeddings ---
    X_train = X_train.astype(str).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    emb_train = model.encode(X_train, show_progress_bar=False)

    # --- Train kNN classifier ---
    k = cfg["models"]["knn_k"]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(emb_train, y_train)

    elapsed = round(time.time() - start, 2)

    print(f" Fine-tuned MiniLM embeddings generated and kNN trained in {elapsed}s")

    return knn, model, emb_train, elapsed
