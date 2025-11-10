"""
train_openai_embed_knn.py
Train and evaluate a kNN classifier using Azure OpenAI embeddings.
"""

import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.features.azure_openai_embedder import embed_text  # must exist and return list/array of embeddings


def run_openai_embed_knn(
    data_path="data/sample_issues.csv",
    k=3,
    test_size=0.2,
    random_state=42,
    text_col="description",
    label_col="category"
):
    """Train and evaluate a kNN classifier using Azure OpenAI embeddings."""
    # 1️ Load data
    print("\n[Azure+kNN] Loading dataset...", flush=True)
    df = pd.read_csv(data_path)
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    # 2️ Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3 Generate embeddings
    print("[Azure+kNN] Generating Azure OpenAI embeddings...", flush=True)
    start = time.time()
    emb_train = embed_text(X_train.tolist())
    emb_test = embed_text(X_test.tolist())
    end_emb = time.time()

    print(f"[Azure+kNN] Embeddings generated in {end_emb - start:.2f} seconds", flush=True)

    # 4 Train kNN
    print("[Azure+kNN] Training kNN classifier...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(emb_train, y_train)

    # 5️ Predict and evaluate
    preds = knn.predict(emb_test)
    f1 = f1_score(y_test.values, preds, average="macro")
    total_time = time.time() - start

    print(f"[Azure+kNN]  Macro-F1: {f1:.3f}", flush=True)
    print(f"[Azure+kNN] Total runtime (incl. embedding): {total_time:.2f}s", flush=True)

    return {"f1": f1, "time": total_time}
