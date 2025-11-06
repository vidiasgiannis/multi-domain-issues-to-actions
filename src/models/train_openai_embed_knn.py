"""
train_openai_embed_knn.py
Train and evaluate a kNN classifier using Azure OpenAI embeddings.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.features.azure_openai_embedder import embed_texts
from sklearn.model_selection import train_test_split
import pandas as pd
import time

def run_openai_embed_knn(data_path="data/sample_issues.csv", k=3, test_size=0.2, random_state=42):
    # Load data
    df = pd.read_csv(data_path)
    X = df["issue"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(" Generating Azure OpenAI embeddings...")
    start = time.time()
    emb_train = embed_texts(X_train.tolist())
    emb_test = embed_texts(X_test.tolist())
    end_emb = time.time()

    print(f"Embeddings generated in {end_emb - start:.2f} seconds")

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(emb_train, y_train)

    # Predict and evaluate
    preds = knn.predict(emb_test)
    f1 = f1_score(y_test, preds, average="macro")
    total_time = time.time() - start

    print(f"Macro-F1: {f1:.3f}")
    print(f"Total runtime (incl. embedding): {total_time:.2f}s")

    return {"f1": f1, "time": total_time}

