"""
evaluate.py
Benchmark multiple models (TF-IDF and Embedding-based) and measure training time.
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sentence_transformers import SentenceTransformer

from src.models.train_baseline import train_baseline
from src.models.train_svm import train_svm
from src.models.train_embed_knn import train_embed_knn
from src.models.train_embed_lr import run_embed_lr


def run_evaluation(cfg):
    # 1️ Load data
    df = pd.read_csv(cfg["data"]["path"])
    X = df["issue"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg["eval"]["random_seed"], stratify=y
    )

    results = {}
    timings = {}

    # 2️ TF-IDF + Logistic Regression
    print("\n Training TF-IDF + Logistic Regression...")
    start = time.time()
    baseline_model = train_baseline(X_train, y_train, cfg)
    preds_base = baseline_model.predict(X_test)
    end = time.time()
    results["TFIDF+LR"] = f1_score(y_test, preds_base, average="macro")
    timings["TFIDF+LR"] = end - start

    # 3️ TF-IDF + SVM
    print("\n Training TF-IDF + SVM...")
    start = time.time()
    svm_model = train_svm(X_train, y_train, cfg)
    preds_svm = svm_model.predict(X_test)
    end = time.time()
    results["TFIDF+SVM"] = f1_score(y_test, preds_svm, average="macro")
    timings["TFIDF+SVM"] = end - start

    # 4️ SBERT + kNN
    print("\n Training SBERT + kNN...")
    emb_model = SentenceTransformer(cfg["models"]["embedder"])
    start = time.time()
    emb_train = emb_model.encode(X_train, show_progress_bar=True)
    emb_test = emb_model.encode(X_test, show_progress_bar=True)
    knn_model = train_embed_knn(emb_train, y_train, k=cfg["models"]["knn_k"])
    preds_knn = knn_model.predict(emb_test)
    end = time.time()
    results["SBERT+kNN"] = f1_score(y_test, preds_knn, average="macro")
    timings["SBERT+kNN"] = end - start

    # 5️5 SBERT + Logistic Regression
    print("\n▶ Training SBERT + Logistic Regression...")
    start = time.time()
    f1_embed_lr = run_embed_lr(cfg["data"]["path"], cfg["models"]["embedder"])
    end = time.time()
    results["SBERT+LR"] = f1_embed_lr
    timings["SBERT+LR"] = end - start

    # 6️ Report results
    print("\n=== BENCHMARK RESULTS ===")
    df_results = pd.DataFrame({
        "Model": results.keys(),
        "Macro-F1": [round(v, 3) for v in results.values()],
        "Time (s)": [round(t, 2) for t in timings.values()]
    }).sort_values(by="Macro-F1", ascending=False)

    print(df_results.to_string(index=False))
    print("\nTotal runtime: {:.2f} seconds".format(sum(timings.values())))

    # Return DataFrame for plotting in notebook
    return df_results


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_evaluation(cfg)
