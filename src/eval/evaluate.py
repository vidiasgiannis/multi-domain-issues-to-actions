# evaluate.py (refactored for persistent results)
import os
import time
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sentence_transformers import SentenceTransformer

from src.models.train_xgboost import train_xgboost
from src.models.train_baseline import train_baseline
from src.models.train_svm import train_svm
from src.models.train_embed_knn import train_embed_knn
from src.models.train_openai_embed_knn import run_openai_embed_knn


def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def compute_metrics(y_true, y_pred):
    """Return multiple metrics in a dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro")
    }


def save_results(model_name, y_true, y_pred, metrics):
    """Save predictions and metrics to /results/model_name/."""
    folder = f"../results/{model_name}"
    os.makedirs(folder, exist_ok=True)

    # Save predictions
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    }).to_csv(f"{folder}/predictions.csv", index=False)

    # Save metrics
    pd.DataFrame([metrics]).to_csv(f"{folder}/metrics.csv", index=False)

    print(f"Saved results for {model_name} to {folder}", flush=True)
    return metrics


def run_tfidf_lr(X_train, X_test, y_train, y_test, cfg):
    log("Training TF-IDF + Logistic Regression...")
    start = time.time()
    model = train_baseline(X_train, y_train, cfg)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "TFIDF_LR"
    metrics["time_sec"] = round(time.time() - start, 2)
    return save_results("tfidf_lr", y_test, y_pred, metrics)


def run_svm(X_train, X_test, y_train, y_test, cfg):
    log("Training TF-IDF + SVM...")
    start = time.time()
    model = train_svm(X_train, y_train, cfg)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "TFIDF_SVM"
    metrics["time_sec"] = round(time.time() - start, 2)
    return save_results("tfidf_svm", y_test, y_pred, metrics)


def run_sbert_knn(X_train, X_test, y_train, y_test, cfg):
    log("Training SBERT + kNN...")
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    embedder = SentenceTransformer(cfg["models"]["embedder"])
    start = time.time()

    emb_train = embedder.encode(X_train, show_progress_bar=False)
    emb_test = embedder.encode(X_test, show_progress_bar=False)
    knn = train_embed_knn(emb_train, y_train, k=cfg["models"]["knn_k"])

    y_pred = knn.predict(emb_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "SBERT_KNN"
    metrics["time_sec"] = round(time.time() - start, 2)
    return save_results("sbert_knn", y_test, y_pred, metrics)


def run_openai_knn(X_train, X_test, y_train, y_test, cfg):
    log("Training Azure OpenAI + kNN...")
    start = time.time()
    res = run_openai_embed_knn(
        data_path=cfg["data"]["path"],
        k=cfg["models"]["knn_k"],
        test_size=cfg["eval"]["test_size"],
        random_state=cfg["eval"]["random_seed"]
    )
    metrics = {
        "model": "Azure_KNN",
        "f1_macro": res["f1"],
        "time_sec": round(res["time"], 2)
    }
    return save_results("openai_knn", y_train, y_train, metrics)  


def run_xgboost(X_train, X_test, y_train, y_test, cfg):
    log("Training TF-IDF + XGBoost...")

    start = time.time()
    model = train_xgboost(X_train, y_train, cfg)
    y_pred_encoded = model.predict(X_test)

    # Decode numeric predictions back to original labels
    label_encoder = model.named_steps["xgb"].label_encoder_
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    elapsed = round(time.time() - start, 2)

    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "TFIDF_XGBOOST"
    metrics["time_sec"] = elapsed

    save_results("tfidf_xgboost", y_test, y_pred, metrics)
    log(f" Done TF-IDF + XGBoost in {elapsed:.2f}s | F1: {metrics['f1_macro']:.3f}")
    return metrics





def aggregate_results():
    """Aggregate all individual model metrics."""
    frames = []
    for sub in os.listdir("../results"):
        metrics_path = f"../results/{sub}/metrics.csv"
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            frames.append(df)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv("../results/metrics_summary.csv", index=False)
    return summary
