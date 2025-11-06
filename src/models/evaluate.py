from .train_baseline import train_baseline
from .train_embed_knn import train_embed_knn
from ..features.embedder import SBERTEmbedder
from ..utils.metrics import classification_metrics, confusion_df, Timer
import pandas as pd

def run_all(df_src, df_tgt, cfg):
    # Baseline
    with Timer() as t_train:
        baseline = train_baseline(df_src.text.values, df_src.category.values, cfg)
    y_pred_tgt = baseline.predict(df_tgt.text.values)
    base_metrics = classification_metrics(df_tgt.category.values, y_pred_tgt)

    # Embeddings + kNN
    embedder = SBERTEmbedder(cfg["models"]["embedder"])
    with Timer() as t_emb_src:
        emb_src = embedder.encode(df_src.text.values)
    with Timer() as t_emb_tgt:
        emb_tgt = embedder.encode(df_tgt.text.values)
    knn = train_embed_knn(emb_src, df_src.category.values, cfg["models"]["knn_k"])
    with Timer() as t_knn_pred:
        y_pred_knn = knn.predict(emb_tgt)
    knn_metrics = classification_metrics(df_tgt.category.values, y_pred_knn)

    summary = pd.DataFrame({
        "TFIDF+LR (Target)": base_metrics,
        "Embeddings+kNN (Target)": knn_metrics
    }).T
    latencies = {
        "baseline_train_s": t_train.elapsed,
        "embed_src_s": t_emb_src.elapsed,
        "embed_tgt_s": t_emb_tgt.elapsed,
        "knn_predict_s": t_knn_pred.elapsed
    }
    return summary, latencies, y_pred_tgt, y_pred_knn
