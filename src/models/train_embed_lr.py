import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

def run_embed_lr(
    data_path="data/sample_issues.csv",
    embed_model="all-MiniLM-L6-v2",
    test_size=0.2,
    random_state=42,
    text_col="description",
    label_col="category"
):
    """Train Logistic Regression on SBERT embeddings and return Macro-F1."""
    # 1 Load data
    df = pd.read_csv(data_path)
    texts = df[text_col].astype(str)
    labels = df[label_col].astype(str)

    # 2 Encode text
    print("\nEncoding text embeddings...", flush=True)
    model = SentenceTransformer(embed_model)
    X = model.encode(texts, show_progress_bar=True)
    y = np.array(labels)

    # 3 Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4 Train Logistic Regression
    print("Training Logistic Regression on embeddings...", flush=True)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # 5️ Evaluate
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")
    print(f" SBERT+LR Macro-F1: {f1:.3f}", flush=True)

    return f1
