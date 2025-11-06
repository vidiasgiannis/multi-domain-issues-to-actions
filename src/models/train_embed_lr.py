# SBERT + Logistic Regression model training

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_embed_lr(data_path="data/sample_issues.csv", embed_model="all-MiniLM-L6-v2",
                 test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    texts, labels = df["issue"].astype(str), df["label"].astype(str)

    model = SentenceTransformer(embed_model)
    X = model.encode(texts, show_progress_bar=True)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"Macro-F1: {f1:.3f}")
    return f1
