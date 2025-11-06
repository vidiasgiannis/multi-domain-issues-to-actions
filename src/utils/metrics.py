from typing import Dict
import time
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

def confusion_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

class Timer:
    def __enter__(self): self.t0=time.time(); return self
    def __exit__(self, *args): self.elapsed=time.time()-self.t0
