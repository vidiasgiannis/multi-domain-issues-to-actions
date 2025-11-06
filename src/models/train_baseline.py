
# TF-IDF + Logistic Regression baseline model training

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
def train_baseline(X_train, y_train, cfg):
    tfidf_cfg = cfg["models"]["tfidf"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(tfidf_cfg["ngram_min"], tfidf_cfg["ngram_max"]),
                                  min_df=tfidf_cfg["min_df"])),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)
    return pipe
