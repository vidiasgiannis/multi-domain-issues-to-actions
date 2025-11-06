
# TF-IDF + Support Vector Machine model training

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

def train_svm(X_train, y_train, cfg):
    tfidf_cfg = cfg["models"]["tfidf"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(tfidf_cfg["ngram_min"], tfidf_cfg["ngram_max"]),
            min_df=tfidf_cfg["min_df"]
        )),
        ("clf", LinearSVC())
    ])
    pipe.fit(X_train, y_train)
    return pipe
