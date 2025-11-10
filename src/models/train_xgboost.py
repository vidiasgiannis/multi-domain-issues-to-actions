from sklearn.calibration import LabelEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def train_xgboost(X_train, y_train, cfg):
    """Train an XGBoost model on TF-IDF text features."""
    tfidf_cfg = cfg["models"]["tfidf"]

    # Encode Labels (XGBoos requires numeric labels)
    label_encoder = LabelEncoder()
    y_trained_encoded = label_encoder.fit_transform(y_train)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(tfidf_cfg["ngram_min"], tfidf_cfg["ngram_max"]),
            min_df=tfidf_cfg["min_df"]
        )),
        ("xgb", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=cfg["eval"]["random_seed"],
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1
        ))
    ])

    pipe.fit(X_train, y_trained_encoded)
    pipe.named_steps["xgb"].label_encoder_ = label_encoder
    return pipe