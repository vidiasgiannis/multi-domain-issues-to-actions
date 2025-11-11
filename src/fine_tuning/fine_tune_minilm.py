"""
fine_tune_minilm.py
Fine-tune the all-MiniLM-L6-v2 model on issue classification data.
"""

import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def fine_tune_minilm(cfg):
    """Fine-tune MiniLM on issue classification using SoftmaxLoss."""
    
    # --- Load parameters from config ---
    data_path = cfg["data"]["path"]
    model_name = cfg["fine_tuning"]["model_name"]
    epochs = cfg["fine_tuning"]["epochs"]
    batch_size = cfg["fine_tuning"]["batch_size"]
    test_size = cfg["eval"]["test_size"]
    output_dir = cfg["fine_tuning"]["output_dir"]
    random_seed = cfg["eval"]["random_seed"]

    os.makedirs(output_dir, exist_ok=True)

    # --- Load dataset and split ---
    df = pd.read_csv(data_path)[["description", "category"]]
    train_df, _ = train_test_split(df, test_size=test_size, stratify=df["category"], random_state=random_seed)

    # --- Encode category labels as integers ---
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["category"])

    # --- Build training examples (text + integer label) ---
    train_examples = [
        InputExample(texts=[row.description, row.description], label=int(row.label))
        for _, row in train_df.iterrows()
    ]

    # --- Initialize model and loss ---
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, drop_last=True)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(le.classes_)
    )

    # --- Fine-tuning process ---
    print(f"Fine-tuning {model_name} for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        show_progress_bar=True,
        warmup_steps=100,                # small warmup for stability
        output_path=output_dir
    )

    # --- Save model and label encoder ---
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
    print(f"Fine-tuned model saved to {output_dir}")

    return model, le
