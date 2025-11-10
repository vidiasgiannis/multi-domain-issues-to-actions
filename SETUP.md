Setup & Operations Guide — Multi-Domain Issues-to-Actions Agent Pipeline
============================================================

This guide covers environment setup, configuration, running the notebook, optional Azure OpenAI for action wording, troubleshooting, and publishing to GitHub.

------------------------------------------------------------
A) Prerequisites (WSL/Ubuntu)
------------------------------------------------------------
Open Ubuntu (WSL) and install:

  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip git make build-essential libopenblas-dev

Note:
If your project lives on Windows (e.g., C:\Users\<You>\Documents\genai\multi-domain-issues-to-actions),
the WSL path is:
  /mnt/c/Users/<You>/Documents/genai/multi-domain-issues-to-actions

------------------------------------------------------------
B) Virtual Environment
------------------------------------------------------------
From the repo root:

  python3 -m venv .venv
  source .venv/bin/activate

Upgrade pip & install dependencies:

  pip install --upgrade pip
  pip install -r requirements.txt

If faiss-cpu fails on first try:

  sudo apt install -y libopenblas-dev
  pip install --no-cache-dir faiss-cpu

------------------------------------------------------------
C) Configuration
------------------------------------------------------------
Edit configs/config.yaml (defaults shown):

data:
  path: ../data/sample_issues.csv   # Path to the synthetic dataset

models:
  tfidf:
    ngram_min: 1                 
    ngram_max: 2                 
    min_df: 2                    
  embedder: all-MiniLM-L6-v2     
  knn_k: 3                                         

eval:
  random_seed: 42                
  test_size: 0.2                 
  metrics: ["accuracy", "f1", "precision", "recall"]


Common tweaks:
- Change source_domain / target_domain to explore different shifts
- Vary knn_k for a quick ablation (1,3,5,7,9)
- Adjust TF-IDF n-grams / min_df for precision–recall trade-offs

------------------------------------------------------------
D) Run the demo Notebook
------------------------------------------------------------
Start Jupyter:

  make demo
  # or
  jupyter notebook notebooks/presentation.ipynb

Follow the cells in order:
1. Load config & data; show distributions
2. Run all models → benchmark table (macro-F1 ranked)
3. Confusion matrix for the best model
4. RAG actions: retrieve top-k KB snippets and generate 3 steps (LLM or fallback)
5. (Optional) Ablations: vary k for kNN, tweak TF-IDF settings

------------------------------------------------------------
E) Optional: Azure OpenAI for Action Wording
------------------------------------------------------------
Everything runs locally by default. To use Azure OpenAI GPT-4o for the final 3-bullet action summary:

1) Create a .env file in the repo root (do not commit):

  AZURE_OPENAI_API_KEY=YOUR_KEY
  AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com
  AZURE_OPENAI_API_VERSION= ...
  AZURE_OPENAI_DEPLOYMENT_ID= ...

2) Install minimal deps:

  pip install --upgrade openai>=1.42.0 python-dotenv

------------------------------------------------------------
F) Quick Recap
------------------------------------------------------------
1. Create venv → pip install -r requirements.txt
2. (If needed) python build_project.py
3. make demo → run notebook cells in order
4. Optional .env → Azure OpenAI for action wording


