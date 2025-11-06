# Multi-Domain Issues-to-Actions
### An Automated Agent for Cross-Domain Issue Classification and Knowledge-Augmented Action Generation

---

## 1. Purpose
The goal of this project is to **transform unstructured issue reports into data-driven actions** using a mix of traditional analytics, semantic modeling, and generative AI.
It demonstrates how a data scientist can build an **end-to-end decision-support system** that classifies issues, retrieves relevant knowledge, and recommends next steps automatically.

---

## 2. Scope of the Project

### Problem Overview
Organizations receive thousands of textual incident reports or support tickets every day.
Manually triaging and responding to them is time-consuming and inconsistent.

This project demonstrates how to:
- Analyze text issues from multiple domains (Retail, Fintech, etc.)
- Detect patterns and classify issues automatically
- Retrieve relevant documentation or procedures
- Generate recommended actions using Generative AI (RAG)

---

## 3. What the System Does

| Stage | Description | Outcome |
|--------|--------------|----------|
| **Data Understanding** | Load and explore the dataset of issues | Identify domain and label distribution |
| **Preprocessing** | Clean text, build TF-IDF and embedding features | Structured numerical data |
| **Model Training** | Train and benchmark multiple models | Compare speed and accuracy |
| **Semantic Search (RAG)** | Retrieve top-k relevant docs using embeddings | Provide context for generation |
| **Action Generation** | Use GPT-based model to summarize next steps | Produce human-readable recommendations |

---

## 4. Models Included

| Model | Technique | Purpose |
|--------|------------|----------|
| TF-IDF + Logistic Regression | Classical ML | Interpretable baseline |
| TF-IDF + SVM | Classical ML | Stronger linear model |
| SBERT + kNN | Semantic embeddings | Meaning-based classification |
| SBERT + Logistic Regression | Semantic + efficient | Embedding-driven classifier |
| Azure OpenAI + kNN | Cloud embeddings | Zero-training semantic approach |
| *(Optional)* GPT-4 Zero-Shot | Large Language Model | Direct classification |

---

## 5. Analytical Focus
This project focuses on **analytical reasoning** and **data understanding**, not just modeling.

Key analyses include:
- Class imbalance and label distribution
- Key TF-IDF terms per category
- Runtime vs. accuracy trade-offs
- Model explainability and insights
- Semantic similarity evaluation

---

## 6. Generative AI Integration (RAG)
The **Retrieval-Augmented Generation (RAG)** module retrieves the **top-k relevant documents** based on cosine similarity of embeddings and feeds them into a GPT-based model.

This ensures that generated recommendations are **context-aware**, **grounded**, and **actionable**.

**Example flow:**
```
Issue → Embedding → Retrieve top-2 KB snippets → GPT generates recommended actions
```

---

## 7. Business Impact
- Faster response time — automated classification and suggested actions
- Consistent support — knowledge-based recommendations
- Reusable pipeline — adaptable across business domains
- Scalable — ready for enterprise deployment with Azure OpenAI

---

## 8. Presentation & Demonstration
The demo includes:
1. Data overview and label distribution
2. Benchmark comparison of all models
3. Example RAG workflow (issue → retrieval → actions)
4. Analytical insights (runtime, accuracy, interpretability)
5. Discussion of improvements (FAISS, fine-tuning, feedback loops)

---

## 9. Repository Structure
```
multi-domain-issues-to-actions/
 ┣ data/                # Sample issues & KB documents
 ┣ src/
 ┃ ┣ features/          # Text preprocessing & embeddings
 ┃ ┣ models/            # Training & evaluation scripts
 ┃ ┣ rag/               # Retrieval and action generation
 ┣ notebooks/           # Presentation notebooks
 ┣ configs/             # Configuration files
 ┣ README.md            # Project overview
 ┣ setup.md             # Installation and environment guide
```

---

## 10. Future Work
- Implement FAISS for faster semantic retrieval
- Add Streamlit dashboard for interactive demos
- Fine-tune small transformer for domain-specific adaptation
- Introduce feedback loop for continuous learning

---

## 11. Key Takeaway
This project bridges **data science** and **generative AI** — showing how structured analytics and LLM reasoning can work together to convert raw text issues into intelligent, explainable actions.
