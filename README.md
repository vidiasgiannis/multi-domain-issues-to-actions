# Multi-Domain Issue Classification and Decision Support
### A Generative AI–Powered Analytical System for Multi-Department Issue Understanding

---

## 1. Purpose
This project demonstrates an **end-to-end decision-support pipeline**
that classifies textual issue reports from different business departments and analyzes their patterns.

The system combines **traditional machine learning**, **semantic embeddings**, and **Generative AI**
to benchmark models and explore how language representations affect performance across domains.

---

## 2. Scope of the Project

### Problem Overview
Organizations receive hundreds of textual issue reports daily —
ranging from IT incidents to analytics errors or marketing performance drops.

Manually routing and prioritizing them is slow and inconsistent.
This project automates that process through classification and analytics.

### What the system demonstrates
- Analyze and visualize issue distributions and priorities
- Clean and preprocess unstructured text
- Benchmark multiple classification models
- Compare **semantic**, **statistical**, and **LLM-based** representations
- Generate analytical insights and improvement directions

---

## 3. System Workflow

| Stage | Description | Output |
|--------|--------------|---------|
| **Data Understanding** | Load and explore synthetic issue dataset | Department & priority distributions |
| **Preprocessing** | Normalize and clean text | Ready-to-train corpus |
| **Model Training** | Fit multiple models (TF-IDF, SBERT, Azure Embeddings) | Classification results |
| **Evaluation** | Compare F1, accuracy, and confusion matrices | Ranked performance table |
| **Interpretation** | Analyze feature importance & embeddings | Explainable insights |

---

## 4. Models Included

| Model | Type | Description |
|--------|------|-------------|
| TF-IDF + Logistic Regression | Classical ML | Lightweight, interpretable baseline |
| TF-IDF + SVM | Classical ML | Strong linear baseline often used in text classification |
| SBERT + kNN | Semantic Embedding | Meaning-based nearest-neighbor classification |
| Azure OpenAI Embeddings + kNN | Cloud Embedding | LLM-based semantic approach |
| *(Optional)* Feedforward Neural Network | Deep Learning | For feature-based comparisons |

---

## 5. Analytical Focus
The notebook highlights analytical reasoning and data exploration rather than only modeling.

Key aspects:
- Class balance and diversity assessment
- Text length and lexical variability
- Model performance comparison (speed vs. accuracy)
- Semantic variance and embedding-space visualization
- Impact of embeddings vs. traditional TF-IDF features

---

## 6. Generative AI Integration (as future direction)
Although this version focuses on classification and analytics,
future extensions may include **Retrieval-Augmented Generation (RAG)**
to produce context-aware action recommendations from classified issues.

**Potential flow:**
```
Issue → Embedding → Retrieve similar issues → Generate recommended next steps
```

---

## 7. Business Impact
- **Automation:** Reduces manual effort in issue triaging
- **Transparency:** Analytical breakdowns for each model
- **Scalability:** Works across departments and industries
- **Extensibility:** Can evolve into a full RAG or agent-based support system

---

## 8. Repository Structure
```
multi-domain-issues-to-actions/
 ┣ data/                # Synthetic issue dataset
 ┣ src/                 # Scripts for data, features, and models
 ┣ notebooks/           # Main presentation notebook (app.ipynb)
 ┣ configs/             # Config and experiment parameters
 ┣ setup.md             # Environment setup guide
 ┣ README.md            # Project overview (this file)
```

---

## 9. Future Work
- Add FAISS for scalable semantic retrieval
- Extend to RAG-based action generation
---

