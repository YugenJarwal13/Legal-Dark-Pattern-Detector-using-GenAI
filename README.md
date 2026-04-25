# 🧠 Legal Dark Pattern Detector using GenAI (RAG + Fine-Tuning)

## 📌 Project Overview

This project focuses on detecting **dark patterns in Terms of Service (ToS)** using Generative AI.

Dark patterns are deceptive or unfair clauses hidden in legal documents that:
- misuse user data
- enforce unfair policies
- reduce user control

We build a system that:
1. Identifies whether a clause is **Predatory or Fair**
2. Classifies the **type of dark pattern**
3. Generates a **human-readable explanation**
4. (Optionally) links to relevant **GDPR regulations**

---

## 🎯 Problem Statement

Terms of Service documents are:
- long and complex  
- rarely read by users  
- often contain hidden risks  

This project aims to:
> Automatically analyze ToS clauses and detect harmful patterns using LLMs.

---

## 🧪 Research Objective

We evaluate:

> **Does fine-tuning + retrieval improve legal reasoning in LLMs compared to base prompting?**

We compare four approaches:
- Base Model
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)
- Fine-Tuned Model (QLoRA)

---

## 🏗️ System Architecture

```text
ToS;DR Dataset
↓
Data Cleaning & Preprocessing
↓
Instruction-Response Dataset (JSONL)
↓
Fine-Tuning (QLoRA in Colab)
↓
Saved LoRA Adapter
↓
VS Code System
├── RAG (ChromaDB + GDPR)
├── Inference (4 modes)
├── SQLite Storage
└── Evaluation + UI (Streamlit)
```

---

## 📊 Dataset

### Primary Dataset:
- **ToS;DR (Terms of Service; Didn’t Read)**

Contains:
- Real-world ToS clauses
- Human-annotated labels:
  - `good` → Fair
  - `bad/blocker` → Predatory
- Category tags (`tosdr_class`)

---

## 🤖 Model

Base Model:
- Mistral-7B-Instruct

Fine-Tuning:
- QLoRA (4-bit quantization)
- Trained on instruction-response legal dataset

---

## 🔍 RAG (Retrieval-Augmented Generation)

We use GDPR as external legal knowledge:
- Article 5 – Principles
- Article 6 – Lawfulness
- Article 7 – Consent
- Article 13–14 – Transparency
- Article 17 – Right to erasure

Stored in:
- ChromaDB vector database

---

## ⚙️ Inference Modes

| Mode | Description |
|------|------------|
| Base | No prompt engineering |
| Prompt | Few-shot prompting |
| RAG | Adds GDPR context |
| Fine-tuned | QLoRA + RAG |

---

## 📈 Evaluation Metrics

### Quantitative:
- Accuracy
- Precision / Recall / F1-score
- ROUGE-L (for explanation quality)
- Citation Accuracy (approximate)

### Qualitative:
- Hallucination analysis
- False positives
- Explanation comparison

---

## 💾 Data Storage

- **ChromaDB** → GDPR retrieval
- **SQLite** → audit history

---

## 🖥️ UI

Built using:
- Streamlit

Features:
- Input ToS clause
- Select model mode
- View structured JSON output
- View past audit history

---

## 📁 Project Structure

```text
project/
├── data/
├── notebooks/
├── models/
├── src/
│ ├── data_prep.py
│ ├── rag.py
│ ├── inference.py
│ ├── evaluate.py
│ └── storage.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Workflow

### Phase 1 (Colab):
- Dataset cleaning
- Instruction dataset creation
- Fine-tuning (QLoRA)

### Phase 2 (VS Code):
- RAG implementation
- Inference system
- Evaluation
- UI

---

## 🧠 Key Contributions

- Comparison of 4 LLM approaches
- Demonstration of hallucination reduction using RAG
- Improvement using parameter-efficient fine-tuning
- Practical legal AI application

---

## ⚠️ Limitations

- No ground-truth legal statute mapping
- Small dataset size
- Simplified legal reasoning

---

## 📌 Future Work

- Larger datasets (CUAD, LegalBench)
- Better legal grounding
- Multi-document reasoning
- Real-time browser extension

---

## 🧑💻 Author

Yugen Jarwal  
B.Tech CSE  

---

## 📜 License

This project is for academic and research purposes.
