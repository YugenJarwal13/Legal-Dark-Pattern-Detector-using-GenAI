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
4. Links to relevant **GDPR regulations** using context-aware retrieval

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

We compare five distinct approaches across a Hybrid Inference Architecture:
1. **Mistral Base** (Offline Lookup)
2. **Mistral Fine-Tuned** (Offline Lookup)
3. **Groq Base** (Live API Inference)
4. **Groq Few-Shot** (Live API Inference)
5. **Groq RAG** (Live API Inference with GDPR context)

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
Saved LoRA Adapter Outputs
↓
VS Code System
├── RAG (ChromaDB + GDPR)
├── Hybrid Inference Pipeline (Live Groq + Offline Mistral)
├── Metrics Evaluator
└── UI Dashboard (Streamlit)
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

## 🤖 Models Used

**Base & Fine-Tuned Models (Colab / Offline):**
- Mistral-7B-Instruct
- QLoRA (4-bit quantization) Fine-Tuning

**Live Inference Models (Groq):**
- Fast LLM inference via Groq API (Zero-shot, Few-shot, RAG)

---

## 🔍 RAG (Retrieval-Augmented Generation)

We use GDPR as external legal knowledge:
- Article 5 – Principles
- Article 6 – Lawfulness
- Article 7 – Consent
- Article 13–14 – Transparency
- Article 17 – Right to erasure

Stored in:
- **ChromaDB** vector database with Smart Re-Ranking logic.

---

## 📈 Evaluation Metrics

### Classification:
- Accuracy, Precision, Recall, F1-score

### NLG Metrics:
- BLEU, ROUGE (to measure textual similarity vs reasoning depth)

### Reliability:
- Hallucination count
- Confidence scores

*Note: The system generates a comprehensive evaluation report in the `data/results/` folder, which is rendered dynamically in the dashboard.*

---

## 🖥️ UI Dashboard

Built using:
- **Streamlit**

Features:
- Input a ToS clause
- View structured comparisons across all 5 models side-by-side
- Distinct visual highlighting for context-aware RAG explanations
- Offline fallback handling for non-live models
- Live rendering of evaluation metrics (Accuracy, F1, Hallucinations)

---

## 📁 Project Structure

```text
project/
├── data/
│   ├── gdpr.txt
│   └── results/ (Evaluation CSVs)
├── notebooks/
├── src/
│   ├── data_prep.py
│   ├── rag.py
│   ├── inference.py
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── colab_loader.py
│   └── utils.py
├── app.py
├── report.md / report.docx
├── requirements.txt
└── README.md
```

---

## 🚀 Workflow

### Phase 1 (Colab):
- Dataset cleaning
- Instruction dataset creation
- Fine-tuning (QLoRA)
- Export model inferences to CSV

### Phase 2 (VS Code):
- RAG implementation
- Hybrid inference system
- Full Pipeline Evaluation
- Interactive UI Dashboard

---

## 🧠 Key Contributions

- Comparison of 5 distinct LLM approaches
- Demonstration of hallucination reduction using RAG (zero hallucinations)
- Innovative hybrid inference architecture solving local compute limits
- Comprehensive evaluation pipeline generating professional metrics reports

---

## 🧑‍💻 Author

Yugen Jarwal  
B.Tech CSE  

---

## 📜 License

This project is for academic and research purposes.
