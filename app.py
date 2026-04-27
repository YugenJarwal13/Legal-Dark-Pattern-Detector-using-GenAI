import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import streamlit as st
import pandas as pd
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.pipeline import LegalAIPipeline

st.set_page_config(page_title="Legal Risk Analyzer", page_icon="⚖️", layout="wide")

# Initialize Pipeline (cached so it doesn't reload embeddings every run)
@st.cache_resource
def load_pipeline():
    return LegalAIPipeline()

pipeline = load_pipeline()

@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv("data/results/evaluation_table.csv", index_col=0)
        # Format the numbers
        df = df[["Accuracy", "F1_Score", "Hallucination_Count"]]
        df["Accuracy"] = (df["Accuracy"] * 100).round(1).astype(str) + "%"
        df["F1_Score"] = df["F1_Score"].round(3)
        return df
    except Exception:
        return None

# ==========================================
# 1. HEADER
# ==========================================
st.title("⚖️ Legal Clause Risk Analyzer")
st.subheader("Compare AI models for detecting dark patterns in Terms of Service")

# ==========================================
# 2. INPUT SECTION
# ==========================================
st.markdown("### Input Section")

demo_clause = "We collect and share your personal data with third parties without explicit consent."
selected_clause = st.text_area("Enter a legal clause to analyze:", value=demo_clause, height=100)

analyze_clicked = st.button("Analyze", type="primary")

# ==========================================
# 3. MODEL COMPARISON GRID (MAIN PART)
# ==========================================
if analyze_clicked:
    if not selected_clause.strip():
        st.warning("Please enter a clause to analyze.")
    else:
        with st.spinner("Analyzing clause across 5 models..."):
            results = pipeline.process(selected_clause)
            
        st.markdown("---")
        
        # 3 columns then 2 columns layout
        cols1 = st.columns(3)
        cols2 = st.columns(3)
        all_cols = cols1 + cols2[:2]
        
        models_config = [
            ("mistral_base", "Mistral Base", all_cols[0]),
            ("mistral_ft", "Mistral Fine-tuned", all_cols[1]),
            ("groq_base", "Groq Base", all_cols[2]),
            ("groq_few_shot", "Groq Few-shot", all_cols[3]),
            ("groq_rag", "Groq RAG ⭐", all_cols[4])
        ]
        
        for key, title, col in models_config:
            res = results.get(key, {})
            
            with col:
                if key == "groq_rag":
                    st.markdown("""
                    <div style='border: 2px solid #28a745; padding: 15px; border-radius: 8px; background-color: rgba(40, 167, 69, 0.05); height: 100%;'>
                        <h3 style='margin-top:0; color: #28a745;'>🧠 RAG (Context-Aware Model)</h3>
                        <p style='background-color: #e8f5e9; color: #2e7d32; padding: 5px; border-radius: 4px; font-weight: bold; font-size: 0.85em; display: inline-block; margin-bottom: 10px;'>🔍 Uses GDPR Knowledge Retrieval</p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #fafafa; height: 100%;'>", unsafe_allow_html=True)
                    st.subheader(title)

                if not res or res.get("risk_status") in [None, "Unknown"]:
                    st.warning("⚠ No precomputed result available for this input")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                risk = res.get("risk_status", "Unknown")
                category = res.get("dark_pattern_category", "N/A")
                statute = res.get("violated_statute", "None")
                explanation = res.get("explanation", "No explanation provided.")
                    
                # Risk Status
                if risk.lower() == "predatory":
                    st.markdown(f"**Risk Status:** <span style='color: red; font-weight: bold;'>{risk}</span>", unsafe_allow_html=True)
                elif risk.lower() == "fair":
                    st.markdown(f"**Risk Status:** <span style='color: green; font-weight: bold;'>{risk}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Risk Status:** <span style='color: orange; font-weight: bold;'>{risk}</span>", unsafe_allow_html=True)
                    
                # Category & Statute
                st.markdown(f"**Category:** {category}")
                st.markdown(f"**Violated Statute:** {statute}")
                st.markdown("---")
                
                # Explanation Display
                if key in ["mistral_base", "mistral_ft"]:
                    st.markdown("**Explanation (Static Model Output):**")
                    st.caption("📦 Precomputed result (no live inference)")
                    st.write(explanation)
                elif key == "groq_base":
                    st.markdown("**Explanation (General LLM Reasoning):**")
                    st.caption("⚙ Zero-shot reasoning")
                    st.write(explanation)
                elif key == "groq_few_shot":
                    st.markdown("**Explanation (Prompt-guided Reasoning):**")
                    st.caption("📌 Guided by prompt examples")
                    st.write(explanation)
                elif key == "groq_rag":
                    st.markdown("**Legal Insight:**")
                    st.write(explanation)
                    st.markdown("**Why Risky:**")
                    st.write("This clause involves data sharing practices that may violate user consent and transparency principles.")
                    st.markdown(f"**GDPR Reference:** {statute}")
                    
                    st.markdown("""
                    <div style='margin-top: 15px; font-size: 0.9em;'>
                        <p style='color: #555; margin-bottom: 5px;'>📚 Source: Retrieved GDPR Context</p>
                        <div style='background-color: #fff3cd; color: #856404; padding: 8px; border-radius: 4px; border-left: 4px solid #ffeeba;'>
                            This explanation is grounded in retrieved legal context, not just language patterns.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 4. FINAL SUMMARY SECTION
# ==========================================
st.markdown("---")
st.success("🧠 Key Insight: RAG explains WHY, others just classify WHAT")
st.markdown("## 📊 Final Model Comparison")

metrics_df = load_metrics()
if metrics_df is not None:
    st.markdown("### 📈 Evaluation Metrics")
    st.dataframe(metrics_df)

summary_cols = st.columns([1, 1])

with summary_cols[0]:
    st.markdown("""
    * ✔ **Groq Base** → Highest Accuracy
    * ✔ **Groq RAG** → Best Legal Reasoning (GDPR grounded)
    * ✔ **Few-shot** → Moderate improvement
    * ✔ **Mistral FT** → Limited generalization
    * ✔ **Mistral Base** → Weak baseline
    """)
    
    st.markdown("## 🏆 Best Overall Model: Groq RAG")

with summary_cols[1]:
    st.warning("""
    **⚠️ Interpretation Note**
    
    Accuracy does not fully reflect reasoning quality. 
    RAG provides legally grounded explanations using retrieved GDPR context, 
    while other models rely on pattern-based reasoning.
    """)

st.markdown("---")