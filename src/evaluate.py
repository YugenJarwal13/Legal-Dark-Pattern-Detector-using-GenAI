import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import normalize_true_label

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def compute_nlg_metrics(self, pred, ref):
        if not ref or not pred:
            return 0.0, 0.0
            
        bleu = sentence_bleu(
            [ref.split()],
            pred.split(),
            smoothing_function=self.smooth
        )
        rouge = self.scorer.score(ref, pred)["rougeL"].fmeasure
        return bleu, rouge

    def evaluate_batch(self, results_df):
        metrics = {}
        models = ["mistral_base", "mistral_ft", "groq_base", "groq_few_shot", "groq_rag"]
        
        y_true = results_df["true_label"].apply(normalize_true_label).tolist()
        y_true_bin = [1 if y == "Predatory" else 0 for y in y_true]

        for model in models:
            if f"{model}_risk_status" not in results_df.columns:
                continue

            preds = results_df[f"{model}_risk_status"].tolist()
            preds_bin = [1 if p == "Predatory" else 0 for p in preds]
            
            acc = accuracy_score(y_true_bin, preds_bin)
            prec = precision_score(y_true_bin, preds_bin, zero_division=0)
            rec = recall_score(y_true_bin, preds_bin, zero_division=0)
            f1 = f1_score(y_true_bin, preds_bin, zero_division=0)
            
            metrics[model] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
            }
            
            # Confidence
            conf_col = f"{model}_confidence"
            if conf_col in results_df.columns:
                confidences = results_df[conf_col].tolist()
                metrics[model]["Avg_Confidence"] = sum(confidences) / len(confidences) if confidences else 0
                hallucinations = sum(1 for c in confidences if c == 1)
                metrics[model]["Hallucination_Count"] = hallucinations
            
            # NLG Metrics
            if "groq" in model:
                bleus, rouges = [], []
                for _, row in results_df.iterrows():
                    pred_exp = row.get(f"{model}_explanation", "")
                    ref_exp = row.get("reference_explanation", "")
                    b, r = self.compute_nlg_metrics(str(pred_exp), str(ref_exp))
                    bleus.append(b)
                    rouges.append(r)
                
                metrics[model]["Avg_BLEU"] = sum(bleus) / len(bleus) if bleus else 0
                metrics[model]["Avg_ROUGE"] = sum(rouges) / len(rouges) if rouges else 0
            else:
                metrics[model]["Avg_BLEU"] = 0
                metrics[model]["Avg_ROUGE"] = 0

        return pd.DataFrame(metrics).T