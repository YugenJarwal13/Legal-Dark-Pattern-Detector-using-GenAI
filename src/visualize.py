import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class Visualizer:
    def __init__(self, output_dir="data/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_classification_metrics(self, df):
        plt.figure(figsize=(10, 6))
        plot_data = df[["Accuracy", "F1_Score"]].reset_index()
        plot_data = plot_data.melt(id_vars="index", var_name="Metric", value_name="Score")
        
        sns.barplot(data=plot_data, x="index", y="Score", hue="Metric", palette="viridis")
        plt.title("Classification Metrics across Models")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "classification_metrics.png"))
        plt.close()

    def plot_nlg_metrics(self, df):
        plt.figure(figsize=(8, 6))
        groq_df = df[df.index.str.contains("groq")]
        if groq_df.empty:
            return
            
        plot_data = groq_df[["Avg_BLEU", "Avg_ROUGE"]].reset_index()
        plot_data = plot_data.melt(id_vars="index", var_name="Metric", value_name="Score")
        
        sns.barplot(data=plot_data, x="index", y="Score", hue="Metric", palette="magma")
        plt.title("NLG Metrics (Groq Models Only)")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        
        plt.figtext(0.5, 0.01, 
                   "*Note: BLEU/ROUGE are computed against proxy references, not human ground truth.", 
                   ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                   
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(self.output_dir, "nlg_metrics.png"))
        plt.close()

    def plot_hallucinations(self, df):
        if "Hallucination_Count" not in df.columns:
            return
            
        plt.figure(figsize=(8, 5))
        plot_data = df[["Hallucination_Count"]].reset_index()
        
        sns.barplot(data=plot_data, x="index", y="Hallucination_Count", hue="index", palette="Reds_r", legend=False)
        plt.title("Hallucination Count by Model")
        plt.xlabel("Models")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "hallucinations.png"))
        plt.close()

    def generate_all_plots(self, df):
        self.plot_classification_metrics(df)
        self.plot_nlg_metrics(df)
        self.plot_hallucinations(df)
