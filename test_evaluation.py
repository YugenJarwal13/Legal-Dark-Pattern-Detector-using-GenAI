import pandas as pd
import json
from src.pipeline import LegalAIPipeline
from src.evaluate import Evaluator
from src.visualize import Visualizer
import os

def extract_reference_explanation(output_str, loader):
    parsed = loader.extract_json(output_str)
    if parsed and "explanation" in parsed:
        return parsed["explanation"]
    return ""

def main():
    print("Initializing Pipeline...")
    pipeline = LegalAIPipeline()
    
    print("Loading dataset...")
    df = pd.read_csv("data/results/results_finetuned.csv")
    
    sample_df = df.sample(n=30, random_state=42).copy()
    
    results_list = []
    
    print(f"Processing {len(sample_df)} clauses...")
    for idx, row in sample_df.iterrows():
        print(f"Processing {idx + 1}/{len(sample_df)}...")
        clause = row["clause"]
        true_label = row["true_label"]
        raw_output = row["output"]
        
        ref_explanation = extract_reference_explanation(raw_output, pipeline.loader)
        
        out = pipeline.process(clause)
        
        row_data = {
            "clause": clause,
            "true_label": true_label,
            "reference_explanation": ref_explanation
        }
        
        for model_name, res in out.items():
            if res:
                row_data[f"{model_name}_risk_status"] = res.get("risk_status", "Unknown")
                row_data[f"{model_name}_category"] = res.get("dark_pattern_category", "General Terms")
                row_data[f"{model_name}_explanation"] = res.get("explanation", "")
                row_data[f"{model_name}_statute"] = res.get("violated_statute", "None")
                row_data[f"{model_name}_confidence"] = res.get("confidence", 0)
                
        results_list.append(row_data)
        
    results_df = pd.DataFrame(results_list)
    
    os.makedirs("data/results", exist_ok=True)
    results_df.to_csv("data/results/evaluation_raw.csv", index=False)
    
    print("Running evaluation...")
    evaluator = Evaluator()
    metrics_df = evaluator.evaluate_batch(results_df)
    
    metrics_df.to_csv("data/results/evaluation_table.csv")
    print("\n=== METRICS ===")
    print(metrics_df)
    
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    visualizer.generate_all_plots(metrics_df)
    
    print("\n✅ Evaluation complete. Check data/plots/ and data/results/evaluation_table.csv")
    print("\n⚠️ IMPORTANT INTERPRETATION RULE:")
    print("BLEU/ROUGE scores are used to measure linguistic similarity with reference explanations, not correctness.")
    print("Higher scores indicate closer phrasing, but not necessarily better legal reasoning.")

if __name__ == "__main__":
    main()
