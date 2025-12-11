import pandas as pd
import os
from config import (
    MODELS, SUBSAMPLE_RATIOS, DROPOUT_RATES, 
    RESULTS_DIR, BATCH_GENE_SIGMA, BATCH_LIBRARY_FACTOR
)
from data_loader import load_raw_data, load_embeddings
from perturbations import apply_subsampling, apply_dropout, apply_batch_effect
from metrics import evaluate_clustering, evaluate_stability, evaluate_batch_integration

def run_benchmark():
    print("Loading Ground Truth Data...")
    adata_full = load_raw_data()
    
    results = []

    print("\n--- Starting Subsampling Benchmark ---")
    for ratio in SUBSAMPLE_RATIOS:
        print(f"Processing Subsample Ratio: {ratio}")
        
        adata_sub = apply_subsampling(adata_full, ratio)
        
        for model in MODELS:
            emb = load_embeddings(model, "subsample", ratio, adata_sub)
            ari = evaluate_clustering(emb, adata_sub)
            
            results.append({
                "Model": model,
                "Experiment": "Subsampling",
                "Parameter": ratio,
                "Metric": "ARI",
                "Value": ari
            })

    print("\n--- Starting Dropout Benchmark ---")
    clean_embeddings_cache = {}
    
    for rate in DROPOUT_RATES:
        print(f"Processing Dropout Rate: {rate}")
        
        adata_noisy = apply_dropout(adata_full, rate)
        
        for model in MODELS:
            emb_noisy = load_embeddings(model, "dropout", rate, adata_noisy)
            
            ari = evaluate_clustering(emb_noisy, adata_noisy)
            results.append({
                "Model": model,
                "Experiment": "Dropout",
                "Parameter": rate,
                "Metric": "ARI",
                "Value": ari
            })
            
            if rate == 0.0:
                clean_embeddings_cache[model] = emb_noisy
            else:
                if model in clean_embeddings_cache:
                    emb_clean = clean_embeddings_cache[model]
                    stability = evaluate_stability(emb_clean, emb_noisy)
                    results.append({
                        "Model": model,
                        "Experiment": "Dropout",
                        "Parameter": rate,
                        "Metric": "Cosine_Stability",
                        "Value": stability
                    })

    print("\n--- Starting Batch Effect Benchmark ---")
    adata_batch = apply_batch_effect(
        adata_full, 
        gene_sigma=BATCH_GENE_SIGMA, 
        library_factor=BATCH_LIBRARY_FACTOR
    )
    
    for model in MODELS:
        print(f"Processing Batch Effect for {model}")
        emb_batch = load_embeddings(model, "batch", "simulated", adata_batch)
        
        ari_bio = evaluate_clustering(emb_batch, adata_batch)
        results.append({
            "Model": model,
            "Experiment": "Batch_Effect",
            "Parameter": "Simulated",
            "Metric": "ARI (Bio Conservation)",
            "Value": ari_bio
        })
        
        sil_batch = evaluate_batch_integration(emb_batch, adata_batch)
        results.append({
            "Model": model,
            "Experiment": "Batch_Effect",
            "Parameter": "Simulated",
            "Metric": "Batch_Silhouette (Lower is Better)",
            "Value": sil_batch
        })

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nBenchmark Complete. Results saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    run_benchmark()
