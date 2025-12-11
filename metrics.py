import numpy as np
import scanpy as sc
import anndata
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from config import CELL_TYPE_COL

def evaluate_clustering(embeddings, adata):
    tmp_adata = anndata.AnnData(X=embeddings)
    tmp_adata.obs = adata.obs.copy()
    
    sc.pp.neighbors(tmp_adata, use_rep='X')
    sc.tl.leiden(tmp_adata, resolution=0.5)
    
    if CELL_TYPE_COL not in tmp_adata.obs:
        print("Warning: Ground truth labels missing. Returning ARI=0.")
        return 0.0
        
    labels_true = tmp_adata.obs[CELL_TYPE_COL]
    labels_pred = tmp_adata.obs['leiden']
    
    return adjusted_rand_score(labels_true, labels_pred)

def evaluate_stability(clean_embeddings, noisy_embeddings):
    norm_clean = np.linalg.norm(clean_embeddings, axis=1)
    norm_noisy = np.linalg.norm(noisy_embeddings, axis=1)
    
    norm_clean[norm_clean == 0] = 1e-9
    norm_noisy[norm_noisy == 0] = 1e-9
    
    dot_products = np.sum(clean_embeddings * noisy_embeddings, axis=1)
    similarities = dot_products / (norm_clean * norm_noisy)
    
    return np.mean(similarities)
