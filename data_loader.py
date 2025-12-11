import os
import numpy as np
import scanpy as sc
import anndata
from config import RAW_DATA_PATH, EMBEDDING_DIR

def load_raw_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Warning: {RAW_DATA_PATH} not found. Generating dummy PBMC-like data.")
        return sc.datasets.pbmc3k_processed()
    return sc.read_h5ad(RAW_DATA_PATH)

def load_embeddings(model_name, condition_type, condition_param, adata_current):
    if model_name == "PCA":
        sc.pp.pca(adata_current, n_comps=50)
        return adata_current.obsm['X_pca']

    filename = f"{condition_type}_{condition_param}.npy"
    file_path = os.path.join(EMBEDDING_DIR, model_name, filename)
    
    if os.path.exists(file_path):
        print(f"Loading {model_name} embeddings from {file_path}")
        emb = np.load(file_path)
        
        if emb.shape[0] != adata_current.shape[0]:
            raise ValueError(f"Shape mismatch! Embedding has {emb.shape[0]} cells, "
                             f"Data has {adata_current.shape[0]} cells.")
        return emb
    else:
        print(f"Placeholder: File {file_path} not found. Generating random embeddings.")
        return np.random.rand(adata_current.shape[0], 512)
