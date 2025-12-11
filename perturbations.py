import numpy as np
import scipy.sparse as sp
from config import RANDOM_SEED

def apply_subsampling(adata, retention_ratio):
    if retention_ratio >= 1.0:
        return adata.copy()

    n_cells = adata.shape[0]
    n_keep = int(n_cells * retention_ratio)
    
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(n_cells, n_keep, replace=False)
    indices.sort()
    
    return adata[indices].copy()

def apply_dropout(adata, dropout_rate):
    if dropout_rate <= 0.0:
        return adata.copy()
    
    adata_noisy = adata.copy()
    X = adata_noisy.X

    if sp.issparse(X):
        X = X.tocoo()
        n_nonzero = X.nnz
        n_drop = int(n_nonzero * dropout_rate)
        
        rng = np.random.default_rng(RANDOM_SEED)
        drop_indices = rng.choice(n_nonzero, n_drop, replace=False)
        
        mask = np.ones(n_nonzero, dtype=bool)
        mask[drop_indices] = False
        
        new_data = X.data[mask]
        new_row = X.row[mask]
        new_col = X.col[mask]
        
        adata_noisy.X = sp.coo_matrix((new_data, (new_row, new_col)), shape=X.shape).tocsr()
        
    else:
        flat_X = X.flatten()
        nonzero_indices = np.nonzero(flat_X)[0]
        n_drop = int(len(nonzero_indices) * dropout_rate)
        
        rng = np.random.default_rng(RANDOM_SEED)
        drop_indices = rng.choice(nonzero_indices, n_drop, replace=False)
        
        flat_X[drop_indices] = 0
        adata_noisy.X = flat_X.reshape(X.shape)

    return adata_noisy
