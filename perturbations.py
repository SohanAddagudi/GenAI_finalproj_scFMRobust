import numpy as np
import scipy.sparse as sp
from config import RANDOM_SEED, BATCH_COL

def apply_subsampling(adata, retention_ratio):
    if retention_ratio >= 1.0:
        return adata.copy()

    n_cells = adata.shape[0]
    n_keep = int(n_cells * retention_ratio)
    
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(n_cells, n_keep, replace=False)
    indices.sort()
    
    return adata[indices].copy()

def apply_splatter_dropout(counts, dropout_mid, dropout_shape):
    X = counts.copy()
    
    if sp.issparse(X):
        data_view = X.data
    else:
        data_view = X

    log_expression = np.log(data_view + 1)
    
    exponent = -dropout_shape * (log_expression - dropout_mid)
    exponent = np.clip(exponent, -100, 100)
    
    dropout_prob = 1 / (1 + np.exp(exponent))
    
    rng = np.random.default_rng(RANDOM_SEED)
    if sp.issparse(X):
        random_vals = rng.random(data_view.shape)
    else:
        random_vals = rng.random(X.shape)
    
    keep_mask = random_vals >= dropout_prob
    
    if sp.issparse(X):
        X.data = data_view * keep_mask
        X.eliminate_zeros()
    else:
        X = X * keep_mask
        
    return X

def apply_dropout(adata, dropout_rate):
    if dropout_rate <= 0.0:
        return adata.copy()

    X = adata.X
    if sp.issparse(X):
        expr_values = X.data
    else:
        expr_values = X.flatten()
        expr_values = expr_values[expr_values > 0]

    log_expr = np.log(expr_values + 1)
    
    midpoint = np.percentile(log_expr, dropout_rate * 100)
    
    new_X = apply_splatter_dropout(adata.X, dropout_mid=midpoint, dropout_shape=-1)
    
    bdata = adata.copy()
    bdata.X = new_X
    return bdata

def apply_batch_effect(adata, n_genes, shift_factor):
    bdata = adata.copy()
    n_cells = bdata.shape[0]
    n_vars = bdata.shape[1]
    
    rng = np.random.default_rng(RANDOM_SEED)
    batch_indices = rng.choice(n_cells, size=n_cells // 2, replace=False)
    
    bdata.obs[BATCH_COL] = "Batch1"
    bdata.obs[BATCH_COL].iloc[batch_indices] = "Batch2"
    
    gene_indices = rng.choice(n_vars, size=n_genes, replace=False)
    
    if sp.issparse(bdata.X):
        bdata.X = bdata.X.tolil()
        bdata.X[batch_indices[:, None], gene_indices] *= shift_factor
        bdata.X = bdata.X.tocsr()
    else:
        bdata.X[batch_indices[:, None], gene_indices] *= shift_factor
        
    return bdata
