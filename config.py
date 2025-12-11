import os

RAW_DATA_PATH = "./data/pbmc3k_processed.h5ad"
EMBEDDING_DIR = "./embeddings"
RESULTS_DIR = "./results"

RANDOM_SEED = 42
CELL_TYPE_COL = "cell_type"
BATCH_COL = "batch"

SUBSAMPLE_RATIOS = [1.0, 0.5, 0.1]
DROPOUT_RATES = [0.0, 0.2, 0.4, 0.6]

BATCH_GENE_SIGMA = 0.4
BATCH_LIBRARY_FACTOR = 0.7

MODELS = ["PCA", "scGPT", "Geneformer", "scFoundation"]
