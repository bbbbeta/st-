#-*- coding : utf-8 -*-
"""
DLPFC with optional LLM-assisted gene selection (Innovation 3).

Usage:
    # Without LLM (standard training):
    python run_DLPFC_with_LLM.py --use_llm false

    # With LLM gene selection (auxiliary strategy, recommended):
    python run_DLPFC_with_LLM.py --use_llm true --strategy auxiliary

    # With LLM gene selection (exclusive strategy):
    python run_DLPFC_with_LLM.py --use_llm true --strategy exclusive
"""
import os
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from ST.utils import calculate_adj_matrix, prefilter_genes
from ST.gene_selector import LLMGeneSelector
from ST import train as train_module

plt.rc('font', family='Times New Roman')

# ==========================================
# Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(description="DLPFC with optional LLM gene selection")
parser.add_argument('--section_id', type=int, default=151673,
                    help='DLPFC section ID (e.g., 151673, 151676)')
parser.add_argument('--k', type=int, default=7,
                    help='Number of clusters (DLPFC: 7 = Layer 1-6 + WM)')
parser.add_argument('--use_llm', type=str, default='false',
                    help='Enable LLM gene selection (true/false)')
parser.add_argument('--strategy', type=str, default='auxiliary',
                    choices=['exclusive', 'auxiliary'],
                    help='Feature construction strategy: exclusive (remove overlap) or auxiliary (LLM denoised via PCA)')
args = parser.parse_args()

SECTION_ID = str(args.section_id)
K = args.k
USE_LLM = args.use_llm.lower() in ('true', '1', 'yes')
STRATEGY = args.strategy

# Paths
DATA_DIR = "Data"
SECTION_DIR = os.path.join(DATA_DIR, "DLPFC", SECTION_ID)
KNOWLEDGE_TABLE_PATH = "Data/knowledge/DLPFC_genes_knowledge_table.csv"
SELECTED_GENES_OUTPUT_PATH = f"Data/knowledge/DLPFC_{SECTION_ID}_selected_genes.json"
FIGURE_SAVE_DIR = "figures"
TISSUE_TYPE = "Human Dorsolateral Prefrontal Cortex (DLPFC)"

# Tu-Zi API Configuration
os.environ["OPENAI_API_KEY"] = "sk-VPxyW6EoN9lEpp9Ps8WM0TyQdDG8hIBfQOhcMNbONRUUElTY"
TUZI_API_BASE_URL = "https://api.tu-zi.com/v1"
TUZI_MODEL = "gpt-5"

os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SELECTED_GENES_OUTPUT_PATH), exist_ok=True)

print("="*80)
print(f"stMMR + LLM Gene Selection (Innovation 3) - DLPFC Section {SECTION_ID}")
print(f"  USE_LLM = {USE_LLM}")
print(f"  STRATEGY = {STRATEGY}")
print("="*80)

# ==========================================
# 1. Data Loading
# ==========================================
print(f"\n[Step 1] Loading Data...")

im_re = pd.read_csv(
    os.path.join(SECTION_DIR, "image_representation/ViT_pca_representation.csv"),
    header=0, index_col=0, sep=','
)
print(f" -> {SECTION_ID}, K={K}")

adata = sc.read_visium(
    path=SECTION_DIR,
    count_file=f"{SECTION_ID}_filtered_feature_bc_matrix.h5"
)
adata.var_names_make_unique()

prefilter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

adata.obsm["im_re"] = im_re

Ann_df = pd.read_csv(
    os.path.join(SECTION_DIR, f"cluster_labels_{SECTION_ID}.csv"),
    sep=',', header=0, index_col=0
)
adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

Ann_df = Ann_df.replace(1, "Layer 1")
Ann_df = Ann_df.replace(2, "Layer 2")
Ann_df = Ann_df.replace(3, "Layer 3")
Ann_df = Ann_df.replace(4, "Layer 4")
Ann_df = Ann_df.replace(5, "Layer 5")
Ann_df = Ann_df.replace(6, "Layer 6")
Ann_df = Ann_df.replace(7, "WM")
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

adata = adata[:, adata.var['highly_variable']]

# ==========================================
# 2. LLM Gene Selection (optional)
# ==========================================
llm_genes = []
use_llm_feature = False

if USE_LLM:
    print("\n" + "="*80)
    print("[Step 2] LLM Gene Selection (Innovation 3)")
    print("="*80)

    # Pre-filter genes by expression before sending to LLM
    adata_full = sc.AnnData(counts=adata.X, obs=adata.obs, var=adata.var)
    adata_full.X = adata.raw.X if adata.raw else adata.X

    candidate_genes = adata.var_names.tolist()

    if os.path.exists(SELECTED_GENES_OUTPUT_PATH):
        print(f" -> Found cached gene selection at {SELECTED_GENES_OUTPUT_PATH}")
        import json
        with open(SELECTED_GENES_OUTPUT_PATH, 'r') as f:
            cached = json.load(f)
        llm_genes = cached.get("selected_genes", [])
        print(f" -> Loaded {len(llm_genes)} cached LLM-selected genes")
    else:
        if not os.path.exists(KNOWLEDGE_TABLE_PATH):
            print(f"WARNING: Knowledge table not found at {KNOWLEDGE_TABLE_PATH}")
            print("         Skipping LLM gene selection.")
        else:
            try:
                gene_selector = LLMGeneSelector(model=TUZI_MODEL, base_url=TUZI_API_BASE_URL)
                print("Starting batch processing...")
                llm_genes, reasoning = gene_selector.select_genes_with_llm(
                    tissue_type=TISSUE_TYPE,
                    gene_list=candidate_genes,
                    knowledge_path=KNOWLEDGE_TABLE_PATH,
                    tokens_per_gene_est=80,
                    max_context_tokens=12000,
                    max_retries=3
                )
                if len(llm_genes) > 0:
                    gene_selector.save_selected_genes(llm_genes, reasoning, SELECTED_GENES_OUTPUT_PATH)
            except Exception as e:
                print(f"Error during LLM gene selection: {e}")

    valid_llm_genes = [g for g in llm_genes if g in adata.var_names]

    if len(valid_llm_genes) > 0:
        adata_temp = adata[:, valid_llm_genes]
        variances = np.var(
            adata_temp.X.toarray() if hasattr(adata_temp.X, 'toarray') else adata_temp.X,
            axis=0
        )
        valid_llm_genes = [g for g, v in zip(valid_llm_genes, variances) if v > 0.05]

    print(f"\nFinal Valid LLM Genes: {len(valid_llm_genes)}")

    if len(valid_llm_genes) == 0:
        print("WARNING: No valid LLM genes after filtering. Falling back to standard training.")
else:
    print("\n[Step 2] LLM Gene Selection skipped (--use_llm false)")

# ==========================================
# 3. Feature Construction
# ==========================================
print("\n" + "="*80)
print("[Step 3] Constructing Feature Sets")
print("="*80)

all_hvg_genes = adata.var_names.tolist()

if USE_LLM and len(valid_llm_genes) > 0:
    if STRATEGY == 'exclusive':
        llm_in_hvg = set(all_hvg_genes) & set(valid_llm_genes)
        genes_for_standard_branch = list(set(all_hvg_genes) - llm_in_hvg)
        print(f" -> Removed {len(llm_in_hvg)} overlapping genes from standard branch.")

        adata_llm = adata[:, valid_llm_genes].copy()
        sc.pp.scale(adata_llm, max_value=10)
        n_comps = min(20, adata_llm.n_vars - 1)
        if n_comps > 1:
            sc.tl.pca(adata_llm, n_comps=n_comps)
            adata.obsm["llm_gene_feature"] = adata_llm.obsm['X_pca']
        else:
            adata.obsm["llm_gene_feature"] = adata_llm.X if not hasattr(adata_llm.X, 'toarray') else adata_llm.X.toarray()

        use_llm_feature = True
        print(f" -> Branch 1 (LLM): Using {len(valid_llm_genes)} genes (PCA shape: {adata.obsm['llm_gene_feature'].shape})")
        print(f" -> Branch 2 (Standard): Using {len(genes_for_standard_branch)} HVGs (exclusive).")
        adata = adata[:, genes_for_standard_branch]
        weight_llm = 15.0
        d_param = 0.5
    else:  # auxiliary
        llm_in_hvg = list(set(valid_llm_genes) & set(all_hvg_genes))
        print(f" -> LLM genes overlapping with HVG: {len(llm_in_hvg)} / {len(valid_llm_genes)}")

        llm_genes_for_pca = llm_in_hvg if len(llm_in_hvg) >= 50 else valid_llm_genes

        adata_llm = adata[:, llm_genes_for_pca].copy()
        sc.pp.scale(adata_llm, max_value=10)
        max_possible_comps = min(adata_llm.n_vars, adata_llm.n_obs) - 1
        n_comps = min(20, max_possible_comps)

        if n_comps > 1:
            sc.tl.pca(adata_llm, n_comps=n_comps)
            adata.obsm["llm_gene_feature"] = adata_llm.obsm['X_pca']
        else:
            X_val = adata_llm.X.toarray() if hasattr(adata_llm.X, 'toarray') else adata_llm.X
            adata.obsm["llm_gene_feature"] = X_val

        use_llm_feature = True
        print(f" -> Branch 1 (LLM): Using {len(llm_genes_for_pca)} genes (PCA shape: {adata.obsm['llm_gene_feature'].shape})")
        print(f" -> Branch 2 (Standard): Using FULL {len(all_hvg_genes)} HVGs (no removal).")
        adata = adata[:, all_hvg_genes]
        weight_llm = 8.0
        d_param = 0.3
else:
    print(" -> Using standard 3000 HVGs.")
    adata = adata[:, all_hvg_genes]
    weight_llm = 15.0
    d_param = 0.5

# ==========================================
# 4. Compute Adjacency Matrix
# ==========================================
print(" -> Computing spatial adjacency...")
adata.obsm["adj"] = calculate_adj_matrix(adata)

# ==========================================
# 5. Model Training
# ==========================================
print("\n" + "="*80)
print("[Step 4] Training Model")
print("="*80)

adata = train_module.train(
    adata,
    k=K,
    n_epochs=200,
    h=[3000, 3000],
    radius=0,      # Visium: no radius refinement
    l=1.0,
    embed=False,
    use_llm_gene=use_llm_feature,
    weight_gene=20.0,
    weight_image=1.0,
    weight_llm=weight_llm,
    weight_fusion=10.0,
    d=d_param,
    a=10,
    b=1,
    c=10,
)

# ==========================================
# 6. Evaluation & Visualization
# ==========================================
print("\n" + "="*80)
print("[Step 5] Results")
print("="*80)

obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['ST'], obs_df['Ground Truth'])
print(f'Final ARI = {ARI:.5f}')

title_suffix = f"_{STRATEGY}_Strategy" if use_llm_feature else "_No_LLM"
sc.pl.spatial(
    adata,
    color=["ST", "Ground Truth"],
    title=[f'stMMR (ARI={ARI:.2f})', 'Ground Truth'],
    show=False,
    wspace=0.4
)
save_path = os.path.join(FIGURE_SAVE_DIR, f"DLPFC_{SECTION_ID}{title_suffix}_comparison.pdf")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {save_path}")

print("\nProcess Completed.")
