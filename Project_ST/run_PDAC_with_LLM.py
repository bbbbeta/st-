#-*- coding : utf-8 -*-
"""
PDAC with optional LLM-assisted gene selection (Innovation 3).

Usage:
    # Without LLM (standard training):
    python run_PDAC_with_LLM.py --use_llm false

    # With LLM gene selection:
    python run_PDAC_with_LLM.py --use_llm true
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
# Configuration
# ==========================================
SECTION_ID = "PDAC"
K = 4
TISSUE_TYPE = "Human Pancreatic Ductal Adenocarcinoma (PDAC)"

# Paths
DATA_DIR = "Data"
KNOWLEDGE_TABLE_PATH = "Data/knowledge/PDAC_knowledge_table.csv"
SELECTED_GENES_OUTPUT_PATH = "Data/knowledge/PDAC_selected_genes.json"
FIGURE_SAVE_DIR = "figures"

# Tu-Zi API Configuration
os.environ["OPENAI_API_KEY"] = "sk-VPxyW6EoN9lEpp9Ps8WM0TyQdDG8hIBfQOhcMNbONRUUElTY"
TUZI_API_BASE_URL = "https://api.tu-zi.com/v1"
TUZI_MODEL = "gpt-5"

# ==========================================
# Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(description="PDAC with optional LLM gene selection")
parser.add_argument('--use_llm', type=str, default='false',
                    help='Enable LLM gene selection (true/false)')
parser.add_argument('--strategy', type=str, default='exclusive',
                    choices=['exclusive', 'auxiliary'],
                    help='Feature construction strategy: exclusive (remove overlap) or auxiliary (LLM denoised via PCA)')
args = parser.parse_args()
USE_LLM = args.use_llm.lower() in ('true', '1', 'yes')
STRATEGY = args.strategy

os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SELECTED_GENES_OUTPUT_PATH), exist_ok=True)

print("="*80)
print(f"stMMR + LLM Gene Selection (Innovation 3) - {SECTION_ID}")
print(f"  USE_LLM = {USE_LLM}")
print(f"  STRATEGY = {STRATEGY}")
print("="*80)

# ==========================================
# 1. Data Loading
# ==========================================
print(f"\n[Step 1] Loading Data...")

im_re = pd.read_csv(
    os.path.join(DATA_DIR, SECTION_ID, "image_representation/ViT_pca_representation.csv"),
    header=0, index_col=0, sep=','
)

counts_file = os.path.join(DATA_DIR, SECTION_ID, "GSM3036911_PDAC-A-ST1-filtered.txt")
coor_file = os.path.join(DATA_DIR, SECTION_ID, "spatial_location.csv")
manual_file = os.path.join(DATA_DIR, SECTION_ID, "layer_manual_PDAC.csv")

try:
    counts = pd.read_csv(counts_file, header=0, index_col=0, sep='\t')
    if counts.shape[1] <= 1:
        counts = pd.read_csv(counts_file, header=0, index_col=0, sep=r'\s+')
except Exception:
    counts = pd.read_csv(counts_file, header=0, index_col=0, sep=None, engine='python')

coor_df = pd.read_csv(coor_file, header=0, index_col=0, sep=',')
label_df = pd.read_csv(manual_file, header=0, index_col=0, sep=',')

print(f" -> Coordinates shape: {coor_df.shape}")

if counts.shape[1] == coor_df.shape[0]:
    print(" -> Transposing counts (Genes x Spots -> Spots x Genes)...")
    counts = counts.T
elif counts.shape[0] != coor_df.shape[0]:
    raise ValueError(f"Shape mismatch! Counts: {counts.shape}, Coords: {coor_df.shape}")

if len(counts) == len(coor_df):
    counts.index = coor_df.index
else:
    common = counts.index.intersection(coor_df.index)
    if len(common) > 0:
        print(f" -> Taking intersection of {len(common)} spots.")
        counts = counts.loc[common]
        coor_df = coor_df.loc[common]
        label_df = label_df.loc[common]
    else:
        print(" -> Warning: Overwriting counts index with coordinate index (assuming same order).")
        counts.index = coor_df.index

adata = sc.AnnData(counts)
adata.var_names_make_unique()

# ==========================================
# 2. Preprocessing
# ==========================================
print("\n[Step 2] Preprocessing...")
prefilter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)

adata.obs["array_row"] = coor_df["y"] * -1
adata.obs["array_col"] = coor_df["x"]
adata.obsm["spatial"] = coor_df.loc[adata.obs_names, ["x", "y"]].to_numpy()
adata.obsm["im_re"] = im_re
adata.obs['Ground Truth'] = label_df["Region"]
adata.obs['ground_truth'] = adata.obs['Ground Truth']

# ==========================================
# 3. LLM Gene Selection (optional)
# ==========================================
llm_genes = []
use_llm_feature = False

if USE_LLM:
    print("\n" + "="*80)
    print("[Step 3] LLM Gene Selection (Innovation 3)")
    print("="*80)

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
        valid_llm_genes = [g for g, v in zip(valid_llm_genes, variances) if v > 1e-6]

    print(f"\nFinal Valid LLM Genes: {len(valid_llm_genes)}")

    if len(valid_llm_genes) == 0:
        print("WARNING: No valid LLM genes after filtering. Falling back to standard training.")

else:
    print("\n[Step 3] LLM Gene Selection skipped (--use_llm false)")

# ==========================================
# 4. Feature Construction
# ==========================================
print("\n" + "="*80)
print("[Step 4] Constructing Feature Sets")
print("="*80)

# Identify Standard HVGs
print(" -> Identifying Top 3000 HVGs...")
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
all_hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()

if USE_LLM and len(valid_llm_genes) > 0:
    if STRATEGY == 'exclusive':
        # Exclusive strategy: remove LLM-HVG overlap
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

    else:  # auxiliary strategy: LLM genes as auxiliary PCA branch
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
# 5. Compute Adjacency Matrix
# ==========================================
print(" -> Computing spatial adjacency...")
adata.obsm["adj"] = calculate_adj_matrix(adata)

# ==========================================
# 6. Model Training
# ==========================================
print("\n" + "="*80)
print("[Step 5] Training Model")
print("="*80)

adata = train_module.train(
    adata,
    knn=K,
    n_epochs=200,
    h=[3000, 3000],
    radius=50,
    l=0.8,
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
# 7. Evaluation & Visualization
# ==========================================
print("\n" + "="*80)
print("[Step 6] Results")
print("="*80)

obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['ST'], obs_df['Ground Truth'])
print(f'Final ARI = {ARI:.5f}')

title_suffix = f"_{STRATEGY}_Strategy" if use_llm_feature else "_No_LLM"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc.pl.scatter(
    adata,
    alpha=1,
    x="array_row",
    y="array_col",
    color="ST",
    legend_fontsize=12,
    show=False,
    size=100000 / adata.shape[0],
    ax=axes[0],
    title=f'stMMR{title_suffix} (ARI={ARI:.2f})'
)

sc.pl.scatter(
    adata,
    alpha=1,
    x="array_row",
    y="array_col",
    color="Ground Truth",
    legend_fontsize=12,
    show=False,
    size=100000 / adata.shape[0],
    ax=axes[1],
    title='Ground Truth'
)

plt.tight_layout()
save_path = os.path.join(FIGURE_SAVE_DIR, f"PDAC{title_suffix}_comparison.pdf")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {save_path}")

print("\nProcess Completed.")
