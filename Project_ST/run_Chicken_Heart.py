#-*- coding : utf-8 -*-
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from stMMR.utils import *
from stMMR.process import *
from stMMR import train_model

from sklearn.neighbors import NearestNeighbors

def _debug_array_stats(name, arr):
    arr_np = np.array(arr)
    finite_mask = np.isfinite(arr_np)
    total = arr_np.size
    finite = finite_mask.sum()
    print(f"[DEBUG] {name}: shape={arr_np.shape}, finite={finite}/{total}, "
          f"min={np.nanmin(arr_np):.4f}, max={np.nanmax(arr_np):.4f}, mean={np.nanmean(arr_np):.4f}")

def get_true_neighbors(coords, n_neighbors=6, eps=1e-2):
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors * 8, len(coords))).fit(coords)
    dists, indices = nbrs.kneighbors(coords)
    dists, indices = dists[:, 1:], indices[:, 1:]
    out = []
    for i in range(len(coords)):
        m = dists[i].min()
        mask = dists[i] <= m * (1 + eps)
        out.append(indices[i][mask])
    return out
def generate_derived(coords, expr, neighbors, fraction=1/3, seed=0):
    rng = np.random.default_rng(seed)
    n_new = max(1, int(len(coords) * fraction))
    chosen = rng.choice(len(coords), size=n_new, replace=False)
    new_c, new_x, src_idx = [], [], []
    for i in chosen:
        neigh = neighbors[i]
        if len(neigh) == 0:
            continue
        idxs = np.concatenate([[i], neigh])
        new_c.append(coords[idxs].mean(axis=0))
        new_x.append(expr[idxs].mean(axis=0))
        src_idx.append(i)
    return np.vstack(new_c), np.vstack(new_x), np.array(src_idx, dtype=int)
def build_augmented_adata(adata, fraction=1/3, eps=1e-2):
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    expr = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    neighbors = get_true_neighbors(coords, n_neighbors=6, eps=eps)
    new_coords, new_expr, src_idx = generate_derived(coords, expr, neighbors, fraction=fraction)

    mask = np.isfinite(new_expr).all(axis=1)
    new_coords, new_expr, src_idx = new_coords[mask], new_expr[mask], src_idx[mask]
    if new_expr.shape[0] == 0:
        print("未生成有效衍生点，返回原始数据")
        return adata.copy()
    _debug_array_stats("new_expr_raw", new_expr)
    _debug_array_stats("all_expr_before_hvg", np.vstack([expr, new_expr]))
    _debug_array_stats("coords_aug", np.vstack([coords, new_coords]))

    all_expr = np.nan_to_num(np.vstack([expr, new_expr]), nan=0.0, posinf=0.0, neginf=0.0)
    all_coords = np.vstack([coords, new_coords])

    new_index = [f"derived_{i}" for i in range(new_expr.shape[0])]
    obs_new = adata.obs.iloc[src_idx].copy()
    obs_new.index = new_index
    obs_concat = pd.concat([adata.obs, obs_new], axis=0)

    aug = sc.AnnData(all_expr, obs=obs_concat, var=adata.var.copy())
    if 'highly_variable' in adata.var:
        aug = aug[:, adata.var['highly_variable']]

    aug.obsm["spatial"] = all_coords

    im_src = adata.obsm["im_re"].iloc[src_idx].copy()
    im_src.index = new_index
    im_aug = pd.concat([adata.obsm["im_re"], im_src], axis=0).fillna(0.0)
    _debug_array_stats("im_aug", im_aug.to_numpy())
    im_aug.index = aug.obs_names
    aug.obsm["im_re"] = im_aug

    adj = calculate_adj_matrix(aug)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    aug.obsm["adj"] = adj
    _debug_array_stats("adj_aug", aug.obsm["adj"])

    return aug

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--section_id', default=14, type=int)
parser.add_argument('--k', default=6, type=int)
args = parser.parse_args()

section_id = args.section_id
k = args.k

# section_id = 14
# k=6
# section_id = 4
# k=5
#
# section_id = 10
# k=7
# #
# section_id = 7
# k=7

im_re = pd.read_csv("Data/chicken_heart/D{}/ViT_pca_representation.csv".format(section_id),
                    header=0, index_col=0, sep=',')
print(section_id, k)
adata = sc.read_10x_h5("Data/chicken_heart/D{}"
    "/chicken_heart_spatial_RNAseq_D{}_filtered_feature_bc_matrix.h5".format(section_id,section_id))
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.obsm["im_re"] = im_re

# genes = adata.var_names.tolist()
# genes_series = pd.Series(adata.var_names, name='gene')
# genes_series.to_csv('result/Chicken_Heart_genes.csv', index=False)

coor_df = pd.read_csv(os.path.join('Data/chicken_heart/D{}'
    '/chicken_heart_spatial_RNAseq_D{}_tissue_positions_list.csv'.format(section_id,section_id),
         ), sep=",",header=None,na_filter=False, index_col=0)
# adata.obs["y_pixel"] = coor_df[4]
# adata.obs["x_pixel"] = coor_df[5]
adata.obs["array_row"] = coor_df[4]*-1
adata.obs["array_col"] = coor_df[5]
adata.obsm["spatial"] = coor_df.loc[adata.obs_names, [4,5]].to_numpy()

Ann_df = pd.read_csv("Data/chicken_heart/D{}/D{}.csv".format(section_id,section_id), sep=",", header=0,
                      na_filter=False, index_col=0)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, "region"]
adata.obs['ground_truth'] = adata.obs['Ground Truth']
adata =  adata[:, adata.var['highly_variable']]
# sc.pp.neighbors(adata, use_rep='X')
# sc.tl.umap(adata)
#
# sc.pl.umap(adata, color='Ground Truth')
adata_base = adata.copy()
adata_base.obsm["adj"] = np.nan_to_num(calculate_adj_matrix(adata_base), nan=0.0, posinf=0.0, neginf=0.0)
adata_base = train_model.train(adata_base, k, n_epochs=200, h=[3000, 3000], lr=0.00005)
obs_base = adata_base.obs[['stMMR', 'Ground Truth']].dropna()
ARI_base = adjusted_rand_score(obs_base['stMMR'], obs_base['Ground Truth'])

print("=== 衍生点增强训练 ===")
adata_aug = build_augmented_adata(adata, fraction=1/3, eps=1e-2)
_debug_array_stats("adata_aug.X", adata_aug.X)
_debug_array_stats("adata_aug.im_re", adata_aug.obsm["im_re"].to_numpy())
_debug_array_stats("adata_aug.adj", adata_aug.obsm["adj"])
adata_aug = train_model.train(adata_aug, k, n_epochs=200, h=[3000, 3000], lr=0.00005)
obs_aug = adata_aug.obs[['stMMR', 'Ground Truth']].dropna()
ARI_aug = adjusted_rand_score(obs_aug['stMMR'], obs_aug['Ground Truth'])

for tag, ad, ari_plot in [("base", adata_base, ARI_base), ("aug", adata_aug, ARI_aug)]:
    obs_df_plot = ad.obs[['stMMR', 'Ground Truth']].dropna()
    ax = sc.pl.scatter(ad, x="array_col", y="array_row", color='stMMR',
                       legend_fontsize=18, show=False, size=100000 / ad.shape[0])
    ax.set_title(f"stMMR ({tag}) ARI={ari_plot:.2}", fontsize=23)
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(f"figures/scatter_stMMR_{tag}_{section_id}.pdf")
    plt.close()

print(f"对比：Baseline ARI={ARI_base:.4f}, Augmented ARI={ARI_aug:.4f}")
# adata.obsm["adj"] = calculate_adj_matrix(adata)
# adata= train_model.train(adata,k,n_epochs=200,h=[3000,3000],lr=0.00005)
#
# obs_df = adata.obs.dropna()
# ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
# print('Adjusted rand index = %.5f' % ARI)
#
# ax = sc.pl.scatter(adata, x="array_col", y="array_row", color='Ground Truth', legend_fontsize=18, show=False,
#                    size=100000 / adata.shape[0])
# title = "Ground Truth"
# ax.set_title(title, fontsize=23)
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# dpi = 600
# plt.savefig("figures/scatter_Truth_{}.pdf".format(section_id))
# plt.close()
#
# ax = sc.pl.scatter(adata, x="array_col", y="array_row", color='stMMR', legend_fontsize=18, show=False,
#                    size=100000 / adata.shape[0])
# title = "{}: ARI={:.2}".format("stMMR", ARI)
# ax.set_title(title, fontsize=23)
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# dpi = 600
# plt.savefig("figures/scatter_stMMR_{}.pdf".format(section_id))
# plt.close()
#
# sc.pp.neighbors(adata, use_rep='emb_pca')
# adata.uns['iroot'] = np.flatnonzero(adata.obs['stMMR'] =="3")[0]  #D14
#
# sc.tl.diffmap(adata)
# sc.tl.dpt(adata, n_branchings=1)
# adata.obs['dpt_pseudotime'] =1-adata.obs['dpt_pseudotime']
# ax = sc.pl.scatter(adata, x="array_col", y="array_row", color='dpt_pseudotime', legend_fontsize=18, show=False,
#                    size=50000 / adata.shape[0])
# title = "{}: ARI={:.2}".format("stMMR", ARI)
# ax.set_title(title, fontsize=23)
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# dpi = 600
# plt.savefig("figures/dpt_{}.pdf".format(section_id), dpi=dpi)
# plt.close()
# obs_df = adata.obs.dropna()
# obs_df.to_csv("result/{}_type_stMMR.csv".format(section_id))
#
# sc.pp.neighbors(adata, use_rep='emb_pca')
# sc.tl.umap(adata)
#
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.umap(adata, color=["stMMR",'Ground Truth',"dpt_pseudotime"], title=['stMMR (ARI=%.2f)' % ARI , "Ground Truth",'pSM'],
#            save="umap{}".format(section_id))
