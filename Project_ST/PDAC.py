
import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

from ST.utils import calculate_adj_matrix, prefilter_genes
from ST import train

# 基本参数
section_id = "PDAC"
k = 4

# 1. 读入数据并构建 AnnData
im_re = pd.read_csv(
    "Data/PDAC/image_representation/ViT_pca_representation.csv",
    header=0,
    index_col=0,
    sep=','
)
counts_file = os.path.join('Data/PDAC/GSM3036911_PDAC-A-ST1-filtered.txt')
coor_file = os.path.join('Data/PDAC/spatial_location.csv')
manual_file = os.path.join('Data/PDAC/layer_manual_PDAC.csv')

counts = pd.read_csv(counts_file, header=0, index_col=0, sep='\t')
coor_df = pd.read_csv(coor_file, header=0, index_col=0, sep=',')
label_df = pd.read_csv(manual_file, header=0, index_col=0, sep=',')

counts = counts.T
counts.index = coor_df.index

adata = sc.AnnData(counts)
adata.var_names_make_unique()

prefilter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)

# 空间坐标
adata.obs["array_row"] = coor_df.loc[adata.obs_names, "y"] * -1
adata.obs["array_col"] = coor_df.loc[adata.obs_names, "x"]
adata.obsm["spatial"] = coor_df.loc[adata.obs_names, ["x", "y"]].to_numpy()

adata.obsm["im_re"] = im_re.loc[adata.obs_names]

common_index = adata.obs_names.intersection(label_df.index)

print("adata 有 spot 数:", adata.n_obs)
print("label_df 有 label 的 spot 数:", len(label_df))
print("两者交集 spot 数:", len(common_index))

adata.obs['Ground Truth'] = np.nan
adata.obs.loc[common_index, 'Ground Truth'] = label_df.loc[common_index, "Region"]
adata.obs['ground_truth'] = adata.obs['Ground Truth']

missing = adata.obs_names.difference(label_df.index)
print("在 label_df 里找不到 label 的 spot 数:", len(missing))
print("示例缺失 spot:", list(missing)[:10])

adata.obsm["adj"] = calculate_adj_matrix(adata)

adata_base = adata.copy()
adata_base = train.train(
    adata_base,
    k,
    n_epochs=50,
    h=[3000, 3000],
    radius=50,
    l=0.8,
    embed=False
)

adata_base.obs['ST_base'] = adata_base.obs['ST'].astype(str)
obs_base = adata_base.obs.dropna(subset=['ST_base', 'Ground Truth']).copy()
obs_base['ST_base'] = obs_base['ST_base'].astype(str)
obs_base['Ground Truth'] = obs_base['Ground Truth'].astype(str)

ARI_base = adjusted_rand_score(
    obs_base['ST_base'],
    obs_base['Ground Truth']
)
print('Base train ARI = %.5f' % ARI_base)

adata_aug = adata.copy()
adata = train.train_with_augmentation(adata, knn=k, n_epochs=200, h=[3000,3000], radius=0, l=1, embed=False)

adata_aug.obs['ST_aug'] = adata_aug.obs['ST'].astype(str)
obs_aug = adata_aug.obs.dropna(subset=['ST_aug', 'Ground Truth']).copy()
obs_aug['ST_aug'] = obs_aug['ST_aug'].astype(str)
obs_aug['Ground Truth'] = obs_aug['Ground Truth'].astype(str)

ARI_aug = adjusted_rand_score(
    obs_aug['ST_aug'],
    obs_aug['Ground Truth']
)
print('Augmented train ARI = %.5f' % ARI_aug)

adata_comb = adata_aug.copy()
adata_comb.obs['ST_base'] = adata_base.obs.loc[adata_comb.obs_names, 'ST_base']
adata_comb.obs['ST_aug'] = adata_aug.obs.loc[adata_comb.obs_names, 'ST_aug']
adata_comb.obs['Ground Truth'] = adata_aug.obs['Ground Truth']

os.makedirs('figures', exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 3)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc.pl.scatter(
    adata_comb,
    color="ST_base",
    title='Base ST (ARI=%.2f)' % ARI_base,
    size=100000 / adata_comb.shape[0],
    ax=axes[0],
    show=False
)

sc.pl.scatter(
    adata_comb,
    color="ST_aug",
    title='Augmented ST (ARI=%.2f)' % ARI_aug,
    size=100000 / adata_comb.shape[0],
    ax=axes[1],
    show=False
)

sc.pl.scatter(
    adata_comb,
    color="Ground Truth",
    title="Ground Truth",
    size=100000 / adata_comb.shape[0],
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.savefig(f'figures/{section_id}_base_vs_aug_vs_gt.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'figures/{section_id}_base_vs_aug_vs_gt.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print(f" `figures/{section_id}_base_vs_aug_vs_gt.*`")