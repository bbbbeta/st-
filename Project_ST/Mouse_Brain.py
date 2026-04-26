#-*- coding : utf-8 -*-
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from ST.utils import *
from ST import train
from datetime import datetime
plt.rc('font',family='Times New Roman')
section_id = "V1_Mouse_Brain_Sagittal_Anterior_Section_1"
k=52

im_re = pd.read_csv(os.path.join('Data',
              section_id, "image_representation/ViT_pca_representation.csv"), header=0, index_col=0, sep=',')
print(section_id, k)

adata = sc.read_visium("Data/{}".format(section_id),
                       count_file="{}_filtered_feature_bc_matrix.h5".format(section_id))
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.obsm["im_re"] = im_re
Ann_df = pd.read_csv("Data/{}/metadata.tsv".format(section_id), sep="	", header=0, na_filter=False,
                     index_col=0)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']
adata.obs['ground_truth'] = adata.obs['Ground Truth']
adata =  adata[:, adata.var['highly_variable']]

adata.obsm["adj"] = calculate_adj_matrix(adata)
# adata = train.train(adata, knn=k, n_epochs=200, h=[3000,3000], radius=0, l=1, embed=False)
adata = train.train_with_augmentation(adata, knn=k, n_epochs=200, h=[3000,3000], radius=0, l=1, embed=False)
obs_df = adata.obs.dropna()

ARI = adjusted_rand_score(obs_df['ST'], obs_df['Ground Truth'])
print('Adjusted rand index = %.5f' % ARI)
os.makedirs('figures', exist_ok=True)
text_kwargs = {'fontfamily': 'Times New Roman'}
plt.rcParams["figure.figsize"] = (8, 3)
try:
    sc.pl.spatial(adata, color=["ST", "Ground Truth"],
                  title=['ST with Augmentation (ARI=%.4f)' % ARI, "Ground Truth"],
                  save=f"_{section_id}_augmentation.pdf")
    print(f"Results saved to figures/show_{section_id}_augmentation.pdf")
except:
    print("Spatial plot failed, using scatter plot instead...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.scatter(adata, x="array_col", y="array_row", color="ST", 
                  title='ST (ARI=%.4f)' % ARI, 
                  size=100000 / adata.shape[0], 
                  ax=axes[0], show=False)
    sc.pl.scatter(adata, x="array_col", y="array_row", color="Ground Truth", 
                  title="Ground Truth", 
                  size=100000 / adata.shape[0], 
                  ax=axes[1], show=False)
    plt.tight_layout()
    plt.savefig(f'figures/{section_id}_augmentation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'figures/{section_id}_augmentation.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Results saved to figures/{section_id}_augmentation.pdf/png")
