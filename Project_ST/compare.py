# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from ST.utils import calculate_adj_matrix, prefilter_genes
from ST import train


def run_experiment(adata, section_id="PDAC", k=4, n_epochs_base=50, n_epochs_aug=200,
                   compare_tensor=False, tensor_temp=0.1, d=1.0):
    os.makedirs('figures', exist_ok=True)

    print("\n" + "=" * 60)
    print("=" * 60)

    adata_base = adata.copy()
    adata_base = train.train(
        adata_base,
        k,
        n_epochs=n_epochs_base,
        h=[3000, 3000],
        radius=50,
        l=0.8,
        embed=False
    )
    adata_base.obs['ST_base'] = adata_base.obs['ST'].astype(str)
    obs_base = adata_base.obs.dropna(subset=['ST_base', 'Ground Truth']).copy()
    obs_base['ST_base'] = obs_base['ST_base'].astype(str)
    obs_base['Ground Truth'] = obs_base['Ground Truth'].astype(str)
    ARI_base = adjusted_rand_score(obs_base['ST_base'], obs_base['Ground Truth'])
    print(f"Base train ARI = {ARI_base:.5f}")

    print("\n" + "=" * 60)
    print("=" * 60)

    adata_aug = adata.copy()
    adata_aug = train.train_with_augmentation(
        adata_aug,
        knn=k,
        n_epochs=n_epochs_aug,
        h=[3000, 3000],
        radius=0,
        l=1,
        embed=False,
        tensor_loss=False   # 不使用张量正则化
    )
    adata_aug.obs['ST_aug'] = adata_aug.obs['ST'].astype(str)
    obs_aug = adata_aug.obs.dropna(subset=['ST_aug', 'Ground Truth']).copy()
    obs_aug['ST_aug'] = obs_aug['ST_aug'].astype(str)
    obs_aug['Ground Truth'] = obs_aug['Ground Truth'].astype(str)
    ARI_aug = adjusted_rand_score(obs_aug['ST_aug'], obs_aug['Ground Truth'])
    print(f"Augmented train ARI = {ARI_aug:.5f}")

    results = {
        'base': {
            'ari': ARI_base,
            'adata': adata_base,
            'label_col': 'ST_base'
        },
        'aug': {
            'ari': ARI_aug,
            'adata': adata_aug,
            'label_col': 'ST_aug'
        }
    }

    if compare_tensor:
        print("\n" + "=" * 60)
        print(f"  tensor_temp = {tensor_temp}, d = {d}")
        print("=" * 60)

        adata_tensor = adata.copy()
        adata_tensor = train.train_with_augmentation(
            adata_tensor,
            knn=k,
            n_epochs=n_epochs_aug,
            h=[3000, 3000],
            radius=0,
            l=1,
            embed=False,
            tensor_loss=True,       # 启用张量低秩正则化
            tensor_temp=tensor_temp,
            d=d                     # tensor loss 权重
        )
        adata_tensor.obs['ST_tensor'] = adata_tensor.obs['ST'].astype(str)
        obs_tensor = adata_tensor.obs.dropna(subset=['ST_tensor', 'Ground Truth']).copy()
        obs_tensor['ST_tensor'] = obs_tensor['ST_tensor'].astype(str)
        obs_tensor['Ground Truth'] = obs_tensor['Ground Truth'].astype(str)
        ARI_tensor = adjusted_rand_score(obs_tensor['ST_tensor'], obs_tensor['Ground Truth'])
        print(f"Aug+Tensor train ARI = {ARI_tensor:.5f}")

        results['tensor'] = {
            'ari': ARI_tensor,
            'adata': adata_tensor,
            'label_col': 'ST_tensor'
        }

    # ─── 汇总对比 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"  Base       ARI = {ARI_base:.5f}")
    print(f"  Aug        ARI = {ARI_aug:.5f}  (vs Base: {ARI_aug - ARI_base:+.5f})")
    if compare_tensor:
        print(f"  Aug+Tensor ARI = {ARI_tensor:.5f}  (vs Base: {ARI_tensor - ARI_base:+.5f}, vs Aug: {ARI_tensor - ARI_aug:+.5f})")

    # ─── 可视化 ────────────────────────────────────────────────────────
    n_methods = 3 if compare_tensor else 2
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods + 1, 4))
    if n_methods == 1:
        axes = [axes]

    methods = [
        ('Base', 'ST_base', ARI_base, adata_base),
        ('Aug (NeighborVote)', 'ST_aug', ARI_aug, adata_aug),
    ]
    if compare_tensor:
        methods.append((f'Aug+Tensor (τ={tensor_temp}, d={d})', 'ST_tensor', ARI_tensor, adata_tensor))

    for ax, (title, label_col, ari_val, adata_val) in zip(axes, methods):
        sc.pl.scatter(
            adata_val,
            x="array_col",
            y="array_row",
            color=label_col,
            title=f'{title}\nARI={ari_val:.4f}',
            size=100000 / adata_val.shape[0],
            ax=ax,
            show=False
        )

    plt.tight_layout()
    plt.savefig(f'figures/{section_id}_base_vs_aug_vs_tensor.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'figures/{section_id}_base_vs_aug_vs_tensor.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n figures/{section_id}_base_vs_aug_vs_tensor.pdf")

    return results


if __name__ == "__main__":
    section_id = "PDAC"
    k = 4

    print(f"load: {section_id}, k={k}")
    im_re = pd.read_csv(
        "Data/PDAC/image_representation/ViT_pca_representation.csv",
        header=0, index_col=0, sep=','
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

    # 预处理
    prefilter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 空间坐标
    adata.obs["array_row"] = coor_df.loc[adata.obs_names, "y"] * -1
    adata.obs["array_col"] = coor_df.loc[adata.obs_names, "x"]
    adata.obsm["spatial"] = coor_df.loc[adata.obs_names, ["x", "y"]].to_numpy()

    # 图像特征
    adata.obsm["im_re"] = im_re.loc[adata.obs_names]

    # Ground Truth
    common_index = adata.obs_names.intersection(label_df.index)
    adata.obs['Ground Truth'] = np.nan
    adata.obs.loc[common_index, 'Ground Truth'] = label_df.loc[common_index, "Region"]
    adata.obs['ground_truth'] = adata.obs['Ground Truth']

    adata.obsm["adj"] = calculate_adj_matrix(adata)

    print(f"adata: {adata.n_obs} spots, {adata.n_vars} genes")
    print(f"Ground Truth spots: {adata.obs['ground_truth'].notna().sum()}")

    # Base vs Aug
    results = run_experiment(
        adata,
        section_id=section_id,
        k=k,
        n_epochs_base=50,
        n_epochs_aug=200,
        compare_tensor=False,   # 改为 True 以启用 Aug+Tensor 实验
        tensor_temp=0.1,      
        d=1.0              
    )
