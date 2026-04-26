import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import os
import random
from ST.utils import get_neighbors

plt.rc('font', family='Times New Roman')


# def visualize_neighbors_verification():
#     section_id = "V1_Mouse_Brain_Sagittal_Anterior_Section_1"
#     data_dir = os.path.join("Data", section_id)
#
#     print(f"Loading data for {section_id}...")
#
#     adata = sc.read_visium(data_dir, count_file=f"{section_id}_filtered_feature_bc_matrix.h5")
#     adata.var_names_make_unique()
#     coords = adata.obsm['spatial']
#     print(f"Total spots: {len(coords)}")
#
#     print("Calculating neighbors using get_neighbors func...")
#     neighbor_indices_list = get_neighbors(coords, n_neighbors=6, eps=1e-1)
#
#     num_samples = 100
#     sample_indices = random.sample(range(len(coords)), num_samples)
#
#     fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
#
#     color_bg = '#9ebca9'
#     color_center = '#42799e'
#     color_neighbor = '#5ca492'
#
#     ax.scatter(coords[:, 0], coords[:, 1], c=color_bg, s=30, edgecolors='none', alpha=0.8, label='Other Spots',
#                zorder=1)
#
#     print(f"Highlighting {num_samples} spots on the global tissue map...")
#
#     for i, center_idx in enumerate(sample_indices):
#         nbrs = neighbor_indices_list[center_idx]
#         center_coord = coords[center_idx]
#
#         lbl_center = 'Selected Spot' if i == 0 else ""
#         lbl_nbr = 'Neighbors' if i == 0 else ""
#
#         if len(nbrs) > 0:
#             for n_idx in nbrs:
#                 ax.plot([center_coord[0], coords[n_idx, 0]],
#                         [center_coord[1], coords[n_idx, 1]],
#                         c=color_neighbor, linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
#
#             ax.scatter(coords[nbrs, 0], coords[nbrs, 1], c=color_neighbor, s=35, edgecolors='white', linewidth=0.5,
#                        label=lbl_nbr, zorder=3)
#
#         ax.scatter(center_coord[0], center_coord[1], c=color_center, s=90, marker='*', edgecolors='white',
#                    linewidth=0.5, label=lbl_center, zorder=4)
#
#     ax.set_title(f"Whole Tissue Neighbor Verification\n(Sampled Spots: {num_samples})", fontsize=14)
#     ax.set_aspect('equal')
#
#     if coords[0][0] > 0:
#         ax.invert_yaxis()
#
#     ax.legend(loc='upper right', frameon=True, framealpha=0.9, fancybox=True)
#
#     plt.tight_layout()
#     os.makedirs('figures', exist_ok=True)
#     save_path = 'figures/verify_neighbors_global.png'
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Visualization saved to {save_path}")
#     plt.show()

def visualize_neighbors_verification():
    # 1. 设置文件路径 (参考 run_PDAC.py)
    section_id = "PDAC"
    data_dir = os.path.join("../Data", "PDAC")
    coor_file = os.path.join(data_dir, 'spatial_location.csv')

    print(f"Loading data for {section_id}...")

    # PDAC 数据的加载方式与 Visium 不同，直接读取 CSV
    if not os.path.exists(coor_file):
        print(f"Error: File not found at {coor_file}")
        return

    coor_df = pd.read_csv(coor_file, header=0, index_col=0, sep=',')

    # 提取坐标 (x, y)
    coords = coor_df[["x", "y"]].to_numpy()

    print(f"Total spots: {len(coords)}")

    print("Calculating neighbors using get_neighbors func...")
    # 注意：PDAC (ST 1.0) 通常是方形网格，理论上有4个最近邻（上下左右）
    # 而不是 Visium 的6个，所以这里 n_neighbors=4
    neighbor_indices_list = get_neighbors(coords, n_neighbors=6, eps=1e-1)

    # 随机选择样本点
    num_samples = 20
    sample_indices = random.sample(range(len(coords)), num_samples)

    # 2. 设置绘图
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # === 配色方案 (保持与 Mouse Brain 一致) ===
    color_bg = '#9ebca9'  # 背景灰绿
    color_center = '#42799e'  # 中心深蓝
    color_neighbor = '#5ca492'  # 邻居青绿

    # 3. 绘制背景：所有原始点
    ax.scatter(coords[:, 0], coords[:, 1], c=color_bg, s=40, edgecolors='none', alpha=0.8, label='Other Spots',
               zorder=1)

    print(f"Highlighting {num_samples} spots on the global tissue map...")

    # 4. 遍历选中的样本点并高亮
    for i, center_idx in enumerate(sample_indices):
        nbrs = neighbor_indices_list[center_idx]
        center_coord = coords[center_idx]

        # 图例标签控制
        lbl_center = 'Selected Spot' if i == 0 else ""
        lbl_nbr = 'Neighbors' if i == 0 else ""

        if len(nbrs) > 0:
            # 4.1 绘制连线
            for n_idx in nbrs:
                ax.plot([center_coord[0], coords[n_idx, 0]],
                        [center_coord[1], coords[n_idx, 1]],
                        c=color_neighbor, linestyle='-', linewidth=1.2, alpha=0.6, zorder=2)

            # 4.2 绘制邻居点
            ax.scatter(coords[nbrs, 0], coords[nbrs, 1], c=color_neighbor, s=45, edgecolors='white', linewidth=0.5,
                       label=lbl_nbr, zorder=3)

        # 4.3 绘制中心点
        ax.scatter(center_coord[0], center_coord[1], c=color_center, s=100, marker='*', edgecolors='white',
                   linewidth=0.5, label=lbl_center, zorder=4)

    # 5. 设置图像属性
    ax.set_title(f"PDAC Tissue Neighbor Verification\n(Sampled Spots: {num_samples}, k=4)", fontsize=14)
    ax.set_aspect('equal')

    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 坐标翻转逻辑：Visium和ST数据通常需要翻转Y轴以匹配病理图
    # 在 run_PDAC.py 中使用了 coor_df["y"]*-1，这里直接翻转坐标轴即可
    ax.invert_yaxis()

    # 优化图例
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fancybox=True)

    plt.tight_layout()
    os.makedirs('../figures', exist_ok=True)
    save_path = '../figures/verify_neighbors_PDAC.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_neighbors_verification()
