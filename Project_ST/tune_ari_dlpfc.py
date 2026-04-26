# -*- coding: utf-8 -*-
"""
Minimal ARI tuning script for DLPFC with grid search.

Features:
- Parameter grid search (neighbor voting + training loss balance)
- Multi-seed evaluation per config
- Automatic CSV/JSON logging
- Best configuration summary

Example:
python tune_ari_dlpfc.py \
  --section_id 151673 \
  --k 7 \
  --n_epochs 120 \
  --seeds 110,123 \
  --grid_knn 6 \
  --grid_l 0.8,1.0,1.2 \
  --grid_a 8,10 \
  --grid_b 0.5,1 \
  --grid_c 8,10,12 \
  --grid_d 0.3,0.5 \
  --grid_tensor_loss false,true \
  --grid_tensor_temp 0.05,0.1
"""

import os
import json
import argparse
import itertools
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score

from ST.utils import calculate_adj_matrix, prefilter_genes
from ST import train as train_module


def parse_csv_values(raw: str, cast_func):
    return [cast_func(x.strip()) for x in raw.split(',') if x.strip() != '']


def parse_bool_csv(raw: str) -> List[bool]:
    mapping = {
        'true': True,
        '1': True,
        'yes': True,
        'y': True,
        'false': False,
        '0': False,
        'no': False,
        'n': False,
    }
    vals = []
    for x in raw.split(','):
        k = x.strip().lower()
        if not k:
            continue
        if k not in mapping:
            raise ValueError(f"Unsupported bool value: {x}")
        vals.append(mapping[k])
    return vals


def load_dlpfc(section_id: str) -> sc.AnnData:
    data_dir = os.path.join('Data', 'DLPFC', section_id)
    im_re_path = os.path.join(data_dir, 'image_representation', 'ViT_pca_representation.csv')
    label_path = os.path.join(data_dir, f'cluster_labels_{section_id}.csv')
    count_file = f'{section_id}_filtered_feature_bc_matrix.h5'

    im_re = pd.read_csv(im_re_path, header=0, index_col=0, sep=',')
    adata = sc.read_visium(path=data_dir, count_file=count_file)
    adata.var_names_make_unique()

    prefilter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    adata = adata[:, adata.var['highly_variable']].copy()
    adata.obsm['im_re'] = im_re.loc[adata.obs_names]

    ann_df = pd.read_csv(label_path, sep=',', header=0, index_col=0)
    adata.obs['ground_truth'] = ann_df.loc[adata.obs_names, 'ground_truth']

    # Keep a text label column aligned with legacy scripts
    ann_df = ann_df.replace(1, 'Layer 1').replace(2, 'Layer 2').replace(3, 'Layer 3')
    ann_df = ann_df.replace(4, 'Layer 4').replace(5, 'Layer 5').replace(6, 'Layer 6').replace(7, 'WM')
    adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'ground_truth']

    adata.obsm['adj'] = calculate_adj_matrix(adata)
    return adata


def evaluate_once(base_adata: sc.AnnData, cfg: Dict[str, Any], seed: int, n_epochs: int, k: int) -> float:
    adata = base_adata.copy()

    adata = train_module.train_with_augmentation(
        adata,
        knn=k,
        n_epochs=n_epochs,
        h=[3000, 3000],
        radius=0,
        l=cfg['l'],
        a=cfg['a'],
        b=cfg['b'],
        c=cfg['c'],
        d=cfg['d'],
        tensor_loss=cfg['tensor_loss'],
        tensor_temp=cfg['tensor_temp'],
        random_seed=seed,
        embed=False,
    )

    obs_df = adata.obs.dropna(subset=['ST', 'Ground Truth']).copy()
    ari = adjusted_rand_score(obs_df['ST'].astype(str), obs_df['Ground Truth'].astype(str))
    return float(ari)


def main():
    parser = argparse.ArgumentParser(description='Minimal ARI tuner for DLPFC')
    parser.add_argument('--section_id', type=int, default=151673)
    parser.add_argument('--k', type=int, default=7, help='Number of clusters for DLPFC')
    parser.add_argument('--n_epochs', type=int, default=120)
    parser.add_argument('--seeds', type=str, default='110,123')

    parser.add_argument('--grid_knn', type=str, default='7')
    parser.add_argument('--grid_l', type=str, default='0.8,1.0,1.2')
    parser.add_argument('--grid_a', type=str, default='8,10,12')
    parser.add_argument('--grid_b', type=str, default='0.5,1.0')
    parser.add_argument('--grid_c', type=str, default='8,10,12')
    parser.add_argument('--grid_d', type=str, default='0.3,0.5')
    parser.add_argument('--grid_tensor_loss', type=str, default='false,true')
    parser.add_argument('--grid_tensor_temp', type=str, default='0.05,0.1')

    parser.add_argument('--out_dir', type=str, default='results/ari_tuning')

    args = parser.parse_args()

    section_id = str(args.section_id)
    seeds = parse_csv_values(args.seeds, int)

    grid = {
        'knn': parse_csv_values(args.grid_knn, int),
        'l': parse_csv_values(args.grid_l, float),
        'a': parse_csv_values(args.grid_a, float),
        'b': parse_csv_values(args.grid_b, float),
        'c': parse_csv_values(args.grid_c, float),
        'd': parse_csv_values(args.grid_d, float),
        'tensor_loss': parse_bool_csv(args.grid_tensor_loss),
        'tensor_temp': parse_csv_values(args.grid_tensor_temp, float),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'dlpfc_{section_id}_grid_results.csv')
    json_path = os.path.join(args.out_dir, f'dlpfc_{section_id}_best_config.json')

    print('=' * 80)
    print(f'DLPFC ARI tuning | section={section_id} | seeds={seeds}')
    print(f'Results CSV: {csv_path}')
    print('=' * 80)

    base_adata = load_dlpfc(section_id)

    rows = []
    best = None

    keys = ['knn', 'l', 'a', 'b', 'c', 'd', 'tensor_loss', 'tensor_temp']
    all_values = [grid[k] for k in keys]
    total = np.prod([len(v) for v in all_values])

    run_idx = 0
    for values in itertools.product(*all_values):
        run_idx += 1
        cfg = dict(zip(keys, values))

        if not cfg['tensor_loss']:
            cfg['tensor_temp'] = -1.0

        seed_aris = []
        for seed in seeds:
            cfg_for_train = dict(cfg)
            train_k = cfg_for_train.pop('knn')

            ari = evaluate_once(
                base_adata=base_adata,
                cfg=cfg_for_train,
                seed=seed,
                n_epochs=args.n_epochs,
                k=train_k,
            )
            seed_aris.append(ari)

        mean_ari = float(np.mean(seed_aris))
        std_ari = float(np.std(seed_aris))

        row = {
            'run_idx': run_idx,
            'total_runs': int(total),
            **cfg,
            'n_epochs': args.n_epochs,
            'seeds': ','.join(map(str, seeds)),
            'seed_aris': ';'.join([f'{x:.6f}' for x in seed_aris]),
            'mean_ari': mean_ari,
            'std_ari': std_ari,
        }
        rows.append(row)

        if best is None or mean_ari > best['mean_ari']:
            best = row

        print(
            f"[{run_idx}/{int(total)}] cfg={cfg} | "
            f"ARI(mean±std)={mean_ari:.5f}±{std_ari:.5f}"
        )

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    assert best is not None

    best_payload = {
        'section_id': section_id,
        'n_epochs': args.n_epochs,
        'seeds': seeds,
        'best_config': {
            'knn': best['knn'],
            'l': best['l'],
            'a': best['a'],
            'b': best['b'],
            'c': best['c'],
            'd': best['d'],
            'tensor_loss': bool(best['tensor_loss']),
            'tensor_temp': best['tensor_temp'],
        },
        'best_mean_ari': best['mean_ari'],
        'best_std_ari': best['std_ari'],
        'csv_path': csv_path,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 80)
    print('Best configuration found:')
    print(json.dumps(best_payload, ensure_ascii=False, indent=2))
    print('=' * 80)


if __name__ == '__main__':
    main()
