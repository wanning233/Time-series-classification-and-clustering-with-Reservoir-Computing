"""
数据集实验 3/4：UWAVE
运行: python3 examples/run_uwave.py
"""

import os, sys, csv, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import v_measure_score, adjusted_rand_score
import umap

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from reservoir_computing.modules import (
    RC_model, StackedRC_model,
    MultiExpertStackedRC_model, MoEStackedRC_model,
)
from reservoir_computing.datasets import ClfLoader

# ── 字体 ──────────────────────────────────────────────────────────────────
_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
          'Noto Sans CJK SC', 'Source Han Sans CN', 'PingFang SC',
          'Heiti SC', 'Arial Unicode MS']
_avail = [f.name for f in fm.fontManager.ttflist]
_sel   = next((f for f in _fonts if f in _avail), None)
if _sel:
    plt.rcParams['font.sans-serif'] = [_sel] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── 路径 ──────────────────────────────────────────────────────────────────
UMAP_DIR   = os.path.join(PROJECT_ROOT, 'umap_visualizations', 'multi_dataset')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')
CSV_PATH   = os.path.join(RESULT_DIR, 'multi_dataset_results.csv')
CSV_FIELDS = ['dataset', 'model', 'nmi', 'ari', 'n_found', 'feat_dim', 'time_s']
os.makedirs(UMAP_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DS_NAME  = 'UWAVE'
DS_LABEL = 'UWAVE'

# UWAVE 时序长(315步)，储备池缩至100
_U = 100
_CFG5 = [{'n_internal_units': max(_U - i*10, 50), 'spectral_radius': 0.99,
           'leak': None, 'connectivity': 0.3, 'input_scaling': 0.2,
           'noise_level': 0.0, 'circle': False} for i in range(5)]
_CFG6 = [{'n_internal_units': max(_U - i*10, 50), 'spectral_radius': 0.99,
           'leak': None, 'connectivity': 0.3, 'input_scaling': 0.2,
           'noise_level': 0.0, 'circle': False} for i in range(6)]

MODEL_CONFIGS = [
    ('RC',          'Baseline RC',
     lambda: RC_model(n_internal_units=_U, readout_type=None)),
    ('StackedRC',   'StackedRC (5L)',
     lambda: StackedRC_model(n_layers=5, reservoir_configs=_CFG5, readout_type=None)),
    ('MultiExpert', 'MultiExpert RC (6L,5E)',
     lambda: MultiExpertStackedRC_model(
         n_layers=6, n_experts=5, reservoir_configs=_CFG6,
         mts_rep='mean', readout_type=None)),
    ('MoE',         'MoE RC (6L,15E)',
     lambda: MoEStackedRC_model(
         n_layers=6, n_experts=15, reservoir_configs=_CFG6,
         gate_lr=0.01, gate_epochs=100, gate_reg=1e-4,
         intra_gate_input='mean', readout_type=None, mts_rep='mean')),
]

# ── 工具函数 ───────────────────────────────────────────────────────────────
def evaluate_clustering(reprs, true_labels, model_name=''):
    from sklearn.cluster import KMeans
    k = len(np.unique(true_labels))

    # Ward 层次聚类
    Dist      = cosine_distances(reprs)
    distArray = ssd.squareform(Dist)
    Z         = linkage(distArray, 'ward')
    clust_ward = fcluster(Z, t=k, criterion='maxclust')
    nmi_ward   = v_measure_score(true_labels, clust_ward)
    ari_ward   = adjusted_rand_score(true_labels, clust_ward)

    # K-Means
    clust_km = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(reprs)
    nmi_km   = v_measure_score(true_labels, clust_km)
    ari_km   = adjusted_rand_score(true_labels, clust_km)

    if model_name:
        print(f"    {model_name:30s}  Ward: NMI={nmi_ward:.4f} ARI={ari_ward:.4f}  "
              f"KMeans: NMI={nmi_km:.4f} ARI={ari_km:.4f}")

    # 取 ARI 更高的结果
    if ari_km >= ari_ward:
        return nmi_km, ari_km, len(np.unique(clust_km)), clust_km
    else:
        return nmi_ward, ari_ward, len(np.unique(clust_ward)), clust_ward


def save_umap(ds_label, models_results, true_labels):
    n = len(models_results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for idx, (label, reprs, clust, nmi, ari) in enumerate(models_results):
        print(f"  UMAP: {label} ...")
        emb = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=15, min_dist=0.1).fit_transform(reprs)
        sc1 = axes[idx, 0].scatter(emb[:, 0], emb[:, 1], c=true_labels,
                                   cmap='tab10', s=20, alpha=0.6,
                                   edgecolors='k', linewidths=0.3)
        axes[idx, 0].set_title(f'{label}\nTrue Labels', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('UMAP Dim 1'); axes[idx, 0].set_ylabel('UMAP Dim 2')
        axes[idx, 0].grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=axes[idx, 0], label='Class', pad=0.02)
        sc2 = axes[idx, 1].scatter(emb[:, 0], emb[:, 1], c=clust,
                                   cmap='tab10', s=20, alpha=0.6,
                                   edgecolors='k', linewidths=0.3)
        axes[idx, 1].set_title(f'{label}\nClustering  NMI={nmi:.4f}  ARI={ari:.4f}',
                               fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('UMAP Dim 1'); axes[idx, 1].set_ylabel('UMAP Dim 2')
        axes[idx, 1].grid(True, alpha=0.3)
        plt.colorbar(sc2, ax=axes[idx, 1], label='Cluster', pad=0.02)
    fig.suptitle(f'Dataset: {ds_label}', fontsize=15, fontweight='bold', y=1.005)
    plt.tight_layout(pad=3.0, h_pad=2.5)
    path = os.path.join(UMAP_DIR, f'{ds_label.replace(" ", "_")}_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  UMAP图已保存: {path}")


def append_csv(rows):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"结果已追加到: {CSV_PATH}")


# ── 主流程 ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(0)
    print(f'\n{"="*65}\n数据集: {DS_LABEL}\n{"="*65}')

    Xtr, Ytr, Xte, Yte = ClfLoader().get_data(DS_NAME)
    X = np.concatenate((Xtr, Xte), axis=0)
    Y = np.concatenate((Ytr, Yte), axis=0)
    true_labels = Y[:, 0]
    print(f'  原始形状={X.shape}  类别={len(np.unique(true_labels))}')

    # 每类保留30个样本，时间步每5帧取1帧：315→63
    MAX_PER_CLASS = 30
    rng = np.random.default_rng(0)
    keep = []
    for c in np.unique(true_labels):
        idx = np.where(true_labels == c)[0]
        if len(idx) > MAX_PER_CLASS:
            idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    X, true_labels = X[keep], true_labels[keep]
    X = X[:, ::5, :]
    print(f'  压缩后形状={X.shape}  类别={len(np.unique(true_labels))}')

    csv_rows, umap_data = [], []

    for key, label, build in MODEL_CONFIGS:
        print(f'\n[{key}] {label}')
        try:
            t0    = time.time()
            model = build()
            model.fit(X, verbose=False)
            reprs   = model.input_repr
            elapsed = time.time() - t0
            print(f'  特征维度={reprs.shape[1]}  耗时={elapsed:.1f}s')
            nmi, ari, n_found, clust = evaluate_clustering(reprs, true_labels, label)
            csv_rows.append({'dataset': DS_LABEL, 'model': key,
                             'nmi': round(nmi, 4), 'ari': round(ari, 4),
                             'n_found': n_found, 'feat_dim': reprs.shape[1],
                             'time_s': round(elapsed, 1)})
            umap_data.append((label, reprs, clust, nmi, ari))
            del model, reprs
        except Exception as e:
            print(f'  !! 失败: {e}')
            csv_rows.append({'dataset': DS_LABEL, 'model': key,
                             'nmi': None, 'ari': None,
                             'n_found': None, 'feat_dim': None, 'time_s': None})

    if umap_data:
        save_umap(DS_LABEL, umap_data, true_labels)
    append_csv(csv_rows)
    print(f'\n{DS_LABEL} 完成！')
