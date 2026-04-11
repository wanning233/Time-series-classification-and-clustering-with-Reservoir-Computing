"""
多数据集四模型聚类对比实验

在四个标准多变量时间序列数据集上，对比四种RC模型最优配置的聚类性能：
  - Japanese_Vowels (640样本, 12变量, 9类)
  - Libras          (360样本,  2变量, 15类)
  - UWAVE           (628样本,  3变量,  8类)
  - ArabicDigits    (8800样本, 13变量, 10类)

四种模型均采用各自在 Japanese_Vowels 上确定的最优配置：
  1. RC_model                  : 单层, 400神经元
  2. StackedRC_model           : 5层渐进递减
  3. MultiExpertStackedRC_model: 6层, 5专家
  4. MoEStackedRC_model        : 6层, 15专家

输出：
  - umap_visualizations/multi_dataset/{dataset}_comparison.png  (每数据集4模型UMAP对比图)
  - results/multi_dataset_results.csv                           (完整数值结果)
"""

import os
import sys
import csv
import time
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

# ── 中文字体配置（与原脚本保持一致）──────────────────────────────────────
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                 'Noto Sans CJK SC', 'Source Han Sans CN',
                 'PingFang SC', 'Heiti SC', 'Arial Unicode MS']
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = next((f for f in chinese_fonts if f in available_fonts), None)
if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
    print(f"使用字体: {selected_font}")
else:
    print("警告: 未找到中文字体，将使用英文标签")
plt.rcParams['axes.unicode_minus'] = False
use_chinese = selected_font is not None

# ── 项目路径 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from reservoir_computing.modules import (
    RC_model,
    StackedRC_model,
    MultiExpertStackedRC_model,
    MoEStackedRC_model,
)
from reservoir_computing.datasets import ClfLoader

np.random.seed(0)

# ── 数据集配置 ────────────────────────────────────────────────────────────
DATASETS = [
    {'name': 'Japanese_Vowels', 'label': 'Japanese Vowels', 'n_classes': 9},
    {'name': 'Libras',          'label': 'Libras',          'n_classes': 15},
    {'name': 'UWAVE',           'label': 'UWAVE',           'n_classes': 8},
    {'name': 'ArabicDigits',    'label': 'Arabic Digits',   'n_classes': 10},
]

# ── 模型最优配置 ──────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        'key': 'RC',
        'label_zh': '基准RC',
        'label_en': 'Baseline RC',
        'build': lambda: RC_model(
            n_internal_units=400,
            readout_type=None,
        ),
    },
    {
        'key': 'StackedRC',
        'label_zh': '层叠RC (5层)',
        'label_en': 'StackedRC (5L)',
        'build': lambda: StackedRC_model(
            n_layers=5,
            reservoir_configs=None,
            readout_type=None,
        ),
    },
    {
        'key': 'MultiExpert',
        'label_zh': '多专家RC (6层5专家)',
        'label_en': 'MultiExpert RC (6L,5E)',
        'build': lambda: MultiExpertStackedRC_model(
            n_layers=6,
            n_experts=5,
            reservoir_configs=None,
            mts_rep='mean',
            readout_type=None,
        ),
    },
    {
        'key': 'MoE',
        'label_zh': 'MoE RC (6层15专家)',
        'label_en': 'MoE RC (6L,15E)',
        'build': lambda: MoEStackedRC_model(
            n_layers=6,
            n_experts=15,
            reservoir_configs=None,
            gate_lr=0.01,
            gate_epochs=100,
            gate_reg=1e-4,
            intra_gate_input='mean',
            readout_type=None,
            mts_rep='mean',
        ),
    },
]

# ── 工具函数 ──────────────────────────────────────────────────────────────
def evaluate_clustering(representations, true_labels, model_name=''):
    """余弦距离 + Ward层次聚类 + NMI/ARI评估（与原脚本保持一致）"""
    Dist = cosine_distances(representations)
    distArray = ssd.squareform(Dist)
    Z = linkage(distArray, 'ward')
    n_clusters_true = len(np.unique(true_labels))
    clust = fcluster(Z, t=n_clusters_true, criterion='maxclust')
    nmi = v_measure_score(true_labels, clust)
    ari = adjusted_rand_score(true_labels, clust)
    n_found = len(np.unique(clust))
    if model_name:
        print(f"      {model_name:30s}  NMI={nmi:.4f}  ARI={ari:.4f}  簇数={n_found}")
    return nmi, ari, n_found, clust


def save_dataset_umap(dataset_label, models_results, true_labels, save_dir):
    """
    为单个数据集生成4模型UMAP对比图（4行×2列：左=真实标签，右=聚类结果）。
    models_results: list of (model_label, representations, clust, nmi, ari)
    """
    os.makedirs(save_dir, exist_ok=True)

    n_models = len(models_results)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 6 * n_models))

    for idx, (model_label, reprs, clust, nmi, ari) in enumerate(models_results):
        print(f"   生成UMAP: {model_label} ...")
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(reprs)

        # 左图：真实标签
        sc1 = axes[idx, 0].scatter(emb[:, 0], emb[:, 1],
                                   c=true_labels, cmap='tab10',
                                   s=20, alpha=0.6,
                                   edgecolors='k', linewidths=0.3)
        axes[idx, 0].set_title(f'{model_label}\nTrue Labels',
                               fontsize=12, fontweight='bold', pad=10)
        axes[idx, 0].set_xlabel('UMAP Dim 1', fontsize=10)
        axes[idx, 0].set_ylabel('UMAP Dim 2', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=axes[idx, 0], label='Class', pad=0.02)

        # 右图：聚类结果
        sc2 = axes[idx, 1].scatter(emb[:, 0], emb[:, 1],
                                   c=clust, cmap='tab10',
                                   s=20, alpha=0.6,
                                   edgecolors='k', linewidths=0.3)
        axes[idx, 1].set_title(
            f'{model_label}\nClustering  NMI={nmi:.4f}  ARI={ari:.4f}',
            fontsize=12, fontweight='bold', pad=10)
        axes[idx, 1].set_xlabel('UMAP Dim 1', fontsize=10)
        axes[idx, 1].set_ylabel('UMAP Dim 2', fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
        plt.colorbar(sc2, ax=axes[idx, 1], label='Cluster', pad=0.02)

    fig.suptitle(f'Dataset: {dataset_label}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(pad=3.0, h_pad=2.5)

    safe_name = dataset_label.replace(' ', '_')
    save_path = os.path.join(save_dir, f'{safe_name}_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   UMAP图已保存: {save_path}")
    return save_path


# ── 主实验循环 ────────────────────────────────────────────────────────────
all_results = []   # list of dict，存储所有数据集×模型的结果

UMAP_DIR = os.path.join(SCRIPT_DIR, '..', 'umap_visualizations', 'multi_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

print('\n' + '=' * 70)
print('多数据集四模型聚类对比实验')
print('=' * 70)

for ds_cfg in DATASETS:
    ds_name  = ds_cfg['name']
    ds_label = ds_cfg['label']

    print(f'\n{"─"*70}')
    print(f'数据集: {ds_label}  ({ds_name})')
    print(f'{"─"*70}')

    # ── 加载数据 ──────────────────────────────────────────────────────
    try:
        print('  加载数据...')
        t0 = time.time()
        Xtr, Ytr, Xte, Yte = ClfLoader().get_data(ds_name)
        X = np.concatenate((Xtr, Xte), axis=0)
        Y = np.concatenate((Ytr, Yte), axis=0)
        true_labels = Y[:, 0]
        print(f'  数据形状: {X.shape}  类别数: {len(np.unique(true_labels))}  '
              f'(加载耗时 {time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'  !! 加载失败: {e}，跳过该数据集')
        continue

    dataset_models_results = []   # 用于UMAP绘图

    # ── 依次训练并评估四种模型 ────────────────────────────────────────
    for m_cfg in MODEL_CONFIGS:
        m_key   = m_cfg['key']
        m_label = m_cfg['label_en']
        print(f'\n  [{m_key}] {m_label}')

        try:
            t0 = time.time()
            model = m_cfg['build']()
            model.fit(X, verbose=False)
            reprs = model.input_repr
            elapsed = time.time() - t0
            print(f'      特征维度={reprs.shape[1]}  训练耗时={elapsed:.1f}s')

            nmi, ari, n_found, clust = evaluate_clustering(
                reprs, true_labels, model_name=m_label)

            all_results.append({
                'dataset':  ds_label,
                'model':    m_key,
                'nmi':      round(nmi, 4),
                'ari':      round(ari, 4),
                'n_found':  n_found,
                'feat_dim': reprs.shape[1],
                'time_s':   round(elapsed, 1),
            })
            dataset_models_results.append((m_label, reprs, clust, nmi, ari))

        except Exception as e:
            print(f'      !! 模型运行失败: {e}')
            all_results.append({
                'dataset':  ds_label,
                'model':    m_key,
                'nmi':      None,
                'ari':      None,
                'n_found':  None,
                'feat_dim': None,
                'time_s':   None,
            })

    # ── 生成该数据集的UMAP对比图 ──────────────────────────────────────
    if dataset_models_results:
        print(f'\n  生成 {ds_label} UMAP对比图...')
        save_dataset_umap(ds_label, dataset_models_results,
                          true_labels, UMAP_DIR)

# ── 保存CSV结果 ───────────────────────────────────────────────────────────
csv_path = os.path.join(RESULT_DIR, 'multi_dataset_results.csv')
fieldnames = ['dataset', 'model', 'nmi', 'ari', 'n_found', 'feat_dim', 'time_s']
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)
print(f'\nCSV结果已保存: {csv_path}')

# ── 打印综合对比表 ────────────────────────────────────────────────────────
print('\n' + '=' * 90)
print('综合对比表  (四数据集 × 四模型，最优配置)')
print('=' * 90)

model_keys = [m['key'] for m in MODEL_CONFIGS]
col_w = 22

# 表头
header = f"{'数据集':<20}" + ''.join(f"{'NMI/ARI  ' + k:<{col_w}}" for k in model_keys)
print(header)
print('-' * 90)

# 按数据集分组输出
for ds_cfg in DATASETS:
    ds_label = ds_cfg['label']
    row_nmi = f"{ds_label:<20}"
    row_ari = f"{'':20}"
    for m_key in model_keys:
        entry = next((r for r in all_results
                      if r['dataset'] == ds_label and r['model'] == m_key), None)
        if entry and entry['nmi'] is not None:
            row_nmi += f"NMI={entry['nmi']:.4f}          "[:col_w]
            row_ari  += f"ARI={entry['ari']:.4f}          "[:col_w]
        else:
            row_nmi += f"{'—':<{col_w}}"
            row_ari  += f"{'—':<{col_w}}"
    print(row_nmi)
    print(row_ari)
    print()

print('=' * 90)
print('实验完成！')
print(f'UMAP图目录 : {os.path.abspath(UMAP_DIR)}')
print(f'CSV结果文件: {os.path.abspath(csv_path)}')
