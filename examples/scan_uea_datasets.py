"""
扫描 UEA 多变量时间序列数据集库，筛选适合无监督聚类实验的数据集
运行: python3 examples/scan_uea_datasets.py

筛选条件：
- 多变量（变量数 V >= 2）
- 类别数 K 在 5-15 之间
- 样本总数在 200-3000 之间
- 时间步 T 在 20-300 之间

依赖：pip install aeon
"""

import warnings
warnings.filterwarnings('ignore')

try:
    from aeon.datasets import load_classification
    from aeon.datasets.tsc_datasets import multivariate as MULTIVARIATE_DATASETS
except ImportError:
    print("请先安装 aeon: pip install aeon")
    exit(1)

import numpy as np

# 筛选条件
MIN_CLASSES    = 5
MAX_CLASSES    = 15
MIN_SAMPLES    = 200
MAX_SAMPLES    = 3000
MIN_TIMEPOINTS = 20
MAX_TIMEPOINTS = 300
MIN_DIMS       = 2

print(f"共有 {len(MULTIVARIATE_DATASETS)} 个多变量数据集，开始扫描...\n")
print(f"筛选条件: 类别数[{MIN_CLASSES},{MAX_CLASSES}], 样本数[{MIN_SAMPLES},{MAX_SAMPLES}], "
      f"时间步[{MIN_TIMEPOINTS},{MAX_TIMEPOINTS}], 变量数>={MIN_DIMS}\n")
print("="*80)

candidates = []
failed     = []

for i, name in enumerate(sorted(MULTIVARIATE_DATASETS)):
    try:
        X_tr, y_tr = load_classification(name, split="train")
        X_te, y_te = load_classification(name, split="test")

        # aeon 返回形状 (N, V, T)
        n_tr  = X_tr.shape[0]
        n_te  = X_te.shape[0]
        n_tot = n_tr + n_te
        n_dim = X_tr.shape[1]
        t_len = X_tr.shape[2]
        n_cls = len(np.unique(np.concatenate([y_tr, y_te])))

        status = f"[{i+1:3d}] {name:<40s} 样本={n_tot:5d} 变量={n_dim:3d} 时间步={t_len:5d} 类别={n_cls:3d}"

        if (MIN_CLASSES    <= n_cls <= MAX_CLASSES and
            MIN_SAMPLES    <= n_tot <= MAX_SAMPLES and
            MIN_TIMEPOINTS <= t_len <= MAX_TIMEPOINTS and
            n_dim >= MIN_DIMS):
            print(f"✅ {status}  ← 符合条件")
            candidates.append({
                'name': name, 'n_total': n_tot, 'n_dims': n_dim,
                'n_timepoints': t_len, 'n_classes': n_cls
            })
        else:
            reasons = []
            if not (MIN_CLASSES <= n_cls <= MAX_CLASSES):
                reasons.append(f"类别={n_cls}")
            if not (MIN_SAMPLES <= n_tot <= MAX_SAMPLES):
                reasons.append(f"样本={n_tot}")
            if not (MIN_TIMEPOINTS <= t_len <= MAX_TIMEPOINTS):
                reasons.append(f"时间步={t_len}")
            if n_dim < MIN_DIMS:
                reasons.append(f"变量={n_dim}")
            print(f"   {status}  不符合: {', '.join(reasons)}")

    except Exception as e:
        print(f"   [{i+1:3d}] {name:<40s} 加载失败: {e}")
        failed.append(name)

print("\n" + "="*80)
print(f"\n✅ 符合条件的数据集共 {len(candidates)} 个：\n")
print(f"{'数据集':<40s} {'样本':>6} {'变量':>5} {'时间步':>7} {'类别':>5}")
print("-"*70)
for d in sorted(candidates, key=lambda x: x['n_classes']):
    print(f"{d['name']:<40s} {d['n_total']:>6d} {d['n_dims']:>5d} {d['n_timepoints']:>7d} {d['n_classes']:>5d}")

if failed:
    print(f"\n加载失败的数据集（{len(failed)}个）: {', '.join(failed)}")

print("\n扫描完成。")
