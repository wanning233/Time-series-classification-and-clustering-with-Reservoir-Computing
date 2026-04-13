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
import numpy as np

# ── 获取完整数据集列表 ────────────────────────────────────────────────────
def get_all_dataset_names():
    """获取所有多变量时间序列数据集名称（合并所有多变量列表，去除单变量）"""
    import aeon.datasets.tsc_datasets as tsc

    # 收集所有多变量相关列表
    MULTIVARIATE_KEYS = [
        'multivariate',
        'multivariate_equal_length',
        'multivariate_unequal_length',
        'multiverse2026',
        'multiverse_core',
        'UEA',
    ]
    # 单变量列表（用于排除）
    UNIVARIATE_KEYS = [
        'univariate', 'univariate2015', 'UCR2015', 'UCR2019',
        'univariate_equal_length', 'univariate_variable_length',
        'univariate_missing_values',
    ]

    multivariate_names = set()
    univariate_names   = set()

    for key in MULTIVARIATE_KEYS:
        val = getattr(tsc, key, None)
        if val and isinstance(val, (list, tuple, set)):
            multivariate_names |= set(val)
            print(f"  {key}: {len(val)} 个")

    for key in UNIVARIATE_KEYS:
        val = getattr(tsc, key, None)
        if val and isinstance(val, (list, tuple, set)):
            univariate_names |= set(val)

    # 只保留多变量、排除纯单变量
    names = multivariate_names - univariate_names
    print(f"\n  多变量合并去重: {len(multivariate_names)} 个")
    print(f"  排除单变量后:   {len(names)} 个")
    return sorted(names)


# ── 扫描并筛选 ────────────────────────────────────────────────────────────
def scan(dataset_names):
    try:
        from aeon.datasets import load_classification
    except ImportError:
        print("请先安装 aeon: pip install aeon")
        return

    MIN_CLASSES, MAX_CLASSES       = 5, 15
    MIN_SAMPLES, MAX_SAMPLES       = 200, 3000
    MIN_TIMEPOINTS, MAX_TIMEPOINTS = 20, 300
    MIN_DIMS                       = 2

    print(f"\n共 {len(dataset_names)} 个数据集，开始扫描...\n")
    print(f"筛选条件: 类别[{MIN_CLASSES},{MAX_CLASSES}], 样本[{MIN_SAMPLES},{MAX_SAMPLES}], "
          f"时间步[{MIN_TIMEPOINTS},{MAX_TIMEPOINTS}], 变量>={MIN_DIMS}\n")
    print("="*85)

    candidates, failed = [], []

    SKIP = 74  # 已扫描前74个，从第75个继续
    for i, name in enumerate(dataset_names):
        if i < SKIP:
            continue
        try:
            X_tr, y_tr = load_classification(name, split="train")
            X_te, y_te = load_classification(name, split="test")

            n_tot = X_tr.shape[0] + X_te.shape[0]
            n_dim = X_tr.shape[1]
            t_len = X_tr.shape[2]
            n_cls = len(np.unique(np.concatenate([y_tr, y_te])))

            line = (f"[{i+1:3d}] {name:<42s} "
                    f"样本={n_tot:5d} 变量={n_dim:4d} 时间步={t_len:5d} 类别={n_cls:3d}")

            ok = (MIN_CLASSES <= n_cls <= MAX_CLASSES and
                  MIN_SAMPLES <= n_tot <= MAX_SAMPLES and
                  MIN_TIMEPOINTS <= t_len <= MAX_TIMEPOINTS and
                  n_dim >= MIN_DIMS)

            if ok:
                print(f"✅ {line}  ← 符合条件")
                candidates.append(dict(name=name, n_total=n_tot,
                                       n_dims=n_dim, n_timepoints=t_len,
                                       n_classes=n_cls))
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
                print(f"   {line}  ✗ {', '.join(reasons)}")

        except Exception as e:
            print(f"   [{i+1:3d}] {name:<42s} 加载失败: {e}")
            failed.append(name)

    print("\n" + "="*85)
    print(f"\n✅ 符合条件的数据集共 {len(candidates)} 个：\n")
    print(f"{'数据集':<42s} {'样本':>6} {'变量':>5} {'时间步':>7} {'类别':>5}")
    print("-"*72)
    for d in sorted(candidates, key=lambda x: x['n_classes']):
        print(f"{d['name']:<42s} {d['n_total']:>6d} {d['n_dims']:>5d} "
              f"{d['n_timepoints']:>7d} {d['n_classes']:>5d}")

    if failed:
        print(f"\n加载失败({len(failed)}个): {', '.join(failed)}")

    print("\n扫描完成。")


# ── 主流程 ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("正在获取数据集列表...\n")
    names = get_all_dataset_names()
    print(f"\n最终合并列表: {len(names)} 个数据集\n")

    if names:
        scan(names)
    else:
        print("无法获取数据集列表")
