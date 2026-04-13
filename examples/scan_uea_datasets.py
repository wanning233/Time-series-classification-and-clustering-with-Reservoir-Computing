"""
扫描 UEA 多变量时间序列数据集库，筛选适合无监督聚类实验的数据集
运行: python3 examples/scan_uea_datasets.py

前74个已扫描结果（直接硬编码，无需重新下载）：
✅ ERing: 300样本, 4变量, 65时间步, 6类
✅ JapaneseVowels: 640样本, 12变量, 25时间步, 9类
✅ Libras: 360样本, 2变量, 45时间步, 15类
✅ NATOPS: 360样本, 24变量, 51时间步, 6类
✅ PEMS-SF: 440样本, 963变量, 144时间步, 7类

本脚本只下载第75个之后的数据集。
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import signal

# ── 前74个已知结果（直接写死，跳过重新下载）─────────────────────────────
KNOWN_CANDIDATES = [
    dict(name='ERing',          n_total=300,  n_dims=4,   n_timepoints=65,  n_classes=6),
    dict(name='JapaneseVowels', n_total=640,  n_dims=12,  n_timepoints=25,  n_classes=9),
    dict(name='Libras',         n_total=360,  n_dims=2,   n_timepoints=45,  n_classes=15),
    dict(name='NATOPS',         n_total=360,  n_dims=24,  n_timepoints=51,  n_classes=6),
    dict(name='PEMS-SF',        n_total=440,  n_dims=963, n_timepoints=144, n_classes=7),
]

# ── 第75个之后的数据集名称（按字母排序后的75-133）────────────────────────
REMAINING = [
    'LenDB', 'LiveFuelMoistureContent_disc', 'Locust2022', 'LongIntervalTask',
    'LowCost', 'MatchingPennies', 'MindReading', 'MotionSenseHAR', 'MotorImagery',
    'NATOPS', 'NewsHeadlineSentiment_disc', 'NewsTitleSentiment_disc',
    'OpenCloseFist', 'Opportunity', 'PAMAP2', 'PEMS-SF', 'PPGDalia_disc',
    'PenDigits', 'PhonemeSpectra', 'PhotoStimulation', 'PronouncedSpeech',
    'RacketSports', 'S2Agri-10pc-17', 'S2Agri-10pc-34', 'S2Agri-17', 'S2Agri-34',
    'SPHERE-WUS', 'STEW', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'ShortIntervalTask', 'SitStand', 'Skoda', 'SongFamiliarity', 'SpokenArabicDigits',
    'StandWalkJump', 'TactileTextureRecognition', 'Tiselac', 'UCDHE-MP',
    'UCDHE-MP-MC', 'UCDHE-Rowing', 'UCDHE-Rowing-MC', 'UCIActivity',
    'UIPRMD-DS-C', 'UIPRMD-HS-C', 'UIPRMD-IL-C', 'UIPRMD-SASLR-C',
    'UIPRMD-SL-C', 'UIPRMD-SSA-C', 'UIPRMD-SSE-C', 'UIPRMD-SSIER-C',
    'UIPRMD-SSS-C', 'UIPRMD-STS-C', 'USCActivity', 'UWaveGestureLibrary',
    'VisualSpeech', 'WISDM', 'WISDM2',
]

MIN_CLASSES, MAX_CLASSES       = 5, 15
MIN_SAMPLES, MAX_SAMPLES       = 200, 3000
MIN_TIMEPOINTS, MAX_TIMEPOINTS = 20, 300
MIN_DIMS                       = 2

def check(n_tot, n_dim, t_len, n_cls):
    return (MIN_CLASSES <= n_cls <= MAX_CLASSES and
            MIN_SAMPLES <= n_tot <= MAX_SAMPLES and
            MIN_TIMEPOINTS <= t_len <= MAX_TIMEPOINTS and
            n_dim >= MIN_DIMS)

def reasons(n_tot, n_dim, t_len, n_cls):
    r = []
    if not (MIN_CLASSES <= n_cls <= MAX_CLASSES):   r.append(f"类别={n_cls}")
    if not (MIN_SAMPLES <= n_tot <= MAX_SAMPLES):   r.append(f"样本={n_tot}")
    if not (MIN_TIMEPOINTS <= t_len <= MAX_TIMEPOINTS): r.append(f"时间步={t_len}")
    if n_dim < MIN_DIMS:                            r.append(f"变量={n_dim}")
    return ', '.join(r)


if __name__ == '__main__':
    from aeon.datasets import load_classification

    candidates = list(KNOWN_CANDIDATES)
    failed = []

    print(f"前74个已知符合条件: {len(KNOWN_CANDIDATES)} 个")
    print(f"继续扫描剩余 {len(REMAINING)} 个数据集...\n")
    print("="*85)

    for i, name in enumerate(REMAINING):
        # 跳过已在已知候选中的
        if any(d['name'] == name for d in KNOWN_CANDIDATES):
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

            if check(n_tot, n_dim, t_len, n_cls):
                print(f"✅ {line}  ← 符合条件")
                candidates.append(dict(name=name, n_total=n_tot,
                                       n_dims=n_dim, n_timepoints=t_len,
                                       n_classes=n_cls))
            else:
                print(f"   {line}  ✗ {reasons(n_tot, n_dim, t_len, n_cls)}")

        except Exception as e:
            print(f"   [{i+1:3d}] {name:<42s} 失败: {e}")
            failed.append(name)

    print("\n" + "="*85)
    print(f"\n✅ 全部符合条件的数据集共 {len(candidates)} 个：\n")
    print(f"{'数据集':<42s} {'样本':>6} {'变量':>5} {'时间步':>7} {'类别':>5}")
    print("-"*72)
    for d in sorted(candidates, key=lambda x: x['n_classes']):
        print(f"{d['name']:<42s} {d['n_total']:>6d} {d['n_dims']:>5d} "
              f"{d['n_timepoints']:>7d} {d['n_classes']:>5d}")

    if failed:
        print(f"\n失败({len(failed)}个): {', '.join(failed)}")
    print("\n扫描完成。")
