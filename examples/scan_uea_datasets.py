"""
扫描 UEA 多变量时间序列数据集库，筛选适合无监督聚类实验的数据集
运行: python3 examples/scan_uea_datasets.py

筛选条件：
- 多变量（变量数 V >= 2）
- 类别数 K 在 5-15 之间
- 样本总数在 200-3000 之间
- 时间步 T 在 20-300 之间

依赖：pip install aeon requests beautifulsoup4
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import requests

# ── 第一步：从官网抓取完整数据集列表 ─────────────────────────────────────
def fetch_dataset_list():
    """从 timeseriesclassification.com 抓取所有多变量数据集名称"""
    print("正在从官网获取完整数据集列表...")
    try:
        from bs4 import BeautifulSoup

        # 尝试多个可能的URL
        urls = [
            "https://www.timeseriesclassification.com/index.php",
            "https://www.timeseriesclassification.com/dataset.php",
            "http://www.timeseriesclassification.com/index.php",
        ]

        html = None
        for url in urls:
            try:
                r = requests.get(url, timeout=15,
                                 headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200 and len(r.text) > 500:
                    html = r.text
                    print(f"  成功访问: {url} (长度={len(html)})")
                    break
            except Exception as e:
                print(f"  {url} 失败: {e}")

        if not html:
            print("所有URL均无法访问")
            return None

        # 调试：打印前500字符和所有链接
        print(f"\n--- HTML前300字符 ---\n{html[:300]}\n---")

        soup = BeautifulSoup(html, 'html.parser')
        all_links = [(a.get('href',''), a.text.strip())
                     for a in soup.find_all('a', href=True)]
        print(f"页面共有 {len(all_links)} 个链接，前10个：")
        for href, text in all_links[:10]:
            print(f"  href={href!r}  text={text!r}")

        # 尝试多种链接格式
        names = []
        patterns = ['Dataset=', 'dataset=', 'd=', 'name=']
        for href, text in all_links:
            for pat in patterns:
                if pat in href:
                    name = href.split(pat)[-1].split('&')[0].strip()
                    if name and len(name) > 2:
                        names.append(name)
                    break

        names = sorted(set(names))
        print(f"官网共找到 {len(names)} 个数据集")
        return names if names else None

    except Exception as e:
        print(f"官网抓取失败({e})，使用 aeon 内置列表")
        return None

# ── 第二步：aeon 内置列表（备用）────────────────────────────────────────
def get_aeon_list():
    try:
        from aeon.datasets.tsc_datasets import multivariate
        print(f"aeon 内置多变量数据集: {len(multivariate)} 个")
        return sorted(multivariate)
    except Exception as e:
        print(f"aeon 列表获取失败: {e}")
        return []

# ── 第三步：用 aeon 逐个加载并检查 ──────────────────────────────────────
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

    for i, name in enumerate(dataset_names):
        try:
            X_tr, y_tr = load_classification(name, split="train")
            X_te, y_te = load_classification(name, split="test")

            # aeon 返回形状 (N, V, T)
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
    # 优先从官网抓完整列表，失败则用 aeon 内置列表
    names = fetch_dataset_list()
    if not names:
        names = get_aeon_list()

    if names:
        scan(names)
    else:
        print("无法获取数据集列表，请检查网络或 aeon 安装。")
