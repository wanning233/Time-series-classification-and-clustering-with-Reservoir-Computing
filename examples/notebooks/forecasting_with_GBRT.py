# 使用 GBRT 进行概率预测

# `reservoir_computing.RC_forecaster` 类允许通过拟合一个将储层状态映射到预测的线性模型来快速执行预测。该线性模型实现为 sklearn 的岭回归器 `sklearn.linear_model.Ridge`。

# 然而，可以使用 sklearn 中的其他回归模型，包括那些计算置信区间的模型，从而获得概率预测。

# 在本示例中，我们将使用 `sklearn.ensemble.HistGradientBoostingRegressor`，这是一个梯度提升回归树（GBRT），允许计算不同的分位数。

# 让我们从导入必要的库开始。

import os
import matplotlib

# 环境探测
KAGGLE_ENV = os.path.exists('/kaggle/working')
try:
    from IPython.display import Image, display, HTML  # Notebook 环境
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

# 后端设置：在 Kaggle/脚本模式使用无界面后端，避免 FigureManager 报错
if KAGGLE_ENV or not IN_NOTEBOOK:
    matplotlib.use('Agg')
elif IN_NOTEBOOK:
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA

from reservoir_computing.reservoir import Reservoir
from reservoir_computing.utils import make_forecasting_dataset
from reservoir_computing.datasets import PredLoader

np.random.seed(0) # 为了可重复性

# 检测是否在 Kaggle 环境
KAGGLE_ENV = os.path.exists('/kaggle/working')

# 创建 picture 文件夹用于保存图片
if KAGGLE_ENV:
    # 在 Kaggle 环境中，保存到 /kaggle/working/picture
    picture_dir = '/kaggle/working/picture'
else:
    # 在本地环境中，保存到脚本文件所在目录
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        picture_dir = os.path.join(script_dir, 'picture')
    except NameError:
        # 如果 __file__ 不可用（如在交互式环境中），使用当前目录
        picture_dir = os.path.join(os.getcwd(), 'picture')

os.makedirs(picture_dir, exist_ok=True)
print(f"图片将保存到: {picture_dir}")

# 定义辅助函数：列出保存的图片文件
def list_saved_images():
    """列出 picture 文件夹中所有保存的图片文件"""
    print("\n" + "="*60)
    print("picture 文件夹中的图片文件:")
    print("="*60)
    if os.path.exists(picture_dir):
        files = sorted([f for f in os.listdir(picture_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if files:
            for i, filename in enumerate(files, 1):
                filepath = os.path.join(picture_dir, filename)
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"{i}. {filename}")
                print(f"   完整路径: {filepath}")
                print(f"   文件大小: {file_size:.2f} KB")
                if IN_NOTEBOOK:
                    print(f"   [可在 Kaggle 输出文件列表中查看]")
        else:
            print("   (暂无图片文件)")
    else:
        print(f"   (文件夹不存在: {picture_dir})")
    print("="*60 + "\n")

# 定义函数：创建 HTML 报告，方便在浏览器中查看所有图片
def create_html_report():
    """创建一个 HTML 报告文件，包含所有保存的图片"""
    if not os.path.exists(picture_dir):
        print("picture 文件夹不存在，无法创建报告")
        return
    
    files = sorted([f for f in os.listdir(picture_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print("没有找到图片文件")
        return
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Forecasting Results - 预测结果报告</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .image-section {
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .image-title {
            font-size: 18px;
            font-weight: bold;
            color: #555;
            margin-bottom: 15px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .info {
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }
        .timestamp {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Forecasting Results - 预测结果报告</h1>
"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(picture_dir, filename)
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        # HTML 文件和图片在同一目录，所以直接使用文件名
        img_src = filename
        
        title = filename.replace('_', ' ').replace('.png', '').title()
        
        html_content += f"""
        <div class="image-section">
            <div class="image-title">{i}. {title}</div>
            <img src="{img_src}" alt="{filename}">
            <div class="info">
                文件名: {filename}<br>
                文件大小: {file_size:.2f} KB<br>
                完整路径: {filepath}
            </div>
        </div>
"""
    
    html_content += f"""
        <div class="timestamp">
            报告生成时间: {timestamp}
        </div>
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(picture_dir, 'results_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML 报告已创建: {html_path}")
    print(f"   可以在浏览器中打开查看所有图片")
    
    # 如果在 Notebook 环境中，尝试自动打开
    if IN_NOTEBOOK:
        try:
            display(HTML(f'<a href="{html_path}" target="_blank">点击打开 HTML 报告</a>'))
        except:
            pass
    
    return html_path

# 加载数据

# 我们将使用数据加载器 `PredLoader` 来获取预测数据集。
# 要查看有哪些可用的数据集，我们可以调用函数 `available_datasets`。通过设置 `details=True` 可以获取额外信息。

downloader = PredLoader()
downloader.available_datasets(details=True)  # 描述可用的数据集

# - 对于本示例，我们将使用 `ElecRome`，这是来自罗马市能源供应网络骨干的电力消耗数据。
# - 原始数据是每 10 分钟采样一次的时间序列。
# - 如果我们不需要如此高的分辨率，可以通过创建一个新时间序列来将数据重采样为每小时分辨率，新序列的条目是原始序列中连续 6 个时间步的平均值。
# - 我们还只取前 `3000` 个时间步，以便更快地拟合模型。

# 下载数据
ts_full = downloader.get_data("ElecRome")

# 将时间序列重采样为每小时频率
ts_hourly = np.mean(ts_full.reshape(-1, 6), axis=1)[:, None]
print("Resampled: ", ts_hourly.shape)

# 仅使用前 3000 个时间步
ts_small = ts_hourly[0:3000,:]
print("Resampled small: ", ts_small.shape)

# 准备数据集

# 为了训练我们的回归模型，我们需要输入和目标数据 `Xtr` 和 `Ytr`。
# 我们还需要测试数据 `Xte` 和 `Yte` 来测试我们的模型，如果需要调整超参数，还需要验证数据 `Xval` 和 `Yval`。
# 我们将使用函数 `make_forecasting_dataset`，给定时间序列 `X`，它执行以下计算：

# 1. 将数据集分割为连续的块：`train`、`val` 和 `test`。块的大小由 `val_percent` 和 `test_percent` 的值给出。如果不需要验证数据，设置 `val_percent=0`（默认值），验证数据将不会被创建。
# 2. 通过将数据移动 `horizon` 个时间步来创建输入数据 `X` 和目标数据 `Y`，其中 `horizon` 是我们想要预测的距离。例如：
#     - `Xtr = train[:-horizon,:]`
#     - `Ytr = train[horizon:,:]`
# 3. 使用 `sklearn.preprocessing` 中的缩放器对象对数据进行归一化。如果没有传递缩放器，则创建一个 `StandardScaler`。缩放器在 `Xtr` 上拟合，然后用于转换 `Ytr`、`Xval` 和 `Xte`。注意 `Yval` 和 `Yte` **不会**被转换。

# 下面的代码示例了该函数的使用。

X = np.arange(36)[:, None]

Xtr, Ytr, Xte, Yte, Xval, Yval, scaler = make_forecasting_dataset(X, horizon=5,
                                                                  test_percent=0.2,
                                                                  val_percent=0.3)
print("Xtr: ", scaler.inverse_transform(Xtr.T)[0])
print("Ytr: ", scaler.inverse_transform(Ytr.T))
print("Xval: ", scaler.inverse_transform(Xval.T)[0])
print("Yval: ", Yval.T)
print("Xte: ", scaler.inverse_transform(Xte.T)[0])
print("Yte: ", Yte.T)

# - 对于本示例，我们想要预测未来 24 小时。
# - 另外，我们不优化模型的超参数，所以不需要验证集（我们保留默认的 `val_percent=0`，验证数据将不会被返回）。

# 生成训练和测试数据集
Xtr, Ytr, Xte, Yte, scaler = make_forecasting_dataset(ts_small,
                                                      horizon=24, # 预测未来 24 小时的预测范围
                                                      test_percent = 0.1)

# 定义储层

# - 接下来，我们通过指定（相当随意地）超参数来创建储层。
# - 然后，我们计算与训练和测试数据相关的储层状态序列 `states_tr` 和 `states_te`。

res = Reservoir(n_internal_units=1000, 
                spectral_radius=0.95, 
                leak=None, 
                connectivity=0.25, 
                input_scaling=0.1, 
                noise_level=0.0, 
                circle=False)   

n_drop=10
states_tr = res.get_states(Xtr[None,:,:], n_drop=n_drop, bidir=False)
states_te = res.get_states(Xte[None,:,:], n_drop=n_drop, bidir=False)

# 降维（可选）

# - 作为可选步骤，我们可以减少储层状态的大小，在这个例子中储层状态相当大。
# - 特别是，由于 `n_internal_units=1000`，我们最终得到一个长度为 `T`、大小为 `1000` 的向量序列。
# - 降维可以加快训练速度，特别是如果我们使用复杂的模型作为读出层，还可以提供一些正则化来改善预测性能。
# - 下面，我们将储层状态的维度从 `1000` 减少到 `75`。

pca = PCA(n_components=75)
states_tr = pca.fit_transform(states_tr[0])
states_te = pca.transform(states_te[0])

# 拟合读出层（线性）

# - 现在我们已经准备好训练读出层，以根据（降维后的）储层状态序列预测期望的输出。
# - 我们首先使用由岭回归器实现的线性读出层。
# - 在读出层训练完成后，我们使用它来计算测试数据的预测 $\hat{Y}_\text{te}$。

# 拟合岭回归模型
ridge = Ridge(alpha=1.0) 
ridge.fit(states_tr, Ytr[n_drop:,:])

# 计算预测
Yhat = ridge.predict(states_te)

# 最后，我们绘制结果。

fig = plt.figure(figsize=(14,4))
plt.plot(Yte[n_drop:,:], 'k--', label="True", linewidth=2)
plt.plot(scaler.inverse_transform(Yhat.reshape(-1, 1)), label="Predicted")
plt.grid()
plt.legend()
plt.title("True vs predicted electricity load")
ridge_img_path = os.path.join(picture_dir, 'forecasting_ridge_result.png')
plt.savefig(ridge_img_path, dpi=150, bbox_inches='tight')
print(f"✅ 图片已保存: {ridge_img_path}")

# 显示图片
if IN_NOTEBOOK:
    # 在 Notebook 中直接显示
    display(Image(ridge_img_path))
else:
    # 在本地环境中显示（如果支持）
    try:
        plt.show(block=False)  # 非阻塞显示
    except:
        pass

plt.close()

# 拟合读出层（GBRT）

# - 将储层状态映射到期望的输出是一个标准的回归问题，可以使用 [sklearn](https://scikit-learn.org/stable/supervised_learning.html) 中的许多标准回归模型之一来解决。
# - 例如，我们可以使用梯度提升回归树，它为我们提供不同分位数的预测。
# - 通过这种方式，我们可以在预测中计算置信区间。
# - 这是实现概率预测的一种非常简单的方法
# - 在下面，我们将为 0.5、0.05 和 0.95 分位数拟合不同的模型，这将为我们提供 90\% 的预测置信区间。

# 分位数 0.5
max_iter = 100
gbrt_median = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5, max_iter=max_iter)
gbrt_median.fit(states_tr, Ytr[n_drop:,0])
median_predictions = gbrt_median.predict(states_te)

# 分位数 0.05
gbrt_percentile_5 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.05, max_iter=max_iter)
gbrt_percentile_5.fit(states_tr, Ytr[n_drop:,0])
percentile_5_predictions = gbrt_percentile_5.predict(states_te)

# 分位数 0.95
gbrt_percentile_95 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.95, max_iter=max_iter)
gbrt_percentile_95.fit(states_tr, Ytr[n_drop:,0])
percentile_95_predictions = gbrt_percentile_95.predict(states_te)

# 绘制带有 90% 置信区间的结果。

# 绘制结果
fig = plt.figure(figsize=(14,4))
plt.plot(Yte[n_drop:,:], 'k--', label="True", linewidth=2)
plt.plot(scaler.inverse_transform(median_predictions[:,None]), label="Median prediction", color="tab:blue")
plt.fill_between(np.arange(len(Yte[n_drop:,:])), scaler.inverse_transform(percentile_5_predictions[:,None]).ravel(), scaler.inverse_transform(percentile_95_predictions[:,None]).ravel(), alpha=0.3, label="90% CI", color="tab:blue")
plt.grid()
plt.legend()
plt.title("Predicted electricity load using Gradient Boosting Regression Trees")
gbrt_img_path = os.path.join(picture_dir, 'forecasting_gbrt_result.png')
plt.savefig(gbrt_img_path, dpi=150, bbox_inches='tight')
print(f"✅ 图片已保存: {gbrt_img_path}")

# 显示图片
if IN_NOTEBOOK:
    # 在 Notebook 中直接显示
    display(Image(gbrt_img_path))
else:
    # 在本地环境中显示（如果支持）
    try:
        plt.show(block=False)  # 非阻塞显示
    except:
        pass

plt.close()

# 列出所有保存的图片文件
print("\n" + "="*70)
print("📊 所有图片已生成完成！")
print("="*70)
list_saved_images()

# 创建 HTML 报告，方便在浏览器中查看所有图片
print("\n正在生成 HTML 报告...")
html_report_path = create_html_report()

print("\n" + "="*70)
print("💡 查看图片的多种方式：")
print("="*70)
print("1. 📁 直接打开文件夹查看:")
print(f"   {picture_dir}")
print("\n2. 🌐 在浏览器中打开 HTML 报告（推荐）:")
print(f"   {html_report_path}")
print("\n3. 📋 在 Kaggle 中:")
print("   - 图片会自动显示在输出单元格中")
print("   - 或在右侧 'Output' 标签页中查看 /kaggle/working/picture/ 文件夹")
print("\n4. 🔍 使用 Python 查看:")
print("   - 调用 list_saved_images() 函数查看文件列表")
print("="*70)
