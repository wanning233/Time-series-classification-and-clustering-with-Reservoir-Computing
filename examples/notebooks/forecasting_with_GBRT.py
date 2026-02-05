# Probabilistic forecasting with GBRT

# The class `reservoir_computing.RC_forecaster` allows to quickly perform forecasting by fitting a linear model that maps the reservoir states into the predictions. The linear is implemented as the ridge regressor from sklearn `sklearn.linear_model.Ridge`.

# It is however possible to use other regression models from sklearn, including those that computes confidence intervals obtaining, in this way, a probabilistic forecasting. 

# In this example we will use `sklearn.ensemble.HistGradientBoostingRegressor`, a Gradient Boost Regression Tree (GBRT) that allows to compute different quantiles.

# Let's start by importing the necessary libraries.

import os
import matplotlib
# 设置 matplotlib 后端和内联显示
try:
    # 尝试设置为内联模式（在 Jupyter/Kaggle Notebook 中）
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    # 如果不是在 Notebook 环境中，使用默认后端
    try:
        matplotlib.use('TkAgg')  # 尝试使用交互式后端
    except:
        pass  # 使用默认后端

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA

from reservoir_computing.reservoir import Reservoir
from reservoir_computing.utils import make_forecasting_dataset
from reservoir_computing.datasets import PredLoader

# 尝试导入 IPython.display（在 Kaggle Notebook 中可用）
try:
    from IPython.display import Image, display, HTML
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

np.random.seed(0) # For reproducibility

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

# Load the data

# We will use the dataloader `PredLoader` to get a forecasting datatset.
# To see what datatsets are available, we can call the function `available_datasets`. By setting `details=True` we can get additional information.

downloader = PredLoader()
downloader.available_datasets(details=True)  # Describe available datasets

# - For this example, we will use `ElecRome` that is the electricity consumption coming from a backbone of the energy supply network in the city of Rome.
# - The original data is a time series sampled every 10 minutes. 
# - If we are not interested in such an high resolution, we can resample the data to hourly resolution by creating a new time series whose entries are the means of 6 consecutive time steps in the original series. 
# - We also take only the first `3000` time steps to be faster in fitting the model.

# Download data
ts_full = downloader.get_data("ElecRome")

# Resample the time series to hourly frequency
ts_hourly = np.mean(ts_full.reshape(-1, 6), axis=1)[:, None]
print("Resampled: ", ts_hourly.shape)

# Use only the first 3000 time steps
ts_small = ts_hourly[0:3000,:]
print("Resampled small: ", ts_small.shape)

# Prepare the datasets

# To train our regression model we need input and target data `Xtr` and `Ytr`. 
# We also need test data `Xte` and `Yte` to test our model and validation data `Xval` and `Yval` if we need to do hyperparameters tuning.
# We will use the function `make_forecasting_dataset` that given a time series `X` does the following computations:

# 1. Splits the dataset in consecutive chunks: `train`, `val` and `test`. The size of the chunks is given by the values `val_percent` and `test_percent`. If we do not need validation data, set `val_percent=0` (default) and the validation data will not be created.
# 2. Create input data `X` and target data `Y` by shifting the data `horizon` time steps, where `horizon` is how far we want to predict. For example:
#     - `Xtr = train[:-horizon,:]`
#     - `Ytr = train[horizon:,:]`
# 3. Normalizes the data using a scaler object from `sklearn.preprocessing`. If no scalers are passed, a `StandardScaler` is created. The scaler is fit on `Xtr` and then used to transform `Ytr`, `Xval`, and `Xte`. Note that `Yval` and `Yte` are **not** transformed.

# The code below exemplifies the use of the function.

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

# - For this example, we want to make forecast 24h ahead. 
# - Also, we are not opitimizing the hyperparameters of the model so we do not need a valiation set (we leave the default `val_percent=0` and the validation data will not be returned).

# Generate training and test datasets
Xtr, Ytr, Xte, Yte, scaler = make_forecasting_dataset(ts_small,
                                                      horizon=24, # forecast horizon of 24h ahead
                                                      test_percent = 0.1)

# Define the Reservoir

# - Next, we create a Reservoir by specifying (rather arbitrarily) the hyperparameters.
# - Then, we compute the sequence of the Reservoir states `states_tr` and `states_te` associated with the training and test data, respecitvely.

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

# Dimensionalty reduction (optional)

# - As optional step, we can reduce the size of the Reservoir states, which in this example is quite large. 
# - In particular, since `n_internal_units=1000`, we end up with a sequence of length `T` of vectors with size `1000`.
# - Dimensionality reduction can speed up the training, especially if we use a sophisticated model as the readout, and can also provide some regularization that improves the prediction performances.
# - Below, we Reduce the dimension of the reservoir states from `1000` to `75`.

pca = PCA(n_components=75)
states_tr = pca.fit_transform(states_tr[0])
states_te = pca.transform(states_te[0])

# Fit the readout (linear)

# - We are now ready to train the readout to predict the desired output given the sequence of (reduced) Reservoir states.
# - We start by using a linear readout implemented by a Ridge regressor.
# - After the readout is trained, we use it to compute the predictions $\hat{Y}_\text{te}$ of the test data.

# Fit the ridge regression model
ridge = Ridge(alpha=1.0) 
ridge.fit(states_tr, Ytr[n_drop:,:])

# Compute the predictions
Yhat = ridge.predict(states_te)

# Finally, we plot the results.

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

# Fit the readout (GBRT)

# - Mapping the Reservoir states to the desired output is a standard regression problem, which can be solved by one of the many standard regression models in [sklearn](https://scikit-learn.org/stable/supervised_learning.html).
# - For example, we can use a Gradient Boost Regression Tree, which gives us predictions for different quantiles.
# - In this way, we can compute confidence intervals in our predictions.
# - This is a very simple way to implement probabilistic forecasting
# - In the following, we will fit a different model for the 0.5, 0.05 and 0.95 quantiles, which will give us a 90\% confidence interval for our prediction.

# Quantile 0.5
max_iter = 100
gbrt_median = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5, max_iter=max_iter)
gbrt_median.fit(states_tr, Ytr[n_drop:,0])
median_predictions = gbrt_median.predict(states_te)

# Quantile 0.05
gbrt_percentile_5 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.05, max_iter=max_iter)
gbrt_percentile_5.fit(states_tr, Ytr[n_drop:,0])
percentile_5_predictions = gbrt_percentile_5.predict(states_te)

# Quantile 0.95
gbrt_percentile_95 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.95, max_iter=max_iter)
gbrt_percentile_95.fit(states_tr, Ytr[n_drop:,0])
percentile_95_predictions = gbrt_percentile_95.predict(states_te)

# Plot the results with the confidence 90% confidence intervals.

# Plot the results
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
