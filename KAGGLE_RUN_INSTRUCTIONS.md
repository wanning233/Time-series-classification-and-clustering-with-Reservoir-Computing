# Kaggle 运行实验指南 - exp_002

## 🚀 快速开始（3 步完成）

### 步骤 1: 打开 Kaggle Notebook

访问：**https://www.kaggle.com/code/wanningggg/notebook-time/edit**

---

### 步骤 2: 上传 Notebook 文件

**方式 A: 直接上传（推荐）**

1. 在 Kaggle Notebook 页面，点击右上角 **"Upload"** 按钮
2. 选择本地文件：
   ```
   /Users/lawliet/Desktop/毕业设计/Time-series-classification-and-clustering-with-Reservoir-Computing/kaggle_exp_002_wide_config.ipynb
   ```
3. 等待上传完成

**方式 B: 从 GitHub 导入**

1. 点击右侧 **"+ Add data"** 或 **"+"** 按钮
2. 选择 **"GitHub"** 标签
3. 授权 Kaggle 访问 GitHub
4. 选择仓库：`wanning233/Time-series-classification-and-clustering-with-Reservoir-Computing`
5. 找到文件：`kaggle_exp_002_wide_config.ipynb`
6. 点击导入

---

### 步骤 3: 运行 Notebook

1. **确认设置**（右侧面板）：
   - ✅ **Internet**: On（必须开启，用于克隆项目）
   - ⚙️ **Accelerator**: GPU 或 CPU（GPU 更快）
   - ⏱️ **Session Timeout**: 最长 9 小时

2. **运行所有单元格**：
   - 点击顶部 **"Run All"** 按钮
   - 或逐个单元格按 `Shift+Enter` 运行

3. **等待运行完成**：
   - 预计时间：10-30 分钟（取决于数据集大小）
   - 可以关闭页面，Kaggle 会继续运行
   - 完成后会收到邮件通知

---

## 📊 查看和下载结果

### 运行完成后

1. 打开 Notebook → 点击 **"Output"** 标签
2. 你会看到以下文件：
   - `results_exp_002_YYYYMMDD_HHMMSS.csv` - 实验结果（NMI, ARI）
   - `config_exp_002_YYYYMMDD_HHMMSS.json` - 实验配置
   - `visualization_exp_002.png` - 可视化图表

### 下载文件

- 点击文件右侧的 **⋮**（三个点）
- 选择 **"Download"**
- 保存到本地分析

---

## 📋 实验配置摘要

| 参数 | 值 |
|------|-----|
| **实验 ID** | exp_002 |
| **名称** | 宽配置测试 - 更强表达能力 |
| **层数** | 5 |
| **单元配置** | [500, 400, 350, 300, 200] |
| **谱半径** | 0.99 |
| **连接度** | 0.3 |
| **数据集** | Japanese_Vowels, Libras |
| **预期结果** | NMI > 0.83, ARI > 0.78 |

---

## 🔧 常见问题

### Q1: 上传失败怎么办？
**A**: 检查文件大小（Kaggle 限制 100MB），或改用 GitHub 导入方式

### Q2: 运行时报错 "ModuleNotFoundError"？
**A**: 确保 Internet 已开启，Notebook 会自动从 GitHub 克隆项目

### Q3: 运行时间太长？
**A**: 
- 使用 GPU 加速（右侧设置 → Accelerator → GPU）
- 减少数据集数量（修改 DATASETS 列表）
- 减少单元数（修改 units_per_layer）

### Q4: 如何运行其他实验？
**A**: 
```bash
# 生成其他实验的 Notebook
cd /Users/lawliet/Desktop/毕业设计/Time-series-classification-and-clustering-with-Reservoir-Computing
python generate_experiment.py --exp exp_003 --output kaggle_exp_003.ipynb
```

---

## 📁 本地文件位置

| 文件 | 路径 |
|------|------|
| Notebook | `kaggle_exp_002_wide_config.ipynb` |
| Python 代码 | `kaggle_exp_002_wide_config.py` |
| 实验配置 | `experiment_config.json` |
| 实验计划 | `EXPERIMENT_PLAN.md` |

---

## 🎯 下一步

1. ✅ 上传并运行 `kaggle_exp_002_wide_config.ipynb`
2. ⏳ 等待运行完成
3. 📥 下载结果文件
4. 📊 对比基准结果（NMI=0.8295）
5. 🔄 根据结果决定下一个实验（exp_003 或其他）

---

## 📞 需要帮助？

查看完整文档：
- [WORKFLOW.md](./WORKFLOW.md) - 完整工作流
- [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) - 10 个实验计划
- [KAGGLE_GUIDE.md](./KAGGLE_GUIDE.md) - Kaggle 详细指南
