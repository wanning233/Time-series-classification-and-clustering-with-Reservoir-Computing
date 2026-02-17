# 层叠 Reservoir 实验工作流

本文档描述完整的实验迭代流程：**本地修改 → GitHub → Kaggle 运行 → 结果分析 → 调优**

---

## 📋 快速开始

### 前置准备

```bash
# 1. 安装 Kaggle API (可选，用于自动化)
pip install kaggle

# 2. 配置 Kaggle API 密钥
# 访问 https://www.kaggle.com/account → Create New API Token
# 下载 kaggle.json 并移动到 ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🔄 完整工作流

### 步骤 1: 本地修改代码

```bash
cd /Users/lawliet/Desktop/毕业设计/Time-series-classification-and-clustering-with-Reservoir-Computing

# 编辑你的实验代码
# 修改配置：experiment_config.json
# 修改脚本：examples/notebooks/ 或 reservoir_computing/
```

### 步骤 2: 提交并推送到 GitHub

```bash
# 方式 A: 使用工作流脚本（推荐）
chmod +x workflow.sh
./workflow.sh

# 方式 B: 手动操作
git add .
git commit -m "实验更新：调整层叠配置为 [500,400,350,300,200]"
git push origin master
```

### 步骤 3: Kaggle 运行

#### 方式 A: 从 GitHub 拉取（推荐）

1. 打开 Notebook: https://www.kaggle.com/code/wanningggg/notebook-time/edit
2. 在第一个单元格运行:
```python
!git pull https://github.com/wanning233/Time-series-classification-and-clustering-with-Reservoir-Computing.git master
```
3. 确保开启 Internet 访问（Settings → Internet → On）
4. 运行所有单元格

#### 方式 B: 上传 Dataset

```bash
# 1. 压缩项目
cd /Users/lawliet/Desktop/毕业设计/
zip -r reservoir-project.zip Time-series-classification-and-clustering-with-Reservoir-Computing

# 2. 上传到 Kaggle Datasets
# 访问 https://www.kaggle.com/datasets → New Dataset → 上传 ZIP

# 3. 在 Notebook 中添加数据集并解压
import zipfile
with zipfile.ZipFile('/kaggle/input/reservoir-project/reservoir-project.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')
```

### 步骤 4: 等待运行完成

- Kaggle 限制：最长 9 小时/次
- 免费 GPU：每周约 30 小时
- 可在 Kaggle App 中接收完成通知

### 步骤 5: 下载结果

#### 方式 A: 手动下载

1. 打开 Notebook → Output 标签
2. 点击文件右侧的 ⋮ → Download
3. 保存到本地分析

#### 方式 B: 使用 API 自动下载

```bash
# 运行辅助脚本
python kaggle_api_helper.py

# 选择选项 2: 下载最新输出
```

#### 方式 C: 命令行下载

```bash
kaggle kernels output wanningggg/notebook-time -p ./kaggle_outputs
```

### 步骤 6: 分析结果并调优

```bash
# 查看下载的结果
ls -la ./kaggle_outputs/

# 分析实验报告
# 阅读生成的结果文件，对比不同配置的性能

# 修改配置进行下一轮实验
# 编辑 experiment_config.json
# 重复步骤 1-6
```

---

## 📊 实验记录模板

每次实验后，在 `experiment_log.md` 中记录：

```markdown
## 实验记录

### 实验 ID: exp_001
- **日期**: 2026-02-17
- **配置**: 
  - 层数：5
  - 单元数：[400, 350, 300, 200, 150]
  - 谱半径：0.99
  - 连接度：0.3
- **数据集**: Japanese_Vowels
- **结果**:
  - NMI: 0.8295
  - ARI: 0.7856
- **观察**: 5 层表现最佳，6 层可能过拟合
- **下一步**: 尝试更宽的单元配置
```

---

## 🎯 调优策略

### 优先级顺序

1. **层数** → 先确定最佳层数（当前 5 层最佳）
2. **单元配置** → 调整每层单元数
3. **谱半径** → 0.90 ~ 0.999
4. **连接度** → 0.2 ~ 0.5
5. **其他参数** → input_scaling, leak, noise

### 推荐实验顺序

```
1. 基准测试 (当前配置)
2. 宽配置测试 [500,400,350,300,200]
3. 窄配置测试 [300,250,200,150,100]
4. 谱半径调优 (0.95, 0.99, 0.999)
5. 连接度调优 (0.2, 0.3, 0.4)
```

---

## 🛠️ 工具脚本

| 脚本 | 用途 |
|------|------|
| `workflow.sh` | 一键提交 + 推送 + Kaggle 指引 |
| `kaggle_api_helper.py` | Kaggle API 自动化（下载/推送） |
| `experiment_config.json` | 实验配置管理 |

---

## ⚠️ 注意事项

1. **Kaggle 限制**
   - 单次运行最长 9 小时
   - 每周 GPU 配额约 30 小时
   - 内存限制约 16GB

2. **数据持久化**
   - `/kaggle/working/` 目录的文件会保留
   - 重要结果及时下载

3. **版本控制**
   - 每次实验前 commit 代码
   - 使用有意义的 commit message
   - 记录实验配置和结果

---

## 📚 相关文档

- [KAGGLE_GUIDE.md](./KAGGLE_GUIDE.md) - Kaggle 详细指南
- [层叠 Reservoir 调优指南.md](./层叠 Reservoir 调优指南.md) - 调优参数说明
- [实验报告 - 层叠 Reservoir 训练结果总结.md](./实验报告 - 层叠 Reservoir 训练结果总结.md) - 历史实验结果

---

## 🆘 常见问题

**Q: Kaggle 无法访问 GitHub？**
A: 确保 Notebook 设置中开启了 Internet 访问

**Q: 内存不足？**
A: 减少 `n_internal_units` 或使用更小的数据集

**Q: 如何批量运行多个实验？**
A: 使用 Kaggle 的"Copy & Edit"创建多个 Notebook 版本并行运行

**Q: 结果如何对比？**
A: 下载所有结果到本地，用 Python/pandas 整理对比表格
