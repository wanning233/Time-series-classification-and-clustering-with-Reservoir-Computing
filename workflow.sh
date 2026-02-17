#!/bin/bash
# 层叠 Reservoir 实验工作流脚本
# 用法：./workflow.sh [commit_message]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_DIR="/Users/lawliet/Desktop/毕业设计/Time-series-classification-and-clustering-with-Reservoir-Computing"
GITHUB_REPO="https://github.com/wanning233/Time-series-classification-and-clustering-with-Reservoir-Computing.git"
KAGGLE_USERNAME="wanningggg"
KAGGLE_NOTEBOOK="notebook-time"

cd "$PROJECT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  层叠 Reservoir 实验工作流${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 步骤 1: 检查 Git 状态
echo -e "${YELLOW}[步骤 1/4] 检查 Git 状态...${NC}"
git status --short

if [ -z "$(git status --short)" ]; then
    echo -e "${GREEN}✓ 没有未提交的更改${NC}"
else
    echo -e "${YELLOW}发现未提交的更改${NC}"
fi
echo ""

# 步骤 2: 提交更改
echo -e "${YELLOW}[步骤 2/4] 提交更改到 Git...${NC}"
read -p "输入提交信息 (默认: '实验更新'): " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-"实验更新"}

git add .
git commit -m "$COMMIT_MSG"
echo -e "${GREEN}✓ 提交完成${NC}"
echo ""

# 步骤 3: 推送到 GitHub
echo -e "${YELLOW}[步骤 3/4] 推送到 GitHub...${NC}"
git push origin master
echo -e "${GREEN}✓ 推送完成${NC}"
echo -e "${BLUE}GitHub 仓库：$GITHUB_REPO${NC}"
echo ""

# 步骤 4: Kaggle 操作指引
echo -e "${YELLOW}[步骤 4/4] Kaggle 操作指引${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  下一步操作${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. 打开 Kaggle Notebook:"
echo -e "   ${GREEN}https://www.kaggle.com/code/$KAGGLE_USERNAME/$KAGGLE_NOTEBOOK/edit${NC}"
echo ""
echo "2. 更新代码（二选一）:"
echo "   A) 从 GitHub 拉取（推荐）:"
echo "      !git pull $GITHUB_REPO master"
echo ""
echo "   B) 重新上传 Dataset:"
echo "      - 压缩项目文件夹为 ZIP"
echo "      - 上传到 Kaggle Datasets"
echo "      - 在 Notebook 中添加数据集"
echo ""
echo "3. 运行 Notebook 并等待结果"
echo ""
echo "4. 下载结果到本地:"
echo "   - 点击 Output → Download"
echo "   - 或使用 Kaggle API 自动下载"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ 工作流完成！${NC}"
echo -e "${BLUE}========================================${NC}"
