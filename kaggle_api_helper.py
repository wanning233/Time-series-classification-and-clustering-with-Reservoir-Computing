#!/usr/bin/env python3
"""
Kaggle API 辅助脚本
用于自动下载 Notebook 输出结果

使用前需要配置:
1. 安装 kaggle: pip install kaggle
2. 下载 kaggle.json: https://www.kaggle.com/account
3. 放置到 ~/.kaggle/kaggle.json
4. 设置权限: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import json
from datetime import datetime

# 配置
KAGGLE_USERNAME = "wanningggg"
KAGGLE_NOTEBOOK = "notebook-time"
OUTPUT_DIR = "./kaggle_outputs"

def setup_kaggle_api():
    """检查 Kaggle API 配置"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print(f"❌ 未找到 Kaggle API 配置文件: {kaggle_json}")
        print("\n配置步骤:")
        print("1. 访问 https://www.kaggle.com/account")
        print("2. 点击 'Create New API Token'")
        print("3. 下载 kaggle.json 文件")
        print(f"4. 移动到 {kaggle_dir}/")
        print("5. 设置权限：chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # 设置权限
    os.chmod(kaggle_json, 0o600)
    print("✓ Kaggle API 配置正确")
    return True

def install_kaggle_package():
    """安装 Kaggle 包"""
    try:
        import kaggle
        print("✓ Kaggle 包已安装")
        return True
    except ImportError:
        print("正在安装 kaggle 包...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
        print("✓ Kaggle 包安装完成")
        return True

def list_notebook_runs():
    """列出 Notebook 运行历史"""
    print(f"\n📋 获取 Notebook 运行历史: {KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}")
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "list", "-m", "-u", KAGGLE_USERNAME],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取运行历史失败: {e}")
        return None

def download_output(run_number=None):
    """下载 Notebook 输出"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if run_number:
        print(f"\n📥 下载运行 #{run_number} 的输出...")
        output_path = os.path.join(OUTPUT_DIR, f"run_{run_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        print("\n📥 下载最新输出...")
        output_path = os.path.join(OUTPUT_DIR, f"latest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # 下载 Notebook 输出
        subprocess.run(
            ["kaggle", "kernels", "output", f"{KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}", "-p", output_path],
            check=True
        )
        print(f"✓ 输出已下载到: {output_path}")
        
        # 列出下载的文件
        print("\n下载的文件:")
        for f in os.listdir(output_path):
            print(f"  - {f}")
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return None

def push_notebook_changes():
    """推送本地 Notebook 更改到 Kaggle"""
    notebook_path = "./kaggle_notebook_template.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ 未找到 Notebook 文件：{notebook_path}")
        return False
    
    print(f"\n📤 推送 Notebook 到 Kaggle: {KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}")
    try:
        subprocess.run(
            ["kaggle", "kernels", "push", "-p", notebook_path],
            check=True
        )
        print("✓ Notebook 推送完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 推送失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("  Kaggle API 辅助工具")
    print("=" * 60)
    
    # 检查依赖
    if not install_kaggle_package():
        return
    
    if not setup_kaggle_api():
        return
    
    # 菜单
    print("\n请选择操作:")
    print("1. 列出运行历史")
    print("2. 下载最新输出")
    print("3. 下载指定运行输出")
    print("4. 推送 Notebook 到 Kaggle")
    print("5. 退出")
    
    choice = input("\n输入选项 (1-5): ").strip()
    
    if choice == "1":
        list_notebook_runs()
    elif choice == "2":
        download_output()
    elif choice == "3":
        run_num = input("输入运行编号: ").strip()
        download_output(run_num)
    elif choice == "4":
        push_notebook_changes()
    elif choice == "5":
        print("退出")
    else:
        print("无效选项")

if __name__ == "__main__":
    main()
