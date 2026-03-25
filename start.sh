#!/bin/bash
# 完整修复架构问题的启动脚本

echo "🚀 PSD to React AI Agent - 修复版启动脚本"
echo "============================================="

BASE_DIR="/Users/tongzonghua/Desktop/workspace/python-project/learn-claude-code/"
cd "$BASE_DIR"

# 2. 创建全新的 ARM64 虚拟环境
if [ ! -d "venv_arm64" ]; then
    echo "📦 创建 ARM64 虚拟环境..."
    arch -arm64 python3 -m venv venv_arm64
fi

echo "🔧 激活 ARM64 虚拟环境..."
source venv_arm64/bin/activate

# 3. 在 ARM64 模式下安装所有依赖
echo ""
echo "📥 安装依赖（ARM64 模式）..."
arch -arm64 venv_arm64/bin/pip install --upgrade pip setuptools wheel
arch -arm64 venv_arm64/bin/pip install -r requirements.txt

# 关键：使用 arch -arm64 启动 Python 进程本身
arch -arm64 venv_arm64/bin/python main.py --reload --log-level info
