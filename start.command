#!/bin/bash
# Hipodcast 一键启动（macOS）——在 Finder 里双击本文件即可。
# 它会自动：建运行环境 → 装依赖 → 检查 key → 打开浏览器 → 启动服务。

cd "$(dirname "$0")" || exit 1

echo "🎙️  Hipodcast 启动中…"
echo "------------------------------------"

# 1) 首次运行：创建独立运行环境
if [ ! -d ".venv" ]; then
  echo "首次启动，正在创建运行环境（只会进行一次）…"
  python3 -m venv .venv || { echo "❌ 创建环境失败，请确认已安装 Python3。"; read; exit 1; }
fi
source .venv/bin/activate

# 2) 安装依赖（已装则秒过）
if [ ! -f ".venv/.deps_ok" ]; then
  echo "正在安装依赖，首次会稍久，请稍候…"
  pip install -q -r requirements.txt && touch ".venv/.deps_ok"
fi

# 3) 检查 key 配置
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "⚠️  还没有配置 key。已为你生成 .env 文件并打开，"
  echo "    请填入 DeepSeek（和阿里云）的 key，保存后再次双击本文件启动。"
  open -e .env
  echo ""
  echo "（按回车键关闭本窗口）"
  read
  exit 0
fi

# 4) 两秒后自动打开浏览器
( sleep 2; open "http://127.0.0.1:8000" ) &

echo ""
echo "✅ 已启动！浏览器会自动打开： http://127.0.0.1:8000"
echo "   想停止：关闭此窗口，或按 Control + C。"
echo "------------------------------------"
echo ""

# 5) 启动服务
python app.py
