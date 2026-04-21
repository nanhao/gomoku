"""AI 包常量定义模块

定义 AI 数据目录和训练数据目录的路径常量。
"""

import os

# 获取 ai 包的目录路径
_AI_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# AI 数据目录: {project_root}/ai/ai_data
AI_DATA_DIR = os.path.join(_AI_PACKAGE_DIR, "ai_data")

# 训练数据目录: {project_root}/ai/train_data
TRAIN_DATA_DIR = os.path.join(_AI_PACKAGE_DIR, "train_data")

# Loss 日志目录: {project_root}/ai/ai_data/loss_logs
LOSS_LOG_DIR = os.path.join(AI_DATA_DIR, "loss_logs")

# Loss 日志记录间隔（每 N 个 epoch 记录一次，默认 200）
LOSS_LOG_INTERVAL = 200
