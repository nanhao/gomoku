"""AI 模块 - 五子棋人工智能

本模块包含 AI 引擎、模型基类、具体 AI 实现和训练数据评价器。
所有 AI 相关逻辑都在此包中，不改动 game 包。
"""

from .base_model import BaseModel
from .engine import AIEngine, create_engine
from .data_recorder import DataRecorder
from .game_manager import AIGameManager
from .evaluator import Evaluator

__all__ = [
    "BaseModel",
    "AIEngine",
    "create_engine",
    "DataRecorder",
    "AIGameManager",
    "Evaluator",
]
