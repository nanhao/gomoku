"""GUI 组件模块

包含五子棋 GUI 的各个独立组件：
- constants: 常量定义（颜色、尺寸等）
- mode_selector: 游戏模式选择界面
- game_board: 游戏棋盘组件
- heatmap: 热力图组件
- replay_viewer: 回放查看器组件
- replay_browser: 回放浏览器组件
- ai_battle: AI 内战组件
"""

from .constants import COLORS, DIMENSIONS, FONTS
from .mode_selector import GameModeSelector
from .game_board import GameBoard
from .heatmap import HeatmapCanvas, HeatmapLegend
from .replay_viewer import ReplayViewer, ReplayWindow
from .replay_browser import ReplayBrowser, RoundSelector, GameSelector
from .ai_battle import AIBattleConfigDialog, AIBattleProgressWindow, AIBattleManager

__all__ = [
    # 常量
    "COLORS",
    "DIMENSIONS",
    "FONTS",
    # 主要组件
    "GameModeSelector",
    "GameBoard",
    "HeatmapCanvas",
    "HeatmapLegend",
    "ReplayViewer",
    "ReplayWindow",
    "ReplayBrowser",
    "RoundSelector",
    "GameSelector",
    "AIBattleConfigDialog",
    "AIBattleProgressWindow",
    "AIBattleManager",
]
