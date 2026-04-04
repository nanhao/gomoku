"""app 包 - 游戏应用层

位于 ai/game 包之上、gui 包之下的薄代理层。
封装游戏会话流程，与 UI 解耦。
"""

from .session import GameSession
from .battle import BattleSession
from .gui_helper import GUIHelper, get_ai_data_dir, create_pve_config

__all__ = [
    "GameSession",
    "BattleSession",
    "GUIHelper",
    "get_ai_data_dir",
    "create_pve_config",
]
