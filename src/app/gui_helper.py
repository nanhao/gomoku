"""GUI 辅助模块

处理与界面无关的业务逻辑，使 GUI 层只保留界面代码。
"""

import os

from ai.constants import AI_DATA_DIR


class GUIHelper:
    """GUI 辅助类 - 处理与界面无关的业务逻辑"""

    def __init__(self, controller, session):
        """初始化 GUIHelper

        Args:
            controller: GomokuController 实例
            session: GameSession 实例
        """
        self.controller = controller
        self.session = session

    def start_game_with_ai_first(self) -> dict:
        """AI 先走时启动游戏

        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "ai_moved": bool,
                "game_over": bool,
                "winner": int          # 仅 game_over=True 时存在
            }
        """
        result = self.session.start_game()

        if result["status"] == "error":
            return {
                "status": "error",
                "error": result.get("error", "启动游戏失败"),
                "ai_moved": False
            }

        return {
            "status": "success",
            "ai_moved": result.get("ai_moved", False),
            "game_over": result.get("game_over", False),
            **({"winner": result["winner"]} if result.get("game_over") else {})
        }

    def handle_player_move(self, row: int, col: int) -> dict:
        """处理玩家落子

        Args:
            row: 行坐标
            col: 列坐标

        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "game_over": bool,
                "winner": int,         # 仅 game_over=True 时存在
                "need_ai_move": bool   # 是否需要 AI 落子
            }
        """
        result = self.session.make_player_move(row, col)

        if result["status"] != "success":
            return {
                "status": "error",
                "error": result.get("error", "落子失败")
            }

        return {
            "status": "success",
            "game_over": result.get("game_over", False),
            "need_ai_move": result.get("need_ai_move", False),
            **({"winner": result["winner"]} if result.get("game_over") else {})
        }

    def handle_ai_move(self) -> dict:
        """处理 AI 落子

        Returns:
            {
                "status": "success" | "error",
                "game_over": bool,
                "winner": int  # 仅 game_over=True 时存在
            }
        """
        result = self.session.do_ai_move()

        if result.get("status") != "success":
            return {
                "status": "error",
                "error": result.get("error", "AI 落子失败")
            }

        return {
            "status": "success",
            "game_over": result.get("game_over", False),
            **({"winner": result["winner"]} if result.get("game_over") else {})
        }

    def get_display_state(self) -> dict:
        """获取显示状态

        Returns:
            {
                "has_game": bool,
                "save_name": str,
                "current_player": int,
                "winner": int,
                "board": List[List[int]],
                "last_move": Tuple[int, int] or None,
                "history": List[dict]
            }
        """
        if self.controller.game is None:
            return {
                "has_game": False,
                "save_name": "",
                "current_player": 0,
                "winner": 0,
                "board": [],
                "last_move": None,
                "history": []
            }

        state = self.controller.get_state()
        history = state.get("history", [])
        last_move = None
        if history:
            last = history[-1]
            last_move = (last["x"], last["y"])

        return {
            "has_game": True,
            "save_name": self.controller.current_save_name or "未知",
            "current_player": state.get("current_player", 1),
            "winner": state.get("winner", 0),
            "board": state.get("board", []),
            "last_move": last_move,
            "history": history
        }

def get_ai_data_dir() -> str:
    """获取 ai_data 目录路径

    Returns:
        ai_data 目录的绝对路径（位于 ai/ai_data/）
    """
    return AI_DATA_DIR


def create_pve_config(selector_result: dict) -> dict:
    """根据选择器结果创建 PvE 配置

    Args:
        selector_result: GameModeSelector 返回的结果

    Returns:
        {
            "game_mode": "pve",
            "ai_player": int,  # 1=AI执黑, 2=AI执白
            "ai_model": str
        }
    """
    player_color = selector_result.get("player_color")
    ai_model = selector_result.get("ai", "random")

    # 玩家执黑 -> AI 执白(2)；玩家执白 -> AI 执黑(1)
    ai_player = 2 if player_color == "black" else 1

    return {
        "game_mode": "pve",
        "ai_player": ai_player,
        "ai_model": ai_model
    }
