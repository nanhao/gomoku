"""游戏会话模块

封装一局游戏（pvp / pve）的会话流程。
app 包是薄的代理层，不包含业务逻辑，所有逻辑委托给 game 包和 ai 包。
"""

from game import GomokuController
from ai import AIGameManager, create_engine


class GameSession:
    """游戏会话 - 管理一局游戏的流程

    职责：
    - 持有游戏模式（pvp / pve）
    - pve 模式下协调 AIGameManager
    - 向 GUI 提供统一的落子接口，返回含 need_ai_move 标志的结果

    不包含任何游戏规则逻辑，全部委托给 GomokuController 和 AIGameManager。
    """

    def __init__(
        self,
        mode: str,
        controller: GomokuController,
        ai_name: str = None,
        ai_player: int = None,
    ):
        """
        初始化游戏会话

        Args:
            mode: 游戏模式，"pvp" 或 "pve"
            controller: GomokuController 实例
            ai_name: AI 名称（pve 模式使用），如 "random"、"fc_v1"
            ai_player: AI 执棋方（1=黑, 2=白），pve 模式使用
        """
        self._mode = mode
        self._controller = controller
        self._ai_player = ai_player
        self._ai_manager = None

        if mode == "pve" and ai_player is not None:
            engine = create_engine(ai_name or "random")
            self._ai_manager = AIGameManager(controller, engine, ai_player)

    @property
    def mode(self) -> str:
        """游戏模式：'pvp' 或 'pve'"""
        return self._mode

    @property
    def ai_player(self):
        """AI 执棋方（1=黑, 2=白），pvp 模式返回 None"""
        return self._ai_player

    def make_player_move(self, x: int, y: int) -> dict:
        """
        玩家落子

        委托给 GomokuController.make_move()，并判断是否需要 GUI 调度 AI 落子。

        Args:
            x: 落子行坐标（0-14）
            y: 落子列坐标（0-14）

        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "game_over": bool,     # 游戏是否结束
                "need_ai_move": bool,  # GUI 是否需要调度 AI 落子
                "winner": int          # 仅 game_over=True 时存在：1=黑, 2=白, 3=平局
            }
        """
        result = self._controller.make_move(x, y)

        if "error" in result:
            return {"status": "error", "error": result["error"]}

        game_over = self._controller.is_over()
        need_ai_move = (
            not game_over
            and self._mode == "pve"
            and self._controller.get_current_player() == self._ai_player
        )

        response = {
            "status": "success",
            "game_over": game_over,
            "need_ai_move": need_ai_move,
        }
        if game_over:
            response["winner"] = self._controller.get_winner()

        return response

    def do_ai_move(self) -> dict:
        """
        执行 AI 落子

        委托给 AIGameManager.ai_move()。

        Returns:
            {
                "status": "success" | "error",
                "game_over": bool,
                "winner": int  # 仅 game_over=True 时存在
            }

        Raises:
            RuntimeError: pvp 模式下调用此方法
        """
        if self._ai_manager is None:
            raise RuntimeError("pvp 模式没有 AI，不能调用 do_ai_move()")

        result = self._ai_manager.ai_move()

        game_over = self._controller.is_over()
        response = {
            "status": result.get("status", "error"),
            "game_over": game_over,
        }
        if game_over:
            response["winner"] = self._controller.get_winner()

        return response
