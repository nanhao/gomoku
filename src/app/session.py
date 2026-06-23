"""游戏会话模块

封装一局游戏（pvp / pve）的会话流程。
app 包是薄的代理层，不包含业务逻辑，所有逻辑委托给 game 包和 ai 包。
"""

from game import GomokuController
from ai import AIGameManager, create_engine
from app.exceptions import ModelLoadError


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
            # 捕获 create_engine 的 ValueError，转换为 ModelLoadError 上抛
            # 由 GUI 层捕获后弹窗提示用户，回退到模式选择界面
            try:
                engine = create_engine(ai_name or "random")
            except ValueError as e:
                raise ModelLoadError(f"模型加载失败：{e}") from e
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

    def start_game(self, save_name: str = None) -> dict:
        """
        启动游戏

        - 创建游戏
        - 如果是 PvE 且 AI 执黑，自动执行 AI 落子

        Args:
            save_name: 存档名称，None 则使用默认名称

        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "ai_moved": bool,      # AI 是否已落子（AI先走时）
                "game_over": bool,     # 游戏是否结束
                "winner": int          # 仅 game_over=True 时存在
            }
        """
        # 创建游戏
        result = self._controller.new_game(save_name)
        if "error" in result:
            return {"status": "error", "error": result["error"], "ai_moved": False}

        # 如果 AI 执黑，执行 AI 落子
        if self._mode == "pve" and self._ai_player == 1:
            ai_result = self._ai_manager.ai_move()
            game_over = self._controller.is_over()
            return {
                "status": ai_result.get("status", "error"),
                "ai_moved": True,
                "game_over": game_over,
                **({"winner": self._controller.get_winner()} if game_over else {})
            }

        return {
            "status": "success",
            "ai_moved": False,
            "game_over": False
        }

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

        # 检查游戏是否已创建
        if self._controller.game is None:
            return {"status": "error", "error": "游戏尚未创建"}

        result = self._ai_manager.ai_move()

        game_over = self._controller.is_over()
        response = {
            "status": result.get("status", "error"),
            "game_over": game_over,
        }
        if game_over:
            response["winner"] = self._controller.get_winner()

        return response
