"""AI 内战会话模块

封装 AI 内战的完整流程：多局对战循环 + 训练样本生成。
app 包是薄的代理层，不包含业务逻辑，全部委托给 ai 包。
"""

import os
from typing import Callable, List

from ai import AIGameManager, DataRecorder, Evaluator, create_engine


class BattleSession:
    """AI 内战会话 - 管理 AI 内战的完整流程

    职责：
    - 创建两个 AI 引擎（委托给 ai.create_engine）
    - 执行多局对战循环（委托给 AIGameManager.start_ai_vs_ai）
    - 对战完成后生成训练样本（委托给 Evaluator）
    - 通过 on_progress 回调向 GUI 报告进度

    不包含任何对战或评估逻辑，全部委托给下层。
    """

    def __init__(
        self,
        ai1_name: str,
        ai2_name: str,
        num_games: int,
        project_root: str,
    ):
        """
        初始化 AI 内战会话

        Args:
            ai1_name: 黑方 AI 名称（如 "random"、"fc_v1"）
            ai2_name: 白方 AI 名称
            num_games: 对战局数
            project_root: 项目根目录，用于确定 ai_data / train_data 路径
        """
        self._ai1_name = ai1_name
        self._ai2_name = ai2_name
        self._num_games = num_games
        self._project_root = project_root
        self._recorder = None

    def run(self, on_progress: Callable[[int, int], None] = None) -> List[dict]:
        """
        执行全部对战并生成训练样本

        流程：
        1. 创建两个 AIEngine（委托给 ai.create_engine）
        2. 创建 DataRecorder
        3. 逐局调用 AIGameManager.start_ai_vs_ai（每局调用 on_progress）
        4. 调用 Evaluator.evaluate_round 生成训练样本

        Args:
            on_progress: 进度回调，每完成一局调用一次，参数为 (current, total)

        Returns:
            对战结果列表，每项为 {"game": 局数, "winner": 获胜方}
        """
        engine1 = create_engine(self._ai1_name)
        engine2 = create_engine(self._ai2_name)

        battle_name = f"{self._ai1_name}_vs_{self._ai2_name}"
        ai_data_dir = os.path.join(self._project_root, "ai_data")
        train_data_dir = os.path.join(self._project_root, "train_data")

        recorder = DataRecorder(battle_name, ai_data_dir)
        self._recorder = recorder

        results = AIGameManager.start_ai_vs_ai(
            engine1, engine2, self._num_games, recorder,
            on_progress=on_progress,
        )

        # 对战完成后生成训练样本
        evaluator = Evaluator(
            ai_data_dir=ai_data_dir,
            train_data_dir=train_data_dir,
        )
        evaluator.evaluate_round(recorder.get_round_dir())

        return results

    def get_recorder(self) -> DataRecorder:
        """
        返回本次对战的 DataRecorder

        GUI 可通过此方法获取 round_dir 用于回放。

        Returns:
            DataRecorder 实例；run() 未调用时返回 None
        """
        return self._recorder
