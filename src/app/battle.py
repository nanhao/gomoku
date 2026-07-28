"""AI 内战会话模块

封装 AI 内战的完整流程：多局对战循环 + 训练样本生成。
app 包是薄的代理层，不包含业务逻辑，全部委托给 ai 包。
"""

import os
from typing import Callable, List

from ai import AIGameManager, DataRecorder, Evaluator
from ai.base_model import ModelPredict
from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR
from app.exceptions import ModelLoadError


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
        ai_data_dir: str = None,
        train_data_dir: str = None,
        temp_config=None,
    ):
        """
        初始化 AI 内战会话

        Args:
            ai1_name: 黑方 AI 名称（如 "random"、"fc_v1"）
            ai2_name: 白方 AI 名称
            num_games: 对战局数
            ai_data_dir: AI 数据目录，默认使用 AI_DATA_DIR 常量
            train_data_dir: 训练数据目录，默认使用 TRAIN_DATA_DIR 常量
            temp_config: 温度采样配置（ai.move_sampler.TempConfig）。
                None 表示全程 argmax（向后兼容），非 None 时透传给
                AIGameManager.start_ai_vs_ai。
        """
        self._ai1_name = ai1_name
        self._ai2_name = ai2_name
        self._num_games = num_games
        self._ai_data_dir = ai_data_dir if ai_data_dir is not None else AI_DATA_DIR
        self._train_data_dir = train_data_dir if train_data_dir is not None else TRAIN_DATA_DIR
        self._temp_config = temp_config
        self._recorder = None

    def run(self, on_progress: Callable[[int, int], None] = None) -> List[dict]:
        """
        执行全部对战并生成训练样本

        流程：
        1. 创建 DataRecorder
        2. 逐局调用 AIGameManager.start_ai_vs_ai（内部每局创建引擎）
        3. 调用 Evaluator.evaluate_round 生成训练样本

        Args:
            on_progress: 进度回调，每完成一局调用一次，参数为 (current, total)

        Returns:
            对战结果列表，每项为 {"game": 局数, "winner": 获胜方}
        """
        battle_name = f"{self._ai1_name}_vs_{self._ai2_name}"
        recorder = DataRecorder(battle_name, self._ai_data_dir)
        self._recorder = recorder

        # start_ai_vs_ai 内部每局调用 create_engine，调用方无需手动创建引擎
        # 捕获 create_engine 的 ValueError 并转换为 ModelLoadError
        try:
            results = AIGameManager.start_ai_vs_ai(
                self._ai1_name, self._ai2_name, self._num_games, recorder,
                on_progress=on_progress,
                temp_config=self._temp_config,
            )
        except ValueError as e:
            raise ModelLoadError(f"模型加载失败: {e}") from e

        # 对战完成后生成训练样本（价值头模型跳过传统棋型评估）
        if _is_value_only_model(self._ai1_name) or _is_value_only_model(self._ai2_name):
            print(f"[跳过评估] 价值头模型不使用传统棋型评估")
        else:
            evaluator = Evaluator(
                ai_data_dir=self._ai_data_dir,
                train_data_dir=self._train_data_dir,
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


def _is_value_only_model(ai_name: str) -> bool:
    """判断 AI 名称是否为价值头模型（cnn_v10）

    通过 registry 加载模型后检测 predict 返回类型。
    加载失败时返回 False（兜底安全策略）。

    Args:
        ai_name: AI 名称，如 "cnn_v10.1"

    Returns:
        True 表示价值头模型
    """
    try:
        from ai.registry import get_model
        model = get_model(ai_name)
        result = model.predict([0] * 226)
        return result.type == ModelPredict.TYPE_VALUE_ONLY
    except Exception:
        return False
