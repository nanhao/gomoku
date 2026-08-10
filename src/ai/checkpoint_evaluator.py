"""Checkpoint 评估器模块

包装 C++ MCTS 对战基础设施，提供 checkpoint vs best 的胜率评估。
"""

import os
import random
from typing import List

from ai.scripts.mcts_config import MctsSelfPlayConfig


class CheckpointEvaluator:
    """Checkpoint 评估器

    使用 C++ MCTS (cpp_mcts.GameRunner) 执行 checkpoint 与 best 模型的对战，
    计算 checkpoint 的胜率。双方交替执方，前 temp_steps 步温度采样。

    直接使用 cpp_mcts.GameRunner（绕过 CppMctsBattleAdapter 的 registry 路径解析），
    因为 checkpoint 是临时文件，不在 registry 中。
    """

    def __init__(self, mcts_config: MctsSelfPlayConfig):
        """初始化评估器

        Args:
            mcts_config: MCTS 对战超参数配置
        """
        self._mcts_config: MctsSelfPlayConfig = mcts_config

    def evaluate(
        self,
        checkpoint_ts_path: str,
        best_ts_path: str,
        num_games: int,
        temp_steps: int,
    ) -> float:
        """执行 checkpoint vs best 对战，返回 checkpoint 胜率

        双方交替执方（checkpoint 偶局执黑、奇局执白），
        前 temp_steps 步温度采样，之后贪婪。

        Args:
            checkpoint_ts_path: checkpoint TorchScript 模型文件路径
            best_ts_path: best TorchScript 模型文件路径
            num_games: 对战局数
            temp_steps: 温度采样步数（0 表示全程贪婪）

        Returns:
            checkpoint 的胜率，值域 [0.0, 1.0]

        Raises:
            FileNotFoundError: TorchScript 模型文件不存在
        """
        # 校验文件存在性
        if not os.path.isfile(checkpoint_ts_path):
            raise FileNotFoundError(f"checkpoint TorchScript 文件不存在: {checkpoint_ts_path}")
        if not os.path.isfile(best_ts_path):
            raise FileNotFoundError(f"best TorchScript 文件不存在: {best_ts_path}")

        import cpp_mcts  # type: ignore[import-not-found]

        # 构建 C++ GameConfig
        cpp_config = cpp_mcts.GameConfig()
        cpp_config.num_simulations = self._mcts_config.simulations
        cpp_config.c_puct = self._mcts_config.c_puct
        cpp_config.dirichlet_epsilon = self._mcts_config.dirichlet_epsilon
        cpp_config.dirichlet_alpha = self._mcts_config.dirichlet_alpha
        cpp_config.virtual_loss = self._mcts_config.virtual_loss
        cpp_config.temperature = self._mcts_config.temperature
        cpp_config.temp_steps = temp_steps
        cpp_config.use_gpu = True

        # 创建 GameRunner（checkpoint=model1, best=model2, 固定 10 个游戏线程）
        runner = cpp_mcts.GameRunner(
            checkpoint_ts_path, best_ts_path, num_games, cpp_config, 10
        )

        # 启动所有游戏线程
        runner.start()

        # 等待完成
        while not runner.is_finished():
            import time
            time.sleep(0.1)

        runner.wait_completion()

        # 获取结果并计算 checkpoint 胜率
        cpp_results = runner.get_results()
        checkpoint_wins = 0
        for r in cpp_results:
            # checkpoint = model1
            if r.winner != 0 and r.winner == r.model1_color:
                checkpoint_wins += 1

        total = len(cpp_results)
        if total == 0:
            return 0.0

        return checkpoint_wins / total
