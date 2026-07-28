"""价值头自对弈 Worker 模块

包含 ValueSelfPlayConfig 和 ValueSelfPlayWorker。
供 ProcessPoolExecutor 的子进程调用，每个 worker 加载一次模型后串行执行多局。
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from ai.registry import get_model
from ai.engine.value_only import ValueOnlyAiEngine
from ai.engine.ai_engine import AIEngine
from ai.constants import STRATEGY_SAMPLING
from game.game import check_winner, is_board_full


@dataclass
class ValueSelfPlayConfig:
    """价值头自对弈超参数配置

    所有字段均有默认值，与 value_self_play.py 的 CLI 默认值一致。
    """
    temperature: float = 1.0
    temp_steps: int = 8
    temp_side: str = "random"
    top_k: Optional[int] = None
    seed: Optional[int] = None


class ValueSelfPlayWorker:
    """价值头自对弈 Worker

    封装模型加载和多局执行逻辑，供 ProcessPoolExecutor 的子进程调用。
    每个 worker 实例加载一次模型，可串行执行多局对弈。

    不可跨进程复用：构造器中加载的模型绑定到当前进程的设备上。
    """

    # 8 个方向
    _DIRECTIONS: List[tuple] = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, 1), (-1, 1), (1, -1),
    ]

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str],
        device: str,
        config: ValueSelfPlayConfig,
        _model=None,
    ):
        """初始化 worker，加载模型到指定设备

        Args:
            model_path: 权重文件路径，用于自动解析模型名和代数
            model_name: 模型注册名.代数（如 "cnn_v10.1"），None 时从 model_path 解析
            device: 推理设备，多进程方案中固定为 "cpu"
            config: 超参数配置
            _model: 可选，直接注入模型实例（测试用），非 None 时跳过加载逻辑
        """
        self._model_path = model_path
        self._model_name = model_name
        self._device = device
        self._config = config

        # 加载模型
        if _model is not None:
            self._model = _model
        else:
            self._model = self._load_model(model_path, model_name, device)

    # ---------- 模型加载 ----------

    @staticmethod
    def _load_model(model_path: str, model_name: Optional[str], device: str):
        """加载模型到指定设备

        与 mcts_self_play.py 中模型加载逻辑一致：
        - 指定 model_name → 用 registry
        - 否则从 model_path 解析模型名和代数
        """
        import torch

        if model_name:
            model = get_model(model_name)
        else:
            # 从路径解析模型名和代数：.../cnn_v10/gen1/cnn_v10.pth → cnn_v10.1
            norm_path = os.path.normpath(model_path)
            parts = norm_path.split(os.sep)
            parsed_name = None
            parsed_gen = None
            for i, p in enumerate(parts):
                if p.startswith("gen"):
                    try:
                        parsed_gen = int(p[3:])
                    except ValueError:
                        continue
                    if i >= 1:
                        parsed_name = parts[i - 1]
                    break
            if parsed_name and parsed_gen is not None:
                spec = f"{parsed_name}.{parsed_gen}"
                model = get_model(spec)
            else:
                raise ValueError(
                    f"无法从路径 '{model_path}' 解析模型名称和代数。"
                    f"请使用 --model-name 显式指定，如 --model-name cnn_v10.1"
                )

        # 移到指定设备
        device_obj = torch.device(device)
        if model.network is not None:
            model.move_to_device(device_obj)
        return model

    # ---------- 公共 API ----------

    @staticmethod
    def worker_entry(
        model_path: str,
        model_name: Optional[str],
        device: str,
        config: ValueSelfPlayConfig,
        game_indices: List[int],
        run_id: str,
    ) -> List[dict]:
        """ProcessPoolExecutor 子进程入口

        在子进程中创建 ValueSelfPlayWorker 并执行对局。
        定义为 @staticmethod 以确保 pickle 序列化可用。

        Args:
            model_path: 权重文件路径
            model_name: 模型注册名
            device: 推理设备
            config: 超参数配置
            game_indices: 对局编号列表
            run_id: 运行唯一标识

        Returns:
            所有对局的样本列表
        """
        worker = ValueSelfPlayWorker(model_path, model_name, device, config)
        return worker.run_games(game_indices, run_id)

    def run_games(
        self,
        game_indices: List[int],
        run_id: str,
    ) -> List[dict]:
        """执行多局自对弈，返回所有样本

        Args:
            game_indices: 要执行的对局编号列表
            run_id: 本次运行唯一标识

        Returns:
            所有对局产生的样本列表，每项为 {"state": [...], "z": int}
        """
        all_samples: List[dict] = []
        total = len(game_indices)
        for i, game_idx in enumerate(game_indices):
            t_start = time.time()
            samples = self._play_one_game(game_idx)
            all_samples.extend(samples)
            elapsed = time.time() - t_start
            result_desc = self._describe_result(samples)
            print(f"[局 {i + 1}/{total}] game_idx={game_idx} — "
                  f"{len(samples)} 步, 结果: {result_desc}, "
                  f"耗时: {elapsed:.1f}s")
        return all_samples

    # ---------- 单局对弈 ----------

    def _play_one_game(self, game_idx: int) -> List[dict]:
        """执行一局自我对弈

        使用 ValueOnlyAiEngine 执行对局循环，收集每步的 state。
        终局后根据结果回填 z 值。

        Args:
            game_idx: 局编号

        Returns:
            本局样本列表，每项 {"state": [...], "z": int}
        """
        import random as _random

        cfg = self._config

        # 创建引擎（双方使用同一模型）
        engine = ValueOnlyAiEngine(
            model=self._model,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )

        # 温度采样方编排
        rng = _random.Random(cfg.seed + game_idx if cfg.seed is not None else None)
        temp_side = cfg.temp_side
        if temp_side == "random":
            temp_side = _random.choice(["black", "white", "both"])

        # 对局循环
        board: List[int] = [0] * 225
        current_player: int = 1
        move_count: int = 0
        samples: List[dict] = []

        while True:
            # 记录当前 state
            state = board + [current_player]
            samples.append({"state": state, "z": 0})

            # 判定本步策略
            is_black = (current_player == 1)
            use_sampling = (
                move_count < cfg.temp_steps
                and (
                    temp_side == "both"
                    or (temp_side == "black" and is_black)
                    or (temp_side == "white" and not is_black)
                )
            )
            strategy = STRATEGY_SAMPLING if use_sampling else "argmax"

            # 转换为 15×15 二维数组供 engine 使用
            board_2d = [[board[i * 15 + j] for j in range(15)] for i in range(15)]

            # AI 决策
            md = engine.get_move(board_2d, current_player, strategy=strategy)
            x, y = md.get_position()
            action = x * 15 + y

            # 落子
            board[action] = current_player
            move_count += 1

            # 终局判定
            winner = check_winner(board, action)
            is_terminal = (winner != 0) or is_board_full(board)

            if is_terminal:
                break

            # 切换玩家
            current_player = 2 if current_player == 1 else 1

        # 回填 z 值
        final_winner = winner if winner != 0 else 0
        for sample in samples:
            player = sample["state"][225]
            if final_winner == 0:
                sample["z"] = 0
            elif final_winner == player:
                sample["z"] = 1
            else:
                sample["z"] = -1

        return samples

    # ---------- 辅助 ----------

    @staticmethod
    def _describe_result(samples: List[dict]) -> str:
        """用人类可读方式描述对局结果"""
        if not samples:
            return "无样本"
        last_z = samples[-1]["z"]
        last_player = samples[-1]["state"][225]
        if last_z == 1:
            return f"玩家{last_player}胜"
        elif last_z == -1:
            other = 2 if last_player == 1 else 1
            return f"玩家{other}胜"
        else:
            return "平局"
