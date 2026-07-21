"""MCTS 自对弈 Worker 模块

包含 MctsSelfPlayConfig（超参数数据类）和 MctsSelfPlayWorker（多局执行器）。
供 ProcessPoolExecutor 的子进程调用，每个 worker 加载一次模型后串行执行多局。
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from ai.registry import get_model
from ai.engine.builder import AIEngineBuilder
from ai.constants import STRATEGY_SAMPLING
from game.game import check_winner, is_board_full


@dataclass
class MctsSelfPlayConfig:
    """MCTS 自对弈超参数配置

    替代旧的 hyperparams 字典，符合"结构化数据用 dataclass"规范。
    所有字段均有默认值，与当前 mcts_self_play.py 的 CLI 默认值一致。
    """
    simulations: int = 800
    c_puct: float = 2.0
    temp_steps: int = 8
    temperature: float = 1.0
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.03
    batch_size: int = 225
    virtual_loss: float = 3.0
    use_prior: bool = True


class MctsSelfPlayWorker:
    """MCTS 自对弈 Worker

    封装模型加载和多局执行逻辑，供 ProcessPoolExecutor 的子进程调用。
    每个 worker 实例加载一次模型，可串行执行多局对弈。

    不可跨进程复用：构造器中加载的模型绑定到当前进程的设备上。
    """

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str],
        device: str,
        config: MctsSelfPlayConfig,
        _model=None,
    ):
        """初始化 worker，加载模型到指定设备

        Args:
            model_path: 权重文件路径，用于自动解析模型名和代数
            model_name: 模型注册名.代数（如 "cnn_v8.1"），None 时从 model_path 解析
            device: 推理设备，多进程方案中固定为 "cpu"
            config: 超参数配置
            _model: 可选，直接注入模型实例（测试用），非 None 时跳过加载逻辑

        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 无法从路径解析模型名或模型加载失败
        """
        self._model_path = model_path
        self._model_name = model_name
        self._device = device
        self._config = config

        # 加载模型：逻辑与当前 run_self_play 中的模型加载一致
        if _model is not None:
            self._model = _model
        else:
            self._model = self._load_model(model_path, model_name, device)

    @staticmethod
    def _load_model(model_path: str, model_name: Optional[str], device: str):
        """加载模型到指定设备

        与 mcts_self_play.py 中 run_self_play 的模型加载逻辑完全一致：
        - 指定 model_name → 用 registry
        - 否则从 model_path 解析模型名和代数
        """
        import torch

        if model_name:
            model = get_model(model_name)
        else:
            # 从路径解析模型名和代数：.../cnn_v8/gen1/cnn_v8.pth → cnn_v8.1
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
                    f"请使用 --model-name 显式指定，如 --model-name cnn_v8.1"
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
        config: "MctsSelfPlayConfig",
        game_indices: List[int],
        run_id: str,
    ) -> List[dict]:
        """ProcessPoolExecutor 子进程入口

        在子进程中创建 MctsSelfPlayWorker 并执行对局。
        定义为 @staticmethod 以确保 pickle 序列化可用，
        同时遵循"非入口脚本只定义类"规范。
        """
        worker = MctsSelfPlayWorker(model_path, model_name, device, config)
        return worker.run_games(game_indices, run_id)

    def run_games(
        self,
        game_indices: List[int],
        run_id: str,
    ) -> List[dict]:
        """执行多局自对弈，返回所有样本

        Args:
            game_indices: 要执行的对局编号列表
            run_id: 本次运行唯一标识（时间戳后缀）

        Returns:
            所有对局产生的样本列表，每项为 {"state": [...], "pi": [...], "z": int}
        """
        all_samples: List[dict] = []
        total = len(game_indices)
        for i, game_idx in enumerate(game_indices):
            t_start = time.time()
            samples = self._play_one_game(game_idx, run_id)
            all_samples.extend(samples)
            elapsed = time.time() - t_start
            print(f"[局 {i + 1}/{total}] game_idx={game_idx} — "
                  f"{len(samples)} 步, 结果: {self._describe_result(samples)}, "
                  f"耗时: {elapsed:.1f}s")
        return all_samples

    def _play_one_game(self, game_idx: int, run_id: str) -> List[dict]:
        """执行一局自我对弈

        与当前 mcts_self_play.py 中 play_one_game() 等价：
        创建 PolicyValueAiEngine → 循环落子 → 记录样本 → 终局回填 z 值。
        每局使用全新 engine 实例（模型复用），确保 MCTS 搜索树不跨局污染。

        Args:
            game_idx: 局编号（用于 log_key）
            run_id: 本次运行唯一标识

        Returns:
            本局样本列表
        """
        cfg = self._config
        log_key_prefix = f"run{run_id}_game{game_idx}"
        engine = AIEngineBuilder.build_policy_value(
            model=self._model,
            num_simulations=cfg.simulations,
            c_puct=cfg.c_puct,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
            dirichlet_alpha=cfg.dirichlet_alpha,
            batch_size=cfg.batch_size,
            virtual_loss=cfg.virtual_loss,
            temperature=cfg.temperature,
            temp_steps=cfg.temp_steps,
            log_key_prefix=log_key_prefix,
            use_prior=cfg.use_prior,
        )

        # 初始化棋盘
        board = [0] * 225
        current_player = 1  # 黑先
        samples: List[dict] = []

        while True:
            state = board + [current_player]
            board_2d = [[board[i * 15 + j] for j in range(15)] for i in range(15)]

            # MCTS 搜索 + 决策
            md = engine.get_move(board_2d, current_player, strategy=STRATEGY_SAMPLING)
            x, y = md.get_position()
            action = x * 15 + y
            visit_counts = md.get_mcts_visit_counts()

            # 计算 π 分布（τ=1.0 归一化）
            total_visits = sum(visit_counts)
            if total_visits > 0:
                pi = [c / total_visits for c in visit_counts]
            else:
                pi = [1.0 / 225] * 225

            # 记录样本（z 暂填 0，终局后回填）
            samples.append({"state": state, "pi": pi, "z": 0})

            # 落子
            board[action] = current_player

            # 终局判定
            winner = check_winner(board, action)
            is_terminal = (winner != 0) or is_board_full(board)

            # 子树复用
            engine.after_move(x, y)

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
