"""MCTS 对战 Worker 模块

包含 MctsBattleWorker（双模型多局对战执行器）和 BattleGameResult（对战结果数据类）。
供 ProcessPoolExecutor 的子进程调用，每个 worker 加载两个模型各一次后串行执行多局对战。

设计文档：design/42/main_docs/mcts_battle_worker_design.md
"""

import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

from ai.registry import get_model
from ai.engine.builder import AIEngineBuilder
from ai.scripts.mcts_game_runner import MctsGameRunner
from ai.scripts.mcts_config import MctsSelfPlayConfig


@dataclass
class BattleGameResult:
    """单局对战结果

    Attributes:
        game_idx: 局编号
        winner: 获胜方（0=平局, 1=黑胜, 2=白胜）
        model1_color: model1 执方（1=黑, 2=白）
        model2_color: model2 执方（1=黑, 2=白）
        model1_won: model1 是否获胜
        model2_won: model2 是否获胜
        move_count: 本局总步数
    """
    game_idx: int
    winner: int
    model1_color: int
    model2_color: int
    model1_won: bool
    model2_won: bool
    move_count: int


class MctsBattleWorker:
    """MCTS 对战 Worker

    封装双模型加载 + 多局对战执行逻辑，供 ProcessPoolExecutor 的子进程调用。
    每个 worker 实例加载两个模型各一次，可串行执行多局对战。
    每局随机分配 model1/model2 的执方。

    不可跨进程复用：构造器中加载的模型绑定到当前进程的设备上。
    """

    def __init__(
        self,
        model1_path: str,
        model1_name: Optional[str],
        model2_path: str,
        model2_name: Optional[str],
        device: str,
        config: MctsSelfPlayConfig,
        _model1=None,
        _model2=None,
    ):
        """初始化 worker，加载两个模型到指定设备

        Args:
            model1_path: 模型1 权重文件路径
            model1_name: 模型1 注册名.代数（如 "cnn_v8.10"），None 时从路径自动解析
            model2_path: 模型2 权重文件路径
            model2_name: 模型2 注册名.代数，None 时从路径自动解析
            device: 推理设备，多进程方案中固定为 "cpu"
            config: MCTS 超参数配置
            _model1: 可选，直接注入模型1 实例（测试用），非 None 时跳过加载逻辑
            _model2: 可选，直接注入模型2 实例（测试用），非 None 时跳过加载逻辑

        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 无法从路径解析模型名或模型加载失败
        """
        self._model1_path = model1_path
        self._model1_name = model1_name
        self._model2_path = model2_path
        self._model2_name = model2_name
        self._device = device
        self._config = config

        # 创建公用的对局运行器
        self._runner = MctsGameRunner(config)

        # 加载模型1
        if _model1 is not None:
            self._model1 = _model1
        else:
            self._model1 = self._load_model(model1_path, model1_name, device)

        # 加载模型2
        if _model2 is not None:
            self._model2 = _model2
        else:
            self._model2 = self._load_model(model2_path, model2_name, device)

    @staticmethod
    def _load_model(model_path: str, model_name: Optional[str], device: str):
        """加载模型到指定设备

        与 MctsSelfPlayWorker._load_model 逻辑完全一致：
        - 指定 model_name → 用 registry
        - 否则从 model_path 解析模型名和代数

        Args:
            model_path: 权重文件路径
            model_name: 模型注册名.代数，None 时从路径自动解析
            device: 推理设备

        Returns:
            加载并移动到指定设备的模型实例

        Raises:
            ValueError: 无法从路径解析模型名
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
                    f"请使用 --model1-name / --model2-name 显式指定"
                )

        # 移到指定设备
        device_obj = torch.device(device)
        if model.network is not None:
            model.move_to_device(device_obj)
        return model

    # ---------- 公共 API ----------

    @staticmethod
    def worker_entry(
        model1_path: str,
        model1_name: Optional[str],
        model2_path: str,
        model2_name: Optional[str],
        device: str,
        config: MctsSelfPlayConfig,
        game_indices: List[int],
        run_id: str,
    ) -> List[BattleGameResult]:
        """ProcessPoolExecutor 子进程入口

        在子进程中创建 MctsBattleWorker 并调用 run_games。
        定义为 @staticmethod 以确保 pickle 序列化可用。

        Args:
            model1_path: 模型1 权重文件路径
            model1_name: 模型1 注册名.代数
            model2_path: 模型2 权重文件路径
            model2_name: 模型2 注册名.代数
            device: 推理设备
            config: MCTS 超参数配置
            game_indices: 要执行的对局编号列表
            run_id: 本次运行唯一标识

        Returns:
            BattleGameResult 列表
        """
        worker = MctsBattleWorker(
            model1_path, model1_name, model2_path, model2_name, device, config
        )
        return worker.run_games(game_indices, run_id)

    def run_games(
        self,
        game_indices: List[int],
        run_id: str,
    ) -> List[BattleGameResult]:
        """执行多局对战，返回每局结果

        Args:
            game_indices: 要执行的对局编号列表
            run_id: 本次运行唯一标识（时间戳后缀）

        Returns:
            BattleGameResult 列表，每项包含单局结果
        """
        results: List[BattleGameResult] = []
        total = len(game_indices)
        for i, game_idx in enumerate(game_indices):
            t_start = time.time()

            # 随机分配执方
            model1_color = random.choice([1, 2])
            model2_color = 3 - model1_color

            # 每局创建全新的 engine 实例（MCTS 搜索树不可跨局复用）
            engine_black, engine_white = self._create_engines(
                model1_color, model2_color, run_id, game_idx
            )

            # 委托 MctsGameRunner 执行对局
            game_result = self._runner.run(
                engine_black, engine_white, run_id, game_idx
            )

            # 判断各方是否获胜
            winner = game_result.winner
            model1_won = (winner == model1_color) and winner != 0
            model2_won = (winner == model2_color) and winner != 0

            results.append(BattleGameResult(
                game_idx=game_idx,
                winner=winner,
                model1_color=model1_color,
                model2_color=model2_color,
                model1_won=model1_won,
                model2_won=model2_won,
                move_count=game_result.move_count,
            ))

            elapsed = time.time() - t_start
            print(f"[局 {i + 1}/{total}] game_idx={game_idx} — "
                  f"model1({model1_color}) vs model2({model2_color}), "
                  f"winner={winner}, 耗时: {elapsed:.1f}s")

        return results

    # ---------- 内部方法 ----------

    def _create_engines(
        self,
        model1_color: int,
        model2_color: int,
        run_id: str,
        game_idx: int,
    ):
        """为当前局创建黑方和白方两个 PolicyValueAiEngine 实例

        Args:
            model1_color: model1 执方（1=黑, 2=白）
            model2_color: model2 执方
            run_id: 本次运行唯一标识
            game_idx: 局编号

        Returns:
            (engine_black, engine_white) 元组
        """
        cfg = self._config
        log_key_prefix = f"run{run_id}_game{game_idx}"

        engine_black = AIEngineBuilder.build_policy_value(
            model=self._model1 if model1_color == 1 else self._model2,
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
        engine_white = AIEngineBuilder.build_policy_value(
            model=self._model2 if model1_color == 1 else self._model1,
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
        return engine_black, engine_white
