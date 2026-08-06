"""CNN v11 训练-评估循环编排器模块

将 SampleBufferReader、CNNV11Trainer、CheckpointEvaluator 组合为完整的
训练闭环：随机 mini-batch 训练 → checkpoint 评估 → 晋升/继续/停止。
"""
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ai import registry
from ai.sample_buffer_reader import SampleBufferReader
from ai.checkpoint_evaluator import CheckpointEvaluator
from ai.loss_logger import LossLogger
from ai.scripts.mcts_config import MctsSelfPlayConfig


@dataclass
class TrainLoopResult:
    """训练循环结果

    Attributes:
        stopped_by: 停止原因（"promoted" 或 "max_failures"）
        promoted_to: 晋升后的 gen spec，未晋升时为 None
        total_updates: 总共执行的 mini-batch 次数
        checkpoints_evaluated: 评估过的 checkpoint 数量
        best_win_rate: 所有 checkpoint 中的最高胜率
        loss_history: 每个 mini-batch 的 total_loss 列表
    """
    stopped_by: str
    promoted_to: Optional[str] = None
    total_updates: int = 0
    checkpoints_evaluated: int = 0
    best_win_rate: float = 0.0
    loss_history: List[float] = field(default_factory=list)


class CnnV11TrainLoop:
    """CNN v11 训练-评估循环编排器

    持有 trainer、buffer reader、evaluator，执行完整的训练闭环：
    1. 从缓冲区随机抽取 mini-batch 训练
    2. 每 N 步保存 checkpoint 并通过 C++ MCTS 评估
    3. 胜率 ≥ 55% 则晋升，连续失败 ≥ max_failures 则停止
    """

    def __init__(
        self,
        model_spec: str,
        buffer_dir: str,
        batch_size: int,
        updates_per_checkpoint: int,
        eval_games: int,
        max_consecutive_failures: int,
        mcts_config: MctsSelfPlayConfig,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cuda",
    ):
        """初始化训练循环

        加载初始模型（作为 best），导出 best TorchScript，
        初始化各组件。

        Args:
            model_spec: 初始模型 spec（如 "cnn_v11.1"），同时作为初始 best
            buffer_dir: 样本缓冲区目录
            batch_size: 每个 mini-batch 的样本数
            updates_per_checkpoint: 多少步后触发 checkpoint 评估
            eval_games: checkpoint 评估对战局数
            max_consecutive_failures: 连续评估失败上限
            mcts_config: MCTS 对战超参数配置
            lr: 学习率
            weight_decay: AdamW 权重衰减
            device: 训练设备（"cuda" 或 "cpu"）

        Raises:
            ValueError: 模型不存在或 spec 无效
        """
        # 解析代数
        parsed = registry._parse_model_spec(model_spec)
        if parsed.name != "cnn_v11":
            raise ValueError(f"不支持的模型: {parsed.name}，仅支持 cnn_v11")
        self._gen: int = parsed.gen if parsed.gen is not None else 1
        self._model_spec: str = model_spec

        # 加载 trainer
        self._trainer = registry.get_trainer(
            model_spec, lr=lr, weight_decay=weight_decay
        )

        # 设置训练设备
        self._trainer.device = torch.device(device)

        # Checkpoint 临时目录
        self._checkpoint_dir: str = os.path.join("temp", "train_checkpoint")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # 导出 best TorchScript（缓存，后续评估复用）
        self._best_ts_path: str = os.path.join(self._checkpoint_dir, "best.pt")
        self._export_torchscript(self._trainer.model.network, self._best_ts_path)

        # 创建组件
        self._reader: SampleBufferReader = SampleBufferReader(buffer_dir)
        self._evaluator: CheckpointEvaluator = CheckpointEvaluator(mcts_config)
        self._loss_logger: LossLogger = LossLogger(
            model_name="cnn_v11", dense_log=True
        )

        self._batch_size: int = batch_size
        self._updates_per_checkpoint: int = updates_per_checkpoint
        self._eval_games: int = eval_games
        self._max_consecutive_failures: int = max_consecutive_failures

        # 密集日志步数计数器（独立于 total_updates，从 1 开始）
        self._step_counter: int = 0

    # ---------- 公开方法 ----------

    def run(self) -> TrainLoopResult:
        """执行训练循环

        Returns:
            TrainLoopResult: 训练循环结果
        """
        total_updates: int = 0
        checkpoints_evaluated: int = 0
        consecutive_failures: int = 0
        best_win_rate: float = 0.0
        loss_history: List[float] = []

        try:
            checkpoint_round = 0
            while True:
                checkpoint_round += 1

                # 每轮 checkpoint 前刷新缓冲区
                self._reader.refresh()
                total_samples = self._reader.total_samples()

                print(f"[checkpoint #{checkpoint_round}] 开始训练 {self._updates_per_checkpoint} 步"
                      f"（缓冲区 {total_samples} 条样本）")

                # 训练 N 个 mini-batch
                round_loss_sum = 0.0
                for i in range(self._updates_per_checkpoint):
                    total_updates += 1
                    total_loss = self._train_one_batch()
                    loss_history.append(total_loss)
                    round_loss_sum += total_loss

                    # 每 10 步或最后一步打印进度
                    if (i + 1) % 10 == 0 or i == self._updates_per_checkpoint - 1:
                        avg_loss = round_loss_sum / (i + 1)
                        print(f"  step {total_updates:>5d}  [{i + 1:>3d}/{self._updates_per_checkpoint}]"
                              f"  loss={total_loss:.4f}  avg_loss={avg_loss:.4f}")

                # 保存 checkpoint
                print(f"  保存 checkpoint...", end=" ", flush=True)
                checkpoint_pth = os.path.join(
                    self._checkpoint_dir, "checkpoint.pth"
                )
                self._trainer.save_model(checkpoint_pth)
                print("pth ✓", end=" ", flush=True)

                # 导出 checkpoint TorchScript
                checkpoint_ts = os.path.join(
                    self._checkpoint_dir, "checkpoint.pt"
                )
                self._export_torchscript(
                    self._trainer.model.network, checkpoint_ts
                )
                print("pt ✓")

                # 评估
                print(f"  评估中（{self._eval_games} 局）...", end=" ", flush=True)
                win_rate = self._evaluator.evaluate(
                    checkpoint_ts,
                    self._best_ts_path,
                    self._eval_games,
                    temp_steps=2,
                )
                print(f"胜率 = {win_rate:.1%}")
                checkpoints_evaluated += 1
                best_win_rate = max(best_win_rate, win_rate)

                # 打印本轮状态
                status = "✓ 通过" if win_rate >= 0.55 else "✗ 未通过"
                print(f"  {status}  |  连续失败: {consecutive_failures + (0 if win_rate >= 0.55 else 1)}/{self._max_consecutive_failures}"
                      f"  |  最高胜率: {best_win_rate:.1%}")
                print()

                if win_rate >= 0.55:
                    # 晋升
                    promoted_gen = self._gen + 1
                    promoted_spec = f"cnn_v11.{promoted_gen}"

                    print(f">>> 晋升! checkpoint → {promoted_spec}")
                    self._promote_checkpoint(
                        checkpoint_pth, checkpoint_ts, promoted_gen
                    )

                    print("保存 loss 日志...")
                    self._loss_logger.flush()
                    self._loss_logger.plot()

                    return TrainLoopResult(
                        stopped_by="promoted",
                        promoted_to=promoted_spec,
                        total_updates=total_updates,
                        checkpoints_evaluated=checkpoints_evaluated,
                        best_win_rate=best_win_rate,
                        loss_history=loss_history,
                    )
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= self._max_consecutive_failures:
                        print(f">>> 连续 {self._max_consecutive_failures} 次失败，停止训练")
                        print("保存 loss 日志...")
                        self._loss_logger.flush()
                        self._loss_logger.plot()
                        return TrainLoopResult(
                            stopped_by="max_failures",
                            promoted_to=None,
                            total_updates=total_updates,
                            checkpoints_evaluated=checkpoints_evaluated,
                            best_win_rate=best_win_rate,
                            loss_history=loss_history,
                        )
        except KeyboardInterrupt:
            print()
            sys.exit(1)

    # ---------- 内部方法 ----------

    def _train_one_batch(self) -> float:
        """执行单次 mini-batch 训练

        从缓冲区随机抽样 → 转换为张量 → 执行 _train_step → 记录 loss。

        Returns:
            total_loss: 本 batch 的总损失值
        """
        # 随机抽样
        samples = self._reader.random_sample(self._batch_size)
        if not samples:
            # 缓冲区为空时，尝试刷新后再抽
            self._reader.refresh()
            samples = self._reader.random_sample(self._batch_size)
        if not samples:
            return 0.0

        # 转换数据
        states = [s["state"] for s in samples]
        pi_targets_list = [s["pi"] for s in samples]
        z_targets_list = [s["z"] for s in samples]

        batch_inputs = self._trainer._convert_inputs(states).to(
            self._trainer.device
        )
        batch_pi = torch.tensor(
            pi_targets_list, dtype=torch.float32, device=self._trainer.device
        )
        batch_z = torch.tensor(
            z_targets_list, dtype=torch.float32, device=self._trainer.device
        ).view(-1, 1)

        # 执行训练步
        result = self._trainer._train_step(batch_inputs, batch_pi, batch_z)

        total_loss = float(result["total_loss"])

        # 记录日志（使用独立步数计数器，不受 checkpoint 刷新影响）
        grad_norm = 0.0
        if result.get("grad_norms"):
            grad_norm = sum(result["grad_norms"].values())
        self._step_counter += 1
        self._loss_logger.log_dense(
            self._step_counter,
            total_loss,
            grad_norm,
        )

        return total_loss

    def _export_torchscript(self, network, output_path: str) -> str:
        """将网络导出为 TorchScript 模型文件

        Args:
            network: PyTorch nn.Module 网络
            output_path: 输出 .pt 文件路径

        Returns:
            输出文件路径
        """
        scripted = torch.jit.script(network)
        torch.jit.save(scripted, output_path)
        return output_path

    def _promote_checkpoint(
        self,
        checkpoint_pth: str,
        checkpoint_ts: str,
        promoted_gen: int,
    ) -> None:
        """将 checkpoint 晋升到正式的 gen 目录

        Args:
            checkpoint_pth: checkpoint PyTorch 权重文件路径
            checkpoint_ts: checkpoint TorchScript 文件路径
            promoted_gen: 晋升后的代数
        """
        from ai.models.cnn_v11.model import CNNV11Model

        # 使用类级别的 DEFAULT_MODEL_PATH 获取基础模型目录
        base_model_dir = os.path.dirname(CNNV11Model.DEFAULT_MODEL_PATH)
        promoted_dir = os.path.join(base_model_dir, f"gen{promoted_gen}")
        os.makedirs(promoted_dir, exist_ok=True)

        # 移动文件
        promoted_pth = os.path.join(promoted_dir, "cnn_v11.pth")
        promoted_ts = os.path.join(promoted_dir, "cnn_v11_torchscript.pt")

        shutil.move(checkpoint_pth, promoted_pth)
        shutil.move(checkpoint_ts, promoted_ts)
