"""CNN v11 off-policy 监督训练编排器模块

将 OffPolicyTrainer 组合为完整训练闭环：
加载模型 → 预检样本 → train_epochs 全量多 epoch 训练（epoch 级打印 + update 级
高密度曲线图）→ 保存 gen{N+1}（pth + TorchScript）。

loss 曲线默认每 log_every_updates（默认 50）次 update 记录一次窗口均值，
X 轴为累计 update 序号；val_total_loss 仍每 epoch 评估一次，
画在各 epoch 的 end_step 位置。

Ctrl+C 由 train_epochs 优雅捕获（先生成曲线图），中断后仍保存模型，
结果通过 OffPolicyTrainResult 标记 interrupted。
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ai import registry
from ai.models.cnn_v11.model import CNNV11Model
from ai.off_policy_trainer import OffPolicyTrainer


@dataclass
class OffPolicyTrainResult:
    """off-policy 训练结果

    Attributes:
        saved_gen: 保存的代数编号（N+1）
        saved_spec: 保存的模型 spec（如 "cnn_v11.2"）
        requested_epochs: 请求的 epoch 总数
        total_epochs: 实际完成的 epoch 数（中断时 < requested_epochs）
        interrupted: 是否被 Ctrl+C 中断
        final_total_loss: 最后完成 epoch 的平均 total_loss（无完成 epoch 时为 0.0）
        final_val_loss: 最后完成 epoch 的平均 val_total_loss（val_split>0 时有效）
        history: train_epochs 返回的完整 epoch 历史
    """
    saved_gen: int
    saved_spec: str
    requested_epochs: int
    total_epochs: int
    interrupted: bool
    final_total_loss: float
    final_val_loss: Optional[float]
    history: List[Dict[str, float]]


class OffPolicyTrainLoop:
    """CNN v11 off-policy 训练编排器

    持有 OffPolicyTrainer，执行"加载模型 → 全量训练 → 保存新 gen"完整流程。
    模型进出遵循多代体系：从 gen{N} 加载，训练结束保存为 gen{N+1}。

    Args:
        model_spec: 初始模型 spec（如 "cnn_v11.1"），必须为 cnn_v11 且带代数
        data_dir: 样本目录（chunk JSON 或 sample.json 格式）
        batch_size: 每个 mini-batch 的样本数
        epochs: 训练 epoch 数
        val_split: 验证集比例，值域 [0, 1)
        lr: 学习率
        weight_decay: AdamW 权重衰减
        device: 训练设备（"cuda" 或 "cpu"），cuda 不可用时自动回退 cpu 并警告
        log_every_updates: 每 N 次 update 记录一次 loss 用于高密度曲线，
            默认 0（关闭，仅 epoch 级记录）

    Raises:
        ValueError: model_spec 非 cnn_v11、未指定代数或模型不存在
    """

    def __init__(
        self,
        model_spec: str,
        data_dir: str,
        batch_size: int,
        epochs: int,
        val_split: float,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        log_every_updates: int = 0,
    ):
        # 解析并校验模型 spec
        parsed = registry._parse_model_spec(model_spec)
        if parsed.name != "cnn_v11":
            raise ValueError(f"不支持的模型: {parsed.name}，仅支持 cnn_v11")
        if parsed.gen is None:
            raise ValueError(f"必须指定代数，如 cnn_v11.1")
        self._gen: int = parsed.gen
        self._model_spec: str = model_spec

        # 设备回退：cuda 不可用时回退 cpu 并警告
        resolved_device = device
        if device == "cuda" and not torch.cuda.is_available():
            print(f"[警告] CUDA 不可用，回退到 CPU")
            resolved_device = "cpu"
        self._device = torch.device(resolved_device)

        # 加载初始模型（模型不存在时抛 ValueError）
        try:
            model = CNNV11Model(fresh=False, gen=self._gen)
        except FileNotFoundError as exc:
            raise ValueError(f"模型不存在: {model_spec}") from exc

        # 构造训练器：OffPolicyTrainer 覆盖 _load_samples，训练能力继承基类
        self._trainer: OffPolicyTrainer = OffPolicyTrainer(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            device=self._device,
            gen=self._gen,
        )

        # 保存训练参数
        self._data_dir: str = data_dir
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._val_split: float = val_split
        self._log_every_updates: int = log_every_updates

    # ---------- 公开方法 ----------

    def run(self) -> OffPolicyTrainResult:
        """执行完整训练流程

        样本预检 → 打印信息 → train_epochs（epoch 循环 + mini-batch + 验证集评估
        + 每 epoch 打印 + 自动曲线图）→ 保存 gen{N+1}（pth + TorchScript）。

        Returns:
            OffPolicyTrainResult: 训练结果

        Raises:
            ValueError: 样本总数为 0（无有效样本）
        """
        # 预检样本：无有效样本时直接报错，不进入训练与保存
        states, _, _ = self._trainer._load_samples([self._data_dir])
        if not states:
            raise ValueError(f"数据目录中无有效样本: {self._data_dir}")

        # 打印数据规模信息（与 train_epochs 内部划分规则一致）
        total_samples = len(states)
        n_val = int(total_samples * self._val_split)
        n_train = total_samples - n_val
        n_batches = math.ceil(n_train / self._batch_size) if n_train > 0 else 0
        print(f"样本总数: {total_samples}（训练 {n_train} / 验证 {n_val}）")
        print(f"训练配置: {self._epochs} epochs × {n_batches} batch/epoch"
              f"（batch_size={self._batch_size}，val_split={self._val_split}）")

        # 执行训练（Ctrl+C 由 train_epochs 优雅捕获，先完成曲线图再返回）
        history = self._trainer.train_epochs(
            sample_dirs=[self._data_dir],
            epochs=self._epochs,
            batch_size=self._batch_size,
            val_split=self._val_split,
            shuffle=True,
            verbose=True,
            log_every_updates=self._log_every_updates,
        )

        # 保存 gen{N+1}（覆盖时打印提示）
        promoted_gen = self._gen + 1
        saved_spec = f"cnn_v11.{promoted_gen}"
        base_model_dir = os.path.dirname(CNNV11Model.DEFAULT_MODEL_PATH)
        promoted_dir = os.path.join(base_model_dir, f"gen{promoted_gen}")
        if os.path.isdir(promoted_dir):
            print(f"[提示] gen{promoted_gen} 已存在，将覆盖")
        os.makedirs(promoted_dir, exist_ok=True)
        pth_path = os.path.join(promoted_dir, "cnn_v11.pth")
        ts_path = os.path.join(promoted_dir, "cnn_v11_torchscript.pt")
        self._trainer.save_model(pth_path)
        self._export_torchscript(self._trainer.model.network, ts_path)
        print(f"模型已保存: {saved_spec} -> {promoted_dir}")

        # 汇总结果：中断 = 实际完成 epoch 数小于请求数
        interrupted = len(history) < self._epochs
        final_total_loss = history[-1]["total_loss"] if history else 0.0
        final_val_loss = (
            history[-1].get("val_total_loss") if history else None
        )
        return OffPolicyTrainResult(
            saved_gen=promoted_gen,
            saved_spec=saved_spec,
            requested_epochs=self._epochs,
            total_epochs=len(history),
            interrupted=interrupted,
            final_total_loss=final_total_loss,
            final_val_loss=final_val_loss,
            history=history,
        )

    # ---------- 内部方法 ----------

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
