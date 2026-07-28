"""CNN v10 训练器实现

负责训练 CNN v10 价值头模型。
使用 MSE 损失（仅价值），不涉及策略头。

训练流程：
    加载 sample.json → 转换为张量 → epoch 循环（shuffle + batch）→ 生成 loss 曲线图
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F

from .model import CNNV10Model
from ai.base_model import board_vector_to_cnn_tensor_relative

# matplotlib 优雅降级
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class CNNV10Trainer:
    """CNN v10 价值头训练器

    训练仅使用 MSE 损失：loss = (v_pred - z_target)²。
    参照 DualHeadTrainer 结构，但仅训练价值头。

    Attributes:
        model: CNNV10Model 实例
        optimizer: AdamW 优化器
        device: 训练设备
    """

    # 默认超参数
    DEFAULT_LR: float = 0.001
    DEFAULT_WEIGHT_DECAY: float = 1e-4

    def __init__(
        self,
        model: Optional[CNNV10Model] = None,
        fresh: bool = False,
        device: Optional[torch.device] = None,
        lr: float = None,
        weight_decay: float = None,
        gen: Optional[int] = None,
    ):
        """初始化训练器

        Args:
            model: 模型实例，None 时根据 fresh 创建
            fresh: 是否创建新模型（不加载已有权重）
            device: 训练设备，None 时自动检测（优先 GPU）
            lr: 学习率，None 时使用 DEFAULT_LR
            weight_decay: AdamW 权重衰减，None 时使用 DEFAULT_WEIGHT_DECAY
            gen: 代数
        """
        self.gen = gen

        # 确定设备
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 创建或保存模型
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model(fresh)

        # 移动到训练设备
        if self.model.network is not None:
            self.model.network.to(self.device)

        # 初始化 AdamW 优化器
        actual_lr = lr if lr is not None else self.DEFAULT_LR
        actual_wd = weight_decay if weight_decay is not None else self.DEFAULT_WEIGHT_DECAY
        self.optimizer = optim.AdamW(
            self.model.network.parameters(),
            lr=actual_lr,
            weight_decay=actual_wd,
        )

    # ---------- 模型创建 ----------

    def _create_model(self, fresh: bool) -> CNNV10Model:
        """创建 CNN v10 模型实例

        Args:
            fresh: 是否创建新模型

        Returns:
            CNNV10Model 实例
        """
        gen = getattr(self, 'gen', None)
        if gen is not None:
            return CNNV10Model(fresh=fresh, gen=gen)
        return CNNV10Model(fresh=fresh)

    # ---------- 样本加载（静态方法） ----------

    @staticmethod
    def _load_samples(sample_dirs: List[str]) -> tuple:
        """加载 sample.json 数据

        期望 sample.json 格式：[{"state": [...226], "z": int}, ...]

        Args:
            sample_dirs: sample.json 所在目录列表

        Returns:
            (states: List[List[int]], zs: List[int]) 元组

        Raises:
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        all_states: List[List[int]] = []
        all_zs: List[int] = []

        for d in sample_dirs:
            path = os.path.join(d, "sample.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Sample file not found: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            for s in samples:
                all_states.append(s["state"])
                all_zs.append(s["z"])

        return all_states, all_zs

    # ---------- 输入转换（静态方法） ----------

    @staticmethod
    def _convert_inputs(inputs: List[List[int]]) -> torch.Tensor:
        """将 226 维 state 列表转换为 (N, 3, 15, 15) 张量

        使用相对视角编码（AlphaGo Zero 风格）。

        Args:
            inputs: state 列表，每个 226 维

        Returns:
            形状为 (N, 3, 15, 15) 的张量
        """
        batch_tensors = []
        for board_vector in inputs:
            tensor = board_vector_to_cnn_tensor_relative(
                board_vector, torch.device("cpu")
            )
            batch_tensors.append(tensor)
        return torch.cat(batch_tensors, dim=0)

    # ---------- 训练入口 ----------

    def train_epochs(
        self,
        sample_dirs: List[str],
        epochs: int,
        batch_size: int = 2048,
        val_split: float = 0.0,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """执行完整训练

        数据加载 → 分割验证集 → epoch 循环（shuffle + batch 训练 + 验证评估）。

        Args:
            sample_dirs: sample.json 所在目录列表
            epochs: 训练轮数
            batch_size: 每批样本数，默认 2048
            val_split: 验证集比例，默认 0
            shuffle: 是否打乱数据，默认 True
            verbose: 是否输出训练日志

        Returns:
            epoch history 列表，每项含 value_loss 和 total_loss
        """
        # 加载数据
        all_states, all_zs = self._load_samples(sample_dirs)

        # 转换张量
        inputs = self._convert_inputs(all_states).to(self.device)
        z_targets = torch.tensor(
            all_zs, dtype=torch.float32, device=self.device
        ).view(-1, 1)

        N = inputs.size(0)

        # 分割验证集
        has_val = val_split > 0
        if has_val:
            n_val = int(N * val_split)
            n_train = max(1, N - n_val)  # 边界保护：至少 1 个训练样本
            if n_val >= N:
                n_val = N - 1
                n_train = 1
            perm = torch.randperm(N)
            train_indices = perm[:n_train]
            val_indices = perm[n_train:]
            train_inputs = inputs[train_indices]
            val_inputs = inputs[val_indices]
            train_z = z_targets[train_indices]
            val_z = z_targets[val_indices]
            n_train_actual = n_train
        else:
            train_inputs = inputs
            train_z = z_targets
            val_inputs = None
            val_z = None
            n_train_actual = N

        # Epoch 循环
        self.model.network.train()
        history: List[Dict[str, float]] = []

        try:
            for epoch in range(epochs):
                # 每 epoch 重新 shuffle
                if shuffle:
                    indices = torch.randperm(n_train_actual)
                    epoch_inputs = train_inputs[indices]
                    epoch_z = train_z[indices]
                else:
                    epoch_inputs = train_inputs
                    epoch_z = train_z

                # 按 batch 训练
                epoch_value_loss = 0.0
                n_batches = 0

                for start in range(0, n_train_actual, batch_size):
                    end = min(start + batch_size, n_train_actual)
                    batch_inputs = epoch_inputs[start:end]
                    batch_z = epoch_z[start:end]

                    # 前向 + 反向
                    self.optimizer.zero_grad()
                    v_pred = self.model.network(batch_inputs)
                    loss = F.mse_loss(v_pred, batch_z)
                    loss.backward()
                    self.optimizer.step()

                    epoch_value_loss += loss.item()
                    n_batches += 1

                # 聚合 epoch 均值
                avg_value_loss = (
                    epoch_value_loss / n_batches if n_batches > 0 else 0.0
                )

                epoch_result: Dict[str, float] = {
                    "epoch": float(epoch + 1),
                    "value_loss": avg_value_loss,
                    "total_loss": avg_value_loss,
                }

                # 验证集评估
                if has_val and val_inputs is not None:
                    val_result = self._evaluate_val(val_inputs, val_z)
                    epoch_result["val_value_loss"] = val_result
                    epoch_result["val_total_loss"] = val_result

                history.append(epoch_result)

                # 输出日志
                if verbose:
                    log_parts = [
                        f"Epoch {epoch + 1}/{epochs}",
                        f"value_loss: {avg_value_loss:.4f}",
                    ]
                    if "val_value_loss" in epoch_result:
                        log_parts.append(
                            f"val_value_loss: {epoch_result['val_value_loss']:.4f}"
                        )
                    print(" | ".join(log_parts))
        except KeyboardInterrupt:
            print("\n[中断] 检测到 Ctrl+C，正在生成图表...")
        finally:
            self._plot_loss_curve(history)

        return history

    # ---------- 验证集评估 ----------

    def _evaluate_val(
        self, val_inputs: torch.Tensor, val_z: torch.Tensor
    ) -> float:
        """在验证集上计算 loss（仅前向，不更新参数）

        Args:
            val_inputs: 验证集输入
            val_z: 验证集目标

        Returns:
            验证集 value_loss
        """
        self.model.network.eval()
        with torch.no_grad():
            v_pred = self.model.network(val_inputs)
            loss = F.mse_loss(v_pred, val_z)
        self.model.network.train()
        return loss.item()

    # ---------- 模型保存 ----------

    def save_model(self, output_path: str) -> str:
        """保存模型权重

        Args:
            output_path: 保存路径

        Returns:
            实际保存路径
        """
        self.model.save(output_path)
        return output_path

    # ---------- 图表生成 ----------

    def _plot_loss_curve(self, history: List[Dict[str, float]]) -> Optional[str]:
        """生成 loss 曲线图并保存为 PNG

        Args:
            history: train_epochs 返回的训练历史

        Returns:
            PNG 文件路径；history 为空或 matplotlib 不可用时返回 None
        """
        if not history:
            return None
        if not _MATPLOTLIB_AVAILABLE:
            return None

        # 输出目录：使用 config/mcts.py 的 MCTS_LOSS_LOG_DIR 作为统一 loss 日志目录
        try:
            from config.mcts import MCTS_LOSS_LOG_DIR
            output_dir = MCTS_LOSS_LOG_DIR
        except ImportError:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "mcts_loss_logs"
            )
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.name
        output_path = os.path.join(
            output_dir, f"{model_name}_{timestamp}.png"
        )

        # 提取数据
        epochs_list = [entry["epoch"] for entry in history]
        total_losses = [entry["total_loss"] for entry in history]
        value_losses = [entry["value_loss"] for entry in history]
        has_val = all("val_value_loss" in entry for entry in history)

        try:
            fig, ax1 = plt.subplots(figsize=(12, 7))

            # 训练损失
            ax1.plot(epochs_list, total_losses, 'b-', label='total_loss',
                     linewidth=1.5)
            ax1.plot(epochs_list, value_losses, '--', color='#17becf',
                     label='value_loss', linewidth=1.0, alpha=0.8)

            # 验证损失
            if has_val:
                val_losses = [entry["val_value_loss"] for entry in history]
                ax1.plot(epochs_list, val_losses, 'g-',
                         label='val_value_loss', linewidth=1.5,
                         marker='o', markersize=3)

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("MSE Loss", color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.legend(loc='upper right')
            ax1.set_title(
                f"Value Loss Curve - {model_name} "
                f"({len(history)} epochs)"
            )
            ax1.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            if hasattr(self, 'optimizer') and self.optimizer is not None:
                print(f"Loss 曲线图已生成: {output_path}")
            return output_path
        except Exception:
            return None
