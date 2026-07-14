"""双头模型训练基类模块

实现 AlphaGo Zero 风格的训练流程：
- 损失函数：l = (z - v)² - πᵀ·p_log
- 优化器：AdamW（weight_decay=1e-4）
- 按设计文档 §9 训练规约实现：_train_step → train_epochs → model.save
- 训练结束后自动生成双 Y 轴损失曲线图（设计文档-25）
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F

# 尝试导入 matplotlib，用于生成损失曲线图
# 若不可用则优雅降级，不阻断训练流程
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def get_device() -> torch.device:
    """获取可用设备（优先 GPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DualHeadTrainer(ABC):
    """双头模型训练基类（MCTS 新体系专用）

    实现 AlphaGo Zero 训练流程：
    - 损失 = MSE(value) + CrossEntropy(policy)
    - AdamW 优化器（weight_decay=1e-4）

    子类（CNNV7Trainer）需实现：
    - _create_model(fresh) → BaseModel
    - _convert_inputs(states) → Tensor (batch, 3, 15, 15)
    - model_name → str

    不继承 BaseTrainer，与旧体系完全隔离。
    """

    # 默认超参数
    DEFAULT_LR: float = 0.001
    DEFAULT_WEIGHT_DECAY: float = 1e-4

    def __init__(
        self,
        model=None,
        fresh: bool = False,
        device: Optional[torch.device] = None,
        lr: float = None,
        weight_decay: float = None,
        gen: Optional[int] = None,
        value_loss_weight: float = 1.0,
        policy_loss_weight: float = 0.0,
    ):
        """初始化训练器

        Args:
            model: 模型实例，None 时根据 fresh 创建
            fresh: 是否创建新模型
            device: 训练设备，None 时自动检测
            lr: 学习率，None 时使用 DEFAULT_LR
            weight_decay: AdamW 权重衰减，None 时使用 DEFAULT_WEIGHT_DECAY
            gen: 代数
            value_loss_weight: 价值损失权重，默认 1.0。设为 0 可禁用来做消融实验
            policy_loss_weight: 策略损失权重，默认 1.0。设为 0 可禁用来做消融实验
        """
        self.gen = gen
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.device = device if device is not None else get_device()

        # 创建或保存模型
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model(fresh)

        # 移动到 GPU
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

    # ---------- 抽象方法 ----------

    @abstractmethod
    def _create_model(self, fresh: bool):
        """创建模型实例"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass

    @abstractmethod
    def _convert_inputs(self, inputs: List[List[int]]) -> torch.Tensor:
        """将 226 维 state 列表转换为模型输入张量

        Args:
            inputs: state 列表，每个元素 226 维

        Returns:
            输入张量，形状取决于模型类型
        """
        pass

    # ---------- 训练入口 ----------

    def _train_step(
        self,
        inputs: torch.Tensor,
        pi_targets: torch.Tensor,
        z_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """单 batch 训练步（设计文档 §9.2 T1-T12）

        这是训练的核心入口方法，所有训练张量操作均通过此方法完成。

        Args:
            inputs: 输入张量 (B, 3, 15, 15)
            pi_targets: π 目标分布 (B, 225)
            z_targets: z 目标价值 (B, 1)，值域 {-1, 0, 1}

        Returns:
            {"value_loss": float, "policy_loss": float, "total_loss": float,
             "grad_norms": dict}
        """
        # T1: 清零梯度
        self.optimizer.zero_grad()

        # T2: 前向传播
        p_log, v = self.model.network(inputs)

        # T3: 校验 p_log 形状
        B = inputs.size(0)
        assert p_log.shape == (B, 225), f"p_log shape: {p_log.shape}, expected ({B}, 225)"

        # T4: 校验 p_log 值域（LogSoftmax 输出 ≤ 0）
        assert p_log.max().item() <= 0, (
            f"p_log 最大值 {p_log.max().item()} > 0！"
            f"LogSoftmax 输出应 ≤ 0，是否被二次取了 log？"
        )

        # T5: 校验 v 形状
        assert v.shape == (B, 1), f"v shape: {v.shape}, expected ({B}, 1)"

        # T6: 校验 v 值域（Tanh 输出 ∈ [-1, 1]）
        assert -1.0 <= v.min().item() <= v.max().item() <= 1.0, (
            f"v 值域错误: [{v.min().item():.4f}, {v.max().item():.4f}]，"
            f"Tanh 输出应在 [-1, 1] 内"
        )

        # T7: 价值损失（MSE）
        value_loss = F.mse_loss(v, z_targets)

        # T8: 策略损失（CrossEntropy via LogSoftmax）
        # p_log 已是 LogSoftmax 输出，绝不二次取 log
        policy_loss = -(pi_targets * p_log).sum(dim=1).mean()

        # T9: 总损失（value/policy loss 可加权以支持消融实验）
        total_loss = self.value_loss_weight * value_loss + self.policy_loss_weight * policy_loss

        # T10: 反向传播
        total_loss.backward()

        # T11: 收集梯度范数
        grad_norms = {}
        for name, param in self.model.network.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()

        # T12: 参数更新
        self.optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": total_loss.item(),
            "grad_norms": grad_norms,
        }

    # ---------- Epoch 循环 ----------

    def train_epochs(
        self,
        sample_dirs: List[str],
        epochs: int,
        batch_size: int = 2048,
        val_split: float = 0.0,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """执行完整训练（设计文档 §9.3 E1-E6）

        数据加载 → 分割验证集 → epoch 循环 → 每个 epoch shuffle + batch 训练。
        所有训练操作通过 _train_step 入口完成。

        Args:
            sample_dirs: sample.json 所在目录列表
            epochs: 训练轮数
            batch_size: 每批样本数，默认 2048
            val_split: 验证集比例，默认 0（无验证集）
            shuffle: 是否打乱数据，默认 True
            verbose: 是否输出训练日志

        Returns:
            epoch history 列表，每项含 value_loss, policy_loss, total_loss
        """
        # L1-L5: 加载数据
        all_states, all_pi, all_z = self._load_samples(sample_dirs)

        # L3: 转换为训练张量
        inputs = self._convert_inputs(all_states).to(self.device)
        pi_targets = torch.tensor(all_pi, dtype=torch.float32, device=self.device)
        z_targets = torch.tensor(all_z, dtype=torch.float32, device=self.device).view(-1, 1)

        # L6: shuffle（首次）
        N = inputs.size(0)

        # L7: 分割验证集
        has_val = val_split > 0
        if has_val:
            n_val = int(N * val_split)
            n_train = N - n_val
            perm = torch.randperm(N)
            train_indices = perm[:n_train]
            val_indices = perm[n_train:]
            train_inputs, val_inputs = inputs[train_indices], inputs[val_indices]
            train_pi, val_pi = pi_targets[train_indices], pi_targets[val_indices]
            train_z, val_z = z_targets[train_indices], z_targets[val_indices]
        else:
            train_inputs, train_pi, train_z = inputs, pi_targets, z_targets
            val_inputs, val_pi, val_z = None, None, None

        # E1-E6: Epoch 循环
        self.model.network.train()
        history = []

        try:
            for epoch in range(epochs):
                # E2: 每 epoch 重新 shuffle
                n_train = train_inputs.size(0)
                if shuffle:
                    indices = torch.randperm(n_train)
                    epoch_inputs = train_inputs[indices]
                    epoch_pi = train_pi[indices]
                    epoch_z = train_z[indices]
                else:
                    epoch_inputs = train_inputs
                    epoch_pi = train_pi
                    epoch_z = train_z

                # E3: 按 batch_size 分 batch，逐批调用 _train_step
                epoch_value_loss = 0.0
                epoch_policy_loss = 0.0
                epoch_total_loss = 0.0
                # 累积各 batch 的梯度范数总和（聚合为标量后取 epoch 均值）
                epoch_grad_sum = 0.0
                n_batches = 0

                for start in range(0, n_train, batch_size):
                    end = min(start + batch_size, n_train)
                    batch_inputs = epoch_inputs[start:end]
                    batch_pi = epoch_pi[start:end]
                    batch_z = epoch_z[start:end]

                    result = self._train_step(batch_inputs, batch_pi, batch_z)

                    epoch_value_loss += result["value_loss"]
                    epoch_policy_loss += result["policy_loss"]
                    epoch_total_loss += result["total_loss"]
                    # 聚合当前 batch 的 grad_norms dict 为标量（sum of all param norms）
                    if "grad_norms" in result:
                        epoch_grad_sum += sum(result["grad_norms"].values())
                    n_batches += 1

                # E4: 聚合 epoch 均值
                avg_value_loss = epoch_value_loss / n_batches if n_batches > 0 else 0.0
                avg_policy_loss = epoch_policy_loss / n_batches if n_batches > 0 else 0.0
                avg_total_loss = epoch_total_loss / n_batches if n_batches > 0 else 0.0
                avg_grad_norm = epoch_grad_sum / n_batches if n_batches > 0 else 0.0

                epoch_result = {
                    "epoch": epoch + 1,
                    "value_loss": avg_value_loss,
                    "policy_loss": avg_policy_loss,
                    "total_loss": avg_total_loss,
                    "grad_norm": avg_grad_norm,
                }

                # E5: 验证集评估（无梯度）
                if has_val and val_inputs is not None:
                    val_result = self._evaluate_val(val_inputs, val_pi, val_z)
                    epoch_result["val_value_loss"] = val_result["value_loss"]
                    epoch_result["val_policy_loss"] = val_result["policy_loss"]
                    epoch_result["val_total_loss"] = val_result["total_loss"]

                history.append(epoch_result)

                # E6: 输出日志
                if verbose:
                    log_parts = [
                        f"Epoch {epoch + 1}/{epochs}",
                        f"value_loss: {avg_value_loss:.4f}",
                        f"policy_loss: {avg_policy_loss:.4f}",
                        f"total_loss: {avg_total_loss:.4f}",
                    ]
                    if "val_total_loss" in epoch_result:
                        log_parts.append(f"val_total_loss: {epoch_result['val_total_loss']:.4f}")
                    print(" | ".join(log_parts))
        except KeyboardInterrupt:
            print("\n[中断] 检测到 Ctrl+C，正在生成图表...")
        finally:
            # 训练结束后（正常或中断）自动生成损失曲线图
            self._plot_loss_curve(history)

        return history

    # ---------- 模型保存 ----------

    def save_model(self, output_path: str) -> str:
        """保存模型权重（设计文档 §9.4 S1）

        Args:
            output_path: 保存路径

        Returns:
            实际保存路径
        """
        self.model.save(output_path)
        return output_path

    # ---------- 内部辅助 ----------

    @staticmethod
    def _load_samples(sample_dirs: List[str]) -> tuple:
        """L1-L5: 加载 sample.json 数据

        Args:
            sample_dirs: sample.json 所在目录列表

        Returns:
            (states: List[List[int]], pis: List[List[float]], zs: List[int])
        """
        all_states = []
        all_pi = []
        all_z = []

        for d in sample_dirs:
            path = os.path.join(d, "sample.json")
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            for s in samples:
                all_states.append(s["state"])
                all_pi.append(s["pi"])
                all_z.append(s["z"])

        return all_states, all_pi, all_z

    def _evaluate_val(self, val_inputs, val_pi, val_z) -> Dict[str, float]:
        """E5: 验证集评估（仅前向，不更新参数）"""
        self.model.network.eval()
        with torch.no_grad():
            p_log, v = self.model.network(val_inputs)
            value_loss = F.mse_loss(v, val_z).item()
            policy_loss = (-(val_pi * p_log).sum(dim=1).mean()).item()
            total_loss = self.value_loss_weight * value_loss + self.policy_loss_weight * policy_loss
        self.model.network.train()
        return {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "total_loss": total_loss,
        }

    # ---------- 图表生成 ----------

    def _plot_loss_curve(self, history: List[Dict[str, float]]) -> str:
        """生成双 Y 轴损失曲线图并保存为 PNG（设计文档-25 §3）

        左 Y 轴（Loss）：
        - total_loss (蓝实线)
        - value_loss (青虚线 #17becf)
        - policy_loss (橙虚线 #ff7f0e)
        - val_total_loss (绿实线，仅当 history 含此字段时)

        右 Y 轴（Gradient Norm）：
        - grad_norm (红实线)

        Args:
            history: train_epochs 返回的训练历史记录列表

        Returns:
            PNG 文件路径；history 为空或 matplotlib 不可用时返回 None
        """
        if not history:
            return None

        if not _MATPLOTLIB_AVAILABLE:
            return None

        # 读取输出目录配置
        from config.mcts import MCTS_LOSS_LOG_DIR
        output_dir = MCTS_LOSS_LOG_DIR
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名：{model_name}_{timestamp}.png
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_name
        output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.png")

        # 提取各指标
        epochs = [entry["epoch"] for entry in history]
        total_losses = [entry["total_loss"] for entry in history]
        value_losses = [entry["value_loss"] for entry in history]
        policy_losses = [entry["policy_loss"] for entry in history]
        grad_norms = [entry.get("grad_norm", 0.0) for entry in history]

        # 判断是否有验证集数据
        has_val = all("val_total_loss" in entry for entry in history)

        try:
            # 创建双 Y 轴图
            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax2 = ax1.twinx()

            # 左轴：损失曲线
            ax1.plot(epochs, total_losses, 'b-', label='total_loss',
                     linewidth=1.5)
            ax1.plot(epochs, value_losses, '--', color='#17becf',
                     label='value_loss', linewidth=1.0, alpha=0.8)
            ax1.plot(epochs, policy_losses, '--', color='#ff7f0e',
                     label='policy_loss', linewidth=1.0, alpha=0.8)

            if has_val:
                val_total_losses = [entry["val_total_loss"]
                                    for entry in history]
                ax1.plot(epochs, val_total_losses, 'g-',
                         label='val_total_loss', linewidth=1.5,
                         marker='o', markersize=3)

            # 右轴：梯度范数
            ax2.plot(epochs, grad_norms, 'r-', label='grad_norm',
                     linewidth=1.0, alpha=0.7)

            # 轴标签与颜色
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color='b')
            ax2.set_ylabel("Gradient Norm", color='r')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')

            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper right')

            # 标题与网格
            ax1.set_title(
                f"Loss Curve - {model_name} "
                f"(Epoch Level, {len(history)} epochs)"
            )
            ax1.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"Loss 曲线图已生成: {output_path}")
            return output_path
        except Exception:
            # 绘图失败不影响训练结果
            return None
