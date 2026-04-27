"""训练器抽象基类模块

定义所有模型训练器必须实现的接口和通用训练逻辑。
支持 GPU 加速训练。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import json
import os
import torch
import torch.optim as optim
from torch import nn

from ai.base_model import BaseModel
from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR, LOSS_LOG_INTERVAL
from ai.loss_logger import LossLogger
from ai.sample_loader import SampleLoader
from ai.self_play_cycle import SelfPlayCycle


def get_device() -> torch.device:
    """获取可用设备（优先 GPU）
    
    Returns:
        torch.device: CUDA 设备（如果可用）或 CPU 设备
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BaseTrainer(ABC):
    """训练器抽象基类
    
    所有模型训练器必须继承此类，实现模型特定的创建逻辑。
    提供通用的训练流程：单轮次训练、批量训练、训练周期。
    
    自动检测并使用 GPU 加速训练。
    
    Attributes:
        model: 模型实例
        optimizer: PyTorch 优化器
        criterion: 损失函数
        device: 训练设备（CPU 或 GPU）
    """
    
    # 默认学习率，子类可覆盖
    DEFAULT_LR = 0.001
    
    def __init__(self, model: Optional[BaseModel] = None, fresh: bool = False,
                 device: Optional[torch.device] = None, loss_logger=None,
                 lr_scheduler_type: str = None, lr_step_size: int = 500,
                 lr_gamma: float = 0.9, lr_patience: int = 100,
                 lr_factor: float = 0.5):
        """初始化训练器

        Args:
            model: 模型实例，如果为 None 则根据 fresh 参数创建
            fresh: 是否强制创建新模型（不加载已保存的模型）
            device: 训练设备，如果为 None 则自动检测（优先 GPU）
            loss_logger: LossLogger 实例，None 表示由训练方法内部自动创建
            lr_scheduler_type: 学习率调度类型，可选 "step" / "plateau" / None
            lr_step_size: StepLR 衰减间隔（epoch），默认 500
            lr_gamma: StepLR 衰减系数，默认 0.9
            lr_patience: ReduceLROnPlateau 耐心值，默认 100
            lr_factor: ReduceLROnPlateau 衰减系数，默认 0.5
        """
        # 设置设备
        self.device = device if device is not None else get_device()

        # 创建或保存模型
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model(fresh)

        # 将模型移到 GPU（如果可用且模型有神经网络）
        if self.model.network is not None:
            self.model.network.to(self.device)

        # 初始化优化器（Adam）
        if self.model.network is not None:
            self.optimizer = optim.Adam(
                self.model.network.parameters(),
                lr=self.DEFAULT_LR
            )
        else:
            self.optimizer = None

        # 损失函数使用 L1Loss（MAE）
        self.criterion = nn.L1Loss(reduction='none')

        # LossLogger 支持依赖注入（测试时可传入 mock，None 则内部自动创建）
        self.loss_logger = loss_logger

        # 学习率调度器配置
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_scheduler = None
        self._setup_lr_scheduler()

        # 打印设备信息
        if self.device.type == "cuda":
            print(f"使用 GPU 加速: {torch.cuda.get_device_name(self.device)}")
        else:
            print("使用 CPU 训练")

        # 初始化样本加载器
        self._sample_loader = SampleLoader(
            device=self.device,
            input_converter=self._convert_inputs
        )

    def _setup_lr_scheduler(self):
        """根据 lr_scheduler_type 初始化学习率调度器"""
        if self.optimizer is None:
            return
        if self.lr_scheduler_type == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma
            )
        elif self.lr_scheduler_type == "plateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_factor,
                patience=self.lr_patience
            )
    
    @abstractmethod
    def _create_model(self, fresh: bool) -> BaseModel:
        """创建模型实例
        
        子类必须实现此方法，返回具体的模型实例。
        
        Args:
            fresh: 是否创建新模型（不加载已保存的模型）
            
        Returns:
            模型实例
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称
        
        子类必须实现此属性，返回模型名称字符串。
        用于数据记录器命名等场景。
        
        Returns:
            模型名称，如 "fc_v1", "cnn_v1"
        """
        pass
    
    def _setup_training(self, **kwargs):
        """设置训练环境，返回 LossLogger

        Args:
            **kwargs: 传递给 LossLogger 的可选参数（session_dir、log_type 等）

        Returns:
            LossLogger 实例
        """
        if self.loss_logger is not None:
            loss_logger = self.loss_logger
        elif kwargs:
            loss_logger = LossLogger(model_name=self.model_name, **kwargs)
        else:
            loss_logger = LossLogger(model_name=self.model_name)
        self.model.network.train() if self.model.network is not None else None
        return loss_logger

    def _finalize_training(self, loss_logger, total_loss, count, save_model=False):
        """生成 loss 曲线图并返回平均损失"""
        if loss_logger:
            loss_logger.flush()  # 确保最后一批密集日志落盘
            plot_path = loss_logger.plot()
            if plot_path:
                print(f"Loss 曲线图已生成: {plot_path}")
        if save_model:
            try:
                model_path = self.model.save()
                print(f"模型已保存: {model_path}")
            except Exception as e:
                print(f"模型保存失败: {e}")
        return total_loss / count if count > 0 else 0.0

    def train_single(self, sample_dir: str, epochs: int = 10, verbose: bool = False, **kwargs) -> float:
        """单轮次训练（单个 sample 文件）

        使用指定目录下的 sample.json 数据进行训练。

        Args:
            sample_dir: sample.json 所在目录路径
            epochs: 训练轮数，默认 10
            verbose: 是否输出每个 epoch 的损失值，默认 False

        Returns:
            训练后的平均损失值

        Raises:
            FileNotFoundError: sample.json 不存在
        """
        # 加载 sample 数据
        sample_path = os.path.join(sample_dir, "sample.json")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        sample_data = self._sample_loader.load(sample_path)

        # 准备训练数据
        inputs_tensor, targets_tensor, masks_expanded = self._sample_loader.prepare_tensors(sample_data)

        # 将 epochs 自动传递给 LossLogger（用于图表标题）
        setup_kwargs = dict(kwargs)
        setup_kwargs.setdefault('epochs', epochs)
        loss_logger = self._setup_training(**setup_kwargs)
        total_loss = 0.0
        completed_epochs = 0
        interrupted = False

        try:
            for epoch in range(epochs):
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model.network(inputs_tensor)

                # 计算损失（使用 mask）
                loss = self._masked_loss(outputs, targets_tensor, masks_expanded)

                # 反向传播
                loss.backward()

                # 计算梯度 L2 范数
                grad_norm = sum(
                    p.grad.norm().item()
                    for p in self.model.network.parameters()
                    if p.grad is not None
                )

                self.optimizer.step()

                total_loss += loss.item()
                completed_epochs += 1

                # 记录 loss 日志（每 LOSS_LOG_INTERVAL 个 epoch）
                if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
                    loss_logger.log(epoch + 1, loss.item(), grad_norm)
                    if verbose:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}, grad_norm: {grad_norm:.4f}, lr: {current_lr:.6f}")

                # 学习率衰减
                self._step_lr_scheduler(loss.item())
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 检测到 Ctrl+C，正在保存模型和图表...")

        avg_loss = self._finalize_training(loss_logger, total_loss, completed_epochs, save_model=interrupted)
        if interrupted:
            print("[中断] 已保存当前进度，可以安全退出")
            raise KeyboardInterrupt
        return avg_loss

    def train_batch(self, sample_dirs: List[str], epochs: int = 1000, verbose: bool = True, **kwargs) -> float:
        """批量训练：将多个 sample 文件数据合并，每个 epoch 只做一次参数更新。

        正确实现"一个 epoch 只更新一次参数"的语义：
        先将所有 sample 文件的数据合并为一个大数据集，
        然后在合并数据集上训练 epochs 次，每次只做一次前向+反向传播。

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            epochs: 训练轮数，默认 1000
            verbose: 是否输出每个 epoch 的损失值，默认 True
            **kwargs: 传递给 LossLogger 的可选参数（session_dir、log_type、round_num 等）

        Returns:
            训练后的平均损失值

        Raises:
            ValueError: sample_dirs 为空列表
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        if not sample_dirs:
            raise ValueError("sample_dirs 不能为空列表")

        # 收集所有 sample 数据（合并为一个数据集）
        all_sample_data = self._sample_loader.load_multiple(sample_dirs)

        # 准备合并后的训练数据（转换为张量）
        inputs_tensor, targets_tensor, masks_tensor = self._sample_loader.prepare_tensors(all_sample_data)

        # 将 epochs 自动传递给 LossLogger（用于图表标题）
        setup_kwargs = dict(kwargs)
        setup_kwargs.setdefault('epochs', epochs)
        loss_logger = self._setup_training(**setup_kwargs)
        total_loss = 0.0
        completed_epochs = 0
        interrupted = False

        try:
            for epoch in range(epochs):
                self.optimizer.zero_grad()

                # 前向传播（基于全部合并数据）
                outputs = self.model.network(inputs_tensor)

                # 计算损失（使用 mask）
                loss = self._masked_loss(outputs, targets_tensor, masks_tensor)

                # 反向传播，只做一次参数更新
                loss.backward()

                # 计算梯度 L2 范数
                grad_norm = sum(
                    p.grad.norm().item()
                    for p in self.model.network.parameters()
                    if p.grad is not None
                )

                self.optimizer.step()

                total_loss += loss.item()
                completed_epochs += 1

                # 记录 loss 日志（每 LOSS_LOG_INTERVAL 个 epoch）
                if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
                    loss_logger.log(epoch + 1, loss.item(), grad_norm)
                    if verbose:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}, grad_norm: {grad_norm:.4f}, lr: {current_lr:.6f}")

                # 学习率衰减
                self._step_lr_scheduler(loss.item())
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 检测到 Ctrl+C，正在保存模型和图表...")

        avg_loss = self._finalize_training(loss_logger, total_loss, completed_epochs, save_model=interrupted)
        if interrupted:
            print("[中断] 已保存当前进度，可以安全退出")
            raise KeyboardInterrupt
        return avg_loss

    def train_mini_batch(
        self,
        sample_dirs: List[str],
        epochs: int = 1000,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> float:
        """Mini-Batch 训练：将多个 sample 文件数据合并，分批次训练。

        每个 epoch 内将数据分成多个 mini-batch，每个 batch 进行一次前向+反向传播。
        相比 train_batch，mini-batch 训练更高效，收敛更快。

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            epochs: 训练轮数，默认 1000
            batch_size: 每批次样本数，默认 32
            shuffle: 每个 epoch 是否打乱数据顺序，默认 True
            verbose: 是否输出训练信息，默认 True

        Returns:
            训练后的平均损失值

        Raises:
            ValueError: sample_dirs 为空列表
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        if not sample_dirs:
            raise ValueError("sample_dirs 不能为空列表")

        # 收集所有 sample 数据（合并为一个数据集）
        all_sample_data = self._sample_loader.load_multiple(sample_dirs)

        # 准备合并后的训练数据（转换为张量）
        inputs_tensor, targets_tensor, masks_tensor = self._sample_loader.prepare_tensors(all_sample_data)

        # 获取样本总数
        n_samples = inputs_tensor.shape[0]

        # 将 epochs 自动传递给 LossLogger（用于图表标题）
        setup_kwargs = dict(kwargs)
        setup_kwargs.setdefault('epochs', epochs)
        loss_logger = self._setup_training(**setup_kwargs)
        total_loss = 0.0
        total_batches = 0
        epoch_loss_sum = 0.0
        epoch_batches = 0
        global_step = 0
        interrupted = False

        # 判断是否启用密集日志（mini-batch 模式下默认启用）
        dense_log = getattr(loss_logger, 'dense_log', False)

        try:
            for epoch in range(epochs):
                # 生成批次索引
                if shuffle:
                    indices = torch.randperm(n_samples, device=self.device)
                else:
                    indices = torch.arange(n_samples, device=self.device)

                # epoch 级别的 GPU 累加器和张量缓冲区
                epoch_loss_sum_tensor = torch.tensor(0.0, device=self.device)
                epoch_batches_count = 0

                # 密集日志：GPU 张量缓冲区（用于批量同步）
                batch_losses = []
                batch_grad_norms = []

                # 按批次训练
                for start_idx in range(0, n_samples, batch_size):
                    # 获取当前批次的索引
                    batch_indices = indices[start_idx:start_idx + batch_size]

                    # 获取批次数据
                    batch_inputs = inputs_tensor[batch_indices]
                    batch_targets = targets_tensor[batch_indices]
                    batch_masks = masks_tensor[batch_indices]

                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model.network(batch_inputs)

                    # 计算损失（使用 mask）
                    loss = self._masked_loss(outputs, batch_targets, batch_masks)

                    # 反向传播
                    loss.backward()

                    # 收集梯度范数和 loss（GPU 张量，不 .item()）
                    if dense_log and loss_logger is not None:
                        grad_norm_tensor = sum(
                            p.grad.norm()
                            for p in self.model.network.parameters()
                            if p.grad is not None
                        )
                        batch_losses.append(loss.detach().clone())
                        batch_grad_norms.append(grad_norm_tensor)

                    self.optimizer.step()

                    # 用张量累加（不 .item()）
                    epoch_loss_sum_tensor += loss
                    epoch_batches_count += 1
                    total_batches += 1

                # epoch 结束后，统一同步到 CPU
                epoch_loss_sum = epoch_loss_sum_tensor.item()
                total_loss += epoch_loss_sum

                # 批量写入密集日志（每个 epoch 只同步一次）
                if dense_log and loss_logger is not None and batch_losses:
                    for loss_t, grad_t in zip(batch_losses, batch_grad_norms):
                        loss_logger.log_dense(global_step, loss_t.item(), grad_t.item())
                        global_step += 1
                    loss_logger.flush()
                    batch_losses.clear()
                    batch_grad_norms.clear()

                # 稀疏日志：每 LOSS_LOG_INTERVAL 个 epoch 记录该 epoch 的平均 loss
                if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
                    epoch_avg_loss = epoch_loss_sum / epoch_batches_count if epoch_batches_count > 0 else 0.0
                    if loss_logger and not dense_log:
                        loss_logger.log(epoch + 1, epoch_avg_loss)
                    if verbose:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"Epoch {epoch + 1}/{epochs}, avg_loss: {epoch_avg_loss:.4f}, lr: {current_lr:.6f}")

                # 学习率衰减
                epoch_avg_loss = epoch_loss_sum / epoch_batches_count if epoch_batches_count > 0 else 0.0
                self._step_lr_scheduler(epoch_avg_loss)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 检测到 Ctrl+C，正在保存模型和图表...")
            # 保存当前 epoch 未 flush 的密集日志
            if dense_log and loss_logger is not None and batch_losses:
                for loss_t, grad_t in zip(batch_losses, batch_grad_norms):
                    loss_logger.log_dense(global_step, loss_t.item(), grad_t.item())
                    global_step += 1
                loss_logger.flush()

        avg_loss = self._finalize_training(loss_logger, total_loss, total_batches, save_model=interrupted)
        if interrupted:
            print("[中断] 已保存当前进度，可以安全退出")
            raise KeyboardInterrupt
        return avg_loss

    def _step_lr_scheduler(self, loss=None):
        """执行学习率衰减步进

        统一封装 StepLR 和 ReduceLROnPlateau 的 step 调用。
        StepLR 不需要参数，ReduceLROnPlateau 需要当前 loss。

        Args:
            loss: 当前 loss 值，用于 ReduceLROnPlateau
        """
        if self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    def train_cycle(
        self,
        rounds: int = 10,
        games: int = 100,
        epochs: int = 1000,
        rate: float = 0.9,
        batch_size: int = None,
        shuffle: bool = False,
        project_root: str = None
    ) -> None:
        """训练周期

        让模型自己跟自己对战 n 个轮次，每个轮次 m 场对局。
        每个轮次结束后，将本轮所有 sample 数据合并训练模型。
        提供 batch_size 时启用 mini-batch 训练，否则使用全量更新。

        Args:
            rounds: 轮次数 n，默认 10
            games: 每轮对局数 m，默认 100
            epochs: 每轮训练的轮数，默认 1000
            rate: 奖励衰减率 r，默认 0.9，传给 Evaluator 用于生成 sample
            batch_size: 每批次样本数，提供时启用 mini-batch 训练，默认 None（全量更新）
            shuffle: 每个 epoch 是否打乱数据顺序，仅 mini-batch 模式有效，默认 False
            project_root: 项目根目录，用于确定数据保存路径
        """
        SelfPlayCycle(self).run(
            rounds=rounds,
            games=games,
            epochs=epochs,
            rate=rate,
            batch_size=batch_size,
            shuffle=shuffle,
            project_root=project_root
        )
    
    def _compute_loss(self, estimated: List[float], sample: List[float], mask: List[int]) -> float:
        """计算损失（Python 纯实现，供单元测试使用）
            
        loss(x,y) = mask(x,y) * abs(estimated_reward(x,y) - sample_reward(x,y))
        
        当前版本 mask 全为 1，等价于对全部 225 个位置求平均 L1 损失。
        mask 机制保留，未来版本可能重新启用（如只对特定位置计算损失）。
            
        Args:
            estimated: 模型预测的 reward 列表（225维）
            sample: sample.json 中的 reward 列表（225维）
            mask: mask 列表（225维，当前版本全为 1）
                
        Returns:
            损失値
        """
        total_loss = 0.0
        mask_sum = 0
            
        for i in range(225):
            if mask[i] == 1:
                estimated_reward = estimated[i]
                sample_reward = sample[i]
                total_loss += abs(estimated_reward - sample_reward)
                mask_sum += 1
            
        # 返回平均损失
        if mask_sum > 0:
            return total_loss / mask_sum
        return 0.0
    
    def _load_sample_data(self, sample_path: str) -> list:
        """加载 sample.json 数据
        
        Args:
            sample_path: sample.json 文件路径
            
        Returns:
            sample 数据列表
            
        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(sample_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_tensors(self, sample_data: list):
        """将 sample 数据转换为训练所需的 PyTorch 张量
            
        Args:
            sample_data: sample 数据列表（可来自单个或多个 sample.json 合并）
                
        Returns:
            (inputs_tensor, targets_tensor, masks_tensor): 三个张量，已移到 GPU
        """
        inputs = []
        targets = []
        masks = []
            
        for item in sample_data:
            inputs.append(item["input"])
            targets.append(item["output"])
            masks.append(item["mask"])
            
        # 转换输入为模型所需的格式
        inputs_tensor = self._convert_inputs(inputs)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)  # (N, 225)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)      # (N, 225)
            
        # 将张量移到 GPU（如果可用）
        inputs_tensor = inputs_tensor.to(self.device)
        targets_tensor = targets_tensor.to(self.device)
        masks_tensor = masks_tensor.to(self.device)
            
        # 注意：targets 已经是 Softmax 概率分布（由 Evaluator 生成）
        # 值域 (0, 1)，所有 mask=1 位置的概率和为 1
        # 不需要额外的归一化操作
            
        # mask 和 targets 都是 225 维，直接返回
        return inputs_tensor, targets_tensor, masks_tensor
    
    def _convert_inputs(self, inputs: list) -> torch.Tensor:
        """将输入数据转换为模型所需的张量格式
        
        子类可以覆盖此方法以支持不同的输入格式。
        默认实现将输入转换为 (N, 226) 的浮点张量，适用于 FC 模型。
        CNN 模型应覆盖此方法以返回 (N, 3, 15, 15) 的 4D 张量。
        
        Args:
            inputs: 输入数据列表，每个元素是 226 维向量
            
        Returns:
            输入张量，形状取决于模型类型
        """
        # 默认实现：返回 (N, 226) 张量，适用于 FC 模型
        return torch.tensor(inputs, dtype=torch.float32)
    
    def _masked_loss(self, outputs, targets_tensor, masks_tensor):
        """计算 masked L1 loss
        
        Args:
            outputs: 模型输出张量 (N, 225)
            targets_tensor: 目标张量 (N, 225)
            masks_tensor: 225 维 mask 张量 (N, 225)
            
        Returns:
            标量损失 tensor
        """
        loss = self.criterion(outputs, targets_tensor)
        loss = loss * masks_tensor
        
        mask_sum = masks_tensor.sum()
        if mask_sum > 0:
            loss = loss.sum() / mask_sum
        else:
            loss = loss.sum()
        
        return loss
