"""训练器抽象基类模块

定义所有模型训练器必须实现的接口和通用训练逻辑。
支持 GPU 加速训练。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
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
    # 默认权重衰减（AdamW 解耦权重衰减），子类可覆盖
    DEFAULT_WEIGHT_DECAY = 0.01
    
    def __init__(self, model: Optional[BaseModel] = None, fresh: bool = False,
                 device: Optional[torch.device] = None, loss_logger=None,
                 lr_scheduler_type: str = None, lr_step_size: int = 500,
                 lr_gamma: float = 0.9, lr_patience: int = 100,
                 lr_factor: float = 0.5, weight_decay: float = None):
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
            weight_decay: AdamW 权重衰减系数，None 时使用 DEFAULT_WEIGHT_DECAY
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

        # 初始化优化器（AdamW，解耦权重衰减）
        if self.model.network is not None:
            wd = weight_decay if weight_decay is not None else self.DEFAULT_WEIGHT_DECAY
            self.optimizer = optim.AdamW(
                self.model.network.parameters(),
                lr=self.DEFAULT_LR,
                weight_decay=wd
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

    @property
    def model_full_name(self) -> str:
        """含代数的完整模型名称（委托给 model.full_name）

        Returns:
            模型完整名称，如 "fc_v1" 或 "cnn_v3.2"
        """
        return self.model.full_name
    
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
            loss_logger = LossLogger(model_name=self.model_full_name, **kwargs)
        else:
            loss_logger = LossLogger(model_name=self.model_full_name)
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

    def _evaluate_val_loss(
        self,
        val_inputs: torch.Tensor,
        val_targets: torch.Tensor,
        val_masks: torch.Tensor
    ) -> float:
        """在验证集上计算 loss（仅前向传播，不更新参数）

        评估时切换到 eval 模式，完成后恢复 train 模式。
        使用 torch.no_grad() 避免计算梯度，节省显存。

        Args:
            val_inputs: 验证集输入张量
            val_targets: 验证集目标张量
            val_masks: 验证集 mask 张量

        Returns:
            验证集 loss 值（标量 float）
        """
        self.model.network.eval()
        with torch.no_grad():
            outputs = self.model.network(val_inputs)
            val_loss = self._masked_loss(outputs, val_targets, val_masks)
        self.model.network.train()
        return val_loss.item()

    def _forward_backward_step(self, inputs, targets, masks) -> Tuple[torch.Tensor, float]:
        """执行一次前向+反向传播+参数更新，返回 (loss张量, grad_norm标量)

        供 train_single 和 train_batch 复用。train_mini_batch 保持独立实现。
        """
        self.optimizer.zero_grad()
        outputs = self.model.network(inputs)
        loss = self._masked_loss(outputs, targets, masks)
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item()
            for p in self.model.network.parameters()
            if p.grad is not None
        )
        self.optimizer.step()
        return loss, grad_norm

    def _run_training_loop(
        self,
        inputs_tensor, targets_tensor, masks_tensor,
        epochs, loss_logger, verbose,
        val_inputs=None, val_targets=None, val_masks=None
    ) -> float:
        """执行训练循环（供 train_single / train_batch 复用）

        包含完整的 for-epoch 循环：前向反向传播、loss 日志记录、verbose 打印、
        学习率衰减、KeyboardInterrupt 处理、_finalize_training 调用。
        提供验证集张量时，每 LOSS_LOG_INTERVAL 个 epoch 评估并记录 val_loss。

        Args:
            inputs_tensor: 训练输入张量
            targets_tensor: 训练目标张量
            masks_tensor: 训练 mask 张量
            epochs: 训练轮数
            loss_logger: LossLogger 实例
            verbose: 是否输出每个 epoch 的损失值
            val_inputs: 验证集输入张量（可选）
            val_targets: 验证集目标张量（可选）
            val_masks: 验证集 mask 张量（可选）

        Returns:
            训练后的平均损失值
        """
        total_loss = 0.0
        completed_epochs = 0
        interrupted = False
        # 是否有验证集（用于决定 x 轴键名）
        has_val = val_inputs is not None

        try:
            for epoch in range(epochs):
                loss, grad_norm = self._forward_backward_step(
                    inputs_tensor, targets_tensor, masks_tensor
                )

                total_loss += loss.item()
                completed_epochs += 1

                # 记录 loss 日志（每 LOSS_LOG_INTERVAL 个 epoch）
                if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
                    val_loss = None
                    if has_val:
                        val_loss = self._evaluate_val_loss(val_inputs, val_targets, val_masks)
                        loss_logger.log_val(epoch + 1, val_loss)
                    loss_logger.log(epoch + 1, loss.item(), grad_norm)
                    if verbose:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        if val_loss is not None:
                            print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}, val_loss: {val_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {current_lr:.6f}")
                        else:
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
        inputs_tensor, targets_tensor, masks_tensor = self._sample_loader.prepare_tensors(sample_data)

        # 将 epochs 自动传递给 LossLogger（用于图表标题）
        setup_kwargs = dict(kwargs)
        setup_kwargs.setdefault('epochs', epochs)
        loss_logger = self._setup_training(**setup_kwargs)

        # 执行训练循环
        return self._run_training_loop(
            inputs_tensor, targets_tensor, masks_tensor, epochs, loss_logger, verbose
        )

    def train_batch(self, sample_dirs: List[str], epochs: int = 1000,
                    verbose: bool = True, val_ratio: float = 0.0, **kwargs) -> float:
        """批量训练：将多个 sample 文件数据合并，每个 epoch 只做一次参数更新。

        正确实现"一个 epoch 只更新一次参数"的语义：
        先将所有 sample 文件的数据合并为一个大数据集，
        然后在合并数据集上训练 epochs 次，每次只做一次前向+反向传播。
        提供 val_ratio > 0 时，自动分割训练集和验证集，并在训练过程中监控 val_loss。

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            epochs: 训练轮数，默认 1000
            verbose: 是否输出每个 epoch 的损失值，默认 True
            val_ratio: 验证集比例，0 表示不分割，默认 0.0
            **kwargs: 传递给 LossLogger 的可选参数（session_dir、log_type、round_num 等）

        Returns:
            训练后的平均损失值

        Raises:
            ValueError: sample_dirs 为空列表或 val_ratio 不合法
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        if not sample_dirs:
            raise ValueError("sample_dirs 不能为空列表")

        # 加载数据：有验证集时分割，无验证集时全部用于训练
        if val_ratio > 0:
            train_data, val_data = self._sample_loader.load_and_split(sample_dirs, val_ratio)
            train_inputs, train_targets, train_masks = self._sample_loader.prepare_tensors(train_data)
            val_inputs, val_targets, val_masks = self._sample_loader.prepare_tensors(val_data)
        else:
            all_sample_data = self._sample_loader.load_multiple(sample_dirs)
            train_inputs, train_targets, train_masks = self._sample_loader.prepare_tensors(all_sample_data)
            val_inputs, val_targets, val_masks = None, None, None

        # 将 epochs 自动传递给 LossLogger（用于图表标题）
        setup_kwargs = dict(kwargs)
        setup_kwargs.setdefault('epochs', epochs)
        setup_kwargs['dense_log'] = False  # train_batch 始终用 epoch-based logging
        loss_logger = self._setup_training(**setup_kwargs)

        # 执行训练循环
        return self._run_training_loop(
            train_inputs, train_targets, train_masks,
            epochs, loss_logger, verbose,
            val_inputs, val_targets, val_masks
        )

    def train_mini_batch(
        self,
        sample_dirs: List[str],
        epochs: int = 1000,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: bool = True,
        val_ratio: float = 0.0,
        **kwargs
    ) -> float:
        """Mini-Batch 训练：将多个 sample 文件数据合并，分批次训练。

        每个 epoch 内将数据分成多个 mini-batch，每个 batch 进行一次前向+反向传播。
        相比 train_batch，mini-batch 训练更高效，收敛更快。
        提供 val_ratio > 0 时，自动分割训练集和验证集，每个 epoch 结束后评估 val_loss。

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            epochs: 训练轮数，默认 1000
            batch_size: 每批次样本数，默认 32
            shuffle: 每个 epoch 是否打乱数据顺序，默认 True
            verbose: 是否输出训练信息，默认 True
            val_ratio: 验证集比例，0 表示不分割，默认 0.0

        Returns:
            训练后的平均损失值

        Raises:
            ValueError: sample_dirs 为空列表或 val_ratio 不合法
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        if not sample_dirs:
            raise ValueError("sample_dirs 不能为空列表")

        # 加载数据：有验证集时分割，无验证集时全部用于训练
        if val_ratio > 0:
            train_data, val_data = self._sample_loader.load_and_split(sample_dirs, val_ratio)
            train_inputs, train_targets, train_masks = self._sample_loader.prepare_tensors(train_data)
            val_inputs, val_targets, val_masks = self._sample_loader.prepare_tensors(val_data)
        else:
            all_sample_data = self._sample_loader.load_multiple(sample_dirs)
            train_inputs, train_targets, train_masks = self._sample_loader.prepare_tensors(all_sample_data)
            val_inputs, val_targets, val_masks = None, None, None

        # 获取样本总数
        n_samples = train_inputs.shape[0]

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
                    batch_inputs = train_inputs[batch_indices]
                    batch_targets = train_targets[batch_indices]
                    batch_masks = train_masks[batch_indices]

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

                # epoch 结束后评估 val loss（mini-batch 模式每个 epoch 都评估）
                val_loss = None
                if val_inputs is not None:
                    val_loss = self._evaluate_val_loss(val_inputs, val_targets, val_masks)
                    # dense 模式用 step，稀疏模式用 epoch 编号
                    x_value = global_step if dense_log else (epoch + 1)
                    loss_logger.log_val(x_value, val_loss)

                # 稀疏日志：每 LOSS_LOG_INTERVAL 个 epoch 记录该 epoch 的平均 loss
                if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
                    epoch_avg_loss = epoch_loss_sum / epoch_batches_count if epoch_batches_count > 0 else 0.0
                    if loss_logger and not dense_log:
                        loss_logger.log(epoch + 1, epoch_avg_loss)
                    if verbose:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        if val_loss is not None:
                            print(f"Epoch {epoch + 1}/{epochs}, avg_loss: {epoch_avg_loss:.4f}, val_loss: {val_loss:.4f}, lr: {current_lr:.6f}")
                        else:
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

    def train_auto(self, sample_dirs: List[str], epochs: int = 1000,
                   batch_size: int = None, shuffle: bool = True,
                   verbose: bool = True, val_ratio: float = 0.0, **kwargs) -> float:
        """自动选择训练方法

        提供 batch_size 时启用 mini-batch 训练，否则使用全量更新。
        统一封装 train_batch 和 train_mini_batch 的分发逻辑。

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            epochs: 训练轮数，默认 1000
            batch_size: 每批次样本数，提供时启用 mini-batch 训练，默认 None（全量更新）
            shuffle: 每个 epoch 是否打乱数据顺序，仅 mini-batch 模式有效，默认 True
            verbose: 是否输出训练信息，默认 True
            val_ratio: 验证集比例，0 表示不分割，默认 0.0
            **kwargs: 传递给 train_batch/train_mini_batch 的额外参数

        Returns:
            训练后的平均损失值
        """
        if batch_size is not None and batch_size > 0:
            return self.train_mini_batch(
                sample_dirs, epochs=epochs, batch_size=batch_size,
                shuffle=shuffle, verbose=verbose, val_ratio=val_ratio, **kwargs
            )
        else:
            return self.train_batch(
                sample_dirs, epochs=epochs, verbose=verbose, val_ratio=val_ratio, **kwargs
            )

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
        project_root: str = None,
        val_ratio: float = 0.0
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
            val_ratio: 验证集比例，0 表示不分割，默认 0.0
        """
        SelfPlayCycle(self).run(
            rounds=rounds,
            games=games,
            epochs=epochs,
            rate=rate,
            batch_size=batch_size,
            shuffle=shuffle,
            project_root=project_root,
            val_ratio=val_ratio
        )
    
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
