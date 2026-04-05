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
from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR


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
    
    def __init__(self, model: Optional[BaseModel] = None, fresh: bool = False, device: Optional[torch.device] = None):
        """初始化训练器
        
        Args:
            model: 模型实例，如果为 None 则根据 fresh 参数创建
            fresh: 是否强制创建新模型（不加载已保存的模型）
            device: 训练设备，如果为 None 则自动检测（优先 GPU）
        """
        # 设置设备
        self.device = device if device is not None else get_device()
        
        # 创建或保存模型
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model(fresh)
        
        # 将模型移到 GPU（如果可用）
        self.model.network.to(self.device)
        
        # 初始化优化器（Adam）
        self.optimizer = optim.Adam(
            self.model.network.parameters(), 
            lr=self.DEFAULT_LR
        )
        
        # 损失函数使用 L1Loss（MAE）
        self.criterion = nn.L1Loss(reduction='none')
        
        # 打印设备信息
        if self.device.type == "cuda":
            print(f"使用 GPU 加速: {torch.cuda.get_device_name(self.device)}")
        else:
            print("使用 CPU 训练")
    
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
    
    def train_single(self, sample_dir: str, epochs: int = 10, verbose: bool = False) -> float:
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
        
        sample_data = self._load_sample_data(sample_path)
        
        # 准备训练数据
        inputs_tensor, targets_tensor, masks_expanded = self._prepare_tensors(sample_data)
        
        # 训练模式
        self.model.network.train()
        
        total_loss = 0.0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model.network(inputs_tensor)
            
            # 计算损失（使用 mask）
            loss = self._masked_loss(outputs, targets_tensor, masks_expanded)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 若 verbose=True，每 200 个 epoch 输出一次损失
            if verbose and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")
        
        # 返回平均损失
        return total_loss / epochs
    
    def train_batch(self, sample_dirs: List[str], epochs: int = 1000, verbose: bool = True) -> float:
        """批量训练：将多个 sample 文件数据合并，每个 epoch 只做一次参数更新。
        
        正确实现"一个 epoch 只更新一次参数"的语义：
        先将所有 sample 文件的数据合并为一个大数据集，
        然后在合并数据集上训练 epochs 次，每次只做一次前向+反向传播。
        
        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            epochs: 训练轮数，默认 1000
            verbose: 是否输出每个 epoch 的损失值，默认 True
            
        Returns:
            训练后的平均损失值
            
        Raises:
            ValueError: sample_dirs 为空列表
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        if not sample_dirs:
            raise ValueError("sample_dirs 不能为空列表")
        
        # 收集所有 sample 数据（合并为一个数据集）
        all_sample_data = []
        for sample_dir in sample_dirs:
            sample_path = os.path.join(sample_dir, "sample.json")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Sample file not found: {sample_path}")
            all_sample_data.extend(self._load_sample_data(sample_path))
        
        # 准备合并后的训练数据（转换为张量）
        inputs_tensor, targets_tensor, masks_tensor = self._prepare_tensors(all_sample_data)
        
        # 训练模式
        self.model.network.train()
        
        total_loss = 0.0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 前向传播（基于全部合并数据）
            outputs = self.model.network(inputs_tensor)
            
            # 计算损失（使用 mask）
            loss = self._masked_loss(outputs, targets_tensor, masks_tensor)
            
            # 反向传播，只做一次参数更新
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 若 verbose=True，每 200 个 epoch 输出一次损失
            if verbose and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")
        
        # 返回平均损失
        return total_loss / epochs
    
    def train_mini_batch(
        self,
        sample_dirs: List[str],
        epochs: int = 1000,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: bool = True
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
        all_sample_data = []
        for sample_dir in sample_dirs:
            sample_path = os.path.join(sample_dir, "sample.json")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Sample file not found: {sample_path}")
            all_sample_data.extend(self._load_sample_data(sample_path))
        
        
        # 准备合并后的训练数据（转换为张量）
        inputs_tensor, targets_tensor, masks_tensor = self._prepare_tensors(all_sample_data)
        
        # 获取样本总数
        n_samples = inputs_tensor.shape[0]
        
        # 训练模式
        self.model.network.train()
        
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            # 生成批次索引
            if shuffle:
                indices = torch.randperm(n_samples, device=self.device)
            else:
                indices = torch.arange(n_samples, device=self.device)
            
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
                self.optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
            
            # 若 verbose=True，每 200 个 epoch 输出一次损失
            if verbose and (epoch + 1) % 200 == 0:
                avg_loss = total_loss / total_batches if total_batches > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs}, avg_loss: {avg_loss:.4f}")
        
        # 返回平均损失
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def train_cycle(
        self,
        rounds: int = 10,
        games: int = 100,
        epochs: int = 1000,
        rate: float = 0.9,
        project_root: str = None
    ) -> None:
        """训练周期
        
        让模型自己跟自己对战 n 个轮次，每个轮次 m 场对局。
        每个轮次结束后，将本轮所有 sample 数据合并，调用 train_batch 训练模型。
        train_batch 保证每个 epoch 只做一次参数更新。
        
        Args:
            rounds: 轮次数 n，默认 10
            games: 每轮对局数 m，默认 100
            epochs: 每轮训练的轮数，默认 1000
            rate: 奖励衰减率 r，默认 0.9，传给 Evaluator 用于生成 sample
            project_root: 项目根目录，用于确定数据保存路径
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        from ai.engine import AIEngine
        from ai.game_manager import AIGameManager
        from ai.data_recorder import DataRecorder
        from ai.evaluator import Evaluator
        
        # 确定项目根目录
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
        
        # 创建数据目录（位于 ai/ 子目录下）
        ai_data_dir = AI_DATA_DIR
        train_data_dir = TRAIN_DATA_DIR
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)
        
        # 创建 AI 引擎（使用当前模型）
        engine = AIEngine(self.model)
        
        print(f"开始训练周期: {rounds} 轮，每轮 {games} 局，奖励衰减率 rate={rate}")
        print(f"数据保存目录: {ai_data_dir}")
        print(f"训练数据目录: {train_data_dir}")
        print("-" * 50)
        
        for round_num in range(1, rounds + 1):
            print(f"\n===== 第 {round_num}/{rounds} 轮 =====")
            
            # 1. 创建数据记录器
            recorder = DataRecorder(
                ai_name=f"{self.model_name}_cycle_{round_num}",
                ai_data_dir=ai_data_dir
            )
            
            # 2. 执行 AI 内战（模型自我对弈）
            print(f"执行 {games} 局自我对弈...")
            results = AIGameManager.start_ai_vs_ai(
                engine1=engine,
                engine2=engine,
                num_games=games,
                recorder=recorder,
                project_root=project_root
            )
            
            # 统计结果
            black_wins = sum(1 for r in results if r["winner"] == 1)
            white_wins = sum(1 for r in results if r["winner"] == 2)
            draws = sum(1 for r in results if r["winner"] == 3)
            print(f"对局结果: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")
            
            # 3. 调用 Evaluator 对每局生成 sample.json，收集所有 sample 目录
            print("生成训练样本...")
            evaluator = Evaluator(
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
                decay_rate=rate
            )
            
            round_dir = recorder.get_round_dir()
            sample_dirs = []
            for game_dir in os.listdir(round_dir):
                game_path = os.path.join(round_dir, game_dir)
                record_file = os.path.join(game_path, "record.json")
                if os.path.isdir(game_path) and os.path.exists(record_file):
                    try:
                        sample_file = evaluator.evaluate_record(record_file)
                        # 收集 sample 所在目录
                        sample_dirs.append(os.path.dirname(sample_file))
                    except Exception as e:
                        print(f"  评价失败 {record_file}: {e}")
            
            print(f"生成 {len(sample_dirs)} 个样本文件")
            
            # 4. 调用 train_batch 将本轮所有 sample 合并训练
            #    每个 epoch 只做一次参数更新，保证正确的训练语义
            if sample_dirs:
                print(f"训练模型 ({epochs} epochs，共 {len(sample_dirs)} 个样本合并训练)...")
                avg_loss = self.train_batch(sample_dirs, epochs=epochs, verbose=True)
                print(f"本轮平均损失: {avg_loss:.4f}")
            
            # 5. 保存更新后的模型
            model_path = self.model.save()
            print(f"模型已保存: {model_path}")
            print("-" * 50)
        
        print(f"\n训练周期完成！共 {rounds} 轮，每轮 {games} 局")
        print(f"最终模型保存于: {self.model.save()}")
    
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
