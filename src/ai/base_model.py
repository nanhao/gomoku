"""AI 模型基类模块

定义所有 AI 模型必须实现的接口。
所有模型输出层使用 LogSoftmax，predict 返回 Softmax 概率分布。

类层次结构：
- BaseModel                    —— 抽象基类，定义通用接口
  - NeuralNetworkModel          —— 神经网络模型抽象基类，提取 PyTorch 共用逻辑
    - FCv1Model                  —— 全连接模型
    - CNNV1Model                 —— 卷积模型 v1
    - CNNV2Model                 —— 卷积模型 v2（轻量化）
    - CNNV3Model                 —— 卷积模型 v3（极简）
  - RandomModel                 —— 随机模型（不使用神经网络）
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn


# ---- ModelPredict：predict() 返回值统一封装 ----

class ModelPredict:
    """模型预测结果统一封装

    统一所有模型的 predict() 返回值，消除当前返回类型不一致的问题：
    - 单头模型（random/fc/cnn v1-6）：仅预测概率分布
    - 双头模型（cnn_v7）：同时预测概率分布 + 局面评估

    通过 type 字段区分：
    - TYPE_POLICY_ONLY：仅包含策略概率分布
    - TYPE_POLICY_VALUE：包含策略概率分布 + 局面价值评估

    支持元组解包：
        result = model.predict(board_vector)
        policy, value = result  # 兼容旧代码

    Attributes:
        policy: 225 维落子概率列表，值域 (0, 1)，总和为 1
        value: 局面价值评估（可选），值域 [-1, 1]，仅 TYPE_POLICY_VALUE 时有效
        type: 结果类型标识，取值为 TYPE_POLICY_ONLY 或 TYPE_POLICY_VALUE
    """

    # 类型常量
    TYPE_POLICY_ONLY = "policy_only"
    TYPE_POLICY_VALUE = "policy_value"

    # 标准 policy 维度
    POLICY_DIM = 225

    def __init__(self, policy: List[float], value: Optional[float] = None):
        """创建预测结果

        Args:
            policy: 225 维落子概率分布，值域 (0, 1)，总和为 1
            value: 局面价值评估，None 表示无价值头输出

        Raises:
            ValueError: policy 长度不为 225
        """
        # 校验 policy 维度
        if len(policy) != self.POLICY_DIM:
            raise ValueError(
                f"policy 长度必须为 {self.POLICY_DIM}，实际为 {len(policy)}"
            )

        self._policy = policy
        self._value = value
        self._type = self.TYPE_POLICY_VALUE if value is not None else self.TYPE_POLICY_ONLY

    @property
    def policy(self) -> List[float]:
        """225 维落子概率分布"""
        return self._policy

    @property
    def value(self) -> Optional[float]:
        """局面价值评估，TYPE_POLICY_ONLY 时为 None"""
        return self._value

    @property
    def type(self) -> str:
        """结果类型：TYPE_POLICY_ONLY 或 TYPE_POLICY_VALUE"""
        return self._type

    # ---- 元组解包支持（兼容旧代码） ----

    def __iter__(self):
        """支持元组解包：policy, value = result"""
        yield self._policy
        yield self._value

    def __getitem__(self, index):
        """支持下标访问：result[0] = policy, result[1] = value"""
        if index == 0:
            return self._policy
        elif index == 1:
            return self._value
        raise IndexError(f"ModelPredict 索引 {index} 超出范围（仅支持 0 和 1）")

    def __len__(self):
        """固定长度为 2（policy + value）"""
        return 2

    def __repr__(self):
        """调试输出"""
        p_len = len(self._policy)
        if self._value is not None:
            return (f"ModelPredict(type={self._type!r}, policy_len={p_len}, "
                    f"value={self._value:.4f})")
        return f"ModelPredict(type={self._type!r}, policy_len={p_len})"


class BaseModel(ABC):
    """AI 模型基类
    
    所有 AI 模型必须继承此类并实现以下接口：
    - predict: 预测每个落子位置的概率
    - name: 模型名称
    - save: 保存模型参数到文件
    - load: 从文件加载模型参数
    - network: 神经网络模块（RandomModel 使用 None）
    
    注意：network 是实例属性而非抽象方法，因为子类在 __init__ 中赋值。
    BaseTrainer 依赖 self.model.network 存在，因此所有子类必须设置此属性。
    """
    
    @abstractmethod
    def predict(self, board_vector: List[int]) -> "ModelPredict":
        """
        预测每个落子位置的概率
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）
            
        Returns:
            ModelPredict 对象：
            - policy: 225 维概率分布，值域 (0, 1)，总和为 1
            - value: 局面价值评估（仅双头模型），单头模型为 None
            - type: 结果类型，TYPE_POLICY_ONLY 或 TYPE_POLICY_VALUE
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass
    
    @property
    def full_name(self) -> str:
        """含代数的完整模型名称

        默认实现返回 name 本身，适用于非多代模型。
        多代模型应在子类中覆盖此属性以包含代数信息。

        Returns:
            模型完整名称，如 "fc_v1" 或 "cnn_v3.2"
        """
        return self.name
    
    @abstractmethod
    def save(self, model_path: str = None) -> str:
        """保存模型参数到文件
        
        Args:
            model_path: 保存路径，为 None 则使用默认路径
            
        Returns:
            模型保存的文件路径
        """
        pass
    
    @abstractmethod
    def load(self, model_path: str = None):
        """从文件加载模型参数
        
        Args:
            model_path: 模型文件路径，为 None 则使用默认路径
        """
        pass
    
    def _ensure_network_attribute(self):
        """确保 network 属性已设置（供 BaseTrainer 初始化时调用）
        
        如果子类未设置 network 属性，抛出 AttributeError。
        """
        if not hasattr(self, 'network') or self.network is None:
            if not hasattr(self, '_network_impl'):
                raise AttributeError(
                    f"{self.__class__.__name__} 必须设置 self.network 属性。"
                    f"对于使用神经网络的模型，在 __init__ 中设置 self.network = <Network>；"
                    f"对于 RandomModel，设置 self.network = None。"
                )


class NeuralNetworkModel(BaseModel):
    """神经网络模型抽象基类
    
    所有使用 PyTorch 神经网络的模型继承此类。
    提供通用的 __init__、predict、save、load 实现，
    子类只需实现 _create_network 和 _prepare_input 两个抽象方法即可。
    
    子类必须定义的类属性：
    - DEFAULT_MODEL_PATH: str —— 默认模型文件路径
    - name: str               —— 模型名称（property）
    
    子类必须实现的抽象方法：
    - _create_network() -> nn.Module   —— 创建并返回神经网络模块
    - _prepare_input(board_vector, device) -> Tensor —— 将输入向量转换为模型输入张量
    """
    
    # 子类必须覆盖此属性，指定默认模型保存路径
    DEFAULT_MODEL_PATH: str = None
    
    def __init__(self, model_path: str = None, fresh: bool = False, gen: int | None = None):
        """初始化神经网络模型

        Args:
            model_path: 模型权重文件路径。指定时必须存在，否则抛出 FileNotFoundError
            fresh: 是否强制创建新模型（使用随机初始化参数，不加载任何权重）
            gen: 代数，指定时基于 DEFAULT_MODEL_PATH 构造 gen{gen}/ 子目录路径并加载

        Raises:
            ValueError: 未指定 model_path 且 fresh=False
            FileNotFoundError: model_path 或 gen 指向的文件不存在
        """
        super().__init__()

        # 保存代数信息
        self.gen = gen

        # 当 gen 不为 None 时，动态覆盖实例的 DEFAULT_MODEL_PATH 为 gen 子目录
        # （save() 依赖此路径，但不作为加载时的静默回退）
        if gen is not None:
            base_dir = os.path.dirname(self.DEFAULT_MODEL_PATH)
            model_filename = os.path.basename(self.DEFAULT_MODEL_PATH)
            self.DEFAULT_MODEL_PATH = os.path.join(base_dir, f"gen{gen}", model_filename)

        # 创建神经网络实例（由子类实现）
        self.network = self._create_network()

        # GPU 移动由 BaseTrainer 统一管理，模型创建时保持在 CPU

        # fresh=True 时，不加载任何模型，使用随机初始化的参数
        if fresh:
            return

        # 确定要加载的权重文件路径
        if model_path:
            # 显式指定了路径 → 文件必须存在，绝不静默回退
            load_path = model_path
            if not os.path.exists(load_path):
                raise FileNotFoundError(
                    f"模型权重文件不存在: {load_path}")
        elif gen is not None:
            # gen 已在上方改写 DEFAULT_MODEL_PATH 为 gen 子目录
            load_path = self.DEFAULT_MODEL_PATH
            if not os.path.exists(load_path):
                raise FileNotFoundError(
                    f"模型权重文件不存在: {load_path}")
        else:
            # 未指定任何路径 → 使用随机初始化权重
            return

        self.load(load_path)
    
    @abstractmethod
    def _create_network(self) -> nn.Module:
        """创建并返回神经网络模块
        
        子类实现此方法，返回对应的 PyTorch 网络实例。
        例如 FCv1Model 返回 FCv1Network()，CNNV1Model 返回 CNNV1Network()。
        
        Returns:
            nn.Module: 神经网络模块实例
        """
        pass
    
    @abstractmethod
    def _prepare_input(self, board_vector: List[int], device: torch.device) -> torch.Tensor:
        """将 226 维 board_vector 转换为模型输入张量
        
        不同的网络结构需要不同的输入格式：
        - FC 模型：直接展平为一维向量 (1, 226)
        - CNN 模型：转换为 4D 张量 (1, 3, 15, 15)
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘，最后 1 位是当前落子方
            device: 目标设备（CPU/GPU）
            
        Returns:
            torch.Tensor: 模型输入张量
        """
        pass
    
    def predict(self, board_vector: List[int]) -> "ModelPredict":
        """预测每个落子位置的概率
        
        通用前向传播流程：
        1. 获取网络所在设备
        2. 将输入转换为张量并移到对应设备
        3. 前向传播（eval 模式，禁用梯度）
        4. 将输出封装为 ModelPredict 返回
            
        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）
                
        Returns:
            ModelPredict 对象，type=TYPE_POLICY_ONLY，value=None
        """
        # 获取设备（与网络所在设备一致）
        device = next(self.network.parameters()).device
            
        # 将输入转换为张量并移到对应设备（由子类实现具体转换逻辑）
        input_tensor = self._prepare_input(board_vector, device)
            
        # 前向传播
        self.network.eval()
        with torch.no_grad():
            output_tensor = self.network(input_tensor)
            
        # 将输出转换为列表 (值域 (0, 1)，exp(log-probability) 还原为概率)
        output_values = output_tensor.squeeze(0).exp().tolist()
            
        # 构建 225 维结果，索引 i 对应位置 (i//15, i%15)
        policy = [float(prob) for prob in output_values]
            
        return ModelPredict(policy=policy)
    
    def save(self, model_path: str = None) -> str:
        """保存模型参数

        Args:
            model_path: 保存路径，默认使用子类的 DEFAULT_MODEL_PATH

        Returns:
            实际保存的路径
        """
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH

        # 自动创建保存目录（若不存在）
        save_dir = os.path.dirname(model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存模型状态字典
        torch.save(self.network.state_dict(), model_path)

        return model_path
    
    @property
    def full_name(self) -> str:
        """含代数的完整模型名称

        多代模型返回 "模型名.代数"（如 "cnn_v3.2"），
        非多代模型返回模型名称本身（如 "fc_v1"），与 name 相同。

        Returns:
            模型完整名称
        """
        if self.gen is not None:
            return f"{self.name}.{self.gen}"
        return self.name
    
    def load(self, model_path: str):
        """加载模型参数
        
        Args:
            model_path: 模型文件路径
        """
        self.network.load_state_dict(torch.load(model_path, weights_only=True))
        self.network.eval()

    def move_to_device(self, device: torch.device = None) -> torch.device:
        """将神经网络移动到指定设备（或自动选择最优设备）。

        当 device 为 None 时，自动检测：CUDA 可用时使用 GPU，否则使用 CPU。
        此方法供 create_samples.py / battle.py 等对弈脚本调用，
        使自我对弈也能利用 GPU 加速推理。

        训练场景由 BaseTrainer 自行管理设备，无需调用此方法。

        Args:
            device: 目标设备，None 时自动检测

        Returns:
            网络最终所在的 torch.device
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.network.to(device)
        return device


def board_vector_to_cnn_tensor(board_vector: List[int], device: torch.device) -> torch.Tensor:
    """将 226 维向量转换为 CNN 输入张量 (1, 3, 15, 15)
    
    这是 CNNV1Model 和 CNNV2Model 共用的输入转换逻辑，
    提取为独立函数以避免代码重复。
    
    转换规则（绝对视角编码）：
    - 通道1（黑棋）：棋盘上值为 1 的位置设为 1.0，其余为 0.0
    - 通道2（白棋）：棋盘上值为 2 的位置设为 1.0，其余为 0.0
    - 通道3（当前方）：当前落子方为黑(1)时全 1.0，为白(2)时全 0.0
    
    Args:
        board_vector: 226 维向量，前 225 位是棋盘，最后 1 位是当前落子方
        device: 目标设备（CPU 或 GPU）
            
    Returns:
        形状为 (1, 3, 15, 15) 的张量，可直接输入 CNN 网络
    """
    # 提取棋盘和当前落子方
    board = board_vector[:225]
    current_player = board_vector[225] if len(board_vector) > 225 else 1
    
    # 创建 3 个通道
    # 通道1: 黑棋位置
    black_channel = [1.0 if x == 1 else 0.0 for x in board]
    # 通道2: 白棋位置
    white_channel = [1.0 if x == 2 else 0.0 for x in board]
    # 通道3: 当前落子方（全1表示黑方，全0表示白方）
    player_channel = [1.0 if current_player == 1 else 0.0] * 225
    
    # 重塑为 15x15
    black_matrix = [black_channel[i*15:(i+1)*15] for i in range(15)]
    white_matrix = [white_channel[i*15:(i+1)*15] for i in range(15)]
    player_matrix = [player_channel[i*15:(i+1)*15] for i in range(15)]
    
    # 转换为张量并堆叠
    black_tensor = torch.tensor(black_matrix, dtype=torch.float32, device=device)
    white_tensor = torch.tensor(white_matrix, dtype=torch.float32, device=device)
    player_tensor = torch.tensor(player_matrix, dtype=torch.float32, device=device)
    
    # 堆叠为 (3, 15, 15) 并添加 batch 维度 -> (1, 3, 15, 15)
    input_tensor = torch.stack([black_tensor, white_tensor, player_tensor], dim=0)
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor


def board_vector_to_cnn_tensor_relative(board_vector: List[int], device: torch.device) -> torch.Tensor:
    """将 226 维向量转换为相对视角 CNN 输入张量 (1, 3, 15, 15)

    相对视角编码（AlphaGo Zero 风格，供 cnn_v7 MCTS 体系使用）：
    - 通道0（我方棋子）：当前落子方的棋子位置设为 1.0，其余为 0.0
    - 通道1（对方棋子）：对方棋子位置设为 1.0，其余为 0.0
    - 通道2（我方颜色）：当前落子方为黑(1)时全 1.0，为白(2)时全 0.0

    与绝对视角编码的区别：通道0/1根据当前落子方动态切换语义，
    从"黑白"变为"我方/对方"，消除绝对颜色信息，提升泛化能力。

    Args:
        board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                     最后 1 位是当前落子方（1=黑, 2=白）
        device: 目标设备（CPU 或 GPU）

    Returns:
        形状为 (1, 3, 15, 15) 的张量，可直接输入 CNN 网络
    """
    board = board_vector[:225]
    current_player = board_vector[225] if len(board_vector) > 225 else 1

    # 通道0: 我方棋子（当前玩家=current_player 的位置）
    my_channel = [1.0 if x == current_player else 0.0 for x in board]
    # 通道1: 对方棋子
    opponent = 2 if current_player == 1 else 1
    opp_channel = [1.0 if x == opponent else 0.0 for x in board]
    # 通道2: 我方颜色指示器
    player_channel = [1.0 if current_player == 1 else 0.0] * 225

    # 重塑为 15x15
    my_matrix = [my_channel[i*15:(i+1)*15] for i in range(15)]
    opp_matrix = [opp_channel[i*15:(i+1)*15] for i in range(15)]
    player_matrix = [player_channel[i*15:(i+1)*15] for i in range(15)]

    # 转换为张量并堆叠
    my_tensor = torch.tensor(my_matrix, dtype=torch.float32, device=device)
    opp_tensor = torch.tensor(opp_matrix, dtype=torch.float32, device=device)
    player_tensor = torch.tensor(player_matrix, dtype=torch.float32, device=device)

    # 堆叠为 (3, 15, 15) 并添加 batch 维度 -> (1, 3, 15, 15)
    input_tensor = torch.stack([my_tensor, opp_tensor, player_tensor], dim=0)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor
