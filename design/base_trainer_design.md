# BaseTrainer 设计文档

## 概述

创建抽象基类 `BaseTrainer`，提取 FC v1 和 CNN v1 训练器的通用逻辑，实现代码复用。

## 类设计

### BaseTrainer (抽象基类)

位置：`ai/base_trainer.py`

```python
class BaseTrainer(ABC):
    """训练器抽象基类
    
    所有模型训练器必须继承此类，实现模型特定的创建逻辑。
    提供通用的训练流程：单轮次训练、批量训练、训练周期。
    """
    
    # 子类可覆盖的默认学习率
    DEFAULT_LR = 0.001
    
    def __init__(self, fresh: bool = False)
        # 创建模型（调用抽象方法 _create_model）
        # 初始化优化器（Adam，使用 DEFAULT_LR）
        # 初始化损失函数（L1Loss）
    
    @abstractmethod
    def _create_model(self, fresh: bool) -> BaseModel
        # 子类实现：创建并返回具体模型实例
    
    @property
    @abstractmethod
    def model_name(self) -> str
        # 子类实现：返回模型名称（如 "fc_v1", "cnn_v1"）
    
    def train_single(self, sample_dir: str, epochs: int = 10, verbose: bool = False) -> float
        # 单轮次训练（单个 sample 文件）
    
    def train_batch(self, sample_dirs: list, epochs: int = 1000, verbose: bool = True) -> float
        # 批量训练：多个 sample 合并训练
    
    def train_cycle(self, rounds: int, games: int, epochs: int, rate: float, project_root: str) -> None
        # 训练周期：自我对弈 -> 生成样本 -> 训练 -> 保存模型
    
    # 以下私有方法为通用实现
    _load_sample_data(sample_path) -> list
    _prepare_tensors(sample_data) -> (inputs, targets, masks)
    _masked_loss(outputs, targets, masks) -> loss_tensor
    _compute_loss(estimated, sample, mask) -> float  # Python 纯实现，供测试用
```

### FCv1Trainer (重构后)

位置：`ai/models/fc_v1/trainer.py`

```python
class FCv1Trainer(BaseTrainer):
    """FC v1 训练器"""
    
    def _create_model(self, fresh: bool) -> FCv1Model
        return FCv1Model(fresh=fresh)
    
    @property
    def model_name(self) -> str
        return "fc_v1"
```

### CNNTrainer (新建)

位置：`ai/models/cnn_v1/trainer.py`

```python
class CNNTrainer(BaseTrainer):
    """CNN v1 训练器"""
    
    def _create_model(self, fresh: bool) -> CNNModel
        return CNNModel(fresh=fresh)
    
    @property
    def model_name(self) -> str
        return "cnn_v1"
```

## 主流程伪代码

### train_single 流程

```
function train_single(sample_dir, epochs, verbose):
    1. 加载 sample.json 数据
    2. 准备张量 (inputs, targets, masks)
    3. 设置模型为训练模式
    4. for epoch in range(epochs):
        - 清零梯度
        - 前向传播
        - 计算 masked loss
        - 反向传播
        - 更新参数
        - 累计损失
        - 若 verbose 且满足条件，打印损失
    5. 返回平均损失
```

### train_cycle 流程

```
function train_cycle(rounds, games, epochs, rate, project_root):
    1. 导入必要的模块 (AIEngine, AIGameManager, DataRecorder, Evaluator)
    2. 创建 AI 引擎（使用当前模型）
    3. for round_num in 1..rounds:
        - 创建 DataRecorder(ai_name=f"{model_name}_cycle_{round_num}")
        - 执行 AIGameManager.start_ai_vs_ai(games 局)
        - 统计对局结果
        - 使用 Evaluator 生成所有 sample
        - 调用 train_batch 训练模型
        - 保存模型
    4. 输出完成信息
```

## train_main.py 修改

```python
def get_trainer(model_name: str, fresh: bool = False):
    if model_name == "fc_v1":
        from ai.models.fc_v1.trainer import FCv1Trainer
        return FCv1Trainer(fresh=fresh)
    elif model_name == "cnn_v1":
        from ai.models.cnn_v1.trainer import CNNTrainer
        return CNNTrainer(fresh=fresh)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
```

## 主流程测试用例

### BaseTrainer 测试（抽象方法测试）

```python
def test_base_trainer_cannot_instantiate_directly():
    # 直接实例化 BaseTrainer 应该报错
    
def test_base_trainer_requires_abstract_methods():
    # 子类必须实现 _create_model 和 model_name
```

### FCv1Trainer 主流程测试

```python
def test_train_single_with_valid_sample():
    # 使用有效 sample 进行单轮次训练
    # 验证返回的损失值是合理的 float

def test_train_batch_with_multiple_samples():
    # 使用多个 sample 进行批量训练
    # 验证返回的平均损失值

def test_model_name_returns_fc_v1():
    # 验证 model_name 返回 "fc_v1"
```

### CNNTrainer 主流程测试

```python
def test_train_single_with_valid_sample():
    # 使用有效 sample 进行单轮次训练

def test_train_batch_with_multiple_samples():
    # 使用多个 sample 进行批量训练

def test_model_name_returns_cnn_v1():
    # 验证 model_name 返回 "cnn_v1"
```

### train_main.py 测试

```python
def test_get_trainer_returns_fc_v1_trainer():
    # 验证 fc_v1 返回 FCv1Trainer

def test_get_trainer_returns_cnn_trainer():
    # 验证 cnn_v1 返回 CNNTrainer

def test_get_trainer_raises_for_invalid_model():
    # 验证无效模型名抛出 ValueError
```

## 边缘测试用例

1. **样本文件不存在**：验证抛出 FileNotFoundError
2. **空 sample_dirs 列表**：验证抛出 ValueError
3. **无效的 JSON 格式**：验证抛出 JSONDecodeError
4. **fresh=True 创建新模型**：验证不加载已有模型
5. **cycle 和 sample-dir 同时使用**：验证参数冲突检测
