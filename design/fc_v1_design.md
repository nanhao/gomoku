# FC v1 模型设计文档

**版本**: v1.2  
**创建日期**: 2026-04-02  
**最后更新**: 2026-04-03

## 1. 概述

FC v1（Fully Connected v1）是五子棋 AI 系统的第一个深度学习模型，采用全连接神经网络架构。该模型通过监督学习从 AI 对战中学习落子策略。

### 1.1 设计目标

- 实现一个可训练的神经网络模型，能够预测五子棋落子位置的 reward 值
- 支持从 sample 数据中进行监督学习训练
- 支持训练周期模式（自我对弈 + 训练迭代）
- 模型参数可持久化保存和加载

### 1.2 模型特点

| 特性 | 说明 |
|------|------|
| 网络类型 | 全连接神经网络（Fully Connected Network） |
| 隐藏层数 | 10 层 |
| 每层神经元数 | 1000 |
| 激活函数 | ReLU |
| 输入维度 | 226（225 位棋盘 + 1 位落子方） |
| 输出维度 | 225（225 个位置的 reward） |
| 损失函数 | L1 Loss（mask 当前全为 1，等价于对所有 225 个位置求平均） |
| 优化器 | Adam (lr=0.001) |

---

## 2. 目录结构

```
ai/fc_v1/
├── __init__.py          # 模块初始化，导出 FCv1Model 和 FCv1Trainer
├── model.py             # 神经网络模型定义（FCv1Network + FCv1Model）
├── trainer.py           # 训练器实现（FCv1Trainer）
├── train.py             # 训练入口脚本（命令行接口）
├── sample_inspector.py  # 样本检视器（SampleInspector）
├── inspect_main.py      # 检视入口脚本（命令行接口）
└── fc_v1.pth            # 模型参数文件（训练后生成）
```

---

## 3. 类设计

### 3.1 FCv1Network（神经网络结构）

**文件**: `ai/fc_v1/model.py`

**职责**: 定义 PyTorch 神经网络结构

**网络架构**:
```
输入层 (226) 
    ↓
隐藏层1 (1000, ReLU)
    ↓
隐藏层2 (1000, ReLU)
    ↓
... (共10层隐藏层)
    ↓
隐藏层10 (1000, ReLU)
    ↓
输出层 (675)
```

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | 无 | - | 初始化网络层 |
| `forward` | `x: torch.Tensor` | `torch.Tensor` | 前向传播 |

**代码框架**:
```python
class FCv1Network(nn.Module):
    """FC v1 神经网络结构"""
    
    def __init__(self):
        super(FCv1Network, self).__init__()
        # 定义 10 个隐藏层和输出层
        self.fc1 = nn.Linear(226, 1000)
        self.relu1 = nn.ReLU()
        # ... fc2-fc10, relu2-relu10
        self.fc_output = nn.Linear(1000, 225)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 依次通过 10 层隐藏层
        x = self.relu1(self.fc1(x))
        # ...
        x = self.fc_output(x)  # 输出 225 维 reward
        return x
```

---

### 3.2 FCv1Model（模型包装类）

**文件**: `ai/fc_v1/model.py`

**职责**: 实现 BaseModel 接口，包装神经网络，提供 predict/save/load 功能

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `model_path: str = None, fresh: bool = False` | - | 初始化模型。若 fresh=True 则创建新模型；若 model_path 存在则加载；否则尝试加载默认路径 |
| `predict` | `board_vector: List[int]` | `List[int]` | 预测每个落子位置的 reward |
| `save` | `model_path: str = None` | `str` | 保存模型参数到文件 |
| `load` | `model_path: str` | - | 从文件加载模型参数 |
| `name` | 无 | `str` | 返回模型名称 "fc_v1" |

**代码框架**:
```python
class FCv1Model(BaseModel):
    """FC v1 模型"""
    
    DEFAULT_MODEL_PATH = "ai/fc_v1/fc_v1.pth"
    
    def __init__(self, model_path: str = None, fresh: bool = False):
        super().__init__()
        self.network = FCv1Network()
        
        # fresh=True 时，不加载任何模型
        if fresh:
            return  # 使用随机初始化的参数
        
        # 如果指定了 model_path 且存在，加载该模型
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        # 否则尝试加载默认路径
        elif os.path.exists(self.DEFAULT_MODEL_PATH):
            self.load(self.DEFAULT_MODEL_PATH)
    
    @property
    def name(self) -> str:
        return "fc_v1"
    
    def predict(self, board_vector: List[int]) -> List[int]:
        # 转换为张量，前向传播，解析输出为 675 维列表
        # reward 值限制在 [-1000000, 1000000] 范围内
        pass
    
    def save(self, model_path: str = None) -> str:
        # 使用 torch.save 保存 network.state_dict()
        pass
    
    def load(self, model_path: str):
        # 使用 torch.load 加载 state_dict
        pass
```

---

### 3.3 FCv1Trainer（训练器）

**文件**: `ai/fc_v1/trainer.py`

**职责**: 负责训练 FC v1 模型，支持单轮次训练和训练周期模式

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|---------|
| `__init__` | `model: FCv1Model = None, fresh: bool = False` | - | 初始化训练器。若 fresh=True 则创建新模型；若 model 为 None 则尝试加载已保存模型 |
| `train_single` | `sample_dir: str, epochs: int = 10, verbose: bool = False` | `float` | 单轮次训练（单个 sample 文件），返回平均损失。verbose=True 时每个 epoch 结束输出 loss |
| `train_batch` | `sample_dirs: list[str], epochs: int = 1000, verbose: bool = True` | `float` | 批量训练：将多个 sample 文件数据合并后训练，每个 epoch 只做一次参数更新，返回平均损失。verbose=True 时每个 epoch 结束输出 loss |
| `train_cycle` | `rounds: int = 10, games: int = 100, epochs: int = 1000, rate: float = 0.9, project_root: str = None` | - | 训练周期模式（自我对弈迭代），内部使用 train_batch 将当轮所有 sample 数据合并训练。rate 为奖励衰减率，传给 Evaluator |

**代码框架**:
```python
class FCv1Trainer:
    """FC v1 训练器"""
    
    def __init__(self, model: FCv1Model = None, fresh: bool = False):
        # fresh=True 时，创建新模型
        if fresh:
            self.model = FCv1Model(fresh=True)
        elif model is not None:
            self.model = model
        else:
            # 默认行为：尝试加载已保存模型
            self.model = FCv1Model()
        
        self.optimizer = optim.Adam(self.model.network.parameters(), lr=0.001)
        self.criterion = nn.L1Loss(reduction='none')
    
    def train_single(self, sample_dir: str, epochs: int = 10, verbose: bool = False) -> float:
        """单轮次训练（单个 sample 文件）"""
        # 1. 加载 sample.json
        # 2. 准备训练数据（input, output, mask）
        # 3. 将 mask 扩展到 675 维
        # 4. 训练循环：前向传播 -> 计算 masked loss -> 反向传播
        #    若 verbose=True，每个 epoch 结束时输出 loss
        # 5. 返回平均损失
        pass
    
    def train_batch(self, sample_dirs: list, epochs: int = 1000, verbose: bool = True) -> float:
        """批量训练：将多个 sample 文件合并，每 epoch 只做一次参数更新"""
        # 1. 遍历 sample_dirs，加载所有 sample.json 数据（合并为一个数据集）
        # 2. 准备合并后的训练数据（input, output, mask）
        # 3. 将 mask 扩展到 675 维
        # 4. 训练循环（epochs 次），每次：
        #    a. 前向传播（基于全部合并数据）
        #    b. 计算 masked loss
        #    c. 反向传播，更新一次参数
        #    d. 若 verbose=True，输出本 epoch 的 loss
        # 5. 返回平均损失
        pass
    
    def train_cycle(self, rounds: int = 10, games: int = 100, epochs: int = 1000, rate: float = 0.9, project_root: str = None) -> None:
        """训练周期模式"""
        # 1. 模型自我对弈 n 个轮次
        # 2. 每个轮次 m 场对局
        # 3. 每轮结束后调用 Evaluator(decay_rate=rate) 生成 sample，再调用 train_batch 合并训练
        pass
```

---

## 4. 训练入口脚本

**文件**: `ai/fc_v1/train.py`

**职责**: 提供命令行接口，支持单轮次训练和训练周期模式

### 4.1 命令格式

```bash
# 单轮次训练（指定 sample 目录）
python ai/fc_v1/train.py --sample-dir train_data/{轮次目录}/{局数目录} [--epochs 10]

# 训练周期（模型自对战）
python ai/fc_v1/train.py --cycle [--rounds 100] [--games 100]
```

### 4.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sample-dir` | str | - | sample.json 所在目录路径（单轮次训练） |
| `--epochs` | int | 1000 | 单轮次训练的训练轮数 |
| `--cycle` | flag | - | 开启训练周期模式 |
| `--rounds` | int | 10 | 训练周期轮次数 n |
| `--games` | int | 100 | 每轮对局数 m |
| `--rate` | float | 0.9 | 奖励衰减率 r，训练周期时传给 Evaluator |
| `--fresh` | flag | - | 从新模型开始训练（不加载已保存的模型） |

### 4.3 使用示例

```bash
# 单轮次训练（默认10轮）
python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1

# 单轮次训练（自定义轮数）
python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1 --epochs 20

# 训练周期（10轮，每轮100局）
python ai/fc_v1/train.py --cycle

# 训练周期（自定义参数）
python ai/fc_v1/train.py --cycle --rounds 50 --games 50

# 从新模型开始训练（不加载已保存模型）
python ai/fc_v1/train.py --cycle --fresh

# 单轮次训练，从新模型开始
python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1 --fresh
```

---

## 5. 训练流程设计

### 5.1 单轮次训练流程

```
1. 验证参数（--sample-dir 或 --cycle 必须指定其一）
2. 创建 FCv1Trainer 实例
3. 检查 sample.json 是否存在
4. 调用 trainer.train_single(sample_dir, epochs, verbose=True)
   a. 加载 sample.json 数据
   b. 准备训练数据
   c. 转换为 PyTorch 张量
   d. 训练循环（epochs 次）：
      i.   前向传播获取预测值（225 维）
      ii.  计算 masked L1 Loss（mask 当前全为 1，等价于对全部 225 位置求平均）
      iii. 反向传播更新参数
      iv.  若 verbose=True，输出当前 epoch 的平均 loss
   f. 返回平均损失
5. 保存模型到 ai/fc_v1/fc_v1.pth
6. 输出训练结果
```

### 5.2 训练周期流程

```
1. 创建 FCv1Trainer 实例
2. 循环 rounds 次：
   a. 创建 DataRecorder
   b. 执行 games 场 AI 内战（模型自我对弈）
   c. 调用 Evaluator(decay_rate=rate) 对每局生成 sample.json
   d. 收集本轮所有 sample 目录，调用 train_batch(sample_dirs, epochs, verbose=True)
      - train_batch 将所有 sample 数据合并为一个数据集
      - 训练 epochs 次，每个 epoch 只做一次参数更新
      - 每个 epoch 结束时输出本 epoch 的平均 loss
   e. 保存更新后的模型
3. 完成所有轮次
```

---

## 6. 损失函数设计

### 6.1 损失函数公式

```
loss(x,y) = mask(x,y) * abs(estimated_reward(x,y) - normalize(sample_reward(x,y)))
```

其中：
- `mask(x,y)`: sample.json 中的 mask 数组，0 或 1
- `estimated_reward`: 模型预测的 reward 值
- `sample_reward`: sample.json 中的 reward 值（经过反向传播优化后的值）
- `normalize(r)`: 将 r 从 [-1000000, 1000000] 归一化到 [-1, 1]，即 `r / 1000000.0`

### 6.2 Target Reward 归一化

**目的**：sample.json 中的 reward 值域为 [-1000000, 1000000]，而网络初始输出约为 ±几十。
量级悬殊导致优化器（lr=0.001）需要极多步数才能收敛。归一化后 target 变为 [-1, 1]，
与网络初始输出量级匹配，大幅加快收敛速度。

**归一化范围**：
- `REWARD_NORMALIZE_FACTOR = 1_000_000.0`
- 归一化公式：`normalized_target = target / REWARD_NORMALIZE_FACTOR`

**适用范围**：
- 归一化**仅在 trainer 内部**（`_prepare_tensors` 方法中）对 targets_tensor 执行
- sample.json 中的原始值**不变**（仍为 ±1,000,000）
- 模型的 `predict` 输出**不受影响**（仍为 [-1000000, 1000000] 的整数）

**实现位置**：`FCv1Trainer._prepare_tensors()`

```python
REWARD_NORMALIZE_FACTOR = 1_000_000.0

# 在 _prepare_tensors 中对 targets 归一化
targets_tensor = targets_tensor / REWARD_NORMALIZE_FACTOR  # [-1000000,1000000] -> [-1,1]
```

### 6.3 实现细节

```python
# 使用 PyTorch L1Loss
self.criterion = nn.L1Loss(reduction='none')

# 计算损失（targets 已在 _prepare_tensors 中归一化到 [-1, 1]）
loss = self.criterion(outputs, targets_tensor)  # 计算每个元素的 L1 损失 (N, 225)
loss = loss * masks_tensor                       # 应用 mask（当前 mask 全为 1，乘以 1 等价于不屏蔽）
loss = loss.sum() / masks_tensor.sum()           # 对 mask=1 的位置求平均（全部 225 位置）
```

### 6.4 Mask 语义说明

sample.json 中的 mask 是 225 维（对应 225 个落子位置），与模型输出维度相同。

**当前版本**：mask 全部初始化为 1，即对所有 225 个位置的 reward 都参与损失计算，等价于标准 L1 Loss。

**设计保留**：mask 机制仍然在代码中保留，未来版本可能重新启用，例如只对特定位置计算损失。

```python
# 当前：mask 全为 1（在 Evaluator._initialize_sample 中初始化）
mask = [1] * 225

# 使用 mask 时
loss = loss * masks_tensor   # mask=1 对所有位置生效，等价于不屏蔽
```

---

## 7. 数据流设计

### 7.1 预测数据流

```
输入: board_vector (226维)
    ↓
FCv1Model.predict()
    ↓
转换为 torch.Tensor (1, 226)
    ↓
FCv1Network.forward()
    ↓
输出: torch.Tensor (1, 225)
    ↓
转换为 List[int]，限制 reward 范围 [-1000000, 1000000]
    ↓
结果: [reward(x0,y0), reward(x0,y1), ..., reward(x14,y14)] (225维)
```

### 7.2 训练数据流

```
输入: sample.json
    ↓
解析为 inputs (N, 226), targets (N, 225), masks (N, 225)
    ↓
转换为 torch.Tensor
    ↓
targets_tensor /= 1_000_000.0  （归一化：[-1000000,1000000] -> [-1,1]）
    ↓
训练循环:
    前向传播 -> outputs (N, 225)
    计算 masked L1 Loss（mask 当前全为 1，等价于对全部 225 位置求平均）
    反向传播
    ↓
更新模型参数
```

---

## 8. 模型持久化

### 8.1 保存格式

- **格式**: PyTorch .pth 格式
- **路径**: `ai/fc_v1/fc_v1.pth`
- **内容**: `network.state_dict()`（包含所有可学习参数）

### 8.2 保存/加载代码

```python
# 保存
torch.save(self.network.state_dict(), model_path)

# 加载
self.network.load_state_dict(torch.load(model_path, weights_only=True))
self.network.eval()
```

### 8.3 默认加载行为

**重要**：FCv1Model 和 FCv1Trainer 默认会尝试加载已保存的模型文件 `fc_v1.pth`。

- 如果 `fc_v1.pth` 存在，自动加载该模型参数继续训练
- 如果 `fc_v1.pth` 不存在，创建新的模型
- 使用 `--fresh` 参数可强制从新模型开始训练

```python
# 默认行为：尝试加载已保存模型
model = FCv1Model()  # 如果 fc_v1.pth 存在，自动加载
trainer = FCv1Trainer()  # 同样会加载已保存模型

# 强制使用新模型
model = FCv1Model(fresh=True)  # 忽略已保存模型，创建新模型
trainer = FCv1Trainer(fresh=True)  # 同样创建新模型

# 手动指定模型路径
model = FCv1Model(model_path="path/to/model.pth")
trainer = FCv1Trainer(model=model)
```

---

## 9. 测试用例

### 9.1 模型测试

**文件**: `tests/ai/test_fc_v1_model.py`

```python
class TestFCv1Model(unittest.TestCase):
    """FC v1 模型测试"""
    
    def test_predict_output_format(self):
        """测试 predict 输出格式"""
        model = FCv1Model()
        result = model.predict([0] * 226)
        self.assertEqual(len(result), 675)
        # 验证格式 [x, y, reward, ...]
        self.assertEqual(result[0], 0)  # x=0
        self.assertEqual(result[1], 0)  # y=0
    
    def test_predict_reward_range(self):
        """测试 reward 值范围"""
        model = FCv1Model()
        result = model.predict([0] * 226)
        rewards = result[2::3]  # 提取所有 reward
        for r in rewards:
            self.assertGreaterEqual(r, -1000000)
            self.assertLessEqual(r, 1000000)
    
    def test_save_and_load(self):
        """测试模型保存和加载"""
        import tempfile
        model = FCv1Model()
        
        # 预测一次
        result1 = model.predict([0] * 226)
        
        # 保存
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        model.save(temp_path)
        
        # 加载新模型
        model2 = FCv1Model(temp_path)
        result2 = model2.predict([0] * 226)
        
        # 结果应该相同
        self.assertEqual(result1, result2)
        
        # 清理
        os.unlink(temp_path)
```

### 9.2 训练器测试

**文件**: `tests/ai/test_fc_v1_trainer.py`

```python
class TestFCv1Trainer(unittest.TestCase):
    """FC v1 训练器测试"""
    
    def test_train_single(self):
        """测试单轮次训练"""
        # 创建临时 sample 数据
        sample_data = [{
            "input": [0] * 226,
            "output": [0] * 675,
            "mask": [1] + [0] * 224
        }]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = os.path.join(tmpdir, "sample.json")
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f)
            
            trainer = FCv1Trainer()
            loss = trainer.train_single(tmpdir, epochs=2)
            
            # 验证返回了损失值
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0)
```

---

## 10. 依赖库

| 库 | 用途 |
|----|------|
| torch | 神经网络框架 |
| torch.nn | 网络层和损失函数 |
| torch.optim | 优化器（Adam） |
| json | sample.json 读写 |
| os | 文件路径操作 |
| argparse | 命令行参数解析 |
| typing | 类型注解 |

---

## 11. 与其他模块的关系

```
┌─────────────────────────────────────────────────────────┐
│                    ai/fc_v1/train.py                     │
│                   （训练入口脚本）                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  ai/fc_v1/trainer.py                     │
│                    （FCv1Trainer）                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   ai/fc_v1/model.py                      │
│          （FCv1Model + FCv1Network）                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  ai/base_model.py                        │
│                   （BaseModel 基类）                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   train_data/...                         │
│              （sample.json 训练数据）                     │
└─────────────────────────────────────────────────────────┘
```

---

## 12. SampleInspector（样本检视器）

### 12.1 职责

读取 sample.json，逐步调用模型 predict，将模型输出与 sample output 做对比并格式化打印。

### 12.2 类设计

**文件**: `ai/fc_v1/sample_inspector.py`

**类**: `SampleInspector`

> 注意：文件命名为 `sample_inspector.py` 而非 `inspect.py`，以避免与 Python 标准库 `inspect` 模块命名冲突。

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `model: FCv1Model` | - | 初始化，持有模型引用 |
| `inspect` | `sample_path: str` | `List[dict]` | 读取 sample.json，对每步调用 predict，返回每步的对比结果列表 |
| `format_step` | `step_result: dict` | `str` | 将单步对比结果格式化为可打印字符串 |

`inspect` 返回的每个 `step_result` 字典结构：
```python
{
    "step": int,               # 步骤编号（从0开始）
    "is_last": bool,           # 是否是最后一步
    "player": int,             # 当前落子方（1=黑, 2=白）
    "occupied_count": int,     # 已占位数量
    "empty_count": int,        # 空位数量
    "model_output": List[int], # 模型输出（225维，原始 predict 结果）
    "target": List[int],       # sample.json 中的原始 output（225维）
    "loss": float,             # 该步的 L1 loss（对所有 225 位置求平均绝对误差，target 已归一化）
    "win_pos": int or None,    # +1e6 位置的索引（无则为 None）
    "win_model_output": float or None,  # 模型在 +1e6 位置的输出值
}
```

`format_step` 格式化输出样例：
```
Step 55 (55子/170空, 落子方: 白, loss=0.1234):
  已占位(55): 模型输出均值=-123456, target均值=-1000000
  空位(170): 模型输出均值=12345, target均值=0
  +1e6位置: 无

Step 109 [最后一步] (109子/116空, 落子方: 黑, loss=0.0312):
  已占位(109): 模型输出均值=-234567, target均值=-1000000
  空位(116): 模型输出均值=45678, target均值=0 (其中1个为+1e6)
  +1e6位置: idx=103 (row=6,col=13), 模型输出=876543
```

### 12.3 入口脚本

**文件**: `ai/fc_v1/inspect_main.py`

**命令格式**：
```bash
python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json
python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json --model-path ai/fc_v1/fc_v1.pth
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sample` | str | 必填 | sample.json 文件路径 |
| `--model-path` | str | None | 模型文件路径，默认加载 fc_v1.pth |

### 12.4 主流程伪代码

```
1. 解析命令行参数（--sample, --model-path）
2. 加载模型：FCv1Model(model_path=args.model_path)
3. 创建 SampleInspector(model)
4. 调用 inspector.inspect(args.sample) 得到 step_results
5. 打印 summary：总步数、胜利步索引
6. 对每个 step_result，调用 inspector.format_step(step_result) 并打印
```

### 12.5 测试用例

**文件**: `tests/ai/test_fc_v1_inspect.py`

```python
class TestSampleInspector(unittest.TestCase):
    # [primary] inspect 返回列表长度等于 sample 步数
    def test_inspect_returns_correct_step_count(self): ...
    # [primary] 每个 step_result 包含必要字段
    def test_step_result_has_required_fields(self): ...
    # [primary] is_last 仅在最后一步为 True
    def test_is_last_only_on_last_step(self): ...
    # [primary] loss 为非负浮点数
    def test_loss_is_non_negative_float(self): ...
    # [primary] format_step 返回非空字符串
    def test_format_step_returns_nonempty_string(self): ...
    # [edge] win_pos 在有 +1e6 的步骤中不为 None
    def test_win_pos_detected_in_last_step(self): ...
    # [edge] win_pos 在无 +1e6 的步骤中为 None
    def test_win_pos_none_in_non_last_step(self): ...
```

---

## 13. 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-04-02 | 初始版本，完成 FC v1 设计文档 |
| v1.1 | 2026-04-03 | train_cycle 新增 rate 参数（奖励衰减率），Evaluator 支持 decay_rate 动态配置，train.py 新增 --rate 参数 |
| v1.2 | 2026-04-03 | mask 语义变更：从有效性标志（空位=0/有棋子=1）改为恒为 1（所有位置均参与 loss 计算）；输出维度从 675 维修正为 225 维；移除 mask 扩展到 675 维的逻辑 |
| v1.3 | 2026-04-03 | 训练时对 target reward 做归一化（[-1000000,1000000]→[-1,1]），仅在 _prepare_tensors 内部生效，sample.json 原始值不变 |
| v1.4 | 2026-04-03 | 新增 SampleInspector（sample_inspector.py）和 inspect_main.py，支持将 sample.json 喂给模型并逐步打印模型输出与 target 的对比 |

---

**文档结束**
