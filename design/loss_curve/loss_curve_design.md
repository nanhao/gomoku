# Loss 日志与 Loss 曲线设计文档

## 需求

为训练过程增加可观测性：
1. **Loss 日志**：训练过程中将 loss 值持久化到日志文件，训练结束后可回溯
2. **Loss 曲线**：训练完成后自动生成 loss 随 epoch 变化的曲线图

## 解决思路

在 `BaseTrainer` 中引入 `LossLogger` 组件：
- 训练过程中按固定间隔记录 loss 到 JSON Lines 文件
- 训练完成后调用绘图方法生成 PNG 曲线图
- `train_single`、`train_batch`、`train_mini_batch`、`train_cycle` 四个方法均接入日志记录
- 通过依赖注入支持灵活的 logger 配置与测试替换

## 设计

### 新增类：LossLogger

```python
class LossLogger:
    """训练 Loss 日志记录器
    
    将训练过程中的 loss 值记录到 JSON Lines 文件，
    并支持生成 loss 曲线图。
    
    支持通过依赖注入传入或禁用，实现灵活的日志配置与测试替换。
    """
    
    def __init__(self, model_name: str, log_dir: str = None):
        """初始化日志记录器
        
        Args:
            model_name: 模型名称，如 "fc_v1"、"cnn_v1"
            log_dir: 日志目录，默认为 ai/ai_data/loss_logs
        """
    
    def log(self, epoch: int, loss: float) -> None:
        """记录一条 loss 日志
        
        Args:
            epoch: 当前 epoch 编号（从1开始）
            loss: 当前 loss 值
        """
    
    def plot(self, output_path: str = None) -> str:
        """根据已记录的日志生成 loss 曲线图
        
        Args:
            output_path: 图片保存路径，默认为日志目录下同名 .png
        
        Returns:
            图片文件路径
        """
    
    def get_log_path(self) -> str:
        """返回当前日志文件路径"""
    
    def clear(self) -> None:
        """清空当前日志文件（新训练开始时调用）"""
```

### 日志文件格式

采用 JSON Lines 格式（每行一个 JSON 对象），文件名包含模型名称、时间戳和 UUID 后缀：

文件路径：`ai/ai_data/loss_logs/{model_name}_{timestamp}_{short_uuid}.jsonl`

其中 `short_uuid` 取 UUID 前 8 位，完全消除同秒冲突的风险。

每行格式：
```json
{"epoch": 1, "loss": 0.1234}
{"epoch": 200, "loss": 0.0987}
```

### 日志记录间隔

使用可配置常量 `LOSS_LOG_INTERVAL` 控制记录间隔（默认 200，与现有 verbose 输出间隔一致）。
- 对于 `train_mini_batch`：记录每个采样点的 epoch avg_loss
- 对于 `train_cycle`：记录每轮训练结束后的 avg_loss，epoch 字段用轮次编号

#### 术语定义

- **epoch**：模型进行一次完整的前向-反向传播。在 `train_single` 中指对单个样本的重复训练次数；在 `train_batch` 中指对全量数据的一次更新；在 `train_mini_batch` 中指按批次遍历全部数据后的一次完整迭代。
- **round（轮次）**：在 `train_cycle` 中，完整的"自我对弈 → 评估 → 训练 → 保存"流程。

#### 多层日志命名规则

`train_cycle` 调用 `train_batch` 时会产生多层日志：
- `train_batch` 内部创建的 `LossLogger` 记录 epoch 粒度 loss，文件名为 `{model_name}_{timestamp1}_{uuid1}.jsonl`
- `train_cycle` 外层的 `LossLogger` 记录 round 粒度 loss，文件名为 `{model_name}_{timestamp2}_{uuid2}.jsonl`

两个日志通过不同的 `timestamp` 与 `uuid` 天然区分，互不干扰。

### 曲线图输出

- 格式：PNG 图片
- 路径：与日志文件同目录，同名 `.png` 后缀
- 内容：X 轴为 epoch/轮次，Y 轴为 loss 值，折线图
- 使用 matplotlib 生成
- **优雅降级**：matplotlib 不可用时输出警告并跳过图表生成，日志文件不受影响，训练不中断

### 伪代码主流程

#### train_single 改造

```
def train_single(sample_dir, epochs, verbose):
    sample_data = load_sample_data(sample_dir)
    inputs, targets, masks = prepare_tensors(sample_data)
    
    # 使用注入的 logger 或自动创建
    loss_logger = self.loss_logger or LossLogger(model_name=self.model_name)
    
    total_loss = 0.0
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = masked_loss(outputs, targets, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # 记录 loss 日志（每 LOSS_LOG_INTERVAL 个 epoch）
        if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
            loss_logger.log(epoch + 1, loss.item())
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")
    
    # 生成 loss 曲线图（matplotlib 缺失时优雅降级）
    if loss_logger:
        loss_logger.plot()
    
    return total_loss / epochs
```

#### train_batch 改造

```
def train_batch(sample_dirs, epochs, verbose):
    all_sample_data = merge_all_samples(sample_dirs)
    inputs, targets, masks = prepare_tensors(all_sample_data)
    
    # 使用注入的 logger 或自动创建
    loss_logger = self.loss_logger or LossLogger(model_name=self.model_name)
    
    total_loss = 0.0
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = masked_loss(outputs, targets, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # 记录 loss 日志（每 LOSS_LOG_INTERVAL 个 epoch）
        if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
            loss_logger.log(epoch + 1, loss.item())
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")
    
    # 生成 loss 曲线图（matplotlib 缺失时优雅降级）
    if loss_logger:
        loss_logger.plot()
    
    return total_loss / epochs
```

#### train_mini_batch 改造

```
def train_mini_batch(sample_dirs, epochs, batch_size, shuffle, verbose):
    all_sample_data = merge_all_samples(sample_dirs)
    inputs, targets, masks = prepare_tensors(all_sample_data)
    N = inputs.shape[0]
    
    # 使用注入的 logger 或自动创建
    loss_logger = self.loss_logger or LossLogger(model_name=self.model_name)
    
    total_loss = 0.0
    total_batches = 0
    epoch_loss_sum = 0.0
    epoch_batches = 0
    
    for epoch in range(epochs):
        indices = shuffle ? randperm(N) : arange(N)
        
        for batch_indices in split_into_batches(indices, batch_size):
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            batch_masks = masks[batch_indices]
            
            outputs = model(batch_inputs)
            loss = masked_loss(outputs, batch_targets, batch_masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_batches += 1
            epoch_loss_sum += loss.item()
            epoch_batches += 1
        
        # 记录 loss 日志（每 LOSS_LOG_INTERVAL 个 epoch 记录该 epoch 的平均 loss）
        if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
            epoch_avg_loss = epoch_loss_sum / epoch_batches
            if loss_logger:
                loss_logger.log(epoch + 1, epoch_avg_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, avg_loss: {epoch_avg_loss:.4f}")
        
        epoch_loss_sum = 0.0
        epoch_batches = 0
    
    # 生成 loss 曲线图（matplotlib 缺失时优雅降级）
    if loss_logger:
        loss_logger.plot()
    
    return total_loss / total_batches
```

#### train_cycle 改造

```
def train_cycle(rounds, games, epochs, rate, project_root):
    # 使用注入的 logger 或自动创建
    loss_logger = self.loss_logger or LossLogger(model_name=self.model_name)
    
    for round_num in range(1, rounds + 1):
        # 1. 自我对弈
        results = start_ai_vs_ai(engine, engine, games, recorder)
        
        # 2. 生成训练样本
        sample_dirs = evaluator.evaluate_records(round_dir)
        
        # 3. 训练
        if sample_dirs:
            # train_batch 内部会创建自己的 LossLogger（内层 epoch 级日志）
            avg_loss = train_batch(sample_dirs, epochs=epochs, verbose=True)
            
            # 记录每轮的 avg_loss（外层 round 级日志）
            if loss_logger:
                loss_logger.log(round_num, avg_loss)
        
        # 4. 保存模型
        model.save()
    
    # 生成 loss 曲线图（外层轮次粒度）
    if loss_logger:
        loss_logger.plot()
```

注意：`train_cycle` 中调用 `train_batch`，`train_batch` 内部会创建自己的 `LossLogger` 记录 epoch 粒度的 loss，而 `train_cycle` 外层的 `LossLogger` 记录轮次粒度的 loss。两者互不干扰，最终生成两张曲线图。

### 新增常量

在 `ai/constants.py` 中添加：

```python
# Loss 日志目录: {project_root}/ai/ai_data/loss_logs
LOSS_LOG_DIR = os.path.join(AI_DATA_DIR, "loss_logs")

# Loss 日志记录间隔（每 N 个 epoch 记录一次，默认 200）
LOSS_LOG_INTERVAL = 200
```

### BaseTrainer 依赖注入支持

`BaseTrainer.__init__` 添加 `loss_logger` 参数，支持注入外部 logger 或禁用日志：

```python
class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, model_name, loss_logger=None):
        """初始化 BaseTrainer
        
        Args:
            model: 神经网络模型
            optimizer: 优化器
            loss_fn: 损失函数
            model_name: 模型名称
            loss_logger: LossLogger 实例，None 表示由训练方法内部自动创建
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.loss_logger = loss_logger  # 支持注入 mock logger 或 None
```

这样：
- **测试时**：可注入 mock LossLogger，验证接口调用，无需 monkeypatch
- **禁用时**：可传入 None，由训练方法决定是否创建（默认行为不变）
- **正常训练**：可传入具体的 LossLogger 实例，实现自定义配置

## 主流程测试用例

### 测试 1：LossLogger 基本日志记录

```python
def test_loss_logger_log(tmp_path):
    """测试日志记录和文件写入"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    
    logger.log(1, 0.5)
    logger.log(200, 0.3)
    logger.log(400, 0.1)
    
    # 读取日志文件验证内容
    log_path = logger.get_log_path()
    assert os.path.exists(log_path)
    
    with open(log_path) as f:
        lines = f.readlines()
    
    assert len(lines) == 3
    entry1 = json.loads(lines[0])
    assert entry1["epoch"] == 1
    assert entry1["loss"] == 0.5
```

### 测试 2：LossLogger 曲线图生成

```python
def test_loss_logger_plot(tmp_path):
    """测试 loss 曲线图生成"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    
    logger.log(1, 0.5)
    logger.log(200, 0.3)
    logger.log(400, 0.1)
    
    img_path = logger.plot()
    
    assert os.path.exists(img_path)
    assert img_path.endswith(".png")
```

### 测试 3：LossLogger clear 功能

```python
def test_loss_logger_clear(tmp_path):
    """测试清空日志功能"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    
    logger.log(1, 0.5)
    logger.clear()
    
    # 清空后日志文件应为空
    log_path = logger.get_log_path()
    with open(log_path) as f:
        content = f.read()
    assert content == ""
```

### 测试 4：train_single 集成日志（依赖注入）

```python
def test_train_single_generates_loss_log(trainer, sample_dir, tmp_path):
    """测试 train_single 训练后通过依赖注入生成 loss 日志"""
    mock_logger = MagicMock(spec=LossLogger)
    trainer.loss_logger = mock_logger
    
    trainer.train_single(sample_dir, epochs=1000, verbose=False)
    
    # 验证 log 和 plot 被调用
    assert mock_logger.log.call_count >= 4  # 约 1000/200 = 5 次
    assert mock_logger.plot.called
```

### 测试 5：train_batch 集成日志（依赖注入）

```python
def test_train_batch_generates_loss_curve(trainer, sample_dirs, tmp_path):
    """测试 train_batch 训练后通过依赖注入生成 loss 曲线图"""
    mock_logger = MagicMock(spec=LossLogger)
    trainer.loss_logger = mock_logger
    
    trainer.train_batch(sample_dirs, epochs=1000, verbose=False)
    
    # 验证 log 和 plot 被调用
    assert mock_logger.log.called
    assert mock_logger.plot.called
```

### 测试 6：边界情况 - 空数据集绘图

```python
def test_loss_logger_plot_empty(tmp_path):
    """测试空数据集情况：不记录日志直接绘图应返回 None"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    img_path = logger.plot()
    assert img_path is None
```

### 测试 7：边界情况 - matplotlib 缺失优雅降级

```python
def test_loss_logger_matplotlib_missing(tmp_path, monkeypatch):
    """测试 matplotlib 不可用时优雅降级"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    logger.log(1, 0.5)
    logger.log(200, 0.3)
    
    # mock matplotlib 导入失败
    import sys
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    
    img_path = logger.plot()
    
    # 应返回 None，但日志文件保持完整
    assert img_path is None
    assert os.path.exists(logger.get_log_path())
```

### 测试 8：边界情况 - 单点数据绘图

```python
def test_loss_logger_single_point(tmp_path):
    """测试仅一个数据点的曲线图生成"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    logger.log(1, 0.5)
    
    img_path = logger.plot()
    
    # 应成功生成图表（matplotlib 可绘制单点）
    assert os.path.exists(img_path)
```

### 测试 9：边界情况 - NaN/Infinity 处理

```python
def test_loss_logger_nan_inf(tmp_path):
    """测试 NaN/Infinity loss 值：日志可保存，绘图不崩溃"""
    logger = LossLogger(model_name="test_model", log_dir=str(tmp_path))
    
    logger.log(1, 0.5)
    logger.log(200, float('nan'))
    logger.log(400, float('inf'))
    
    log_path = logger.get_log_path()
    assert os.path.exists(log_path)
    
    # plot 应处理异常值而不抛出异常
    img_path = logger.plot()
    assert img_path is None or os.path.exists(img_path)
```

### 测试 10：边界情况 - 日志目录不可写

```python
def test_loss_logger_directory_readonly(tmp_path):
    """测试日志目录不可写时抛出异常"""
    log_dir = tmp_path / "readonly"
    log_dir.mkdir()
    os.chmod(log_dir, 0o444)
    
    try:
        logger = LossLogger(model_name="test_model", log_dir=str(log_dir))
        with pytest.raises((PermissionError, IOError, OSError)):
            logger.log(1, 0.5)
    finally:
        os.chmod(log_dir, 0o755)
```

## 文件修改范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ai/loss_logger.py` | 新增 | LossLogger 类实现，含 UUID 文件名、matplotlib 优雅降级 |
| `ai/base_trainer.py` | 修改 | `__init__` 添加 `loss_logger` 参数；四个训练方法使用常量与依赖注入 |
| `ai/constants.py` | 修改 | 新增 `LOSS_LOG_DIR` 和 `LOSS_LOG_INTERVAL` 常量 |
| `tests/ai/test_loss_logger.py` | 新增 | LossLogger 单元测试（10 个用例，含边界情况） |
| `tests/ai/test_base_trainer.py` | 修改 | 新增 2 个集成测试用例，采用依赖注入替代 monkeypatch |
