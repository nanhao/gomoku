# Mini-Batch 训练功能实现

## 需求内容

用户提出在训练时 GPU 跑满但 CPU 空闲，希望提高 CPU 利用率。经分析，训练阶段 GPU 满载是正常现象，但当前训练使用全批量梯度下降（Full Batch GD），效率不高。用户要求实现 mini-batch 训练以提高训练效率。

## 解决思路

在 `BaseTrainer` 中新增 `train_mini_batch` 方法：
- 将合并后的数据集分成小批次（batch_size=32）
- 每个 epoch 内按批次进行前向、反向传播，多次更新参数
- 支持每个 epoch 打乱数据顺序（shuffle=True）
- 保留原有 `train_batch` 方法不变

## 执行过程

### 1. 设计阶段
- 创建设计文档 `design/mini_batch_training_design.md`
- 定义方法签名、参数说明、伪代码主流程
- 设计主流程测试用例和边缘测试用例

### 2. 测试用例编写
- 主流程测试（8 个）：`tests/ai/primary/test_mini_batch_training.py`
  - 返回值类型测试
  - 多批次更新测试
  - shuffle 功能测试
  - 参数更新测试
  - verbose 输出测试
  - 默认参数测试

- 边缘测试（10 个）：`tests/ai/edge/test_mini_batch_training_edge.py`
  - 空数据异常测试
  - batch_size 边界测试
  - 单样本测试
  - 零 epoch 测试
  - 多目录合并测试

### 3. 编码实现
修改文件：`ai/base_trainer.py`
新增方法：`train_mini_batch`
关键实现：
```python
def train_mini_batch(self, sample_dirs, epochs=1000, batch_size=32, shuffle=True, verbose=True):
    # 合并数据
    # 每个 epoch 内按 batch 分割
    # 每个 batch 做一次前向+反向+更新
```

### 4. 测试验证
- 18 个测试全部通过
- 全量测试 208/209 通过（1 个失败与本次功能无关）

## 执行结果

- 新增方法 `train_mini_batch`，支持 mini-batch 训练
- batch_size 默认 32，shuffle 默认 True
- 每个 epoch 内多次更新参数，训练效率更高
- 与 `train_batch` 对比：

| 特性 | train_batch | train_mini_batch |
|------|-------------|------------------|
| 更新频率 | 每 epoch 1 次 | 每 epoch N/batch_size 次 |
| GPU 内存占用 | 高（全部数据） | 低（batch_size） |
| 随机性 | 无 | 有（shuffle） |

## 涉及文件

### 新增文件
- `design/mini_batch_training_design.md` - 设计文档
- `tests/ai/primary/test_mini_batch_training.py` - 主流程测试
- `tests/ai/edge/test_mini_batch_training_edge.py` - 边缘测试
- `task_reports/2026-04-05/mini_batch训练功能实现_1.md` - 本报告

### 修改文件
- `ai/base_trainer.py` - 新增 `train_mini_batch` 方法
