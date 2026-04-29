# 测试集耗时分析与精简报告

## 1. 现状概览

| 指标 | 值 |
|------|-----|
| 测试文件数 | 73 |
| AST用例数（含subTest） | 2152 |
| pytest收集用例数 | 1092 |
| 全量预估耗时 | 8~15 分钟 |

## 2. 用例数量分布 Top 15

| 排名 | 文件 | AST用例数 | 所属模块 |
|------|------|-----------|----------|
| 1 | test_shape_detector.py | 76 | ai |
| 2 | test_neural_network_model.py | 66 | ai |
| 3 | test_main.py (cli) | 66 | cli |
| 4 | test_reward_calculator.py | 58 | ai |
| 5 | test_registry.py | 58 | ai/unit |
| 6 | test_loss_logger.py | 52 | ai |
| 7 | test_evaluator_shape_detection.py (primary) | 52 | ai/primary |
| 8 | test_controller.py | 48 | game |
| 9 | test_cnn_model.py | 48 | ai |
| 10 | test_heatmap.py | 48 | gui |
| 11 | test_session.py (primary) | 46 | app/primary |
| 12 | test_ai_battle.py | 46 | gui |
| 13 | test_fc_v1_model.py | 44 | ai |
| 14 | test_performance_level2_numba.py | 44 | ai |
| 15 | test_display_updater.py | 42 | gui |

## 3. 实测速度分层

### 3.1 分层总览

| 速度档位 | 秒/用例 | 典型文件 | 瓶颈原因 |
|----------|---------|----------|----------|
| 极慢 | 3.0~12.0 | train_cycle | 实际AI对局+模型训练 |
| 慢 | 1.0~3.0 | performance_level2 | ProcessPool/JIT编译 |
| 中等 | 0.3~0.5 | GUI, loss_logger | matplotlib绘图, Tk初始化 |
| 快 | 0.14~0.25 | CNN模型, shape_detector | 纯Python/PyTorch计算 |

### 3.2 极慢测试明细（>1秒/用例）

| 用例 | 耗时(秒) | 文件 | 原因 |
|------|----------|------|------|
| test_train_cycle_basic | 12.29 | test_fc_v1_trainer_cycle.py | 实际对局2局+训练 |
| test_train_cycle_creates_model_file | 3.78 | test_fc_v1_trainer_cycle.py | 实际对局1局+训练 |
| test_train_batch_verbose_outputs_epoch_loss | 3.56 | test_fc_v1_trainer_batch.py | 400 epoch训练 |
| test_trainer_compute_loss | 3.08 | test_fc_v1_trainer.py | 模型初始化开销 |
| test_train_cycle_with_existing_model | 2.97 | test_fc_v1_trainer_cycle.py | 实际对局+训练 |

**5个用例合计 25.68秒，占 FC v1 Trainer 总耗时(33.61秒)的 76%**

### 3.3 各模块实测耗时

| 文件/文件组 | pytest用例 | 耗时(秒) | 秒/用例 |
|-------------|-----------|----------|---------|
| test_fc_v1_trainer (3文件) | 24+5子 | 33.61 | 1.40 |
| test_performance (3文件) | 54+9子 | 55.86 | 1.03 |
| test_train_main + test_battle | 34 | 13.94 | 0.41 |
| GUI (test_main + test_mode_selector) | 29 | 10.15 | 0.35 |
| test_loss_logger | 26 | 8.49 | 0.33 |
| test_self_play + test_game_mgr | 33 | 7.99 | 0.24 |
| test_neural_network_model | 33 | 6.82 | 0.21 |
| test_shape_detector | 38 | 5.58 | 0.15 |
| CNN模型 (3文件) | 60 | 8.29 | 0.14 |

## 4. 时间消耗热点分析

### 热点1：实际训练/对局类测试（占时最大）

train_cycle 类测试是最大瓶颈。3个用例占 19秒。

共同特征：真正运行了 AI 对局 + 模型训练的完整流程。

### 热点2：重复fixture开销

FC v1 Trainer 被拆成 8 个文件，每个都有独立的 setUp/tearDown：

```
8文件 × 各做一遍 mkdtemp + patch模型路径 + 创建sample.json
= 约 0.3s × 8 = 2.4s 纯fixture开销
```

同理：
- test_evaluator_shape_detection.py (primary) 和 test_evaluator_shape_detection_edge.py (edge)
  共享相同 Evaluator 初始化逻辑，但各自 setUp
- test_mini_batch_training.py 和 test_mini_batch_training_edge.py 同上
- test_loss_logger.py 和 test_loss_logger_val.py 同上

### 热点3：同功能多模型重复测试

CNN v1/v2/v3 三个模型的测试结构几乎相同：

```
test_cnn_model.py    → 48用例 (CNNv1Network + CNNV1Model)
test_cnn_v2_model.py → 32用例 (CNNv2Network + CNNV2Model)
test_cnn_v3_model.py → 40用例 (CNNv3Network + CNNV3Model)
```

三个文件都测试：forward_single, forward_batch, softmax_output_range,
softmax_row_sum, structure, name, predict_shape, save_and_load ...

如果三个模型都继承自 NeuralNetworkModel，而 test_neural_network_model.py 已覆盖了基类行为，
子类只需测试差异部分。

### 热点4：高epoch数测试

```
test_train_batch_verbose_outputs_epoch_loss: epochs=400 → 3.56s
test_verbose_true_outputs_epoch_loss:        epochs=400 → ~3s
test_verbose_output_format:                  epochs=200 → ~1.5s
```

这些测试目的是验证"每200 epoch输出一次loss"，但为此实际运行了200~400个epoch。
完全可以用 mock 或更小的 epoch 间隔来验证。

## 5. 精简策略与预估收益

### 策略A：引入 pytest marker 分层执行（P0，最快见效）

**目标**：CI 快速反馈 3~4 分钟，全量验证 8~10 分钟

**做法**：

在 conftest.py 中注册 marker：
```python
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 耗时>1秒的测试")
```

给极慢/慢测试打标记，CI 运行 `pytest -m "not slow"`。

**预估收益**：CI 从 10分钟 → 3~4分钟

**风险**：零。不删不改任何用例，仅加标记。

---

### 策略B：合并同单元碎片文件（P1）

**目标**：消除重复 fixture 开销，减少维护成本

| 合并前 | 合并后 |
|--------|--------|
| FC v1 Trainer: 8文件 → | 2文件 |
| Evaluator shape detection: 2文件 → | 1文件 |
| Mini-batch training: 2文件 → | 1文件 |
| Loss logger: 2文件 → | 1文件 |

**预估收益**：fixture 重复开销 -20%，代码行数 -30%

---

### 策略C：多模型共享测试基类（P2）

**目标**：3个CNN模型测试合并为1个参数化测试

```python
# 用参数化替代3份几乎相同的测试
@pytest.mark.parametrize("model_cls,network_cls", [
    (CNNv1Model, CNNv1Network),
    (CNNv2Model, CNNv2Network),
    (CNNv3Model, CNNv3Network),
])
class TestCNNModelCommon:
    def test_predict_shape(self, model_cls): ...
    def test_save_and_load(self, model_cls): ...
```

**预估收益**：~80用例 → ~30用例，覆盖率不变

---

### 策略D：降低慢测试的"实际工作量"（P3）

**目标**：train_cycle 从 12秒/用例 → 2秒/用例

可选方案：
1. 用 mock 替换 run_self_play_and_evaluate，验证调用链而非AI质量
2. 减小参数：games=1, epochs=1
3. verbose测试改用更小的epoch间隔（如每2个epoch输出一次）

**预估收益**：极慢测试合计从 ~30秒 → ~6秒

**风险**：中等，需验证 mock 等价性

---

## 7. 策略D方案1深度分析：mock run_self_play_and_evaluate

### 7.1 当前慢测试的调用链

```
FCv1Trainer.train_cycle()                    ← 入口
  → SelfPlayCycle(trainer).run()             ← 编排器
      → run_self_play_and_evaluate()         ← 耗时根源（AI对局+评价）
          → AIEngine(model)                  ← 创建AI引擎
          → DataRecorder(...)                ← 数据记录器
          → AIGameManager.start_ai_vs_ai()   ← ★ 实际AI对局，极慢
          → Evaluator.evaluate_round()       ← 评价，中等耗时
      → trainer.train_auto(sample_dirs)      ← 训练，快（epochs=1时<0.5s）
      → trainer.model.save()                 ← 保存，快
```

**耗时根因**：`AIGameManager.start_ai_vs_ai()` 执行完整对局，
每局平均50+步棋，每步涉及模型推理 → 即使 games=1 也需 3~5 秒。

### 7.2 当前3个慢测试验证了什么

| 测试 | 耗时 | 参数 | 断言内容 |
|------|------|------|----------|
| test_train_cycle_basic | 12.29s | rounds=1, games=2 | ai_data/train_data目录存在且含子目录 |
| test_train_cycle_creates_model_file | 3.78s | rounds=1, games=1 | 训练后模型文件可保存 |
| test_train_cycle_with_existing_model | 2.97s | rounds=1, games=1 | 传入已有模型后训练仍可用(predict返回225) |

**关键洞察**：这3个测试验证的是 **SelfPlayCycle 的编排逻辑**
（目录创建、方法调用链、模型保存），而非 AI 对局质量或训练效果。

### 7.3 已有覆盖：test_self_play_cycle.py

项目中已有 `test_self_play_cycle.py`，用完整的 mock 测试了编排逻辑：

```python
# 该文件已有11个mock测试用例（毫秒级），覆盖：
- 参数传递：games, rate, max_workers, epochs, project_root → run_self_play_and_evaluate
- 编排逻辑：多轮迭代、每轮保存模型、空数据跳过训练
- 中断处理：interrupted=True 时跳出循环
- 日志记录：loss_logger.log / loss_logger.plot
```

这些 mock 测试与3个慢测试的验证目标**高度重叠**。

### 7.4 Mock方案详细设计

**mock 点**：`ai.self_play_cycle.run_self_play_and_evaluate`

与 test_self_play_cycle.py 不同的是，这里保留真实的 FCv1Trainer 和模型，
仅 mock 对弈+评价步骤。

#### 层级1：mock run_self_play_and_evaluate，保留真实 train_auto（推荐）

```python
@patch('ai.self_play_cycle.run_self_play_and_evaluate')
def test_train_cycle_basic(self, mock_self_play):
    """验证 train_cycle 编排逻辑（mock对弈，真实训练）"""
    trainer = FCv1Trainer()
    trainer.loss_logger = MagicMock()
    trainer.loss_logger.dense_log = False
    
    # 准备预置sample.json，让 train_auto 有数据可读
    sample_dir = os.path.join(self.temp_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    sample_data = [{"input": [0]*226, "output": [0]*225, "mask": [1]+[0]*224}]
    with open(os.path.join(sample_dir, "sample.json"), 'w') as f:
        json.dump(sample_data, f)
    
    # mock 返回值：指向预置的sample
    mock_self_play.return_value = {
        "round_dir": self.ai_data_dir,
        "train_data_dir": self.train_data_dir,
        "sample_dirs": [sample_dir],
        "sample_count": 1,
        "results": [{"winner": 1}],
        "recorder": MagicMock(),
        "interrupted": False
    }
    
    trainer.train_cycle(rounds=1, games=1, epochs=1, project_root=self.temp_dir)
    
    # 断言：run_self_play_and_evaluate 被正确调用
    mock_self_play.assert_called_once()
    # 断言：目录被创建（由 SelfPlayCycle.run 第71-73行创建）
    self.assertTrue(os.path.exists(self.ai_data_dir))
    self.assertTrue(os.path.exists(self.train_data_dir))
    # 断言：模型可保存
    saved = trainer.model.save(os.path.join(self.temp_dir, "model.pth"))
    self.assertTrue(os.path.exists(saved))
```

**预估耗时**：0.5~1秒（mock省掉AI对局的3~12秒，保留train_auto的<0.5秒）

#### 层级2：同时 mock run_self_play_and_evaluate + train_auto（不推荐）

```python
@patch('ai.self_play_cycle.run_self_play_and_evaluate')
def test_train_cycle_basic(self, mock_self_play):
    trainer = MagicMock()  # 连 trainer 也 mock
    ...
```

**不推荐原因**：全部 mock 后测试变为"mock 调 mock"，验证价值接近于零。
test_self_play_cycle.py 已经做了这个层级的测试。

### 7.5 等价性验证

| 验证项 | 原始测试 | Mock层级1 | 等价？ |
|--------|----------|-----------|--------|
| ai_data目录被创建 | ✅ 对弈创建 | ✅ SelfPlayCycle.run()在调用前创建(L71-73) | ✅ |
| train_data目录被创建 | ✅ 评价创建 | ✅ 同上 | ✅ |
| 目录含子目录数据 | ✅ 对弈产出 | ❌ mock返回预置目录 | ⚠️ 见下文 |
| 模型可保存 | ✅ 训练后save | ✅ train_auto真实执行后save | ✅ |
| 模型仍可predict | ✅ | ✅ train_auto仍真实执行 | ✅ |
| train_auto被调用 | ✅ 隐含验证 | ✅ mock返回sample_dirs→触发train_auto | ✅ |

**⚠️ "目录含子目录数据"的等价性分析**：

原始测试断言 `ai_data_dir` 下有子目录，这是对弈产生的 game_N 目录。
mock 后 `round_dir` 指向预置目录，不含子目录 → 原始断言会失败。

**解决**：调整断言语义。原始断言验证的本质是"train_cycle 后数据目录非空"，
mock 后可改为验证 `run_self_play_and_evaluate 被正确调用` + `train_auto 被调用`，
这与 test_self_play_cycle.py 中的编排逻辑验证等价。

**结论**：mock 后的测试与 test_self_play_cycle.py 的编排逻辑验证**高度重叠**，
唯一额外价值是"FCv1Trainer + 真实模型参与整个流程"。

### 7.6 最终建议

**将3个慢测试合并为1个最小化集成测试，用 mock 层级1**：

| 原始 | Mock后 | 耗时变化 |
|------|--------|----------|
| test_train_cycle_basic (12.29s) | 合并为1个mock测试 | — |
| test_train_cycle_creates_model_file (3.78s) | 合并为1个mock测试 | — |
| test_train_cycle_with_existing_model (2.97s) | 保留或合并 | — |
| **3个合计 19.04s** | **1个mock测试 ~1s** | **-95%** |

**与 test_self_play_cycle.py 的关系**：

- test_self_play_cycle.py（11个mock测试）：验证编排逻辑的**行为正确性**（参数传递、调用次数、中断处理）
- mock后的 train_cycle 测试（1个）：验证**FCv1Trainer参与编排时的集成正确性**（真实模型+真实train_auto）

两者互补而非重复：一个验证"调用链正确"，一个验证"真实组件能跑通"。

### 7.7 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| mock掩盖run_self_play_and_evaluate的真实行为 | 中 | run_self_play_and_evaluate自身有独立测试(test_self_play_generator.py) |
| mock返回值与真实返回结构不一致 | 低 | test_self_play_cycle.py已验证返回结构消费方式 |
| 丢失"真实对局→真实训练"的端到端验证 | 中 | 可将1个最小端到端测试移入integration标记，仅nightly运行 |

---

### 策略E：清理已废弃模型测试（P1）

如果 cnn_v1 / cnn_v2 已不再维护训练，对应测试可以归档。

| 模型 | 测试文件 | 用例数 |
|------|----------|--------|
| cnn_v1 | test_cnn_model.py, test_cnn_trainer.py | ~73 |
| cnn_v2 | test_cnn_v2_model.py, test_cnn_v2_trainer.py | ~47 |

**预估收益**：如废弃可减少 ~120用例

**前提**：需确认哪些模型已废弃

## 6. 建议执行顺序

| 优先级 | 动作 | 预估收益 | 风险 | 前置条件 |
|--------|------|----------|------|----------|
| P0 | 加 pytest marker 分层 | CI 10min→3min | 零 | 无 |
| P1 | 确认废弃模型并清理 | -120用例 | 低 | 用户确认 |
| P2 | 合并FC v1 Trainer 8→2 | fixture -20% | 低 | 无 |
| P3 | CNN多模型参数化 | -50用例 | 低 | 无 |
| P4 | train_cycle mock化 | 极慢→快 | 中 | 验证等价 |
