# CNN v10 自对弈与训练脚本设计

## 1. 概述

为 cnn_v10 价值头模型创建独立的自我对弈脚本和训练脚本，并确保 battle.py 兼容 cnn_v10。

**核心变更：**
- 新增 `value_self_play.py`：cnn_v10 自我对弈生成训练样本
- 新增 `CNNV10Trainer` + `value_train.py`：cnn_v10 价值头训练
- 修改 `battle.py`：确保 cnn_v10 对战兼容

**样本格式（价值头专用）：** `{"state": [226个int], "z": int}`
cnn_v10 没有策略头，样本不包含 pi 字段。state 为 226 维 board_vector，z 为终局结果（1 / -1 / 0）。

---

## 2. 自我对弈脚本

### 2.1 value_self_play.py

新增 `src/ai/scripts/value_self_play.py`。

参数参照 `mcts_self_play.py`，去掉 MCTS 专属参数，保留通用参数：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model` | str | 是 | - | 权重文件路径 |
| `--model-name` | str | 否 | None | 模型注册名.代数，默认从路径解析 |
| `--games` | int | 是 | - | 对弈局数 |
| `--output` | str | 否 | `{model}_value_{timestamp}` | 输出目录 |
| `--device` | str | 否 | cuda | 推理设备 |
| `--workers` | int | 否 | None | 并行进程数，1=单进程，None=CPU核数 |
| `--temperature` | float | 否 | 1.0 | 温度值 |
| `--temp-steps` | int | 否 | 8 | 前N步启用温度采样 |
| `--temp-side` | str | 否 | random | 采样方：black/white/random/both |

**与 mcts_self_play 的区别：**
- 无 `--simulations`、`--c-puct`、`--dirichlet-epsilon`、`--dirichlet-alpha`、`--batch-size`、`--virtual-loss`、`--no-prior`
- 引擎使用 `ValueOnlyAiEngine` 而非 `PolicyValueAiEngine`
- 样本不含 pi 字段

**输出：** 同 `mcts_self_play.py`，生成 `sample.json` + `summary.json`。
**周期保存：** 每 50 局保存一次，支持 Ctrl+C 中断保存。

### 2.2 value_self_play_worker.py

新增 `src/ai/scripts/value_self_play_worker.py`。

| 类 | 说明 |
|----|------|
| `ValueSelfPlayConfig` | dataclass：temperature, temp_steps, temp_side, top_k, seed |
| `ValueSelfPlayWorker` | 加载模型，串行执行多局对弈，返回样本列表 |

Worker 通过 `ValueOnlyAiEngine` 执行对局，使用 `AIGameManager.play_single_game`。

**对局流程参见：[value_self_play_flow.puml](value_self_play_flow.puml)**

### 2.3 样本生成逻辑

每局结束后，遍历对局每一步的 `board_vector`（即 state），为每步生成一条样本：

```
z = 终局结果（当前玩家视角）:
  若当前玩家胜 → +1
  若当前玩家负 → -1
  平局       → 0
```

样本不包含 pi，因为 cnn_v10 无策略头，训练仅用 state + z。

---

## 3. CNNV10Trainer

新增 `src/ai/models/cnn_v10/trainer.py`。

### 3.1 对外接口

| 接口 | 说明 |
|------|------|
| `CNNV10Trainer(model, fresh, device, lr, weight_decay, gen)` | 构造器 |
| `train_epochs(sample_dirs, epochs, batch_size, val_split, shuffle, verbose) -> List[dict]` | 训练入口 |
| `save_model(output_path) -> str` | 保存权重 |

### 3.2 训练算法

- **损失函数：** MSE（均方误差），`loss = (v_pred - z_target)²`
- **优化器：** AdamW，weight_decay 默认 1e-4
- **学习率：** 默认 0.001
- **训练流程：** 加载 sample.json → 转换 state 为张量 → epoch 循环（shuffle + batch）→ 生成 loss 曲线图

### 3.3 样本加载

`_load_samples(sample_dirs)` 从 sample.json 加载 `state` 和 `z`，返回 `(states, zs)` 元组。
- state: 226 维 → 经 `_convert_inputs` 转为 `(N, 3, 15, 15)` 张量
- z: 转为 `(N, 1)` 张量

### 3.4 registry 更新

`src/ai/registry.py` 中 cnn_v10 条目更新 `trainer_class_path`：

```
"cnn_v10": RegistryEntry(
    "ai.models.cnn_v10.model.CNNV10Model",
    "ai.models.cnn_v10.trainer.CNNV10Trainer", True),
```

---

## 4. value_train.py

新增 `src/ai/scripts/value_train.py`。

参数参照 `mcts_train.py`，去掉 policy 相关字段：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model` | str | 是 | - | 初始权重文件路径 |
| `--model-name` | str | 否 | None | 模型注册名.代数 |
| `--train-from-dir` | str[] | 是 | - | 训练数据目录，可多个 |
| `--epochs` | int | 是 | - | 训练轮数 |
| `--output` | str | 否 | None | 输出权重路径，默认覆盖 --model |
| `--batch-size` | int | 否 | 2048 | batch size |
| `--lr` | float | 否 | 0.001 | 学习率 |
| `--l2-reg` | float | 否 | 1e-4 | AdamW weight_decay |
| `--val-split` | float | 否 | 0.1 | 验证集比例 |
| `--no-shuffle` | flag | 否 | False | 关闭数据打乱 |
| `--device` | str | 否 | cuda | 训练设备 |

**训练流程参见：[value_train_flow.puml](value_train_flow.puml)**

主流程：解析参数 → 加载模型 → 创建 CNNV10Trainer → train_epochs → 保存模型 → 输出 loss。

---

## 5. battle.py 兼容

### 5.1 src/ai/battle.py（CLI）

已兼容。`create_engine("cnn_v10.1")` 自动路由到 `ValueOnlyAiEngine`（builder.py 第 206 行）。

需更新 help 文本示例，加入 cnn_v10 用法。

### 5.2 src/app/battle.py（GUI BattleSession）

`BattleSession.run()` 在对战结束后调用 `Evaluator.evaluate_round()` 生成训练样本（传统棋型评估）。

对于 cnn_v10，评估步骤不适用（无策略头，不使用传统棋型评估）。修改方案：
- 当 ai1 或 ai2 为 `TYPE_VALUE_ONLY` 类型时，跳过 `evaluate_round()` 调用
- 判定方式：通过 `create_engine` 后检测引擎类型（`isinstance(engine, ValueOnlyAiEngine)`）

### 5.3 play_single_game 数据记录兼容

`game_manager.py` 的 `play_single_game` 中 `recorder.record_move` 调用传递 `mcts_probs`、`mcts_value`、`mcts_q_values`。`ValueOnlyAiEngine` 的 `MoveDecision` 返回这些字段为 None，`DataRecorder.record_move` 需确保 None 值可正常处理（当前已兼容）。

---

## 6. 类图

参见 [value_class.puml](value_class.puml)

---

## 7. 测试用例

### 7.1 ValueSelfPlayConfig

**TC-SPC-01** 默认值
- 创建 `ValueSelfPlayConfig()`
- 期望：temperature=1.0, temp_steps=8, temp_side="random", top_k=None, seed=None

**TC-SPC-02** 自定义值
- 创建带自定义参数的 config
- 期望：各字段值与传入一致

### 7.2 CNNV10Trainer

**TC-TRN-01** 构造器
- `CNNV10Trainer(model=model_instance)`
- 期望：不抛异常，optimizer 已初始化

**TC-TRN-02** _load_samples 加载单目录
- 准备临时目录含 sample.json（state + z 格式）
- 期望：返回 (states, zs)，长度正确

**TC-TRN-03** train_epochs 单 epoch
- 准备少量样本（state + z）
- 调用 `train_epochs(sample_dirs, epochs=1)`
- 期望：返回 history 列表，含 value_loss

**TC-TRN-04** val_split 验证集
- `train_epochs(..., val_split=0.2)`
- 期望：history 每项含 val_value_loss

**TC-TRN-05** save_model
- 训练后调用 `save_model(temp_path)`
- 期望：文件存在，加载后预测一致

**TC-TRN-06** KeyboardInterrupt
- 模拟中断，调用 train_epochs
- 期望：生成 loss 曲线图，不丢数据

**TC-TRN-07** fresh=True
- `CNNV10Trainer(fresh=True, gen=1)`
- 期望：创建新模型，不加载已有权重

**TC-TRN-08** 无 sample.json 报错
- 传入空目录
- 期望：FileNotFoundError

### 7.3 value_self_play 脚本

**TC-SP-01** 单局自对弈
- `--model cnn_v10.pth --games 1 --workers 1`
- 期望：生成 sample.json + summary.json，sample 格式正确（含 state 和 z，无 pi）

**TC-SP-02** 多进程自对弈
- `--model cnn_v10.pth --games 4 --workers 2`
- 期望：生成样本数正确

**TC-SP-03** 中断恢复
- 启动后 Ctrl+C
- 期望：已生成样本不丢失，summary 含 completed_games

**TC-SP-04** --model-name 显式指定
- `--model weights.pth --model-name cnn_v10.1 --games 1`
- 期望：正确解析模型 spec

### 7.4 value_train 脚本

**TC-VT-01** 单目录训练
- `--model cnn_v10.pth --train-from-dir samples/ --epochs 5`
- 期望：训练完成，loss 下降

**TC-VT-02** 多目录训练
- `--train-from-dir dir1 dir2 --epochs 3`
- 期望：合并数据训练

**TC-VT-03** --output 指定输出路径
- `--output new_weights.pth --epochs 1`
- 期望：权重保存到指定路径

### 7.5 registry

**TC-REG-01** get_trainer
- `get_trainer("cnn_v10.1", model=model_instance)`
- 期望：返回 CNNV10Trainer 实例

**TC-REG-02** 未指定代数报错
- `get_trainer("cnn_v10")`
- 期望：ValueError

### 7.6 battle.py

**TC-BTL-01** CLI cnn_v10 对战
- `python src/ai/battle.py --ai1 cnn_v10.1 --ai2 random --games 1`
- 期望：正常完成，输出对战结果

**TC-BTL-02** GUI BattleSession cnn_v10
- `BattleSession("cnn_v10.1", "random", 1).run()`
- 期望：正常完成，跳过 evaluate_round

**TC-BTL-03** cnn_v10 vs cnn_v10
- 双方均为 cnn_v10
- 期望：正常完成，不抛异常
