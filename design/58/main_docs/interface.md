# 接口设计

## 1. `OffPolicyTrainer`（`src/ai/off_policy_trainer.py`）

继承 `CNNV11Trainer`（其父类为 `DualHeadTrainer`），仅覆盖样本加载逻辑，其余训练能力（`_train_step`、`_convert_inputs`、`save_model`、`_plot_loss_curve` 等）全部继承复用。

**公开接口：**

| 成员 | 类型/签名 | 说明 |
|------|-----------|------|
| `_load_samples(sample_dirs: List[str]) -> Tuple[List[List[int]], List[List[float]], List[float]]` | 静态方法（覆盖基类同名方法） | 从目录列表加载全部样本，返回 `(states, pis, zs)`。`train_epochs` 内部经 `self._load_samples` 多态调用 |

**`_load_samples` 加载规则（按优先级）：**

1. 目录存在 `manifest.json` → 按 manifest 的 `chunks` 列表逐个加载 chunk JSON 文件（sample_buffer / renjunet_samples 格式）
2. 无 `manifest.json` 但存在 `sample.json` → 加载之（兼容旧样本目录）
3. 均无 → 扫描目录下所有 `.json` 文件（跳过 manifest.json）逐个加载

**容错与校验规则：**

- 损坏的 JSON 文件：跳过并打印警告，不中断加载
- manifest 中记录但磁盘缺失的文件：静默跳过
- 单条样本缺失 `state` / `pi` / `z` 字段：跳过并计数
- `state` 长度 ≠ 226 或 `pi` 长度 ≠ 225：跳过并计数
- 无有效样本时返回空列表（调用方据此报错）

## 2. `OffPolicyTrainLoop`（`src/ai/off_policy_train_loop.py`）

训练编排器。持有 `OffPolicyTrainer`，执行"加载模型 → 全量训练 → 保存新 gen"完整流程。

**公开接口：**

| 成员 | 类型/签名 | 说明 |
|------|-----------|------|
| `__init__(model_spec: str, data_dir: str, batch_size: int, epochs: int, val_split: float, lr: float, weight_decay: float, device: str)` | 构造 | 解析代数并加载初始模型（`OffPolicyTrainer(gen=N)`）。`model_spec` 非 cnn_v11 或模型不存在时抛 `ValueError` |
| `run() -> OffPolicyTrainResult` | 方法 | 执行完整训练：加载样本 → `train_epochs` → 保存 `gen{N+1}` → 返回结果。Ctrl+C 时优雅结束（仍保存模型并标记中断） |

**`run()` 行为要点：**

1. 调用 `trainer.train_epochs(sample_dirs=[data_dir], epochs, batch_size, val_split, shuffle=True, verbose=True)`
2. `train_epochs` 内部完成：样本加载 → 验证集划分 → epoch 循环（每 epoch shuffle + mini-batch 切分 + 逐批 `_train_step` + 验证集评估 + 打印汇总）→ 自动生成曲线图
3. 训练结束（正常或中断）后保存模型：`gen{N+1}/cnn_v11.pth`（`save_model`）+ `gen{N+1}/cnn_v11_torchscript.pt`（`torch.jit.script` 导出）
4. `gen{N+1}` 已存在时覆盖，并打印提示
5. 样本总数为 0 时抛 `ValueError` 并给出明确信息
6. 打印信息包含：样本总数、训练/验证样本数、epoch 数、batch 数/epoch、保存路径

## 3. `OffPolicyTrainResult`（dataclass，`src/ai/off_policy_train_loop.py`）

`run()` 的返回结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `saved_gen` | `int` | 保存的代数编号（N+1） |
| `saved_spec` | `str` | 保存的模型 spec（如 `"cnn_v11.2"`） |
| `requested_epochs` | `int` | 请求的 epoch 总数 |
| `total_epochs` | `int` | 实际完成的 epoch 数（中断时 < requested_epochs） |
| `interrupted` | `bool` | 是否被 Ctrl+C 中断 |
| `final_total_loss` | `float` | 最后完成 epoch 的平均 total_loss |
| `final_val_loss` | `Optional[float]` | 最后完成 epoch 的平均 val_total_loss（`val_split > 0` 时有效，否则 None） |
| `history` | `List[Dict[str, float]]` | `train_epochs` 返回的完整 epoch 历史（含 value/policy/total loss、grad_norm、val_* 字段） |

## 4. CLI 入口（`src/ai/scripts/mcts_train_cnn_v11_offpolicy.py`）

**参数表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | None | 初始模型 spec（如 `cnn_v11.1`）。不指定时自动识别 cnn_v11 最高代数 |
| `--data-dir` | str | `renjunet_samples` | 样本目录（chunk JSON 或 sample.json 格式） |
| `--batch-size` | int | 2048 | 每个 mini-batch 的样本数 |
| `--epochs` | int | 100 | 训练 epoch 数 |
| `--validate-set` | float | 0.1 | 验证集比例，值域 [0, 1) |
| `--lr` | float | 0.001 | 学习率 |
| `--weight-decay` | float | 1e-4 | AdamW 权重衰减 |
| `--device` | str | `cuda` | 训练设备（cuda 不可用时自动回退 cpu 并警告） |

**参数校验规则（违反任一即打印错误并退出，exit code ≠ 0）：**

- `--batch-size` 必须 > 0
- `--epochs` 必须 > 0
- `--validate-set` 必须满足 0 ≤ value < 1
- `--lr` 必须 > 0
- `--weight-decay` 必须 ≥ 0

**行为要点：**

1. `--model` 未指定时经 `registry.get_highest_gen("cnn_v11")` 自动识别并打印
2. 启动时打印完整配置（模型、数据目录、batch、epoch、验证集比例、学习率、设备）
3. 运行结束打印结果摘要（保存的 gen、完成 epoch 数、最终 loss/val-loss）。曲线图路径不重复打印——已由 `train_epochs` 内部 E7 打印"Loss 曲线图已生成: {path}"（用户确认）
4. 不注册 `os._exit` 级 SIGINT handler——Ctrl+C 由 `train_epochs` 优雅捕获（生成曲线图），中断后仍保存模型
5. 输出文件命名：曲线图 `mcts_loss_logs/cnn_v11_{YYYYMMDD_HHMMSS}.png`（由 `_plot_loss_curve` 自动生成）
