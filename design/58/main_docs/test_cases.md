# 测试用例集

符号约定：`==>` 表示预期结果；测试均在临时目录中构造数据，禁止触碰真实样本与模型文件。

## 1. OffPolicyTrainer — 样本加载（test_off_policy_trainer.py）

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-OPT-01 | 空目录 | 临时目录，无任何 JSON 文件 | `_load_samples` 返回三个空列表 |
| TC-OPT-02 | manifest + chunk 目录加载 | 临时目录含 manifest.json（chunks 列 2 个文件，共 50 条样本）+ 2 个 chunk JSON | 返回 states/pi/z 各 50 条；`len(state) == 226`、`len(pi) == 225` 全部成立；与 chunk 文件内容一致 |
| TC-OPT-03 | 损坏 chunk 跳过 | 1 个正常 chunk（30 条）+ 1 个损坏 chunk（非法 JSON） | 返回 30 条；打印 1 条警告；不抛异常 |
| TC-OPT-04 | manifest 缺失，目录扫描 | 无 manifest.json，目录含 3 个 chunk_*.json（各 10 条） | 返回 30 条，全部加载 |
| TC-OPT-05 | sample.json 兼容 | 无 manifest.json，目录含 1 个 sample.json（20 条标准样本） | 返回 20 条，字段格式正确 |
| TC-OPT-06 | 混合格式 | 目录同时含 manifest.json（chunk 40 条）与 sample.json（20 条） | 返回 40 条（manifest 优先），sample.json 不重复加载 |
| TC-OPT-07 | 字段缺失样本跳过 | chunk 含 10 条正常 + 2 条缺 `z` 字段 + 1 条缺 `state` 字段 | 返回 10 条；跳过计数为 3，打印提示 |
| TC-OPT-08 | 维度不符跳过 | chunk 含 10 条正常 + 2 条 `state` 长度 225 + 1 条 `pi` 长度 224 | 返回 10 条；跳过计数为 3 |
| TC-OPT-09 | 无 manifest 且无 sample.json 时跳过 manifest 自身 | 目录含 manifest.json + 1 个 chunk | 按 manifest 加载，不把 manifest.json 当样本文件解析 |
| TC-OPT-10 | 空样本目录经 train_epochs 调用 | `OffPolicyTrainer._load_samples([空目录])` 由 `train_epochs` 内部触发 | 不抛异常，history 为空（基类已增加空样本防御：`_load_samples` 返回空列表时直接返回空 history，用户确认方案 A） |

## 2. OffPolicyTrainLoop — 训练编排（test_off_policy_train_loop.py）

测试使用最小数据规模（如 100 条样本、batch 32、epoch 2、val_split 0.1），并指向临时数据目录与临时模型 gen 目录（通过 monkeypatch 替换 `CNNV11Model.DEFAULT_MODEL_PATH` 基目录）。

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-OPL-01 | 正常训练保存新 gen | model_spec=`cnn_v11.1`，epochs=2 | `saved_gen == 2`、`saved_spec == "cnn_v11.2"`；`gen2/` 下存在 `cnn_v11.pth` 与 `cnn_v11_torchscript.pt` 且非空；`total_epochs == 2`、`interrupted == False` |
| TC-OPL-02 | gen{N+1} 已存在 | 预置 `gen2/` 目录含旧权重，训练 model_spec=`cnn_v11.1` | 覆盖成功（新文件时间戳更新）；打印"已存在，将覆盖"提示；不抛异常 |
| TC-OPL-03 | 模型不存在 | model_spec=`cnn_v11.99`（无对应 gen 目录） | 构造或 `run()` 抛 `ValueError`，错误信息含模型 spec |
| TC-OPL-04 | 非 cnn_v11 spec | model_spec=`cnn_v7.1` | 抛 `ValueError`，信息明确"仅支持 cnn_v11" |
| TC-OPL-05 | 样本总数为 0 | data_dir 为空目录 | `run()` 抛 `ValueError`，信息含"无有效样本"；不产生任何 gen 目录 |
| TC-OPL-06 | val_split=0 | val_split=0，epochs=2 | 训练正常完成；history 各项无 `val_*` 字段；`final_val_loss == None` |
| TC-OPL-07 | val_split>0 时 val 计算 | val_split=0.1，100 条样本，epochs=2 | 每 epoch 打印含 `val_total_loss`；`final_val_loss` 为有限数值；history 各项含 `val_value_loss`/`val_policy_loss`/`val_total_loss` |
| TC-OPL-08 | 打印包含梯度范数 | 正常训练 epochs=1 | 每个 epoch 打印行含 `grad_norm` 字段，数值 ≥ 0 |
| TC-OPL-09 | 曲线图生成 | 正常训练 epochs=2（history 非空） | 在 `mcts_loss_logs/` 下生成 `cnn_v11_*.png` 文件且非空；返回路径非 None |
| TC-OPL-10 | 中断后仍保存 | epochs=100，训练第 1 epoch 第 1 个 batch 期间模拟 KeyboardInterrupt | `total_epochs(0) < 100`、`interrupted == True`；`gen{N+1}` 已保存且可加载；history 为空 → 曲线图不生成（用户确认：保持第 1 epoch 中断，断言不再要求曲线图） |
| TC-OPL-11 | 设备回退 | device=`"cuda"` 且 CUDA 不可用（monkeypatch `torch.cuda.is_available` 返回 False） | 回退 CPU 训练完成；打印回退警告；结果正常 |
| TC-OPL-12 | 训练 loss 呈下降趋势 | 100 条样本，epochs=5，小 lr | `history` 的 `total_loss` 序列整体不上升（允许局部波动），末尾 < 首项 |
| TC-OPL-13 | 批次切分正确 | 100 条样本，batch_size=32 | 每个 epoch 内 batch 数 == 4（32+32+32+4），`history` 各指标为 batch 均值 |

## 3. CLI 入口（含于 test_off_policy_train_loop.py 或独立 CLI 测试）

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-CLI-01 | 默认值生效 | 仅传 `--data-dir <临时目录>`（--model 不传） | 自动识别 cnn_v11 最高代数并打印；其余参数使用文档默认值（batch=2048、epochs=100、validate-set=0.1） |
| TC-CLI-02 | batch 边界校验 | `--batch-size 0` | 打印"必须 > 0"，exit code ≠ 0 |
| TC-CLI-03 | epochs 边界校验 | `--epochs -1` | 打印错误，exit code ≠ 0 |
| TC-CLI-04 | validate-set 上界 | `--validate-set 1.0` | 打印"必须 < 1"，exit code ≠ 0 |
| TC-CLI-05 | validate-set 下界 | `--validate-set -0.1` | 打印"必须 >= 0"，exit code ≠ 0 |
| TC-CLI-06 | 类型校验 | `--epochs abc` | argparse 报错，exit code ≠ 0 |
| TC-CLI-07 | 数据目录不存在 | `--data-dir <不存在的路径>` | 打印明确错误（目录不存在），exit code ≠ 0 |
| TC-CLI-08 | 模型不存在 | `--model cnn_v11.99` | 打印明确错误（模型不存在），exit code ≠ 0，不产生 gen 目录 |

## 4. 回归确认（基类改动影响面）

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-REG-01 | 既有 train_epochs 打印兼容 | 现有 `test_dual_head_trainer.py` 全量通过 | 基类 E6 打印行追加 grad_norm 后，既有测试不受影响（无断言变更） |
| TC-REG-02 | `_plot_loss_curve` 未回归 | 现有 `test_dual_head_trainer_plot.py` 全量通过 | 曲线图功能保持原行为 |
