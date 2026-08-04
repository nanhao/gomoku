# 测试用例集 — CNN v11

---

## 一、CNNV11Network 单元测试

### TC1.1 forward — 输出形状正确

- **前置**: 创建 `CNNV11Network()` 实例
- **操作**: `network(x)`，`x` 形状 `(4, 3, 15, 15)`（batch_size=4）
- **验证**: 返回 `(policy, value)` 元组；`policy.shape == (4, 225)`；`value.shape == (4, 1)`

### TC1.2 forward — 策略头值域

- **前置**: 网络实例
- **操作**: `policy, _ = network(x)`
- **验证**: `policy` 所有值 ≤ 0（LogSoftmax 输出）；`policy.exp().sum(dim=1)` 每行之和 ≈ 1.0（容差 1e-4）

### TC1.3 forward — 价值头值域

- **前置**: 网络实例
- **操作**: `_, value = network(x)`
- **验证**: `value` 所有值 ∈ [-1, 1]（Tanh 输出）

### TC1.4 forward — 双头独立输出

- **前置**: 网络实例，两组相同输入 `x1 == x2`（batch 中两条相同样本）
- **操作**: `(p1, v1), (p2, v2) = network(x1)[0], network(x2)[0]`
- **验证**: `p1` 与 `p2` 各位置相等（容差 1e-6）；`v1 == v2`（确定性输出）

### TC1.5 forward — 梯度可传播（双头均参与）

- **前置**: 网络实例，输入 `x`
- **操作**: `policy, value = network(x)`；`loss = policy.sum() + value.sum()`；`loss.backward()`
- **验证**: 所有可训练参数的 `.grad` 不为 None；策略头 `policy_fc.weight.grad` 非零；价值头 `value_fc2.weight.grad` 非零

### TC1.6 forward — eval 模式不改变确定性

- **前置**: 网络实例，固定输入 `x`
- **操作**: 分别以 `train()` 和 `eval()` 模式各前向 3 次，记录输出
- **验证**: 同一模式下 3 次输出完全一致（deterministic）；train 与 eval 模式下输出完全一致（BN 无 dropout，单样本 batch 时 BN 退化为恒等变换）

### TC1.7 参数量范围

- **前置**: 网络实例
- **操作**: 统计 `sum(p.numel() for p in network.parameters())`
- **验证**: 参数量在 830 万 ~ 860 万之间

### TC1.8 残差块数

- **前置**: 网络实例
- **操作**: 统计网络中 `nn.Conv2d` 层数（排除 `conv_out` 和 `value_conv_out` 两个 1×1 卷积）
- **验证**: Conv2d 层数为 15（1 层初始 Conv1 + 14 层残差块内 Conv2~Conv15），对应 7 个残差块

---

## 二、CNNV11Model 单元测试

### TC2.1 构造器 — 默认创建新模型

- **操作**: `CNNV11Model(fresh=True)`
- **验证**: 模型实例创建成功；`model.name == "cnn_v11"`；`model.network` 为 `CNNV11Network` 实例

### TC2.2 构造器 — 加载已保存权重

- **前置**: 存在有效的 `cnn_v11.pth` 权重文件
- **操作**: `CNNV11Model(model_path="cnn_v11.pth")`
- **验证**: 模型加载成功；网络参数与保存时一致

### TC2.3 构造器 — 权重文件不存在

- **前置**: `model_path` 指向不存在的文件
- **操作**: `CNNV11Model(model_path="nonexistent.pth")`
- **验证**: 抛出 `FileNotFoundError`

### TC2.4 构造器 — gen 参数

- **前置**: `{model_dir}/gen1/cnn_v11.pth` 存在
- **操作**: `CNNV11Model(gen=1)`
- **验证**: 加载 gen1 目录下的权重；`model.gen == 1`；`model.full_name == "cnn_v11.1"`

### TC2.5 predict — 空棋盘输出形状

- **前置**: 模型实例
- **操作**: `result = model.predict([0] * 226)`
- **验证**: `result.type == "policy_value"`；`len(result.policy) == 225`；`result.value` 为 float，值域 [-1, 1]

### TC2.6 predict — 策略概率和为 1

- **前置**: 模型实例
- **操作**: `result = model.predict(board_vector)`
- **验证**: `sum(result.policy) ≈ 1.0`（容差 1e-4）；所有概率值 > 0

### TC2.7 predict_batch — 批量推理

- **前置**: 模型实例，构造 3 个不同 `board_vector`
- **操作**: `results = model.predict_batch([bv1, bv2, bv3])`
- **验证**: `len(results) == 3`；每项 `type == "policy_value"`；policy 长度均为 225

### TC2.8 predict_batch — 空输入

- **前置**: 模型实例
- **操作**: `results = model.predict_batch([])`
- **验证**: 返回空列表 `[]`

### TC2.9 predict_batch — 与逐条 predict 结果一致

- **前置**: 模型实例，固定三个 board_vector
- **操作**: 分别用 `predict_batch` 和三次 `predict` 获取结果
- **验证**: 两种方式获取的 policy 和 value 完全一致（容差 1e-6）

### TC2.10 save & load — 往返一致性

- **前置**: 模型实例，临时文件路径 `tmp_path`
- **操作**: `model.save(tmp_path)` → 新建 `CNNV11Model(model_path=tmp_path)` → 对相同输入做 predict
- **验证**: save 前和 load 后的 predict 输出完全一致（容差 1e-6）

### TC2.11 move_to_device — GPU 可用时迁移

- **前置**: CUDA 可用
- **操作**: `model.move_to_device()` （默认 auto）
- **验证**: 所有网络参数在 GPU 上（`p.device.type == "cuda"`）

### TC2.12 _prepare_input — 相对视角编码正确

- **前置**: 模型实例
- **操作**: 构造 board_vector（黑方视角），检查编码结果
- **验证**: 通道 0（我方）= 黑棋位置为 1.0；通道 1（对方）= 白棋位置为 1.0；通道 2（颜色）= 全 1.0（黑方）

---

## 三、CNNV11Trainer 单元测试

### TC3.1 构造器 — fresh=True

- **操作**: `CNNV11Trainer(fresh=True, device=torch.device("cpu"))`
- **验证**: `trainer.model` 为 `CNNV11Model` 实例；`trainer.model_name == "cnn_v11"`；`trainer.optimizer` 为 `AdamW` 实例

### TC3.2 构造器 — 传入已有模型

- **前置**: 已创建 `CNNV11Model(fresh=True)`
- **操作**: `CNNV11Trainer(model=existing_model)`
- **验证**: `trainer.model` 为传入的模型实例（同一对象）

### TC3.3 构造器 — gen 参数

- **前置**: `{model_dir}/gen1/cnn_v11.pth` 存在
- **操作**: `CNNV11Trainer(gen=1, device=torch.device("cpu"))`
- **验证**: `trainer.model.gen == 1`

### TC3.4 _convert_inputs — 输出形状

- **前置**: trainer 实例，构造 10 个 board_vector（每个 226 维）
- **操作**: `tensor = trainer._convert_inputs(states)`
- **验证**: `tensor.shape == (10, 3, 15, 15)`；dtype 为 float32

### TC3.5 _train_step — 损失下降

- **前置**: trainer 实例，构造小 batch 训练数据（2 条样本，含 state/pi/z）
- **操作**: 执行一次 `_train_step`，记录 loss；再执行一次
- **验证**: 第二次 loss ≤ 第一次 loss（或差异在合理波动范围内）

### TC3.6 _train_step — 返回字典结构

- **前置**: trainer 实例，构造训练数据
- **操作**: `result = trainer._train_step(inputs, pi_targets, z_targets)`
- **验证**: `result` 包含 `"value_loss"`, `"policy_loss"`, `"total_loss"`, `"grad_norms"`；所有 loss 为 float

### TC3.7 _train_step — p_log 校验

- **前置**: trainer 实例，正常训练数据
- **操作**: 调用 `_train_step`
- **验证**: 内部 p_log 形状校验通过（不抛 AssertionError）；p_log 值域校验通过（max ≤ 0）

### TC3.8 train_epochs — 空目录

- **前置**: trainer 实例，`sample_dirs` 指向不存在 `sample.json` 的目录
- **操作**: `trainer.train_epochs(sample_dirs=["empty_dir"], epochs=1)`
- **验证**: 不崩溃；返回空 history 或包含零样本 epoch

### TC3.9 train_epochs — 单 epoch 完整流程

- **前置**: trainer 实例，临时目录包含 1 条合法样本的 `sample.json`（state/pi/z）
- **操作**: `history = trainer.train_epochs(sample_dirs=[tmp_dir], epochs=1, batch_size=1)`
- **验证**: `len(history) == 1`；history[0] 包含 `value_loss`, `policy_loss`, `total_loss`

### TC3.10 save_model — 文件产出

- **前置**: trainer 实例，临时输出路径
- **操作**: `trainer.save_model(tmp_path)`
- **验证**: 文件存在；文件大小 > 0

---

## 四、注册表集成测试

### TC4.1 get_model — 加载 cnn_v11 模型

- **前置**: `{model_dir}/cnn_v11/gen1/cnn_v11.pth` 存在
- **操作**: `model = get_model("cnn_v11.1")`
- **验证**: 返回 `CNNV11Model` 实例；`model.name == "cnn_v11"`；`model.gen == 1`

### TC4.2 get_model — 未指定代数报错

- **操作**: `get_model("cnn_v11")`
- **验证**: 抛出 `ValueError`（多代模型必须指定代数）

### TC4.3 get_trainer — 加载 cnn_v11 训练器

- **前置**: 模型文件存在
- **操作**: `trainer = get_trainer("cnn_v11.1")`
- **验证**: 返回 `CNNV11Trainer` 实例

### TC4.4 list_models_with_full_name — 包含 cnn_v11

- **前置**: 至少一个 gen 目录下有 `.pth` 文件
- **操作**: `models = list_models_with_full_name()`
- **验证**: `"cnn_v11.1"`（或相应代数）在列表中

### TC4.5 create_engine — 路由到 PolicyValueAiEngine

- **前置**: cnn_v11 模型存在
- **操作**: `engine = create_engine("cnn_v11.1")`
- **验证**: engine 为 `PolicyValueAiEngine` 实例

---

## 五、端到端集成测试

### TC5.1 MCTS 自对弈 — 单局生成样本

- **前置**: 已训练的 `cnn_v11.1.pth` 权重（或 fresh 随机权重）
- **操作**: `mcts_self_play.py --model cnn_v11.1.pth --model-name cnn_v11.1 --games 1 --simulations 10 --device cpu`
- **验证**: 输出目录包含 `sample.json`；sample.json 为合法 JSON 数组；每项含 `state`/`pi`/`z` 字段

### TC5.2 训练脚本 — 单 epoch 训练

- **前置**: TC5.1 产生的 sample.json
- **操作**: `mcts_train.py --model cnn_v11.1.pth --model-name cnn_v11.1 --train-from-dir <output_dir> --epochs 1 --batch-size 64 --device cpu`
- **验证**: 训练正常完成；输出 epoch 日志；生成 loss 曲线图 `.png`

### TC5.3 对战 — cnn_v11 与 random 对战

- **前置**: cnn_v11 和 random 模型
- **操作**: `mcts_battle.py --model1 cnn_v11.1.pth --model1-name cnn_v11.1 --model2 cnn_v11.1.pth --model2-name random --games 2 --simulations 10 --device cpu`
- **验证**: 正常完成；输出双方胜率统计

---

## 六、边界与异常测试

### TC6.1 棋盘已满时 predict 不崩溃

- **前置**: 模型实例
- **操作**: 构造全满棋盘（无空位）的 board_vector，调用 `predict`
- **验证**: 不抛异常；返回正常 `ModelPredict`

### TC6.2 无效落子方值

- **前置**: 模型实例
- **操作**: `board_vector[225] = 3`（非法落子方），调用 `predict`
- **验证**: 不崩溃（board_vector_to_cnn_tensor_relative 有容错处理）

### TC6.3 超长 board_vector

- **前置**: 模型实例
- **操作**: `board_vector` 长度 300（超过 226 维），调用 `predict`
- **验证**: 不崩溃；截取前 225+1 位或使用已有容错逻辑

### TC6.4 空 sample.json

- **前置**: trainer 实例
- **操作**: 目录下 `sample.json` 内容为 `[]`（空数组），执行 `train_epochs`
- **验证**: 不崩溃；无样本时跳过训练或输出提示

### TC6.5 极大批量推理

- **前置**: 模型实例
- **操作**: `predict_batch` 传入 500 个 board_vector
- **验证**: 不 OOM（CPU 测试）；返回 500 个 ModelPredict；结果正确
