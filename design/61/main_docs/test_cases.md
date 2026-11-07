# cnn_v12 测试用例集

> 意图 61 | 单元测试放 `tests/ai/unit/`，集成测试放 `tests/ai/integration/`
>
> **强制约束**：任何用例禁止向 `src/ai/models/` 真实目录写入或删除文件；凡需落盘的用例一律使用 `tempfile.TemporaryDirectory()` 并在用例结束时清理。

---

## 一、CNNV12Network（`tests/ai/unit/test_cnn_v12_network.py`）

| ID | 场景 | 输入/操作 | 预期 |
|----|------|-----------|------|
| TC1.1 | 批量前向输出形状 | `network(torch.randn(4, 3, 15, 15))` | `policy.shape == (4, 225)`，`value.shape == (4, 1)` |
| TC1.2 | 单样本前向 | `network(torch.randn(1, 3, 15, 15))` | `policy.shape == (1, 225)`，`value.shape == (1, 1)` |
| TC1.3 | 策略头值域 | 批量前向 | `policy` 全部 ≤ 0（LogSoftmax） |
| TC1.4 | 策略头归一化 | `policy.exp().sum(dim=1)` | 每行之和 ≈ `1.0`（容差 `1e-5`） |
| TC1.5 | 价值头值域 | 批量前向 | `value` 全部落在 `[-1, 1]`（Tanh） |
| TC1.6 | 参数量精确值 | `sum(p.numel() for p in network.parameters())` | 恰好 `1_346_541` |
| TC1.7 | 架构常量 | 读取 `CHANNELS` / `NUM_RES_BLOCKS` / `res_blocks` | `CHANNELS == 128`；`NUM_RES_BLOCKS == 4`；`len(res_blocks) == 4`；每个块的 `conv1.in_channels == conv1.out_channels == 128` |
| TC1.8 | 相对 cnn_v11 的瘦身比例 | 同时实例化 `CNNV11Network` 与 `CNNV12Network` | `v12_params < v11_params / 6`，且 `1_346_541 * 6 < 8_436_333` |
| TC1.9 | 与 cnn_v11 权重不兼容 | `torch.load` 出的 v11 `state_dict` 载入 v12 | 抛 `RuntimeError`（形状不匹配），证明必须从零训练 |
| TC1.10 | TorchScript 可导出 | `torch.jit.script(network)` 后前向 `(2,3,15,15)` | 不抛异常；输出形状与 eager 一致 |
| TC1.11 | 梯度可回传 | 前向 + `policy.sum().backward()` | 全部参数 `grad` 非 `None`（无断链层） |

---

## 二、CNNV12Model（`tests/ai/unit/test_cnn_v12_model.py`）

| ID | 场景 | 输入/操作 | 预期 |
|----|------|-----------|------|
| TC2.1 | 模型名 | `CNNV12Model(fresh=True).name` | `"cnn_v12"` |
| TC2.2 | 默认权重路径 | `CNNV12Model.DEFAULT_MODEL_PATH` | 以 `cnn_v12.pth` 结尾，父目录名为 `cnn_v12` |
| TC2.3 | 预测结果类型 | `CNNV12Model(fresh=True).predict([0]*226)` | `type == ModelPredict.TYPE_POLICY_VALUE` |
| TC2.4 | 预测结果语义 | 同上 | `policy` 长度 225、全部 ∈ `(0, 1]`、和为 `1.0 ± 1e-4`；`value` ∈ `[-1, 1]` |
| TC2.5 | 代数路径改写 | 构造 `CNNV12Model(fresh=True, gen=3)` | `DEFAULT_MODEL_PATH` 含 `gen3` 目录段 |
| TC2.6 | 缺权重报错 | `CNNV12Model(gen=99)` | 抛 `FileNotFoundError` |
| TC2.7 | 完整名 | `CNNV12Model(fresh=True, gen=2).full_name` | `"cnn_v12.2"` |
| TC2.8 | 批量与单条一致 | 同一批 3 个 `board_vector` 分别 `predict` 与 `predict_batch` | 逐元素差 < `1e-5` |
| TC2.9 | 空批量 | `predict_batch([])` | 返回 `[]`，不抛异常 |
| TC2.10 | 存取往返 | `save()` 到临时目录 → `load()` → 再预测 | 两次 `predict` 输出完全一致 |
| TC2.11 | 相对视角编码与 cnn_v11 逐位一致 | 同一 226 维 `board_vector` 分别过 `CNNV11Model` 与 `CNNV12Model` 的输入转换 | 两个 `(1,3,15,15)` 张量 `torch.equal` 为 `True`（证明相对视角编码被继承而非重写） |

---

## 三、CNNV12Trainer（`tests/ai/unit/test_cnn_v12_trainer.py`）

| ID | 场景 | 输入/操作 | 预期 |
|----|------|-----------|------|
| TC3.1 | 训练器模型名 | `get_trainer` 或直接构造 | `model_name == "cnn_v12"` |
| TC3.2 | 模型创建与代数透传 | `CNNV12Trainer(fresh=True, gen=2)` | `trainer.model` 为 `CNNV12Model` 实例，`trainer.model.gen == 2` |
| TC3.3 | 单步训练可执行 | 随机构造 `inputs (4,3,15,15)`、`pi (4,225)`、`z (4,)` 调 `_train_step` | 返回有限 `total_loss`（非 NaN / Inf） |
| TC3.4 | 输入张量形状 | 一条 226 维 state 经训练器输入转换 | 形状 `(1, 3, 15, 15)` |
| TC3.5 | 优化器超参 | 检查 `trainer.optimizer` | `AdamW`，`weight_decay == 1e-4`，`lr` 为传入值 |
| TC3.6 | 曲线图改名 | `CNNV12Trainer(fresh=True).model_name` | 用于图名时得到 `cnn_v12_*` 而非 `cnn_v11_*` |

---

## 四、registry 集成（`tests/ai/integration/test_cnn_v12_registry.py`）

前置条件：用例只读。`gen1/cnn_v12.pth` 不存在时，依赖真实权重的用例抛 `unittest.SkipTest`，禁止为跑通测试而创建真实权重。

| ID | 场景 | 输入/操作 | 预期 |
|----|------|-----------|------|
| TC4.1 | 注册表包含条目 | 读取 `MODEL_REGISTRY` | 存在键 `"cnn_v12"`，`supports_multi_gen is True` |
| TC4.2 | 未指定代数报错 | `get_model("cnn_v12")` | 抛 `ValueError` |
| TC4.3 | 按代数加载 | `get_model("cnn_v12.1")` | 返回 `CNNV12Model`，`name == "cnn_v12"`，`gen == 1` |
| TC4.4 | 训练器加载 | `get_trainer("cnn_v12.1", device=cpu)` | 返回 `CNNV12Trainer`，`model_name == "cnn_v12"` |
| TC4.5 | 模型清单可见性 | `list_models_with_full_name()` | `gen1/` 下有 `.pth` 时含 `"cnn_v12.1"`；无 `.pth` 时不含（不展示空代数） |
| TC4.6 | 最高代数 | `get_highest_gen("cnn_v12")` | 返回 `gen{N}/` 中存在 `.pth` 的最大 N；全无时抛 `ValueError` |
| TC4.7 | 引擎路由 | `create_engine("cnn_v12.1")` | 返回 `PolicyValueAiEngine`（本意图下走 Python MCTS 回退，而非 C++ 适配器） |
| TC4.8 | 未知模型不误命中 | `get_model("cnn_v13.1")` | 抛 `ValueError` |

---

## 五、回归约束

| ID | 内容 |
|----|------|
| TC5.1 | `tests/ai/unit/test_cnn_v11_network.py`、`test_cnn_v11_model.py`、`test_cnn_v11_trainer.py` 全部保持通过（本次不改动 cnn_v11 任何文件） |
| TC5.2 | `tests/ai/integration/test_cnn_v11_registry.py` 保持通过 |
| TC5.3 | `python -m pytest tests/ -v` 全量通过 |
| TC5.4 | `list_models_with_full_name()` 在 cnn_v12 尚无权重时不因该条目报错（空 gen 目录场景已由基类跳过逻辑覆盖） |

---

## 六、用例与需求映射

| 需求点 | 覆盖用例 |
|--------|----------|
| 残差塔 7 → 4 | TC1.7、TC1.6 |
| 通道 256 → 128 | TC1.7、TC1.8 |
| 双头输出语义不变 | TC1.1、TC1.3、TC1.4、TC1.5、TC2.3、TC2.4 |
| 策略/价值头结构与 cnn_v11 一致 | TC1.1、TC1.6（参数量精确值 1,346,541 只有在价值头 FC 隐藏层保持 256 时才成立） |
| 多代权重体系 | TC2.5、TC2.6、TC2.7、TC4.3、TC4.5、TC4.6 |
| registry 无侵入扩展 | TC4.1 ~ TC4.4 |
| 不影响既有模型 | TC5.1 ~ TC5.4 |
| 权重不可从 cnn_v11 热启动 | TC1.9 |
