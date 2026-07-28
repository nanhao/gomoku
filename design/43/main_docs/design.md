# CNN v10 价值头模型设计

## 1. 概述

cnn_v10 是在 cnn_v9 基础上去掉策略头、仅保留价值头的单输出模型。
配合新的"枚举-评估"落子算法，替代原有的 MCTS 搜索机制。

**核心变更：**
- 网络：cnn_v9 去掉策略头，仅输出价值 `[-1, 1]`
- 引擎：新增 `ValueOnlyAiEngine`，实现枚举候选位置 + 模型评估 + 概率归一化的落子决策
- 模型类型：新增 `TYPE_VALUE_ONLY`
- 暂不处理训练

## 2. 网络架构

### 2.1 CNNV10Network

在 CNNV9Network 基础上删除策略头全部组件（`conv_out`、`policy_fc1`、`policy_fc2`、`log_softmax`），
仅保留共享特征层和价值头。

```
输入 (batch, 3, 15, 15)
→ Conv1(3→48) → BN1 → ReLU1
→ [残差块1] Conv2→BN2→ReLU2→Conv3→BN3→[+shortcut]→ReLU3
→ [残差块2] Conv4→BN4→ReLU4→Conv5→BN5→[+shortcut]→ReLU5
→ 共享特征 (batch, 48, 15, 15)
→ value_conv_out(48→1,1×1) → Flatten(225) → FC(225→64) → ReLU → FC(64→1) → Tanh
→ 价值 (batch, 1)
```

forward 返回单个 `(batch, 1)` 张量，值域 `[-1, 1]`。

### 2.2 参数量估算

cnn_v9 策略头约 29,000 参数（conv_out + policy_fc1 + policy_fc2），删除后 CNNV10Network 约 **100,000** 参数。

## 3. 落子决策算法

对局中的 AI 落子分为 5 步：

```
输入：board (15×15), current_player (1/2)
输出：落子坐标 (x, y)

步骤1 - 预处理
  遍历全部 225 个位置，对每个空位判断是否"孤立"：
    孤立空位：在上下左右及四条斜线共 8 个方向上，
              相邻 1~2 格范围内均无任何棋子。
  孤立空位的评估值设为 -1，不参与后续模型评估。

步骤2 - 模拟落子与终局检测
  对每个非孤立空位 p：
    模拟 current_player 在 p 落子，得到新局面 g。
    若 g 为终局（五连/棋盘满）：
      置 p 的评估值 = 1（胜）/ -1（负）/ 0（平）
    否则：
      将 g 加入待评估列表 G。

步骤3 - 模型评估
  对 G 中的每个局面 g_i：
    以对手视角构建 board_vector（current_player = 对方），
    输入 CNNV10Model，得 v_i ∈ [-1, 1]。
    取反：v_i' = -v_i（转回当前落子方视角）。
  将 v_i' 填入对应位置的评估值。

步骤4 - 概率归一化
  将全部 225 个位置的评估值做 masked softmax 归一化：
    - 已有棋子的位置 mask 掉（概率 = 0）
    - 孤立空位以 -1 参与 softmax
    - 候选位置以 v_i' 或终局值参与 softmax
  得到 225 维概率分布。

步骤5 - 落子决策
  基于概率分布，按指定策略选择落子：
    - argmax：选概率最大的位置（并列时随机打破）
    - sampling：温度加权采样（复用 temperature_sample）
```

**交互流程图参见：[cnn_v10_flow.puml](cnn_v10_flow.puml)**

## 4. 涉及文件

| 文件 | 变更类型 |
|------|---------|
| `src/ai/models/cnn_v10/__init__.py` | 新增 |
| `src/ai/models/cnn_v10/model.py` | 新增（CNNV10Network + CNNV10Model） |
| `src/ai/engine/value_only.py` | 新增（ValueOnlyAiEngine） |
| `src/ai/engine/builder.py` | 修改（新增 build_value_only + create_engine 路由） |
| `src/ai/registry.py` | 修改（新增 cnn_v10 条目） |
| `src/ai/base_model.py` | 修改（ModelPredict 新增 TYPE_VALUE_ONLY） |
| `tests/ai/test_cnn_v10_models.py` | 新增 |
| `tests/ai/test_value_only_engine.py` | 新增 |
| `tests/ai/unit/test_registry.py` | 修改（新增 cnn_v10 用例） |

## 5. 对外接口

### 5.1 ModelPredict（修改）

| 变更 | 说明 |
|------|------|
| 新增常量 | `TYPE_VALUE_ONLY = "value_only"` |
| 构造参数 | `policy` 改为 `Optional[List[float]]`，`TYPE_VALUE_ONLY` 时可为 `None` |

### 5.2 CNNV10Network

| 接口 | 说明 |
|------|------|
| `forward(x: Tensor) -> Tensor` | 输入 `(batch,3,15,15)`，输出 `(batch,1)` 价值张量，值域 `[-1,1]` |

### 5.3 CNNV10Model

继承 `NeuralNetworkModel`。

| 接口 | 说明 |
|------|------|
| `predict(board_vector: List[int]) -> ModelPredict` | 返回 `type=TYPE_VALUE_ONLY`，`value` ∈ `[-1,1]` |
| `predict_batch(board_vectors: List[List[int]]) -> List[ModelPredict]` | 批量评估，返回等长列表 |

`save` / `load` / `move_to_device` 继承自 `NeuralNetworkModel`，行为不变。

### 5.4 ValueOnlyAiEngine

继承 `AIEngine`。

| 接口 | 说明 |
|------|------|
| `get_move(board, current_player, strategy) -> MoveDecision` | 执行 5 步算法，返回落子决策 |
| `after_move(x, y)` | 空操作（无搜索树） |

构造参数：`model`（BaseModel）、`temperature`、`top_k`、`rng`。

### 5.5 AIEngineBuilder（修改）

| 新增接口 | 说明 |
|------|------|
| `build_value_only(model, temperature, top_k, rng) -> ValueOnlyAiEngine` | 创建价值引擎 |

`create_engine` 路由新增：`TYPE_VALUE_ONLY` → `build_value_only`。

## 6. 类图

参见 [cnn_v10_class.puml](cnn_v10_class.puml)

## 7. 测试用例

### 7.1 CNNV10Network

**TC-NET-01** 前向输出形状
- 输入 `(4, 3, 15, 15)` 随机张量
- 期望：输出形状 `(4, 1)`

**TC-NET-02** 输出值域
- 输入 `(1, 3, 15, 15)` 随机张量
- 期望：输出值 ∈ `[-1, 1]`

**TC-NET-03** 无策略头组件
- 检查网络 `__dict__`
- 期望：不存在 `conv_out`、`policy_fc1`、`policy_fc2`、`log_softmax` 属性

**TC-NET-04** 价值头组件完整
- 检查网络属性
- 期望：存在 `value_conv_out`、`value_fc1`、`value_fc2`、`value_tanh`

**TC-NET-05** 卷积通道数
- 检查各卷积层
- 期望：`conv1: 3→48`，`conv2~5: 48→48`，`value_conv_out: 48→1(1×1)`

**TC-NET-06** 残差连接存在性
- 网络中应包含 BatchNorm2d 层

**TC-NET-07** 参数量
- 期望：参数量约 100,000（±5%）

### 7.2 CNNV10Model

**TC-MOD-01** predict 返回类型
- 输入 `[0]*226`，调用 `predict()`
- 期望：`result.type == TYPE_VALUE_ONLY`

**TC-MOD-02** predict 返回值域
- 输入 `[0]*226`
- 期望：`result.value` ∈ `[-1, 1]`

**TC-MOD-03** 确定性
- 相同输入两次 `predict()`
- 期望：两次 `value` 相等

**TC-MOD-04** 不同棋局面产生不同输出
- 黑方首步天元 vs 白方首步天元
- 期望：两次 `value` 不同（网络非对称）

**TC-MOD-05** predict_batch 等长
- 输入 3 个不同 board_vector
- 期望：返回 3 个 ModelPredict

**TC-MOD-06** predict_batch 与 predict 一致
- 同一 board_vector 分别经 `predict` 和 `predict_batch`（单元素列表）
- 期望：两者 `value` 相等

**TC-MOD-07** save/load 往返
- 创建 fresh 模型 → save → 新实例 load
- 同一输入预测
- 期望：value 一致

**TC-MOD-08** fresh=True 不加载文件
- 权重文件存在时，`CNNV10Model(fresh=True)`
- 期望：成功创建，不抛异常

**TC-MOD-09** model_path 缺失
- 指定不存在的 `model_path`
- 期望：`FileNotFoundError`

### 7.3 ValueOnlyAiEngine

**TC-ENG-01** 空棋盘落子
- `board` 全空，`current_player=1`，`strategy="argmax"`
- 期望：返回合法坐标 `(x, y)`，棋盘该位置为空

**TC-ENG-02** 棋盘已满
- 模拟 225 子全满
- 期望：`ValueError`

**TC-ENG-03** 唯一空位强制选择
- 仅剩 1 个空位
- 期望：选中该空位

**TC-ENG-04** argmax 确定性
- 相同局面两次 `get_move(strategy="argmax")`
- 期望：两次落子坐标相同

**TC-ENG-05** sampling 策略
- `get_move(strategy="sampling")`，`temperature=1.0`
- 期望：返回合法坐标

**TC-ENG-06** 孤立空位被排除
- 构造棋盘：仅天元有棋子，其余全部为孤立空位（间距 >2）
- `get_move()`
- 期望：落子位置在天元 2 格范围内（非孤立区域）

**TC-ENG-07** 终局位置直接评估
- 构造棋盘：黑方已有四连，仅剩 1 个空位可五连
- `get_move(strategy="argmax")` 以黑方视角
- 期望：选中可五连的位置

**TC-ENG-08** 终局位置直接评估（对方赢）
- 构造棋盘：白方已有四连，黑方落子
- 某位置落子后白方五连 → 终局值 = -1（黑方输）
- 另一个位置不导致终局
- 期望：不选白方可五连的位置

**TC-ENG-09** after_move 不抛异常
- 任意合法坐标调用 `after_move(x, y)`
- 期望：不抛异常

**TC-ENG-10** MoveDecision 包含 probs
- `get_move()` 返回的 MoveDecision
- 期望：`get_probs()` 返回 225 维非 None 列表，总和 ≈ 1.0

### 7.4 孤立空位检测

**TC-ISO-01** 天元邻接非孤立
- 棋盘仅天元 (7,7) 有黑子
- 检测 (7,8)（相邻 1 格）
- 期望：非孤立

**TC-ISO-02** 天元距离 2 内非孤立
- 棋盘仅天元 (7,7) 有黑子
- 检测 (7,9)（距离 2）
- 期望：非孤立

**TC-ISO-03** 天元距离 3 外为孤立
- 棋盘仅天元 (7,7) 有黑子
- 检测 (7,10)（距离 3）
- 期望：孤立

**TC-ISO-04** 斜线方向距离 2 内非孤立
- 棋盘仅天元 (7,7) 有黑子
- 检测 (9,9)（右下距离 2）
- 期望：非孤立

**TC-ISO-05** 角落初始为孤立
- 空棋盘
- 检测 (0,0)
- 期望：孤立（周围无棋子）

**TC-ISO-06** 已有棋子不算孤立检测
- 已有棋子的位置不在检测范围内（预处理仅对空位）

### 7.5 注册表

**TC-REG-01** get_model 正常
- `get_model("cnn_v10.1")`
- 期望：返回 CNNV10Model 实例

**TC-REG-02** 未指定代数报错
- `get_model("cnn_v10")`
- 期望：`ValueError`（多代模型）

**TC-REG-03** create_engine 路由
- `create_engine("cnn_v10.1")`
- 期望：返回 ValueOnlyAiEngine 实例

### 7.6 ModelPredict

**TC-MP-01** TYPE_VALUE_ONLY 构造
- `ModelPredict(value=0.5)`
- 期望：`type == TYPE_VALUE_ONLY`，`policy is None`，`value == 0.5`

**TC-MP-02** TYPE_VALUE_ONLY 解包
- `policy, value = ModelPredict(value=0.5)`
- 期望：`policy is None`，`value == 0.5`
