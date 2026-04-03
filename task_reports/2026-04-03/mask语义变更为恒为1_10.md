# mask 语义变更为恒为 1

**日期**: 2026-04-03  
**序号**: 10

---

## 1. 需求内容

用户修改 `requirements/key_points.md`，将评价器生成 sample.json 时 mask 的初始化逻辑从"有效性标志（空位=0/有棋子=1）"改为"恒为 1"：

```
mask = [1] * 225
positions = 棋盘上所有 225 个位置
```

即：**不再区分有子位置和空位位置的 mask 值**，所有 225 个位置的 mask 全为 1，所有位置的 reward 均参与 loss 计算。

**重要约束**：mask 只是暂时让它不生效，未来可能重新启用，因此代码中 mask 机制仍须保留。

损失函数不变：
```
loss(x,y) = mask(x,y) * abs(estimated_reward(x,y) - sample_reward(x,y))
```
由于 mask 全为 1，等价于对所有 225 个位置求标准平均 L1 损失。

---

## 2. 解决思路

按照 TDD 流程：
1. 先更新设计文档（fc_v1_design.md），同步 mask 语义、修正输出维度描述（675→225）、移除 mask 扩展逻辑
2. 修改现有测试用例中失效的 `mask=0` 断言
3. 新增专项测试类 `TestEvaluatorMaskAlwaysOne` 验证恒为 1 的行为
4. 运行测试确认预期失败（TDD 红灯阶段）
5. 修改 `evaluator.py` 实现代码
6. 更新 `trainer.py` 相关注释
7. 运行全量测试确认通过（TDD 绿灯阶段）

---

## 3. 执行过程

### Step 1：读取现有代码
- 读取 `ai/evaluator.py`、`ai/fc_v1/trainer.py`、`tests/ai/test_evaluator.py`、`design/fc_v1_design.md`
- 确认需要改动的具体位置

### Step 2：更新设计文档
**文件**：`design/fc_v1_design.md`（版本 v1.0 → v1.2）
- 1.2 节表格：输出维度 675 → 225；损失函数描述更新
- 3.1 节 FCv1Network：输出层 `nn.Linear(1000, 675)` → `nn.Linear(1000, 225)`
- 5.1 节流程：移除"将 mask 从 225 维扩展到 675 维"步骤
- 6.2 节代码：`masks_expanded` → `masks_tensor`，注释同步
- 6.3 节：从"Mask 扩展逻辑（675维）"改为"Mask 语义说明（恒为1）"
- 7.1 节预测数据流：输出 675 维 → 225 维
- 7.2 节训练数据流：移除 mask 扩展步骤
- 12 节版本历史：新增 v1.2 条目

### Step 3：修改现有测试
**文件**：`tests/ai/test_evaluator.py`
- `test_initialize_sample` 第 86 行：`mask[0] == 0` → `mask[0] == 1`（空位 mask 现在也是 1）

### Step 4：新增测试类
**文件**：`tests/ai/test_evaluator.py`
新增 `TestEvaluatorMaskAlwaysOne` 类，包含 5 个测试用例：
- `test_mask_all_ones_empty_board`：空棋盘下 mask 全为 1
- `test_mask_all_ones_occupied_board`：有棋子时 mask 仍全为 1
- `test_mask_length_is_225`：mask 长度为 225
- `test_mask_preserved_after_compute_rewards`：_compute_rewards 后 mask 仍全为 1
- `test_sample_json_mask_all_ones`：sample.json 中写入的 mask 全为 1

### Step 5：运行测试确认预期失败
运行结果：14 个测试中 5 个失败（符合 TDD 预期）

### Step 6：修改 evaluator.py
**文件**：`ai/evaluator.py`
- `_initialize_sample`：
  - `mask = [0] * 225` → `mask = [1] * 225`
  - 删除循环内 `mask[i] = 1` 和 `mask[i] = 0` 两行赋值
  - 删除冗余的 `board_idx = i` 变量
  - 更新 docstring 说明
- `_compute_rewards`：
  - 删除第 159 行冗余的 `sample_dict[step_idx]["mask"][reward_idx] = 1`
  - 删除第 170 行冗余的 `sample_dict[step_idx]["mask"][reward_idx] = 1`
  - 添加注释说明 mask 初始化时已全为 1

### Step 7：修改 trainer.py 注释
**文件**：`ai/fc_v1/trainer.py`
- `_compute_loss` docstring：新增"当前版本 mask 全为 1"说明
- `_masked_loss` docstring：将"只对 mask=1 的位置计算"改为"当前版本 mask 全为 1，等价于对全部 225 个位置求平均"

### Step 8：运行全量测试
- 运行 `tests/ai` 下全量测试
- 结果：**115 个测试全部通过**（比原来增加了 5 个新测试）

---

## 4. 执行结果

- 全量测试 115/115 通过，0 失败，0 错误
- mask 语义已从"有效性标志（空位=0/有棋子=1）"改为"恒为 1"
- mask 机制代码结构保留，未来可重新启用

---

## 5. 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `requirements/key_points.md` | 只读（用户已修改） | 需求文档，mask 语义变更来源 |
| `design/fc_v1_design.md` | 更新 | 版本 v1.1 → v1.2，同步 mask 语义、修正输出维度 |
| `tests/ai/test_evaluator.py` | 更新 | 修正旧断言、新增 TestEvaluatorMaskAlwaysOne 类（5个用例） |
| `ai/evaluator.py` | 更新 | _initialize_sample mask 全为1，_compute_rewards 移除冗余赋值 |
| `ai/fc_v1/trainer.py` | 更新 | _compute_loss 和 _masked_loss docstring 注释同步 |
