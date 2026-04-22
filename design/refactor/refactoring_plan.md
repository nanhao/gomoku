# 大文件重构计划书

> **代码主题**：`ai/evaluator.py`（749行）、`ai/base_trainer.py`（591行）、`gui/main.py`（531行）
> **生成日期**：2026-04-22
> **方法**：双代理辩论 + 裁判综合

---

## 一、裁判综述

### 1.1 两份参赛计划概览

| 维度 | 保守派计划 | 激进派计划 |
|------|-----------|-----------|
| **核心哲学** | 最小化改动，渐进拆分，保持兼容 | 彻底重构，理想架构，打破兼容 |
| **Evaluator 拆分** | 4个新文件（shape_detector, reward_calculator, line_tracer, probability_utils） | 7个新文件 + 1个IO模块（引入ShapeType枚举、RewardConfig、ThreatAnalyzer、SampleBuilder等） |
| **BaseTrainer 拆分** | 2个新文件（sample_loader, self_play_cycle）+ 内部提取私有方法 | 5个新文件（model_trainer, sample_dataset, training_pipeline, callbacks, trainer_factory），删除子类Trainer |
| **GUI 拆分** | 2个新文件（load_game_dialog, display_updater） | 4个新文件（launcher, main_window, load_game_dialog, game_presenter），引入MVP模式 |
| **预估工期** | 2-3周 | 5-6周 |
| **调用方影响** | 小（保留Facade委托） | 大（需修改battle.py、train_main.py、删除trainer子类） |
| **技术风险** | 低 | 中高（PyTorch DataLoader引入、MVP与tkinter适配） |

### 1.2 裁判评分

| 评价维度（权重） | 保守派 | 激进派 | 说明 |
|-----------------|--------|--------|------|
| **可行性（25%）** | 9/10 | 6/10 | 保守派每步可独立验证；激进派引入DataLoader、MVP与tkinter的适配存在不确定性 |
| **与现状适配度（20%）** | 9/10 | 5/10 | 激进派要求删除fc_v1/cnn_v1的trainer子类，影响面大；项目当前无DataLoader使用经验 |
| **测试保障能力（20%）** | 8/10 | 7/10 | 激进派的可测试性设计更优，但需要大量新测试支撑；保守派可复用现有测试 |
| **长期价值（20%）** | 6/10 | 9/10 | 激进派的配置驱动、回调机制、MVP分离具有更好的长期扩展性 |
| **实施工作量（15%）** | 9/10 | 5/10 | 保守派工作量约为激进派的40%-50% |
| **加权总分** | **8.15** | **6.35** | |

### 1.3 裁判结论

> **保守派在可行性上显著占优，激进派在长期架构价值上表现更好。**
>
> 本项目是一个**持续演进的AI训练工程**，而非一次性交付的框架产品。核心诉求是在**不中断现有训练流程**的前提下逐步改善代码结构。因此，综合计划以保守派为骨架， selectively 吸收激进派的关键设计思想。
>
> **被吸收的思想**：
> - 配置驱动（RewardConfig 数据类，集中管理 reward 数值）
> - 职责彻底分离（train_cycle 必须提取为独立模块，斩断循环依赖）
> - 事件处理与视图分离（GUI 中引入 Presenter 概念，但不强行套用 MVP）
>
> **被舍弃的思想**：
> - PyTorch DataLoader 统一训练模式（项目当前无此实践，引入成本高、收益不明确）
> - 删除 Trainer 子类（改动面过大，且子类在未来可能承载模型特定的训练策略）
> - GUI 完全 MVP 化（tkinter 的命令式风格与 MVP 的声明式风格存在阻抗）

---

## 二、综合重构计划

### 2.1 目标文件一：`ai/evaluator.py`（749行 → 目标200行以内）

#### 现状问题

1. **职责爆炸**：单个类承载棋型检测、reward计算、Softmax转换、五连追溯、文件I/O五种职责
2. **棋型检测与reward计算耦合**：`detect_shapes` 与 `shape_to_reward` 互相调用，无法独立测试
3. **冲四递归嵌套深**：`_compute_four_reward` 内部模拟落子、找防守位、再检测棋型，认知负荷极高
4. **魔术数字分散**：`0.9 * MAX_REWARD`、`0.5 * MAX_REWARD` 等硬编码分布在5个方法中
5. **Softmax作为实例方法**：通用数值工具与业务类绑定

#### 重构目标结构

```
ai/
├── shape_detector.py      # 纯棋型检测引擎（无状态，可独立测试）
├── reward_calculator.py   # Reward计算核心（含配置驱动 + 冲四递归）
├── line_tracer.py         # 五连追溯奖励
├── probability_utils.py   # Softmax / 数值工具（通用纯函数）
└── evaluator.py           # Facade：文件I/O + 流程编排 + 委托兼容层
```

#### 具体拆分方案

**`ai/shape_detector.py`** — 棋型检测引擎
- **职责**：模拟落子后检测四个方向的棋型
- **设计**：纯函数集合，无需实例化
- **核心接口**：
  - `detect_shapes(board, position, player) -> List[Dict]`
  - `detect_single_direction(board, x, y, player, dx, dy) -> Optional[str]`
- **迁移内容**：`DIRECTIONS`、`DIRECTION_NAMES`、`detect_shapes`、`_detect_shape_in_direction`

**`ai/reward_calculator.py`** — Reward计算核心
- **职责**：将棋型映射为数值reward，处理冲四递归防守模拟
- **关键改进（吸收激进派思想）**：引入 `RewardConfig` 数据类，集中管理所有 reward 数值
  ```python
  @dataclass
  class RewardConfig:
      max_reward: int = 1_000_000
      values: Dict[str, float] = field(default_factory=lambda: {
          "连五": 1.0, "活四": 0.9, "多冲四": 0.9, "冲四": 0.5,
          "多活三": 0.8, "活三": 0.5, "眠三": 0.1, "活二": 0.3,
      })
  ```
- **核心接口**：
  - `compute_position_reward(board, x, y, current_player, config) -> int`
  - `shape_to_reward(shape_info, config, ...) -> int`
  - `_compute_four_reward(...)`（内部方法）
- **依赖**：通过import使用 `shape_detector`，不持有实例状态

**`ai/line_tracer.py`** — 五连追溯奖励
- **职责**：从终局棋盘找到五连位置，追溯历史步骤，反向传播衰减奖励
- **设计**：轻量级类，接收 `max_reward`、`decay_rate`
- **核心接口**：
  - `find_five_in_a_row(board, winner) -> List[Tuple[int, int]]`
  - `trace_winning_line(end_board, winner, history) -> List[Tuple[int, float]]`
  - `blend_rewards(shape_reward, line_bonus, alpha) -> float`

**`ai/probability_utils.py`** — 通用概率工具
- **职责**：带mask的数值稳定Softmax
- **设计**：模块级纯函数
- **核心接口**：
  - `masked_softmax(rewards, mask, scale=1.0/100000.0) -> List[float]`

**`ai/evaluator.py`（保留为Facade）**
- **保留职责**：`evaluate_record()`、`evaluate_round()`（文件I/O、流程编排）
- **委托方法**：原 `detect_shapes`、`shape_to_reward` 等方法保留，内部调用新模块，附加 `DeprecationWarning`
- **内部实现**：`_compute_all_rewards()` 调用 `LineTracer`、`RewardCalculator`、`masked_softmax` 组合结果

#### 实施步骤

| 步骤 | 内容 | 行数变化 | 风险 |
|------|------|---------|------|
| **E1** | 提取 `shape_detector.py`。迁移棋型检测相关方法和常量。Evaluator保留委托。 | -120行 | 极低。纯逻辑移动 |
| **E2** | 提取 `line_tracer.py`。迁移五连追溯相关方法。Evaluator初始化时创建LineTracer实例。 | -100行 | 低。逻辑独立 |
| **E3** | 提取 `reward_calculator.py`。引入RewardConfig，迁移reward计算和冲四递归。 | -200行 | 中。冲四递归涉及模拟落子，需重点测试 |
| **E4** | 提取 `probability_utils.py`。迁移Softmax。清理Evaluator中无人调用的委托。 | -50行 | 极低 |

#### 兼容性与回退

- `evaluate_record(record_file)`、`evaluate_round(round_dir)` **签名与行为完全不变**
- 被拆分的内部方法保留为委托，调用方可继续使用，但会收到 `DeprecationWarning`
- 每步完成后运行 evaluator 测试，确保已知 record.json 生成的 sample.json 逐字节一致
- 冲四递归的旧实现在 E3 中保留为注释块，可瞬间切回

---

### 2.2 目标文件二：`ai/base_trainer.py`（591行 → 目标250行以内）

#### 现状问题

1. **训练器、加载器、编排器三职合一**：一个类同时管理模型生命周期、加载JSON/转张量、编排自我对弈全周期
2. **三种训练模式代码重复**：`train_single`、`train_batch`、`train_mini_batch` 在数据校验、合并、LossLogger、plot上存在大面积重复（约60%）
3. **`train_cycle` 内部引入循环依赖**：方法体内延迟import `ai.engine`、`ai.game_manager` 等，长达120行，是最不稳定的部分
4. **数据加载与训练循环耦合**：`_load_sample_data`、`_prepare_tensors` 与训练循环混在一起

#### 重构目标结构

```
ai/
├── sample_loader.py       # 样本数据加载与预处理（JSON → Tensor）
├── self_play_cycle.py     # 训练周期编排（自我对弈→评价→训练→保存）
└── base_trainer.py        # 模型生命周期 + 训练循环 + 委托兼容层
```

#### 具体拆分方案

**`ai/sample_loader.py`** — 样本数据加载器
- **职责**：统一负责从 `sample.json` 到 PyTorch 张量的全链路转换
- **设计**：
  ```python
  class SampleLoader:
      def __init__(self, device: torch.device, input_converter=None): ...
      def load(self, sample_path: str) -> list: ...
      def load_multiple(self, sample_dirs: List[str]) -> list: ...
      def prepare_tensors(self, sample_data: list) -> Tuple[Tensor, Tensor, Tensor]: ...
  ```
- **迁移内容**：`_load_sample_data`、`_prepare_tensors`、`_convert_inputs`

**`ai/self_play_cycle.py`** — 自我对弈训练周期
- **职责**：原 `train_cycle` 的完整逻辑——自我对弈、调用Evaluator生成样本、合并训练、保存模型
- **关键改进**：将方法内部的延迟import改为文件顶部正常import，彻底斩断循环依赖
- **设计**：
  ```python
  class SelfPlayCycle:
      def __init__(self, trainer: BaseTrainer, loss_logger=None): ...
      def run(self, rounds=10, games=100, epochs=1000, rate=0.9, project_root=None): ...
  ```

**`ai/base_trainer.py` 内部重构**
- **不新增文件**，在现有文件中提取私有方法消除重复：
  - `_collect_and_prepare(sample_dirs) -> Tuple[Tensor, Tensor, Tensor]`：统一 `train_batch` / `train_mini_batch` 的数据收集
  - `_setup_training_logger() -> LossLogger`：统一logger获取
  - `_finalize_training(loss_logger)`：统一plot与清理
- **三种训练模式的差异**：保留在各自的公共方法中，仅将重复代码块提取到私有方法

#### 实施步骤

| 步骤 | 内容 | 行数变化 | 风险 |
|------|------|---------|------|
| **T1** | 提取 `sample_loader.py`。迁移数据加载与预处理。三种训练方法统一使用 `self._sample_loader`。 | -80行 | 低。无副作用的数据转换 |
| **T2** | 内部提取 `_collect_and_prepare`、`_setup_training_logger`、`_finalize_training`，消除重复代码块。 | -60行 | 中低。需注意三种模式的细微差异 |
| **T3** | 提取 `self_play_cycle.py`。将 `train_cycle` 整体迁移。BaseTrainer保留单句委托。 | -120行 | 中。涉及多个外部系统交互 |

#### 兼容性与回退

- `train_single()`、`train_batch()`、`train_mini_batch()`、`train_cycle()` **签名与返回值完全不变**
- `_load_sample_data()`、`_prepare_tensors()`、`_convert_inputs()` 保留为委托，指向 `self._sample_loader`
- `BaseTrainer` 为抽象基类，子类（`FCv1Trainer`、`CNNTrainer`）**无需改动**
- `train_cycle` 的委托实现保留旧实现的缩略注释，可瞬间切回

---

### 2.3 目标文件三：`gui/main.py`（531行 → 目标200行以内）

#### 现状问题

1. **`GomokuGUI` 同时承担界面构建、事件处理、游戏流程控制、显示更新**
2. **`_load_game()` 内联完整对话框**：33行的Toplevel+Listbox+Button实现，无法复用
3. **模块级函数与类隐式耦合**：`run_ai_vs_ai_mode()` 和 `main()` 直接操作 `GomokuGUI` 初始化细节
4. **显示更新逻辑分散在事件处理方法中**

#### 重构目标结构

```
gui/
├── components/
│   └── load_game_dialog.py   # 加载存档对话框（新）
├── display_updater.py        # 游戏状态渲染器（新）
├── ai_battle.py              # AI内战运行器（新）
└── main.py                   # 模式分发 + GomokuGUI精简版
```

#### 具体拆分方案

**`gui/components/load_game_dialog.py`** — 加载游戏对话框
- **职责**：弹出存档选择对话框，返回选中的存档名或None
- **设计**：
  ```python
  class LoadGameDialog:
      @staticmethod
      def show(parent, saves: List[str]) -> Optional[str]: ...
  ```
- **收益**：`_load_game()` 从35行压缩为5行

**`gui/display_updater.py`** — 游戏状态渲染器
- **职责**：根据 `display_state` 字典刷新棋盘、标签、历史记录列表
- **设计**：
  ```python
  class GameDisplayUpdater:
      def __init__(self, board, save_name_label, status_label, 
                   game_status_label, history_listbox, helper): ...
      def update(self, display_state: dict): ...
      def update_history(self, history: list): ...
      def show_game_over(self, winner): ...
  ```
- **收益**：`GomokuGUI` 中所有"读状态→写控件"的逻辑被隔离

**`gui/ai_battle.py`** — AI内战运行器
- **职责**：原 `run_ai_vs_ai_mode()` 的完整逻辑
- **收益**：`main.py` 不再直接依赖 `AIBattleProgressWindow` 的实例化细节

**`gui/main.py` 中的 `GomokuGUI`**
- **吸收激进派思想（选择性）**：引入事件处理分离，但不强行套用MVP
- **保留设计**：`GomokuGUI` 仍然持有 `controller`、`session`、`helper`，因为 tkinter 的命令式风格天然适合"View 直接持有业务对象"
- **核心变化**：事件处理方法只保留"调用 helper → 调用 updater"的骨架，显示逻辑委托给 `GameDisplayUpdater`

#### 实施步骤

| 步骤 | 内容 | 行数变化 | 风险 |
|------|------|---------|------|
| **G1** | 提取 `gui/components/load_game_dialog.py`。封装Toplevel对话框。 | -30行 | 极低。纯UI组件移动 |
| **G2** | 提取 `gui/display_updater.py`。迁移 `_update_display`、`_update_history`、`_show_game_over`。 | -80行 | 低。仅操作已有控件 |
| **G3** | 提取 `gui/ai_battle.py`。迁移 `run_ai_vs_ai_mode()`。 | -20行 | 极低。函数级移动 |

#### 兼容性与回退

- `GomokuGUI.__init__(root, game_mode, ai_player, ai_model)` **签名不变**
- 被拆分的方法保留为委托，内部调用新组件
- `main()` 和 `run_ai_vs_ai_mode()` 在 `gui/main.py` 中保留暴露（通过从新文件import重新导出）
- 外部 `python -m gui.main` 的启动方式**完全不变**

---

## 三、全局实施路线图

### 3.1 执行顺序建议

按**业务核心度**和**风险收益比**排序：

```
第一阶段（第1周）：Evaluator域
  └─ E1: shape_detector 提取
  └─ E2: line_tracer 提取
  └─ E3: reward_calculator 提取（含RewardConfig）
  └─ E4: probability_utils 提取

第二阶段（第2周）：BaseTrainer域
  └─ T1: sample_loader 提取
  └─ T2: 内部消除三种训练模式重复代码
  └─ T3: self_play_cycle 提取

第三阶段（第3周）：GUI域
  └─ G1: load_game_dialog 提取
  └─ G2: display_updater 提取
  └─ G3: ai_battle 提取

第四阶段（第4周）：清理与固化
  └─ 删除Evaluator中无调用的委托方法
  └─ 运行全量测试
  └─ 更新设计文档
```

### 3.2 各阶段验收标准

| 阶段 | 验收标准 |
|------|---------|
| **第一阶段** | `Evaluator.evaluate_record()` 对已知 record.json 生成的 sample.json 与重构前逐字节一致；新提取模块的单元测试覆盖率 >= 80% |
| **第二阶段** | `train_single`/`train_batch`/`train_mini_batch` 三种模式在相同输入下损失曲线误差 < 1e-5；`train_cycle` 完成1轮x2局自我对弈不报错 |
| **第三阶段** | `python -m gui.main` 三种模式（PvP/PvE/AI内战）正常启动；`_load_game` 对话框可正常选择存档 |
| **第四阶段** | 三个目标文件行数均降至250行以内；全量回归测试通过 |

### 3.3 测试策略

1. **对比测试（第一阶段核心）**：新旧模块对同一输入产生相同输出，才能进入下一阶段
2. **每步验证**：每完成一个Step，运行现有测试确保不破坏
3. **Evaluator 特殊保障**：生成已知 record.json 的 sample，对比MD5
4. **Trainer 特殊保障**：固定随机种子，对比三种训练模式的loss曲线

---

## 四、风险管控

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 冲四递归逻辑迁移引入bug | 中 | 高 | E3保留旧实现为注释块；对同一record对比输出 |
| train_cycle外部依赖变化导致失效 | 中 | 高 | T3保留单句委托，旧实现保留注释；先运行1轮x2局验证 |
| GUI控件引用传递出错 | 低 | 中 | G2中GameDisplayUpdater只接收已有控件引用，不创建新窗口 |
| 调用方遗漏导致导入错误 | 低 | 中 | 全局搜索所有 `Evaluator(`、`BaseTrainer(`、`GomokuGUI(` 引用点；保留Facade兼容层至少一个迭代周期 |
| 重构范围失控 | 中 | 高 | **严格边界**：不修改 `game/`、`app/`（除battle.py外）、`cli/` 目录；不修改模型结构 |

---

## 五、预期收益

### 5.1 行数变化

| 文件 | 当前行数 | 目标行数 | 降幅 |
|------|---------|---------|------|
| `ai/evaluator.py` | 749 | ~180 | 76% |
| `ai/base_trainer.py` | 591 | ~220 | 63% |
| `gui/main.py` | 531 | ~180 | 66% |

### 5.2 质量改善

| 维度 | 改善点 |
|------|--------|
| **可测试性** | 棋型检测、reward计算、五连追溯可独立单元测试，无需加载完整Evaluator |
| **可维护性** | reward数值集中配置（RewardConfig），修改一处全局生效；bug定位从"在749行中找"变为"在150行模块中找" |
| **可读性** | 每个文件职责单一，新开发者可在5分钟内理解模块边界 |
| **可扩展性** | 新增棋型只需修改 `shape_detector.py` + `RewardConfig`；新增训练模式只需在 `BaseTrainer` 中添加方法（共享数据加载和清理逻辑） |

---

## 六、附录：新旧文件映射表

| 旧文件/方法 | 新归属 | 说明 |
|------------|--------|------|
| `Evaluator.detect_shapes` | `ai/shape_detector.py` | 纯函数提取 |
| `Evaluator._detect_shape_in_direction` | `ai/shape_detector.py` | 纯函数提取 |
| `Evaluator.shape_to_reward` | `ai/reward_calculator.py` | 引入RewardConfig |
| `Evaluator._compute_four_reward` | `ai/reward_calculator.py` | 冲四递归 |
| `Evaluator._find_four_defense` | `ai/reward_calculator.py` | 辅助方法 |
| `Evaluator._shape_reward_value` | `ai/reward_calculator.py` | 辅助方法 |
| `Evaluator.find_five_in_a_row` | `ai/line_tracer.py` | 五连追溯 |
| `Evaluator.trace_winning_line` | `ai/line_tracer.py` | 五连追溯 |
| `Evaluator.blend_rewards` | `ai/line_tracer.py` | 奖励融合 |
| `Evaluator._softmax` | `ai/probability_utils.py` | 通用纯函数 |
| `BaseTrainer._load_sample_data` | `ai/sample_loader.py` | 数据加载 |
| `BaseTrainer._prepare_tensors` | `ai/sample_loader.py` | 张量转换 |
| `BaseTrainer._convert_inputs` | `ai/sample_loader.py` | 输入转换 |
| `BaseTrainer.train_cycle` | `ai/self_play_cycle.py` | 训练周期编排 |
| `GomokuGUI._load_game` (内联对话框) | `gui/components/load_game_dialog.py` | 组件化 |
| `GomokuGUI._update_display` | `gui/display_updater.py` | 渲染分离 |
| `GomokuGUI._update_history` | `gui/display_updater.py` | 渲染分离 |
| `GomokuGUI._show_game_over` | `gui/display_updater.py` | 渲染分离 |
| `run_ai_vs_ai_mode` | `gui/ai_battle.py` | 函数级移动 |
