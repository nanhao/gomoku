# 找 Bug 比赛 —— 最终报告

> 比赛时间：2026-04-24
> 项目范围：`d:\projects\gomoku`（五子棋 AI 项目）
> 审查方式：三位选手独立并行审查

---

## 一、比赛概述

本次比赛由三位风格迥异的选手参加，分别从**边界条件与异常处理**、**测试覆盖与质量**、**架构设计与代码异味**三个维度对项目进行全面审查。

三位选手共提交了 **125 条** bug/缺陷报告，经过去重与交叉验证后，确认 **~109 条** 为有效独特缺陷。

---

## 二、参赛选手与战绩

| 排名 | 选手代号 | 专注领域 | 提交总数 | 有效独特数 | 原始得分 |
|:---:|:---|:---|:---:|:---:|:---:|
| **🏆 第一名** | **选手C：架构审查官** | 设计缺陷、性能、资源泄漏、代码异味 | **50** | **~39** | **110 分** |
| **🥈 第二名** | **选手B：测试侦探** | 测试覆盖率、断言缺失、测试隔离 | **42** | **~40** | **97 分** |
| **🥉 第三名** | **选手A：边界猎手** | 边界条件、异常处理、输入验证 | **33** | **~26** | **73 分** |

### 评分标准
- Critical = 4 分
- High = 3 分
- Medium = 2 分
- Low = 1 分

### 获胜理由
**选手C（架构审查官）** 以 50 条报告、110 分的总成绩夺冠。其优势在于：
- 发现了最多的 **Critical** 级别缺陷（含类名冲突、浅拷贝污染、内存泄漏等根本性设计问题）
- 覆盖了最广泛的代码区域（从底层算法到 GUI 组件到测试基础设施）
- 提供了大量可操作的修复建议

**选手B（测试侦探）** 虽然总分第二，但其 **独有贡献最大**（约 40 条测试领域缺陷几乎无重叠），对测试基础设施的改善价值极高。

**选手A（边界猎手）** 虽然总分第三，但在 **运行时安全** 方面贡献突出，发现了多个可能导致崩溃的边界条件缺陷。

---

## 三、交叉验证统计

### 重叠发现（多位选手共同发现）

| # | 缺陷描述 | 发现者 | 严重程度 |
|---|:---|:---:|:---:|
| 1 | `shape_detector.py` 原地修改 board 后异常导致状态污染 | A, C | Critical |
| 2 | `ai/engine.py` model_map 缺少 `cnn_v2` | A, C | High |
| 3 | `game/controller.py` 裸 `except Exception` 掩盖错误 | A, C | High |
| 4 | `ai/loss_logger.py` `open().close()` 不安全的文件创建 | A, C | High |
| 5 | `gui/components/ai_battle.py` 后台线程异常不传播 | A, C | High |
| 6 | `gui/components/mode_selector.py` 与 `ai_battle.py` 重复代码 | A, C | Low |
| 7 | `gui/components/replay_browser.py` 字符串替换推断路径 | A, C | Medium |
| 8 | `ai/base_trainer.py` + `RandomModel` 初始化崩溃 | A, C | Critical |
| 9 | `ai/evaluator.py` 多进程异常信息丢失 | A, C | High |
| 10 | `tests/test_controller.py` `test_draw` 被长期跳过 | B, C | Low |
| 11 | `ai/models/__init__.py` 导出不完整（缺少 cnn_v2） | B, C | Medium |

**重叠率分析**：
- A 与 C 重叠约 **11 条**（A 的 33%），说明两人在源码缺陷上有较高交集
- B 与 A/C 几乎无重叠，说明测试领域的缺陷与其他维度互补性极强
- **总计 11 条重叠缺陷**，去重后有效缺陷约 **109 条**

---

## 四、完整缺陷清单（去重后）

### 4.1 Critical（12 条）

> 可能导致崩溃、数据丢失、状态污染或严重设计故障

#### CR-001: BaseTrainer 初始化 RandomModel 时崩溃
- **文件**: `ai/base_trainer.py`
- **行号**: ~79-84
- **描述**: `self.model.network.to(self.device)` 中，若模型为 `RandomModel`，其 `network` 属性返回 `None`，调用 `None.to()` 会抛出 `AttributeError`
- **发现者**: A, C
- **修复**: 初始化前检查 `self.model.network is not None`

#### CR-002: AI 对战满盘时 ValueError 未捕获导致全盘崩溃
- **文件**: `ai/game_manager.py`
- **行号**: ~168-188, 264-284
- **描述**: 棋盘已满时 `engine.get_move()` 抛出 `ValueError`，异常未被捕获，整个 AI 对战流程崩溃
- **发现者**: A
- **修复**: 在循环内添加 try-except 捕获 `ValueError`

#### CR-003: DataRecorder game_id 不匹配导致数据丢失
- **文件**: `ai/data_recorder.py`
- **行号**: ~78-132
- **描述**: `record_move` 和 `save_game_result` 严格检查 `game_id`，不匹配时抛 `ValueError`，导致数据丢失
- **发现者**: A
- **修复**: 将异常改为警告并自动纠正，或确保调用方同步

#### CR-004: Evaluator 多进程并行时 BrokenProcessPool 未处理
- **文件**: `ai/evaluator.py`
- **行号**: ~346-367
- **描述**: `ProcessPoolExecutor` 中 worker 进程若因 OOM 被杀死，`future.result()` 抛出 `BrokenProcessPool`，未被捕获，进程池可能永久损坏
- **发现者**: A
- **修复**: 增加 `BrokenProcessPool` 捕获并回退到串行模式

#### CR-005: shape_detector 原地修改 board 后异常导致状态永久污染
- **文件**: `ai/shape_detector.py`
- **行号**: ~218-234
- **描述**: `detect_shapes` 原地修改 `board[idx] = player` 后恢复，但若中间代码抛异常，恢复不会执行，调用者棋盘被永久修改
- **发现者**: A, C
- **修复**: 使用 `try...finally` 确保恢复，或深拷贝 board

#### CR-006: heatmap tooltip 窗口未正确管理导致内存泄漏
- **文件**: `gui/components/heatmap.py`
- **行号**: ~172-192
- **描述**: `create_tooltip()` 每次创建新 `Toplevel`，鼠标快速移动时可能创建大量窗口且销毁不及时
- **发现者**: C
- **修复**: 使用单例 tooltip 窗口，仅更新内容

#### CR-007: game/save.py 浅拷贝导致外部修改影响存档
- **文件**: `game/save.py`
- **行号**: ~16-33
- **描述**: `state_copy = dict(game_state)` 只做浅拷贝，`board` 和 `history` 的嵌套列表仍可能被外部修改
- **发现者**: C
- **修复**: 使用 `copy.deepcopy(game_state)`

#### CR-008: CNN v1/v2 trainer 类名冲突
- **文件**: `ai/models/cnn_v1/trainer.py`, `ai/models/cnn_v2/trainer.py`
- **描述**: 两个文件都定义了 `CNNTrainer`，同时导入时后者覆盖前者
- **发现者**: C
- **修复**: 改名为 `CNNV1Trainer` / `CNNV2Trainer`

#### CR-009: CNN v1/v2 model 类名冲突
- **文件**: `ai/models/cnn_v1/model.py`, `ai/models/cnn_v2/model.py`
- **描述**: 两个文件都定义了 `CNNModel` / `CNNNetwork`
- **发现者**: C
- **修复**: 改名为 `CNNV1Model`/`CNNV1Network` 和 `CNNV2Model`/`CNNV2Network`

#### CR-010: GUI 测试在无头环境全部静默跳过
- **文件**: `tests/gui/test_*.py`
- **描述**: `teardown_method` 中 `except Exception: raise SkipTest(...)` 导致任何异常都被当作"GUI不可用"跳过
- **发现者**: B
- **修复**: 区分真正不可用和测试失败，仅 tkinter 初始化失败时才 skip

#### CR-011: test_zero_epochs 的危险异常处理使测试永远通过
- **文件**: `tests/ai/edge/test_mini_batch_training_edge.py`
- **行号**: ~73-83
- **描述**: `try/except ZeroDivisionError: pass` 掩盖了"期望的异常根本没有发生"的事实
- **发现者**: B
- **修复**: 使用 `pytest.raises(ZeroDivisionError)` 明确断言

#### CR-012: Controller 平局测试被长期跳过
- **文件**: `tests/test_controller.py`
- **行号**: ~112-118
- **描述**: `raise unittest.SkipTest("TODO: 补充 Controller 层平局测试")` 让核心路径完全缺失覆盖
- **发现者**: B, C
- **修复**: 立即实现平局场景测试

---

### 4.2 High（29 条）

#### HI-001: engine 缺少 cnn_v2 支持
- **文件**: `ai/engine.py`
- **行号**: ~34-47
- **描述**: `model_map` 只有 random/fc_v1/cnn_v1，请求 cnn_v2 时静默回退到 RandomModel
- **发现者**: A, C

#### HI-002: battle.py 缺少 cnn_v2 支持
- **文件**: `ai/battle.py`
- **行号**: ~34-38
- **描述**: `_AI_MODEL_MAP` 同样缺少 cnn_v2
- **发现者**: C

#### HI-003: GomokuGame 初始化不验证 state 结构完整性
- **文件**: `game/game.py`
- **行号**: ~7-20
- **描述**: 直接访问 state 键，不验证 board 是否为 15x15
- **发现者**: A

#### HI-004: Controller.make_move 使用裸 except Exception
- **文件**: `game/controller.py`
- **行号**: ~77-103
- **描述**: 捕获所有异常统一包装，掩盖真正的程序错误（TypeError/IndexError）
- **发现者**: A, C

#### HI-005: SampleLoader.prepare_tensors 不验证 sample_data 结构
- **文件**: `ai/sample_loader.py`
- **行号**: ~62-95
- **描述**: 直接访问 `item["input"]` 等键，不验证长度
- **发现者**: A

#### HI-006: ReplayViewer._load_data 不做文件存在性检查
- **文件**: `gui/components/replay_viewer.py`
- **行号**: ~48-62
- **描述**: 直接 `open(self.record_file)`，不存在时抛 FileNotFoundError
- **发现者**: A

#### HI-007: LossLogger 初始化时文件句柄未安全关闭
- **文件**: `ai/loss_logger.py`
- **行号**: ~86
- **描述**: `open(...).close()` 若 open 失败可能泄漏描述符
- **发现者**: A, C

#### HI-008: AIBattleProgressWindow 后台线程异常导致 UI 永久挂起
- **文件**: `gui/components/ai_battle.py`
- **行号**: ~258-282
- **描述**: 线程异常不传播，UI 永远停留在"正在进行"
- **发现者**: A, C

#### HI-009: Evaluator._compute_all_rewards 中 input_vector 长度检查边界问题
- **文件**: `ai/evaluator.py`
- **行号**: ~181-184
- **描述**: `len < 226` 时静默跳过，无警告，可能导致样本缺失
- **发现者**: A

#### HI-010: LossLogger 裸 except 静默吞掉 matplotlib 错误
- **文件**: `ai/loss_logger.py`
- **行号**: ~237, 392
- **描述**: `except Exception:` 完全静默，不记录 traceback
- **发现者**: C

#### HI-011: Evaluator 多进程评价中异常信息丢失
- **文件**: `ai/evaluator.py`
- **行号**: ~340-342, 365-367
- **描述**: 只打印 `e` 的字符串，不打印 traceback
- **发现者**: A, C

#### HI-012: shape_detector JIT 回退路径静默失败
- **文件**: `ai/shape_detector.py`
- **行号**: ~307-309
- **描述**: `except Exception:` 完全静默吞掉 Numba JIT 异常
- **发现者**: C

#### HI-013: BaseTrainer 大量重复的训练循环代码
- **文件**: `ai/base_trainer.py`
- **行号**: ~190-460
- **描述**: `train_single`/`train_batch`/`train_mini_batch` 大量重复，违反 DRY
- **发现者**: C

#### HI-014: CNN v1/v2 模型大量重复代码
- **文件**: `ai/models/cnn_v1/model.py`, `cnn_v2/model.py`
- **描述**: `_vector_to_tensor`/`predict`/`save`/`load` 几乎完全相同
- **发现者**: C

#### HI-015: 多处直接访问 controller.game 内部属性
- **文件**: `gui/main.py`, `app/gui_helper.py`, `app/session.py`, `cli/main.py`
- **描述**: 违反封装原则，内部实现改变时易崩溃
- **发现者**: C

#### HI-016: Evaluator 委托方法过多增加维护负担
- **文件**: `ai/evaluator.py`
- **行号**: ~230-385
- **描述**: 十几个方法只是简单委托给 `_reward_calculator` 或 `_line_tracer`
- **发现者**: C

#### HI-017: test_save.py time.sleep(1.1) 脆弱测试
- **文件**: `tests/test_save.py`
- **行号**: ~116-119
- **描述**: 依赖真实时间流逝，在 CI 中不稳定
- **发现者**: B

#### HI-018: Controller 测试清理不彻底
- **文件**: `tests/test_controller.py`
- **行号**: ~30-44
- **描述**: 只删 `.json`，子目录残留
- **发现者**: B

#### HI-019: Session 测试共享固定目录
- **文件**: `tests/app/primary/test_session.py`
- **行号**: ~40-55
- **描述**: 不同测试类共享 `_test_saves`，并发时可能冲突
- **发现者**: B

#### HI-020: data_recorder 测试使用固定相对路径
- **文件**: `tests/ai/test_data_recorder.py`
- **行号**: ~15-20
- **描述**: `"test_ai_data"` 固定路径，中断或并发时交叉污染
- **发现者**: B

#### HI-021: evaluator 测试使用固定相对路径
- **文件**: `tests/ai/primary/test_evaluator_*.py`
- **行号**: ~30-40
- **描述**: 固定路径交叉污染
- **发现者**: B

#### HI-022: battle 测试使用项目根目录固定路径
- **文件**: `tests/ai/test_battle.py`
- **行号**: ~30-40
- **描述**: 并发运行（pytest-xdist）时多个 worker 同时读写同一目录
- **发现者**: B

#### HI-023: fc_v1_trainer_batch 断言依赖内部常量
- **文件**: `tests/ai/test_fc_v1_trainer_batch.py`
- **行号**: ~200-230
- **描述**: 断言依赖 `LOSS_LOG_INTERVAL = 200`，白盒测试脆弱
- **发现者**: B

#### HI-024: CLI 完全缺失测试
- **文件**: `cli/main.py`
- **描述**: 所有 CLI 命令零覆盖
- **发现者**: B

#### HI-025: GUI 主界面完全缺失测试
- **文件**: `gui/main.py`
- **描述**: 集成逻辑零覆盖
- **发现者**: B

#### HI-026: SelfPlayCycle 完全缺失测试
- **文件**: `ai/self_play_cycle.py`
- **描述**: 训练周期编排核心零覆盖
- **发现者**: B

#### HI-027: train_main 完全缺失测试
- **文件**: `ai/train_main.py`
- **描述**: 参数解析和工厂函数零覆盖
- **发现者**: B

#### HI-028: line_tracer 完全缺失测试
- **文件**: `ai/line_tracer.py`
- **描述**: 五连追溯奖励核心算法零覆盖
- **发现者**: B

#### HI-029: reward_calculator 完全缺失测试
- **文件**: `ai/reward_calculator.py`
- **描述**: 棋型映射和冲四递归零覆盖
- **发现者**: B

---

### 4.3 Medium（42 条）

#### MD-001: GameModeSelector 未处理可用 AI 列表为空
- **文件**: `gui/components/mode_selector.py`
- **发现者**: A

#### MD-002: line_tracer board 长度未验证
- **文件**: `ai/line_tracer.py`
- **发现者**: A

#### MD-003: CLI handle_show/short 加载失败后行为不一致
- **文件**: `cli/main.py`
- **发现者**: A

#### MD-004: HeatmapCanvas.create_tooltip 未检查父窗口状态
- **文件**: `gui/components/heatmap.py`
- **发现者**: A

#### MD-005: GomokuSave.save_game 不处理不可写目录
- **文件**: `game/save.py`
- **发现者**: A

#### MD-006: ReplayBrowser 路径替换不可靠
- **文件**: `gui/components/replay_browser.py`, `gui/ai_battle.py`
- **描述**: 使用 `replace("ai_data", "train_data")`，可能错误替换多处
- **发现者**: A, C

#### MD-007: AIEngine.board_to_vector 不验证 board 尺寸
- **文件**: `ai/engine.py`
- **发现者**: A

#### MD-008: SelfPlayCycle 模型保存失败中断训练周期
- **文件**: `ai/self_play_cycle.py`
- **发现者**: A

#### MD-009: Evaluator.evaluate_record 中 JSON 文件损坏未处理
- **文件**: `ai/evaluator.py`
- **发现者**: A

#### MD-010: gui/main.py _open_replay 可能访问已销毁窗口
- **文件**: `gui/main.py`
- **发现者**: A

#### MD-011: shape_detector 缺少直接单元测试（含 JIT 回退路径）
- **文件**: `ai/shape_detector.py`
- **发现者**: B

#### MD-012: GUI 测试使用无意义断言（assert True）
- **文件**: `tests/gui/test_*.py`
- **发现者**: B

#### MD-013: mode_selector 测试未真正调用被测方法
- **文件**: `tests/gui/test_mode_selector.py`
- **发现者**: B

#### MD-014: gui 初始化测试验证不足
- **文件**: `tests/gui/test_gomoku_gui.py`
- **发现者**: B

#### MD-015: 测试中修改类属性污染全局状态
- **文件**: `tests/ai/test_fc_v1_model.py`, `test_cnn_model.py`
- **发现者**: B

#### MD-016: test_engine_vs_itself 验证不足
- **文件**: `tests/ai/test_engine.py`
- **描述**: 未验证交替落子、无重复
- **发现者**: B

#### MD-017: base_trainer 抽象方法测试覆盖不足
- **文件**: `tests/ai/test_base_trainer.py`
- **发现者**: B

#### MD-018: battle_session_run 验证不足
- **文件**: `tests/ai/test_battle.py`
- **发现者**: B

#### MD-019: gui_helper 测试与实现紧耦合
- **文件**: `tests/app/test_gui_helper.py`
- **发现者**: B

#### MD-020: test_state_isolation 未验证 history 深拷贝
- **文件**: `tests/test_game.py`
- **发现者**: B

#### MD-021: test_record_format 验证不足
- **文件**: `tests/ai/test_data_recorder.py`
- **发现者**: B

#### MD-022: ai/models/__init__.py 导出不完整
- **文件**: `ai/models/__init__.py`
- **描述**: 缺少 cnn_v2 导出
- **发现者**: B, C

#### MD-023: cnn_v2/__init__.py 为空
- **文件**: `ai/models/cnn_v2/__init__.py`
- **发现者**: B

#### MD-024: test_session 边界验证不足
- **文件**: `tests/app/edge/test_session.py`
- **发现者**: B

#### MD-025: battle_session 初始化验证不足
- **文件**: `tests/app/primary/test_battle.py`
- **发现者**: B

#### MD-026: 训练器测试只验证"不报错"
- **文件**: `tests/ai/test_fc_v1_trainer.py`, `test_cnn_trainer.py`
- **发现者**: B

#### MD-027: mini_batch 测试边界不足
- **文件**: `tests/ai/primary/test_mini_batch_training.py`
- **发现者**: B

#### MD-028: fc_v1_inspect 测试验证不足
- **文件**: `tests/ai/test_fc_v1_inspect.py`
- **发现者**: B

#### MD-029: display_updater 缺少单元测试
- **文件**: `gui/display_updater.py`
- **发现者**: B

#### MD-030: app/session 无游戏状态分支未覆盖
- **文件**: `app/session.py`
- **发现者**: B

#### MD-031: probability_utils / sample_loader 缺少直接测试
- **文件**: `ai/probability_utils.py`, `ai/sample_loader.py`
- **发现者**: B

#### MD-032: test_game 胜利后拒绝落子未验证
- **文件**: `tests/test_game.py`
- **发现者**: B

#### MD-033: game.py _is_draw 每次 O(n²) 遍历
- **文件**: `game/game.py`
- **发现者**: C

#### MD-034: loss_logger.log 每次打开文件（I/O 性能差）
- **文件**: `ai/loss_logger.py`
- **发现者**: C

#### MD-035: base_trainer 死代码（_load_sample_data / _prepare_tensors）
- **文件**: `ai/base_trainer.py`
- **发现者**: C

#### MD-036: 方法内部导入（违反 PEP 8）
- **文件**: `ai/game_manager.py`, `game/controller.py`
- **发现者**: C

#### MD-037: gui_helper.get_ai_data_dir 无意义包装
- **文件**: `app/gui_helper.py`
- **发现者**: C

#### MD-038: loss_logger 全局修改 matplotlib 后端
- **文件**: `ai/loss_logger.py`
- **发现者**: C

#### MD-039: ai_battle 未处理窗口关闭事件
- **文件**: `gui/components/ai_battle.py`
- **发现者**: C

#### MD-040: replay_viewer 创建新 tk.Tk()
- **文件**: `gui/components/replay_viewer.py`
- **发现者**: C

#### MD-041: gui_helper.create_pve_config 缺少输入验证
- **文件**: `app/gui_helper.py`
- **发现者**: C

#### MD-042: battle.py run_battle 未返回 recorder 路径
- **文件**: `ai/battle.py`
- **发现者**: C

---

### 4.4 Low（26 条）

#### LO-001: get_color_name / get_winner_name 缺少输入验证
- **文件**: `ai/battle.py`
- **发现者**: A

#### LO-002: BaseTrainer._compute_loss 长度未验证
- **文件**: `ai/base_trainer.py`
- **发现者**: A

#### LO-003: LossLogger.plot has_grad_norm 仅检查第一个元素
- **文件**: `ai/loss_logger.py`
- **发现者**: A

#### LO-004: GUIHelper.get_player_name 对无效 player 返回固定值
- **文件**: `app/gui_helper.py`
- **发现者**: A

#### LO-005: Controller.new_game 不验证存档名合法性
- **文件**: `game/controller.py`
- **发现者**: A

#### LO-006: AIGameManager.start_pve_game 返回类型不一致
- **文件**: `ai/game_manager.py`
- **发现者**: A

#### LO-007: cli/main.py print_board 不验证 board 类型
- **文件**: `cli/main.py`
- **发现者**: A

#### LO-008: run_tests.py setUp/tearDown 检测逻辑不准确
- **文件**: `tests/run_tests.py`
- **发现者**: A

#### LO-009: 多处使用 except Exception 过于宽泛
- **文件**: `ai/evaluator.py`, `ai/self_play_cycle.py`, `ai/loss_logger.py`, `game/controller.py`, `gui/components/replay_browser.py`, `ai/shape_detector.py`
- **发现者**: A

#### LO-010: run_tests.py 维护两套测试发现机制
- **文件**: `tests/run_tests.py`
- **发现者**: B

#### LO-011: helpers.py 辅助函数无测试
- **文件**: `tests/ai/helpers.py`
- **发现者**: B

#### LO-012: replay 测试依赖硬编码 JSON
- **文件**: `tests/gui/test_replay_viewer.py`, `test_replay_browser.py`
- **发现者**: B

#### LO-013: test_fc_v1_softmax_input 文件名不匹配
- **文件**: `tests/ai/test_fc_v1_softmax_input.py`
- **发现者**: B

#### LO-014: save.py list_saves 对非 JSON 文件静默忽略
- **文件**: `game/save.py`
- **发现者**: C

#### LO-015: base_trainer 未验证 loss_logger 接口
- **文件**: `ai/base_trainer.py`
- **发现者**: C

#### LO-016: gui/main.py _new_game 未检查 AI 先手
- **文件**: `gui/main.py`
- **发现者**: C

#### LO-017: game_manager project_root 参数未使用
- **文件**: `ai/game_manager.py`
- **发现者**: C

#### LO-018: self_play_cycle 未处理非 KeyboardInterrupt 异常
- **文件**: `ai/self_play_cycle.py`
- **发现者**: C

#### LO-019: 大量测试文件修改 sys.path
- **文件**: `tests/` 下几乎所有文件
- **发现者**: C

#### LO-020: temp 目录脚本硬编码绝对路径
- **文件**: `temp/` 下多个脚本
- **发现者**: C

#### LO-021: 模型 DEFAULT_MODEL_PATH 硬编码
- **文件**: `ai/models/fc_v1/model.py` 等
- **发现者**: C

#### LO-022: constants.py LOSS_LOG_INTERVAL 魔法数字
- **文件**: `ai/constants.py`
- **发现者**: C

#### LO-023: game_board 每次重绘全部棋子
- **文件**: `gui/components/game_board.py`
- **发现者**: C

#### LO-024: app/session.py start_game 返回值不一致
- **文件**: `app/session.py`
- **发现者**: C

#### LO-025: cli/main.py handle_show/handle_short 代码重复
- **文件**: `cli/main.py`
- **发现者**: C

#### LO-026: probability_utils 未验证输入长度
- **文件**: `ai/probability_utils.py`
- **发现者**: C

---

## 五、选手独有贡献（去重后仅由该选手发现）

### 选手A 独有贡献（~21 条）

| # | 缺陷 | 文件 | 严重程度 |
|---|:---|:---|:---:|
| 1 | AI 对战满盘 ValueError 未捕获 | `ai/game_manager.py` | Critical |
| 2 | DataRecorder game_id 不匹配 | `ai/data_recorder.py` | Critical |
| 3 | Evaluator BrokenProcessPool | `ai/evaluator.py` | Critical |
| 4 | GomokuGame 初始化不验证 state | `game/game.py` | High |
| 5 | SampleLoader 不验证数据 | `ai/sample_loader.py` | High |
| 6 | ReplayViewer 不做文件检查 | `gui/components/replay_viewer.py` | High |
| 7 | Evaluator input_vector 边界 | `ai/evaluator.py` | High |
| 8 | GameModeSelector AI 列表为空 | `gui/components/mode_selector.py` | Medium |
| 9 | line_tracer board 长度 | `ai/line_tracer.py` | Medium |
| 10 | CLI handle_show/short | `cli/main.py` | Medium |
| 11 | HeatmapCanvas tooltip | `gui/components/heatmap.py` | Medium |
| 12 | GomokuSave 不可写目录 | `game/save.py` | Medium |
| 13 | AIEngine board 尺寸 | `ai/engine.py` | Medium |
| 14 | SelfPlayCycle 保存失败 | `ai/self_play_cycle.py` | Medium |
| 15 | Evaluator JSON 损坏 | `ai/evaluator.py` | Medium |
| 16 | gui/main _open_replay | `gui/main.py` | Medium |
| 17 | get_color_name 验证 | `ai/battle.py` | Low |
| 18 | _compute_loss 长度 | `ai/base_trainer.py` | Low |
| 19 | LossLogger has_grad_norm | `ai/loss_logger.py` | Low |
| 20 | GUIHelper.get_player_name | `app/gui_helper.py` | Low |
| 21 | Controller.new_game 存档名 | `game/controller.py` | Low |

### 选手B 独有贡献（~39 条）

选手B在测试领域几乎 100% 独有，核心贡献包括：
- **3 条 Critical**：GUI 测试全部静默跳过、test_zero_epochs 危险模式、Controller 平局测试被跳过
- **14 条 High**：6 个核心模块完全零测试覆盖（CLI、GUI 主界面、SelfPlayCycle、train_main、line_tracer、reward_calculator），以及测试隔离缺陷、脆弱测试
- **18 条 Medium**：断言不足、Mock 问题、包接口不完整、训练验证缺失
- **4 条 Low**：run_tests.py 双机制、helpers 无测试、硬编码 JSON fixture、文件名不匹配

### 选手C 独有贡献（~38 条）

选手C在架构层面的独有贡献：
- **4 条 Critical**：heatmap 内存泄漏、save.py 浅拷贝、CNN 类名冲突（2 条）
- **8 条 High**：裸 except 静默吞错、BaseTrainer 重复代码、CNN 模型重复代码、封装违反、委托方法膨胀
- **20 条 Medium**：性能问题（_is_draw O(n²)、log 每次打开文件）、设计问题（死代码、方法内导入、全局 matplotlib 后端、窗口关闭事件缺失）、API 不一致
- **9 条 Low**：硬编码路径、魔法数字、game_board 全量重绘、session 返回值不一致、代码重复

---

## 六、修复优先级 Top 10

| 优先级 | 缺陷 | 理由 |
|:---:|:---|:---|
| **P0** | CR-005: shape_detector 原地修改 board | 数据污染，棋盘状态可能被永久破坏 |
| **P0** | CR-007: save.py 浅拷贝 | 存档数据可能被外部修改污染 |
| **P0** | CR-011: test_zero_epochs 危险异常处理 | 测试永远通过，隐藏真实 bug |
| **P0** | CR-008/009: CNN 类名冲突 | 导入时静默覆盖，训练器可能创建错误模型 |
| **P1** | CR-001: BaseTrainer + RandomModel 崩溃 | 训练流程直接崩溃 |
| **P1** | CR-002: AI 对战满盘崩溃 | 满盘对局直接崩溃 |
| **P1** | CR-004: Evaluator BrokenProcessPool | 进程池损坏后无法恢复 |
| **P1** | HI-001/002: engine/battle 缺少 cnn_v2 | 用户请求 cnn_v2 时静默回退到随机 |
| **P1** | CR-010: GUI 测试全部静默跳过 | CI 报告失真，误以为测试通过 |
| **P1** | CR-003: DataRecorder game_id 不匹配 | 训练数据丢失 |

---

## 七、按模块统计缺陷密度

| 模块 | Critical | High | Medium | Low | 合计 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ai/evaluator.py` | 1 | 3 | 1 | 0 | **5** |
| `ai/game_manager.py` | 1 | 0 | 0 | 0 | **1** |
| `ai/data_recorder.py` | 1 | 0 | 0 | 0 | **1** |
| `ai/base_trainer.py` | 1 | 1 | 1 | 1 | **4** |
| `ai/loss_logger.py` | 0 | 2 | 1 | 1 | **4** |
| `ai/shape_detector.py` | 1 | 1 | 0 | 0 | **2** |
| `ai/engine.py` | 0 | 1 | 1 | 0 | **2** |
| `ai/battle.py` | 0 | 1 | 0 | 1 | **2** |
| `ai/self_play_cycle.py` | 0 | 0 | 1 | 1 | **2** |
| `ai/sample_loader.py` | 0 | 1 | 0 | 0 | **1** |
| `ai/probability_utils.py` | 0 | 0 | 0 | 1 | **1** |
| `ai/line_tracer.py` | 0 | 0 | 1 | 0 | **1** |
| `ai/reward_calculator.py` | 0 | 0 | 0 | 0 | **0（完全无测试）** |
| `ai/train_main.py` | 0 | 0 | 0 | 0 | **0（完全无测试）** |
| `ai/models/` | 2 | 2 | 1 | 1 | **6** |
| `game/game.py` | 0 | 1 | 0 | 0 | **1** |
| `game/controller.py` | 0 | 1 | 0 | 2 | **3** |
| `game/save.py` | 1 | 0 | 1 | 1 | **3** |
| `gui/components/` | 1 | 2 | 4 | 1 | **8** |
| `gui/main.py` | 0 | 1 | 1 | 1 | **3** |
| `gui/display_updater.py` | 0 | 0 | 1 | 0 | **1** |
| `app/session.py` | 0 | 0 | 0 | 2 | **2** |
| `app/gui_helper.py` | 0 | 0 | 1 | 2 | **3** |
| `cli/main.py` | 0 | 1 | 1 | 2 | **4** |
| `tests/` | 3 | 14 | 24 | 8 | **49** |

**结论**：测试目录（`tests/`）是缺陷最密集的区域（49 条），其次是 GUI 组件（8 条）和 AI 模型（6 条）。

---

## 八、比赛总结

### 核心发现
1. **测试基础设施薄弱**：约 45% 的缺陷集中在测试领域，多个核心模块（CLI、SelfPlayCycle、line_tracer、reward_calculator）完全无测试覆盖
2. **异常处理过度宽泛**：项目中存在 10+ 处裸 `except Exception`，掩盖真实错误
3. **CNN v2 支持不完整**：模型存在但 engine、battle、包导出均未注册
4. **状态管理风险**：shape_detector 原地修改 board、save.py 浅拷贝、DataRecorder game_id 校验过严
5. **代码重复严重**：BaseTrainer 三个训练方法、CNN v1/v2 模型、GUI 组件中多处重复代码

### 建议修复策略
1. **第一阶段（本周）**：修复 12 条 Critical 缺陷，重点关注状态污染、崩溃、测试跳过问题
2. **第二阶段（下周）**：为 6 个零测试覆盖的核心模块编写基础测试；统一使用 `tempfile.mkdtemp()` 隔离测试
3. **第三阶段（持续）**：提取公共逻辑消除重复；将魔法数字提取为常量；缩小 except 范围

---

*报告生成时间：2026-04-24*  
*主持人：找 Bug 比赛组委会*
