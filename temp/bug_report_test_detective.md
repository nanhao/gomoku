# 🐛 测试侦探 Bug 清单报告

> 侦查范围：`d:\projects\gomoku` 整个项目  
> 侦查时间：2026-04-24  
> 统计：共发现 **42 个有效缺陷**，其中 Critical 3 个，High 14 个，Medium 18 个，Low 7 个。

---

## 一、Critical（测试完全错误，会隐藏真实 Bug）

### BUG-001
- **文件路径**：`tests/gui/test_mode_selector.py`（line 45-62）、`tests/gui/test_game_board.py`（line 57-101）等全部 GUI 测试
- **行号范围**：多个文件 teardown_method（line 24-29）
- **Bug 描述**：所有 GUI 测试在 `teardown_method` 中捕获任何异常后抛出 `unittest.SkipTest("GUI环境不可用: {e}")`。这意味着在无头环境（CI/CD、GitHub Actions）中，**几乎所有 GUI 测试都会被静默跳过**，而不是失败。开发者会误以为 GUI 测试通过了，实际上完全没有执行。
- **严重程度**：Critical
- **修复建议**：区分"真正的 GUI 环境不可用"和"测试断言失败"。只在检测到 tkinter 无法初始化（如 `$DISPLAY` 缺失）时才 skip，其他异常应正常抛出失败。

### BUG-002
- **文件路径**：`tests/ai/edge/test_mini_batch_training_edge.py`
- **行号范围**：line 73-83
- **Bug 描述**：`test_zero_epochs` 使用 `try/except ZeroDivisionError: pass` 的危险模式。该测试期望 `epochs=0` 时引发 `ZeroDivisionError`，但实际上 `range(0)` 不会进入循环，不会触发除零。测试在"期望的异常未发生"时反而静默通过，掩盖了真实行为。
- **严重程度**：Critical
- **修复建议**：明确测试意图。如果期望 `epochs=0` 报错，应使用 `pytest.raises(ZeroDivisionError)` 或 `self.assertRaises`；如果允许 `epochs=0`，则应断言返回值为 0 或空列表。

### BUG-003
- **文件路径**：`tests/test_controller.py`
- **行号范围**：line 112-118
- **Bug 描述**：`test_draw` 测试被 `raise unittest.SkipTest("TODO: 补充 Controller 层平局测试")` 完全跳过。平局逻辑是游戏核心路径之一，跳过意味着 Controller 的平局场景在测试套件中完全缺失。这不是临时 skip，而是被当作长期占位符使用。
- **严重程度**：Critical
- **修复建议**：立即实现该测试，构造满盘且无胜者的局面，验证 `make_move` 返回 `game_over=True` 且 `winner=0`（或 3，取决于业务定义）。

---

## 二、High（重要路径未覆盖 / 显著测试缺陷）

### BUG-004
- **文件路径**：`tests/test_save.py`
- **行号范围**：line 116-119
- **Bug 描述**：`test_save_same_name_same_day` 使用 `time.sleep(1.1)` 确保时间戳不同。这是**脆弱测试（Flaky Test）**：依赖真实时间流逝，在慢速 CI 环境可能不稳定；增加测试套件运行时间；且没有 mock 时间源。
- **严重程度**：High
- **修复建议**：使用 `unittest.mock.patch('time.time')` 或 `freezegun` 库来冻结/控制时间，移除 `time.sleep`。

### BUG-005
- **文件路径**：`tests/test_controller.py`
- **行号范围**：line 30-44（setup_method）
- **Bug 描述**：`setup_method` 使用固定目录 `_test_saves` 并在清理时只删除 `.json` 文件：`[f for f in os.listdir(test_save_dir) if f.endswith('.json')]`。如果测试运行中创建了子目录或其他类型文件，它们会残留，导致测试隔离失败，后续测试可能受污染。
- **严重程度**：High
- **修复建议**：使用 `tempfile.mkdtemp()` 创建唯一临时目录，并在 teardown 中使用 `shutil.rmtree()` 完整删除整个目录。

### BUG-006
- **文件路径**：`tests/app/primary/test_session.py`
- **行号范围**：line 40-55（`_make_controller`）
- **Bug 描述**：`_make_controller` 使用 `_test_saves` 作为固定目录，不同测试类共享同一个物理路径。虽然使用 `time.time()` 生成存档名来避免冲突，但如果并发执行或时间精度问题仍可能冲突。更关键的是，teardown 只删除了 `.json` 文件，残留目录结构。
- **严重程度**：High
- **修复建议**：每个测试实例使用 `tempfile.mkdtemp()` 创建独立目录，并在 teardown 中完整删除。

### BUG-007
- **文件路径**：`tests/ai/test_data_recorder.py`
- **行号范围**：line 15-20
- **Bug 描述**：`self.test_dir = "test_ai_data"` 使用**相对路径固定名称**。虽然 `setUp/tearDown` 会尝试清理，但如果测试中断（如 Ctrl+C、断点调试）或并发执行，该目录会残留并污染后续测试运行。
- **严重程度**：High
- **修复建议**：使用 `tempfile.mkdtemp(prefix="test_ai_data_")` 生成唯一临时目录。

### BUG-008
- **文件路径**：`tests/ai/primary/test_evaluator_shape_detection.py`、`tests/ai/primary/test_evaluator_winning_line.py`
- **行号范围**：line 30-40（各文件 setup）
- **Bug 描述**：同样使用 `self.test_dir = "test_evaluator_shape_data"` / `"test_evaluator_winning_data"` 固定相对路径。并发运行或中断时会导致交叉污染。
- **严重程度**：High
- **修复建议**：统一使用 `tempfile.mkdtemp()` 进行隔离。

### BUG-009
- **文件路径**：`tests/ai/test_battle.py`
- **行号范围**：line 30-40
- **Bug 描述**：`self.test_ai_data_dir = os.path.join(project_root, "test_battle_ai_data")` 使用项目根目录下的固定路径。虽然 setUp/tearDown 会清理，但并发运行（pytest-xdist）时多个 worker 会同时读写同一目录，导致不可预期的失败或数据污染。
- **严重程度**：High
- **修复建议**：使用 `tempfile.mkdtemp(dir=project_root)` 在项目根目录下创建唯一临时目录。

### BUG-010
- **文件路径**：`tests/ai/test_fc_v1_trainer_batch.py`
- **行号范围**：line 200-230（`test_train_batch_multiple_sample_dirs_merged`）
- **Bug 描述**：该测试重定向 `sys.stdout` 并断言"2个epoch均小于200，所以不应有任何Epoch输出"。这是一种**脆弱的白盒测试**：它依赖 `LOSS_LOG_INTERVAL = 200` 的内部常量，如果该常量被修改为 100，测试会虚假失败。测试应当验证功能行为（如损失下降、模型更新），而非输出日志的具体内容。
- **严重程度**：High
- **修复建议**：断言训练后的模型参数发生变化，或断言返回的 avg_loss 是合理数值，而非检查 stdout 内容。

### BUG-011
- **文件路径**：`cli/main.py`
- **行号范围**：整个文件（1-261 行）
- **Bug 描述**：**完全缺失测试**。CLI 模块包含完整的命令解析、游戏循环、棋盘渲染、存档管理等功能，但 `tests/` 下没有任何针对 `cli/main.py` 的测试。这意味着所有 CLI 命令（`new`、`put`、`list`、`print`、`show`、`short`）都未被自动化测试覆盖。
- **严重程度**：High
- **修复建议**：创建 `tests/cli/test_main.py`，使用 `unittest.mock.patch('builtins.input')` 模拟用户输入，测试各命令的解析和执行逻辑。

### BUG-012
- **文件路径**：`gui/main.py`
- **行号范围**：整个文件（1-456 行）
- **Bug 描述**：**完全缺失测试**。`GomokuGUI` 类是 GUI 主程序，协调所有组件交互（菜单、棋盘、状态显示、AI 对战调用等）。虽然组件有单独测试，但主程序的集成逻辑（如 `_on_board_click` 调用 helper 后更新显示、`_start_game_with_ai_first` 流程）完全没有被测试。
- **严重程度**：High
- **修复建议**：至少对 `GomokuGUI` 的业务逻辑进行单元测试，将 tkinter 相关调用 mock 掉，验证方法调用链和状态转换。

### BUG-013
- **文件路径**：`ai/self_play_cycle.py`
- **行号范围**：整个文件（1-220 行）
- **Bug 描述**：**完全缺失测试**。`SelfPlayCycle` 是训练周期的核心编排类，负责自我对弈、数据记录、评估、训练、模型保存的完整流程。该类没有任何单元测试或集成测试，是 AI 训练 pipeline 中风险最高的未覆盖区域。
- **严重程度**：High
- **修复建议**：创建 `tests/ai/test_self_play_cycle.py`，使用 mock 替换 `AIGameManager.start_ai_vs_ai`、`Evaluator.evaluate_round`、`trainer.train_batch` 等外部依赖，验证编排逻辑和参数传递。

### BUG-014
- **文件路径**：`ai/train_main.py`
- **行号范围**：整个文件（1-285 行）
- **Bug 描述**：**完全缺失测试**。命令行参数解析、训练器工厂 `get_trainer`、主训练流程都没有测试。`get_trainer` 中的模型名称字符串匹配（如 `"cnn_v2"`）一旦拼写错误不会被捕获。
- **严重程度**：High
- **修复建议**：使用 `unittest.mock.patch('sys.argv')` 测试参数解析；对 `get_trainer` 测试各模型名称的映射是否正确。

### BUG-015
- **文件路径**：`ai/line_tracer.py`
- **行号范围**：整个文件（1-143 行）
- **Bug 描述**：**完全缺失测试**。`LineTracer` 类负责五连追溯奖励计算，包含 `find_five_in_a_row`、`trace_winning_line`、`blend_rewards` 三个核心方法。 evaluator 的测试可能间接覆盖部分逻辑，但 `trace_winning_line` 中复杂的 history 匹配和反向传播衰减没有任何直接测试。
- **严重程度**：High
- **修复建议**：创建 `tests/ai/test_line_tracer.py`，构造已知五连的终局棋盘和 history，验证追溯到的步骤索引和奖励值是否符合衰减公式。

### BUG-016
- **文件路径**：`ai/reward_calculator.py`
- **行号范围**：整个文件（1-239 行）
- **Bug 描述**：**完全缺失测试**。`RewardCalculator` 负责将棋型映射为 reward 值，包含冲四递归防守模拟等复杂逻辑。`compute_position_reward` 的双向检测、`_compute_four_reward` 的递归模拟、`_find_four_defense` 的防守位置计算等核心算法均未被直接测试。
- **严重程度**：High
- **修复建议**：创建 `tests/ai/test_reward_calculator.py`，针对各棋型构造棋盘状态，验证 reward 计算的正确性；特别测试冲四递归路径。

### BUG-017
- **文件路径**：`ai/shape_detector.py`
- **行号范围**：整个文件（1-313 行）
- **Bug 描述**：**缺少直接单元测试**。虽然有 `test_evaluator_shape_detection.py` 间接覆盖，但 `detect_shapes`、`detect_single_direction`、Numba JIT 回退路径、棋型合并逻辑（多冲四/多活三）等没有独立测试。特别是 Numba 不可用时回退到纯 Python 的分支从未被测试。
- **严重程度**：High
- **修复建议**：创建 `tests/ai/test_shape_detector.py`，直接测试各函数；使用 `unittest.mock.patch('ai.shape_detector._NUMBA_AVAILABLE', False)` 强制测试纯 Python 回退路径。

---

## 三、Medium（测试质量差 / 覆盖不足 / 潜在问题）

### BUG-018
- **文件路径**：`tests/gui/test_*.py`（全部 8 个文件）
- **行号范围**：多处
- **Bug 描述**：大量 GUI 测试使用无意义断言如 `assert True`（`test_draw_pieces_empty` line 63）、`assert board.canvas is not None`。这些测试只验证了"函数没有抛出异常"，没有验证实际行为（如画布上绘制了正确数量的图形元素）。
- **严重程度**：Medium
- **修复建议**：使用 `canvas.find_all()` 验证图形对象数量；验证棋子颜色、坐标位置；使用 `tk.Event` 模拟点击后验证回调参数。

### BUG-019
- **文件路径**：`tests/gui/test_mode_selector.py`
- **行号范围**：line 45-83
- **Bug 描述**：`test_select_pve_black`、`test_select_pve_white` 等测试**没有真正调用被测方法**。它们直接给 `selector.result` 赋值，然后断言赋值正确。这完全没有测试 `_show_color_selection` 对话框的逻辑、颜色按钮回调、AI 选择联动等实际代码。
- **严重程度**：Medium
- **修复建议**：使用 mock 或直接调用内部方法（如 `_select_pvp` 被真实调用），而非直接操作 result 属性。

### BUG-020
- **文件路径**：`tests/gui/test_gomoku_gui.py`
- **行号范围**：line 40-80
- **Bug 描述**：测试中的 `GomokuGUIWrapper` 虽然验证了初始化，但 `test_gui_initialization` 仅断言 `gui.controller is not None`，没有验证 session、helper、display_updater 等关键协作对象的正确组装。
- **严重程度**：Medium
- **修复建议**：增加断言验证 `gui.session.mode`、`gui.helper.controller` 等组装关系。

### BUG-021
- **文件路径**：`tests/ai/test_fc_v1_model.py`、`tests/ai/test_cnn_model.py`
- **行号范围**：line 15-30
- **Bug 描述**：测试中修改类属性 `FCv1Model.DEFAULT_MODEL_PATH` 和 `CNNModel.DEFAULT_MODEL_PATH` 进行隔离。虽然这在单线程下有效，但如果并发执行或测试中断后恢复，类属性可能被永久污染，影响其他测试或后续运行。
- **严重程度**：Medium
- **修复建议**：使用 `unittest.mock.patch.object(FCv1Model, 'DEFAULT_MODEL_PATH', ...)` 作为上下文管理器，确保测试结束后自动恢复。

### BUG-022
- **文件路径**：`tests/ai/test_engine.py`
- **行号范围**：line 100-140
- **Bug 描述**：`test_engine_vs_itself` 让 AIEngine 自我对弈一局，但仅断言 `winner in [1, 2, 3]`。没有验证对局过程是否合法（如是否交替落子、是否有重复落子、棋盘状态是否正确演化）。
- **严重程度**：Medium
- **修复建议**：解析 `history` 验证落子交替、无重复位置；验证最终棋盘与 history 一致；验证获胜方确实形成了五连。

### BUG-023
- **文件路径**：`tests/ai/test_base_trainer.py`
- **行号范围**：line 50-100
- **Bug 描述**：`BaseTrainer` 的抽象方法验证测试仅检查 `NotImplementedError`，但没有测试 `_ensure_network_attribute` 被调用、lr_scheduler 的初始化逻辑、epoch 级日志记录等分支。
- **严重程度**：Medium
- **修复建议**：增加测试验证 `_ensure_network_attribute` 在 network=None 时抛出 AttributeError；测试各 lr_scheduler 类型的初始化。

### BUG-024
- **文件路径**：`tests/ai/test_battle.py`
- **行号范围**：line 200-250
- **Bug 描述**：`test_battle_session_run` 仅验证 `results` 长度和 winner 范围，没有验证对局质量（如是否有足够步数、是否交替落子、recorder 是否正确记录）。
- **严重程度**：Medium
- **修复建议**：验证 recorder 中 history 的完整性；检查每步的 input/output 格式是否符合 Evaluator 预期。

### BUG-025
- **文件路径**：`tests/app/test_gui_helper.py`
- **行号范围**：line 100-150
- **Bug 描述**：`test_get_display_state_no_game` 等测试直接操作 controller 的内部状态，而非通过公共 API。这导致测试与被测实现紧密耦合，一旦内部状态表示改变，测试会虚假失败。
- **严重程度**：Medium
- **修复建议**：通过 `controller.new_game()`、`controller.make_move()` 等公共 API 建立测试前置条件。

### BUG-026
- **文件路径**：`tests/test_game.py`
- **行号范围**：line 180-220
- **Bug 描述**：`test_state_isolation` 测试深拷贝保护，但仅验证了外部修改 `board_copy[0][0]` 不影响原状态。没有测试 `get_state()` 返回的 `history` 列表是否也是深拷贝（历史记录被外部修改可能导致内部状态污染）。
- **严重程度**：Medium
- **修复建议**：增加断言验证 `history` 和 `board` 均为深拷贝对象。

### BUG-027
- **文件路径**：`tests/ai/test_data_recorder.py`
- **行号范围**：line 100-130
- **Bug 描述**：`test_record_format` 仅检查 record.json 中存在 `history` 和 `winner` 字段，没有验证 `history` 中每个 step 的数据结构（是否包含 `position`、`input`、`output` 等字段）。
- **严重程度**：Medium
- **修复建议**：详细验证 step 的数据结构、input 向量长度（226）、output 向量长度（225）等。

### BUG-028
- **文件路径**：`ai/models/__init__.py`
- **行号范围**：line 1-19
- **Bug 描述**：`ai/models/__init__.py` 没有导出 `CNNv2Model`，但 `ai/models/cnn_v2/` 目录和模型文件存在，且测试直接导入 `ai.models.cnn_v2.model.CNNv2Model`。这导致包接口不完整，用户无法通过 `from ai.models import CNNv2Model` 使用该模型。
- **严重程度**：Medium
- **修复建议**：在 `ai/models/__init__.py` 中补充 `from .cnn_v2 import CNNv2Model` 并加入 `__all__`。

### BUG-029
- **文件路径**：`ai/models/cnn_v2/__init__.py`
- **行号范围**：整个文件（0 字节）
- **Bug 描述**：`cnn_v2/__init__.py` 为空文件，没有导出 `CNNv2Model` 和 `CNNTrainer`。这导致无法通过 `from ai.models.cnn_v2 import CNNv2Model` 导入。
- **严重程度**：Medium
- **修复建议**：在 `cnn_v2/__init__.py` 中添加 `from .model import CNNv2Model` 和 `from .trainer import CNNTrainer`。

### BUG-030
- **文件路径**：`tests/app/edge/test_session.py`
- **行号范围**：line 50-80
- **Bug 描述**：`test_make_player_move_no_game` 测试未创建游戏时的落子行为，但仅验证了返回 dict 中存在 `"error"` 键，没有验证 `need_ai_move` 是否为 False，也没有验证错误消息内容。
- **严重程度**：Medium
- **修复建议**：增加断言 `assert result["need_ai_move"] is False` 和错误消息的具体内容验证。

### BUG-031
- **文件路径**：`tests/app/primary/test_battle.py`
- **行号范围**：line 30-60
- **Bug 描述**：`test_battle_session_initialization` 仅验证 `session.num_games` 和 `session.ai1_name`，没有验证 `ai2_name`、`project_root` 等其它初始化参数是否正确存储；也没有验证 `get_recorder()` 初始状态。
- **严重程度**：Medium
- **修复建议**：增加对 `ai2_name`、`recorder` 初始状态的断言。

### BUG-032
- **文件路径**：`tests/ai/test_fc_v1_trainer.py`、`tests/ai/test_cnn_trainer.py`
- **行号范围**：多处
- **Bug 描述**：训练器测试普遍只验证"训练不报错"和"返回 float 类型损失"，没有验证：1) 训练后模型参数是否真正更新；2) 多次 epoch 后损失是否呈下降趋势；3) 对相同样本重复训练是否收敛。
- **严重程度**：Medium
- **修复建议**：记录训练前后的模型状态（如某层权重），使用 `torch.equal` 验证参数发生变化；验证损失值在合理范围内且随训练下降。

### BUG-033
- **文件路径**：`tests/ai/primary/test_mini_batch_training.py`
- **行号范围**：line 100-150
- **Bug 描述**：mini-batch 测试没有验证 `shuffle=True` 和 `shuffle=False` 的区别，没有验证 batch_size 大于样本数时的行为，也没有验证 `batch_size=1` 的边界情况。
- **严重程度**：Medium
- **修复建议**：增加 shuffle 效果验证（比较两次 epoch 的 batch 顺序）；测试 batch_size 边界条件。

### BUG-034
- **文件路径**：`tests/ai/test_fc_v1_inspect.py`
- **行号范围**：line 50-80
- **Bug 描述**：模型可解释性测试仅验证了输出形状，没有验证：1) 对合法位置赋予非零概率；2) 对已占位置概率为零（mask 正确应用）；3) 概率和严格等于 1。
- **严重程度**：Medium
- **修复建议**：增加对 mask 有效性、概率归一化、非法位置概率为零的断言。

### BUG-035
- **文件路径**：`gui/display_updater.py`
- **行号范围**：整个文件（1-111 行）
- **Bug 描述**：**缺少单元测试**。`GameDisplayUpdater` 负责所有界面刷新逻辑，但 `tests/gui/` 下没有对应的测试文件。虽然 `test_gomoku_gui.py` 间接涉及，但 `update_history`、`show_game_over` 等方法的独立逻辑未被验证。
- **严重程度**：Medium
- **修复建议**：创建 `tests/gui/test_display_updater.py`，mock tkinter 组件，验证各更新方法的行为。

---

## 四、Low（风格问题 / 轻微建议）

### BUG-036
- **文件路径**：`tests/run_tests.py`
- **行号范围**：line 30-50
- **Bug 描述**：自定义测试运行器通过 `unittest.TestLoader().discover('tests/ai', pattern='test_*.py')` 发现 AI 测试，但没有包含 `tests/ai/primary/` 和 `tests/ai/edge/` 子目录的显式发现逻辑。虽然 `discover` 默认递归，但如果目录结构变化可能导致测试遗漏。
- **严重程度**：Low
- **修复建议**：显式在 discover 中包含子目录，或直接使用 `pytest` 作为统一运行器，避免维护两套测试发现机制。

### BUG-037
- **文件路径**：`tests/ai/helpers.py`
- **行号范围**：line 1-73
- **Bug 描述**：`create_test_sample_file` 等辅助函数被多处使用，但本身没有测试。如果辅助函数出现 bug（如生成错误格式的 sample.json），会导致大量测试虚假通过或失败。
- **严重程度**：Low
- **修复建议**：为关键辅助函数添加简单单元测试，或至少使用 `jsonschema` 验证生成的数据结构。

### BUG-038
- **文件路径**：`tests/gui/test_replay_viewer.py`、`tests/gui/test_replay_browser.py`
- **行号范围**：多处
- **Bug 描述**：这些测试依赖临时 JSON 文件作为 fixture，但 JSON 文件内容是硬编码字符串。如果 `record.json` 或 `sample.json` 的格式在未来演进，这些测试数据会过时。
- **严重程度**：Low
- **修复建议**：使用工厂函数动态生成 fixture 数据，确保数据结构始终与当前代码期望一致。

### BUG-039
- **文件路径**：`tests/ai/test_fc_v1_softmax_input.py`
- **行号范围**：line 30-60
- **Bug 描述**：文件名暗示测试"softmax input"，但代码实际测试的是模型预测输出的概率分布特性。测试名称与内容不完全匹配，容易造成维护困惑。
- **严重程度**：Low
- **修复建议**：将文件重命名为 `test_fc_v1_predict_distribution.py`，使文件名准确反映测试内容。

### BUG-040
- **文件路径**：`app/session.py`
- **行号范围**：line 100-130
- **Bug 描述**：`do_ai_move` 方法在 `controller.game is None` 时返回 `{"status": "error", ...}`，但没有测试覆盖这个分支。类似地，`make_player_move` 中 `controller.game is None` 的分支也未被 app 层测试直接覆盖。
- **严重程度**：Low
- **修复建议**：在 `tests/app/primary/test_session.py` 中增加对 `do_ai_move` 无游戏状态的测试。

### BUG-041
- **文件路径**：`ai/probability_utils.py`、`ai/sample_loader.py`
- **行号范围**：整个文件
- **Bug 描述**：这两个工具模块缺少直接测试。`masked_softmax` 的数值稳定性处理、`SampleLoader.load_multiple` 的合并逻辑、`prepare_tensors` 的设备迁移等都没有独立验证。
- **严重程度**：Low
- **修复建议**：创建 `tests/ai/test_probability_utils.py` 和 `tests/ai/test_sample_loader.py`，针对边界条件（如全 mask=0、空文件列表）进行测试。

### BUG-042
- **文件路径**：`tests/test_game.py`
- **行号范围**：line 100-130
- **Bug 描述**：胜利条件测试（横/纵/对角）只验证了胜利方检测，没有验证获胜后立即拒绝后续落子（`make_move` 在游戏结束后应返回错误或拒绝）。
- **严重程度**：Low
- **修复建议**：增加测试：在已获胜的棋盘上再次落子，验证返回 `"error": "游戏已结束"` 或类似提示。

---

## 五、总结统计

| 严重程度 | 数量 | 主要类别 |
|---------|------|---------|
| Critical | 3 | 跳过/忽略、危险异常处理 |
| High | 14 | 完全缺失测试、测试隔离、脆弱测试 |
| Medium | 18 | 断言不足、Mock问题、覆盖盲区 |
| Low | 7 | 风格、命名、边界分支缺失 |
| **总计** | **42** | |

### 最需要优先修复的 Top 5

1. **BUG-002**：`test_zero_epochs` 的危险 `except ZeroDivisionError: pass` 模式 —— 立即修复，避免测试永远通过。
2. **BUG-001**：GUI 测试在无头环境中全部静默跳过 —— 修复 teardown 逻辑，区分真正不可用和测试失败。
3. **BUG-003**：`test_draw` 被长期跳过 —— 实现 Controller 平局测试。
4. **BUG-004**：`time.sleep(1.1)` 脆弱测试 —— 改为 mock 时间。
5. **BUG-005 / BUG-006 / BUG-007**：多处测试隔离问题 —— 统一使用 `tempfile.mkdtemp()` 和 `shutil.rmtree()`。
