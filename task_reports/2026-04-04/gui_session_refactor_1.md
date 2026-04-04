# GUI 与 Session 职责分离重构任务报告

## 1. 需求内容

将 `gui/main.py` 中涉及游戏流程控制的代码迁移到 `app/session.py`，使 GUI 层只负责界面展示，游戏流程控制由 Session 层负责。

**触发原因**：修复 "人机对战模式下，如果选择让 AI 先走，第一步 AI 不会动" 的 Bug 时，发现 GUI 层混杂了游戏流程控制逻辑，违反了分层架构原则。

## 2. 解决思路

### 架构问题分析
- **问题**：`gui/main.py` 中直接调用 `controller.new_game()` 和 `controller.game` 判断，GUI 层越界处理了业务逻辑
- **目标**：GUI 只负责界面事件转发和展示更新，所有游戏流程控制由 `GameSession` 统一管理

### 重构方案
1. 在 `GameSession` 中新增 `start_game()` 方法，统一处理游戏创建和 AI 先走逻辑
2. 修改 `GomokuGUI`，通过 `session.start_game()` 和 `session.do_ai_move()` 与业务层交互
3. 移除 GUI 中对 `controller.game` 的直接判断

## 3. 执行过程

### 步骤 1：详细设计（设计文档）
- 创建 `design/gui_session_refactor_design.md`
- 定义目标架构、接口设计、主流程伪代码
- 设计主流程测试用例和边缘测试用例

### 步骤 2：编写测试用例
- **主流程测试** (`tests/app/primary/test_session.py`)：
  - `test_pvp_start_game_creates_game` - PvP 模式创建游戏
  - `test_pve_ai_first_start_game_creates_and_ai_moves` - AI 先走创建游戏并落子
  - `test_pve_player_first_start_game_only_creates` - 玩家先走只创建游戏
  - `test_start_game_with_auto_generated_name` - 自动生成存档名

- **边缘测试** (`tests/app/edge/test_session.py`)：
  - `test_start_game_with_duplicate_name_returns_error` - 存档名冲突
  - `test_do_ai_move_when_game_not_started_raises_error` - 游戏未创建调用 AI 落子
  - `test_start_game_twice_returns_error` - 重复启动游戏

### 步骤 3：编码实现

**修改 `app/session.py`**：
- 新增 `start_game(save_name)` 方法：
  - 调用 `controller.new_game()` 创建游戏
  - 如果 AI 执黑，自动执行 AI 落子
  - 返回统一格式的结果字典
- 增强 `do_ai_move()`：添加游戏未创建的检查

**修改 `gui/main.py`**：
- 新增 `_start_game_with_ai_first()` 方法：调用 `session.start_game()` 并更新界面
- 修改 `__init__`：AI 先走时调用 `_start_game_with_ai_first`
- 修改 `_new_game()`：使用 `session.start_game()` 替代 `controller.new_game()`
- 简化 `_ai_move()`：移除 `controller.game is None` 检查，由 Session 层处理

### 步骤 4：跑通全量测试
- 245 个测试通过，5 个子测试通过
- 修复了测试间存档名称冲突的问题

## 4. 执行结果

### 架构改进
- **职责分离**：GUI 只负责界面，Session 负责游戏流程控制
- **修复 Bug**：AI 先走时，现在会先创建游戏再执行 AI 落子
- **接口统一**：所有游戏操作通过 `GameSession` 的方法完成

### 测试覆盖
- 新增 7 个测试用例（4 主流程 + 3 边缘）
- 全量 245 个测试通过

### 文件变更
| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `app/session.py` | 修改 | 新增 `start_game()` 方法，增强 `do_ai_move()` |
| `gui/main.py` | 修改 | 使用 `session.start_game()`，简化 `_ai_move()` |
| `tests/app/primary/test_session.py` | 修改 | 新增 `start_game` 主流程测试 |
| `tests/app/edge/test_session.py` | 修改 | 新增 `start_game` 边缘测试 |
| `design/gui_session_refactor_design.md` | 新增 | 设计文档 |

## 5. 涉及文件

### 修改的文件
1. `d:/projects/gomoku/app/session.py`
2. `d:/projects/gomoku/gui/main.py`
3. `d:/projects/gomoku/tests/app/primary/test_session.py`
4. `d:/projects/gomoku/tests/app/edge/test_session.py`

### 新增的文件
1. `d:/projects/gomoku/design/gui_session_refactor_design.md`
2. `d:/projects/gomoku/task_reports/2026-04-04/gui_session_refactor_1.md`

## 6. 后续工作

重构完成后，可以顺利修复原 Bug：
- AI 先走时，`session.start_game()` 会先创建游戏
- 然后自动执行 AI 落子
- GUI 只需根据返回结果更新界面
