# 整合 GUI 测试到 tests/gui/ 目录

## 需求内容

用户发现项目中存在两套 GUI 测试文件：
- `gui/test_*.py`：旧版测试（包含已修复的 grab_set 问题和边界坐标修复）
- `tests/gui/test_*.py`：新建测试（内容较少）

用户选择方案 B：以 `tests/gui/` 为准，将旧版内容合并进去，修改 `run_tests.py` 导入路径，删除 `gui/` 下的测试文件。

## 解决思路

1. 将 `gui/` 下六个旧测试文件的内容（更丰富、已修复）迁移到 `tests/gui/` 对应文件
2. 修正 `project_root` 路径计算（从两级上跳改为三级上跳）
3. 修改 `run_tests.py` 顶层导入路径：`gui.test_*` → `tests.gui.test_*`
4. 运行测试验证全部通过
5. 删除 `gui/` 下六个旧测试文件

## 执行过程

### 步骤 1：确认当前文件状态

读取 `tests/gui/` 下所有文件，发现 `test_replay_browser.py` 在上一会话中已经是正确版本（包含 `_create_selector` 工厂方法，使用 `__new__` 绕过 `grab_set()`）。

### 步骤 2：修改 run_tests.py 导入路径

将 `run_tests.py` 顶部的导入：
```python
from gui.test_mode_selector import TestGameModeSelector
# ...
```
改为：
```python
from tests.gui.test_mode_selector import TestGameModeSelector
# ...
```

### 步骤 3：运行测试

```
测试结果: 129/129 通过
SUCCESS: 所有测试通过！
```

### 步骤 4：删除旧测试文件

删除以下 6 个文件：
- `gui/test_mode_selector.py`
- `gui/test_game_board.py`
- `gui/test_heatmap.py`
- `gui/test_replay_viewer.py`
- `gui/test_replay_browser.py`
- `gui/test_ai_battle.py`

## 涉及文件

| 文件 | 操作 |
|------|------|
| `tests/run_tests.py` | 修改导入路径 `gui.` → `tests.gui.` |
| `tests/gui/test_mode_selector.py` | 已在上一会话更新（旧版内容） |
| `tests/gui/test_game_board.py` | 已在上一会话更新（旧版内容，保留边界修复） |
| `tests/gui/test_heatmap.py` | 已在上一会话更新（旧版内容） |
| `tests/gui/test_replay_viewer.py` | 已在上一会话更新（旧版内容，含详细断言） |
| `tests/gui/test_replay_browser.py` | 已在上一会话更新（含 `_create_selector` 工厂方法） |
| `tests/gui/test_ai_battle.py` | 保持用户修改版本不变 |
| `gui/test_mode_selector.py` | 删除 |
| `gui/test_game_board.py` | 删除 |
| `gui/test_heatmap.py` | 删除 |
| `gui/test_replay_viewer.py` | 删除 |
| `gui/test_replay_browser.py` | 删除 |
| `gui/test_ai_battle.py` | 删除 |
