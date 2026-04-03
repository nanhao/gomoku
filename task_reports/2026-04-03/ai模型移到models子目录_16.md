# 任务报告：AI 模型目录重组到 models 子目录

## 1. 需求内容

在 `ai` 目录下创建子目录 `models`，然后把 `ai/random` 和 `ai/fc_v1` 放到 `models` 目录下。做完后回归全体测试用例。

## 2. 解决思路

这是纯目录结构调整，步骤如下：
1. 创建 `ai/models/` 目录
2. 将 `ai/random/` 和 `ai/fc_v1/` 移动到 `ai/models/` 下
3. 创建 `ai/models/__init__.py`，导出所有模型
4. 全量扫描并更新所有 import 路径（含 GUI 扫描目录逻辑、测试文件）
5. 回归全量测试

## 3. 执行过程

### 步骤 1-3：移动目录并创建 __init__.py

用 PowerShell `Move-Item` 命令移动两个子目录到 `ai/models/`，并新建 `ai/models/__init__.py` 导出 `RandomModel`、`FCv1Model`、`FCv1Trainer`。

### 步骤 4：更新 import 路径

通过全局搜索 `from ai.(random|fc_v1)` 找出所有引用点，逐一更新：

| 文件 | 旧路径 | 新路径 |
|------|--------|--------|
| `ai/engine.py` | `from ai.random` / `from ai.fc_v1` | `from ai.models.random` / `from ai.models.fc_v1` |
| `ai/battle.py` | 字符串 `"ai.random.model.*"` | `"ai.models.random.model.*"` |
| `ai/models/fc_v1/train.py` | `from ai.fc_v1.trainer` + 路径层级 | `from ai.models.fc_v1.trainer` + 多一层 dirname |
| `ai/models/fc_v1/inspect_main.py` | `from ai.fc_v1.*` + 路径层级 | `from ai.models.fc_v1.*` + 多一层 dirname |
| `tests/app/primary/test_session.py` | `from ai.random/fc_v1` | `from ai.models.random/fc_v1` |
| `tests/ai/test_fc_v1_inspect.py` | `from ai.fc_v1.*` | `from ai.models.fc_v1.*` |
| `tests/ai/test_engine.py` | `from ai.random.model` | `from ai.models.random.model` |
| `tests/ai/test_battle.py` | `from ai.random.model` | `from ai.models.random.model` |
| `tests/ai/test_fc_v1_trainer.py` | `from ai.fc_v1.*` | `from ai.models.fc_v1.*` |
| `tests/ai/test_fc_v1_model.py` | `from ai.fc_v1.model` | `from ai.models.fc_v1.model` |
| `tests/ai/test_game_manager.py` | `from ai.random.model` | `from ai.models.random.model` |
| `tests/gui/test_gomoku_gui.py` | `from ai.random` | `from ai.models.random` |
| `tests/gui/test_ai_battle.py` | `from ai.random` | `from ai.models.random` |

### 遇到的问题及解决

**问题**：第一轮回归出现 1 个失败 `test_get_available_ais`。

**原因**：`gui/components/mode_selector.py` 和 `gui/components/ai_battle.py` 中的 `_get_available_ais()` 方法通过扫描 `ai/` 目录下子目录来枚举可用 AI，移动后 `ai/` 下已无 `random`/`fc_v1` 子目录。

**解决**：将两处扫描目录从 `ai/` 改为 `ai/models/`，同步更新测试文件中的验证逻辑。

## 4. 执行结果

全量测试 **158/158 全部通过**。

## 5. 涉及文件

**新增文件：**
- `ai/models/__init__.py`

**移动（目录）：**
- `ai/random/` → `ai/models/random/`
- `ai/fc_v1/` → `ai/models/fc_v1/`

**修改文件：**
- `ai/engine.py`
- `ai/battle.py`
- `ai/models/fc_v1/train.py`
- `ai/models/fc_v1/inspect_main.py`
- `gui/components/mode_selector.py`
- `gui/components/ai_battle.py`
- `tests/app/primary/test_session.py`
- `tests/ai/test_fc_v1_inspect.py`
- `tests/ai/test_engine.py`
- `tests/ai/test_battle.py`
- `tests/ai/test_fc_v1_trainer.py`
- `tests/ai/test_fc_v1_model.py`
- `tests/ai/test_game_manager.py`
- `tests/gui/test_gomoku_gui.py`
- `tests/gui/test_ai_battle.py`
- `tests/gui/test_mode_selector.py`
