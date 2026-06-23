# app ↔ gui 循环依赖分析与解决方案

## 1. 问题描述

`app` 包与 `gui` 包之间存在循环依赖，违反了架构设计中 `gui → app → ai/game` 的单向依赖原则。

根据 [app_design.md](../app_design.md)，`app` 包的定位是位于 `ai`/`game` 包之上、`gui` 包之下的薄代理层。`app` 不应依赖 `gui`，但目前存在反向依赖。

## 2. 依赖关系分析

### 2.1 当前依赖图

![依赖关系图](dependency_graph.png)

详见：[dependency_graph.puml](dependency_graph.puml)

### 2.2 正向依赖（gui → app，符合架构）

| 文件 | 导入内容 | 用途 |
|------|---------|------|
| `gui/main.py` | `from app import GameSession, GUIHelper, get_ai_data_dir, create_pve_config` | PvE/PvP 游戏流程 |
| `gui/main.py` | `from app.exceptions import ModelLoadError` | 捕获模型加载异常并弹窗 |
| `gui/components/ai_battle.py` | `from app import BattleSession` | AI 内战流程 |
| `gui/components/ai_battle.py` | `from app.exceptions import ModelLoadError` | 捕获模型加载异常并弹窗 |

### 2.3 反向依赖（app → gui，违反架构循环依赖根因）

| 文件 | 导入内容 | 用途 |
|------|---------|------|
| `app/gui_helper.py` | `from gui.components.constants import ANIMALS` | `GUIHelper.get_player_name()` 格式化玩家名称 |

**这是 `app` → `gui` 唯一的导入**，导致了整个循环依赖。

### 2.4 ANIMALS 常量分析

```python
# gui/components/constants.py
ANIMALS = {
    "black": "🐻",     # 小熊 emoji
    "white": "🐰",     # 小兔 emoji
    "last_move": "🐾", # 脚印 emoji
}
```

`ANIMALS` 是纯 UI 展示常量（Unicode 表情符号），用于界面显示。在 `app/gui_helper.py` 中的使用：

```python
# app/gui_helper.py 第 160-163 行
def get_player_name(self, player: int, winner: int = 0) -> str:
    if player == 1:
        return f"小熊 {ANIMALS['black']}"   # → "小熊 🐻"
    else:
        return f"小兔 {ANIMALS['white']}"   # → "小兔 🐰"
```

### 2.5 app 包内其他文件无 gui 依赖

确认 `app/session.py`、`app/battle.py`、`app/exceptions.py` 均无任何 `gui` 包的导入，只有 `gui_helper.py` 引入了循环依赖。

## 3. 解决方案

### 3.1 方案对比

| 方案 | 描述 | 改动量 | 架构纯净度 | 推荐 |
|------|------|--------|-----------|------|
| **A: 构造函数注入** | GUIHelper 通过构造函数接收 animal 字符串 | 小 | ★★★ | ✅ |
| B: 移除 ANIMALS 硬编码 | GUIHelper 直接硬编码 emoji 字符 | 最小 | ★★☆ | 备选 |
| C: 新建共享常量模块 | 提取 ANIMALS 到 src 级别共享模块 | 中 | ★★☆ | ❌ |

### 3.2 推荐方案：构造函数注入（方案 A）

**核心思路**：让 `GUIHelper` 不再直接导入 `ANIMALS`，而是通过构造函数从 GUI 层接收展示用的动物表情字符串。默认值保持向后兼容。

**修改点：**

#### ① `app/gui_helper.py`

- **删除**：`from gui.components.constants import ANIMALS`（第 9 行）
- **修改**：`GUIHelper.__init__` 构造函数新增 `black_animal` 和 `white_animal` 参数
- **修改**：`get_player_name` 使用实例属性替代 `ANIMALS` 常量

```python
# 修改前
class GUIHelper:
    def __init__(self, controller, session):
        self.controller = controller
        self.session = session

    def get_player_name(self, player, winner=0):
        if player == 1:
            return f"小熊 {ANIMALS['black']}"
        else:
            return f"小兔 {ANIMALS['white']}"

# 修改后
class GUIHelper:
    def __init__(self, controller, session,
                 black_animal: str = "🐻",
                 white_animal: str = "🐰"):
        self.controller = controller
        self.session = session
        self._black_animal = black_animal
        self._white_animal = white_animal

    def get_player_name(self, player, winner=0):
        if player == 1:
            return f"小熊 {self._black_animal}"
        else:
            return f"小兔 {self._white_animal}"
```

#### ② `gui/main.py`

- **修改**：创建 `GUIHelper` 时传入 `ANIMALS` 中的值

```python
# 修改前（第 69 行）
self.helper = GUIHelper(self.controller, self.session)

# 修改后
self.helper = GUIHelper(
    self.controller, self.session,
    black_animal=ANIMALS["black"],
    white_animal=ANIMALS["white"],
)
```

### 3.3 修改后依赖关系

```
┌─────────────────────────────────────────┐
│              gui 包                      │  上层（纯 UI）
│  持有 ANIMALS 常量                       │
│  创建 GUIHelper 时传入展示字符串           │
└────────────────┬────────────────────────┘
                 │ 单向依赖（无反向）
┌────────────────▼────────────────────────┐
│              app 包                      │  应用层（薄代理）
│  GUIHelper 通过构造函数接收展示参数       │
│  不再导入 gui.* 任何内容                  │
└──────┬───────────────────┬─────────────┘
       │                   │
┌──────▼──────┐    ┌───────▼─────────────┐
│   ai 包     │    │     game 包          │
└─────────────┘    └─────────────────────┘
```

循环依赖完全消除。

## 4. 影响范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `app/gui_helper.py` | 修改 | 删除 `from gui...import ANIMALS`；构造函数新增 2 个可选参数 |
| `gui/main.py` | 修改 | 创建 `GUIHelper` 时传入 animal 参数 |

## 5. 验证方法

修改完成后，运行以下命令验证循环依赖已消除：

```bash
# 检查 app 包中不再有 gui 的导入
rg "from gui|import gui" src/app/

# 应返回空结果
```

同时运行现有测试确保无回归：

```bash
python -m pytest tests/ -x -q
```

## 6. 设计决策说明

**为什么不选择方案 B（硬编码）**：虽然硬编码更简单，但 `GUIHelper` 保留参数化能力更符合开闭原则——未来若 `ANIMALS` 发生变化（如支持主题切换、i18n），无需修改 `app` 层代码。

**为什么不选择方案 C（共享模块）**：`ANIMALS` 常量是纯 UI 层关注点，不应提升到项目级共享模块。这会破坏模块边界和职责分离。
