# GUI 与 Session 职责分离重构设计文档

## 1. 问题描述

当前 `gui/main.py` 中混杂了游戏流程控制逻辑，违反了分层架构原则：
- GUI 层应该只负责界面展示和事件转发
- 游戏流程控制应该由 `app/session.py` 负责

## 2. 目标架构

```
┌─────────────────────────────────────┐
│           gui/main.py               │
│  - 界面创建与展示                    │
│  - 事件捕获（点击、按钮）            │
│  - 调用 session 方法执行业务         │
│  - 根据 session 返回结果更新界面     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         app/session.py              │
│  - 游戏流程控制（新建、落子、AI触发） │
│  - 协调 controller 和 ai_manager    │
│  - 返回操作结果和下一步指示          │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│   game/     │  │    ai/      │
│ controller  │  │  manager    │
└─────────────┘  └─────────────┘
```

## 3. 需要迁移的逻辑

### 3.1 AI 先走逻辑（gui/main.py 第 72-74 行）

**当前代码：**
```python
# 如果是人机对战且 AI 执黑，自动开始
if game_mode == "pve" and ai_player == 1:
    self.root.after(500, self._ai_move)
```

**问题：** 游戏尚未创建就调用 `_ai_move()`，导致 AI 无法落子

**重构方案：**
- 在 `GameSession` 中添加 `start_game()` 方法
- 该方法负责创建游戏，并在 AI 先走时执行 AI 落子
- GUI 只需调用 `session.start_game()`，无需关心内部流程

### 3.2 AI 自动触发逻辑（gui/main.py 第 229-230 行）

**当前代码：**
```python
if result["need_ai_move"]:
    self.root.after(500, self._ai_move)
```

**分析：** 这部分逻辑已经通过 `need_ai_move` 标志由 session 控制，GUI 只是根据标志决定是否调度 AI 落子。这部分可以保留在 GUI 中，因为涉及 UI 调度（`root.after`）。

### 3.3 AI 落子执行（gui/main.py 第 232-248 行）

**当前代码：**
```python
def _ai_move(self):
    """AI 落子"""
    if self.controller.game is None or self.controller.is_over():
        return

    result = self.session.do_ai_move()
    # ... 更新界面
```

**分析：** 这个检查 `self.controller.game is None` 的逻辑应该下沉到 session 中，GUI 应该直接调用 `session.do_ai_move()` 并处理返回结果。

## 4. 接口设计

### 4.1 GameSession 新增/修改接口

```python
class GameSession:
    def start_game(self, save_name: str = None) -> dict:
        """
        启动游戏
        
        - 创建游戏
        - 如果是 PvE 且 AI 执黑，自动执行 AI 落子
        
        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "ai_moved": bool,      # AI 是否已落子（AI先走时）
                "game_over": bool,     # 游戏是否结束
                "winner": int          # 仅 game_over=True 时存在
            }
        """
    
    def do_ai_move(self) -> dict:
        """
        执行 AI 落子（已存在，需增强返回值）
        
        Returns:
            {
                "status": "success" | "error" | "skipped",
                "error": str,          # 仅 error 时存在
                "game_over": bool,
                "winner": int          # 仅 game_over=True 时存在
            }
        """
```

### 4.2 GomokuGUI 修改

```python
class GomokuGUI:
    def __init__(self, root, game_mode="pvp", ai_player=None, ai_model="random"):
        # ... 初始化代码 ...
        
        # 如果是人机对战且 AI 执黑，启动游戏（由 session 处理 AI 落子）
        if game_mode == "pve" and ai_player == 1:
            self.root.after(500, self._start_game_with_ai_first)
    
    def _start_game_with_ai_first(self):
        """AI 先走：启动游戏并更新界面"""
        result = self.session.start_game()
        if result["status"] == "success":
            self._update_board()
            if result.get("game_over"):
                # 显示游戏结束
                pass
    
    def _new_game(self):
        """新建游戏"""
        # 获取存档名称...
        result = self.session.start_game(save_name)
        # 更新界面...
```

## 5. 主流程伪代码

### 5.1 AI 先走流程

```python
# GUI 层
if game_mode == "pve" and ai_player == 1:
    self.root.after(500, self._start_game_with_ai_first)

def _start_game_with_ai_first(self):
    result = self.session.start_game()
    if result["status"] == "success":
        self._update_board()
        if result.get("game_over"):
            self._show_game_over(result["winner"])

# Session 层
def start_game(self, save_name=None):
    # 创建游戏
    result = self._controller.new_game(save_name)
    if "error" in result:
        return {"status": "error", "error": result["error"]}
    
    # 如果 AI 执黑，执行 AI 落子
    if self._mode == "pve" and self._ai_player == 1:
        ai_result = self._ai_manager.ai_move()
        return {
            "status": "success",
            "ai_moved": True,
            "game_over": self._controller.is_over(),
            "winner": self._controller.get_winner() if self._controller.is_over() else None
        }
    
    return {"status": "success", "ai_moved": False, "game_over": False}
```

### 5.2 玩家落子后 AI 触发流程

```python
# GUI 层（保持不变）
def _on_board_click(self, row, col):
    result = self.session.make_player_move(row, col)
    if result["status"] == "success":
        self._update_board()
        if result["game_over"]:
            self._show_game_over(result["winner"])
            return
        if result["need_ai_move"]:
            self.root.after(500, self._ai_move)

def _ai_move(self):
    result = self.session.do_ai_move()
    if result["status"] == "success":
        self._update_board()
        if result["game_over"]:
            self._show_game_over(result["winner"])

# Session 层（do_ai_move 已存在，逻辑正确）
```

## 6. 主流程测试用例

### 6.1 AI 先走测试

```python
def test_ai_first_move_starts_game_and_makes_move():
    """AI 先走时，start_game 应创建游戏并执行 AI 落子"""
    session = GameSession(mode="pve", controller=controller, ai_player=1)
    result = session.start_game()
    
    assert result["status"] == "success"
    assert result["ai_moved"] == True
    # 验证棋盘上有一个黑子
    black_count = count_black_pieces(controller.get_state()["board"])
    assert black_count == 1
```

### 6.2 玩家先走测试

```python
def test_player_first_move_starts_game_without_ai_move():
    """玩家先走时，start_game 只创建游戏，不执行 AI 落子"""
    session = GameSession(mode="pve", controller=controller, ai_player=2)
    result = session.start_game()
    
    assert result["status"] == "success"
    assert result["ai_moved"] == False
    # 验证棋盘为空
    black_count = count_black_pieces(controller.get_state()["board"])
    assert black_count == 0
```

### 6.3 PvP 模式测试

```python
def test_pvp_start_game_creates_game():
    """PvP 模式 start_game 只创建游戏"""
    session = GameSession(mode="pvp", controller=controller)
    result = session.start_game()
    
    assert result["status"] == "success"
    assert controller.game is not None
```

## 7. 边缘测试用例

### 7.1 存档名称冲突

```python
def test_start_game_with_duplicate_name_returns_error():
    """存档名称已存在时应返回错误"""
    controller.new_game("test_save")
    session = GameSession(mode="pvp", controller=controller)
    result = session.start_game("test_save")
    
    assert result["status"] == "error"
```

### 7.2 AI 落子前游戏结束

```python
def test_ai_move_when_game_over_returns_skipped():
    """游戏已结束时 do_ai_move 应返回 skipped"""
    # 创建已结束的游戏...
    result = session.do_ai_move()
    assert result["status"] == "skipped"
```

## 8. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `app/session.py` | 修改 | 添加 `start_game()` 方法，增强 `do_ai_move()` 返回值 |
| `gui/main.py` | 修改 | 使用 `session.start_game()`，简化 `_ai_move()` 调用 |
| `tests/app/test_session.py` | 新增/修改 | 添加 `start_game` 和重构后的测试用例 |
| `tests/gui/test_gomoku_gui.py` | 修改 | 更新测试以适配新的接口 |

## 9. 兼容性说明

- `make_player_move` 和 `do_ai_move` 的返回值保持不变，向后兼容
- 新增 `start_game` 方法，不影响现有代码
- GUI 的 `_on_board_click` 逻辑保持不变
