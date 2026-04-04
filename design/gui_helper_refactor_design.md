# GUI Helper 重构设计文档

## 1. 需求内容

将 `gui/main.py` 中与界面无关的代码迁移到 `app/gui_helper.py`，使 GUI 层只保留界面相关代码，提升整体测试覆盖率到 95% 以上。

## 2. 目标架构

```
app/
├── __init__.py
├── battle.py          # 已存在：AI 内战业务逻辑
├── session.py         # 已存在：游戏会话管理
└── gui_helper.py      # 新增：GUI 辅助函数（无界面依赖）

gui/
├── main.py            # 修改后：只保留界面代码
└── components/        # 已存在：UI 组件
```

## 3. 接口设计

### 3.1 GUIHelper 类

```python
class GUIHelper:
    """GUI 辅助类 - 处理与界面无关的业务逻辑"""
    
    def __init__(self, controller, session):
        self.controller = controller
        self.session = session
    
    def start_game_with_ai_first(self) -> dict:
        """AI 先走时启动游戏
        
        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "ai_moved": bool,
                "game_over": bool,
                "winner": int          # 仅 game_over=True 时存在
            }
        """
    
    def handle_player_move(self, row, col) -> dict:
        """处理玩家落子
        
        Returns:
            {
                "status": "success" | "error",
                "error": str,          # 仅 error 时存在
                "game_over": bool,
                "winner": int,         # 仅 game_over=True 时存在
                "need_ai_move": bool   # 是否需要 AI 落子
            }
        """
    
    def handle_ai_move(self) -> dict:
        """处理 AI 落子
        
        Returns:
            {
                "status": "success" | "error",
                "game_over": bool,
                "winner": int  # 仅 game_over=True 时存在
            }
        """
    
    def get_display_state(self) -> dict:
        """获取显示状态
        
        Returns:
            {
                "has_game": bool,
                "save_name": str,
                "current_player": int,  # 1=黑, 2=白
                "winner": int,          # 0=未结束, 1=黑胜, 2=白胜, 3=平局
                "board": List[List[int]],
                "last_move": Tuple[int, int] or None,
                "history": List[dict]
            }
        """
    
    def get_winner_text(self, winner: int) -> str:
        """获取获胜方文本"""
    
    def get_player_name(self, player: int, winner: int = 0) -> str:
        """获取玩家名称（含动物主题）"""
```

### 3.2 独立函数

```python
def get_ai_data_dir() -> str:
    """获取 ai_data 目录路径"""

def create_pve_config(selector_result: dict) -> dict:
    """根据选择器结果创建 PvE 配置
    
    Args:
        selector_result: GameModeSelector 返回的结果
        
    Returns:
        {
            "game_mode": "pve",
            "ai_player": int,  # 1=AI执黑, 2=AI执白
            "ai_model": str
        }
    """

def run_ai_vs_ai_mode(num_games: int, ai1_name: str, ai2_name: str, 
                      on_progress=None, on_complete=None):
    """运行 AI 内战模式（不含 GUI 弹窗）
    
    Args:
        num_games: 对战局数
        ai1_name: AI 1 名称
        ai2_name: AI 2 名称
        on_progress: 进度回调函数 (current, total) -> None
        on_complete: 完成回调函数 (results) -> None
    """
```

## 4. 主流程伪代码

### 4.1 AI 先走流程

```python
# gui/main.py
def _start_game_with_ai_first(self):
    result = self.helper.start_game_with_ai_first()
    if result["status"] == "error":
        messagebox.showerror("错误", result.get("error"))
        return
    self._update_board()
    if result.get("game_over"):
        self._show_game_over(result["winner"])

# app/gui_helper.py
def start_game_with_ai_first(self):
    result = self.session.start_game()
    return {
        "status": result["status"],
        "error": result.get("error"),
        "ai_moved": result.get("ai_moved"),
        "game_over": result.get("game_over"),
        "winner": result.get("winner")
    }
```

### 4.2 玩家落子流程

```python
# gui/main.py
def _on_board_click(self, row, col):
    result = self.helper.handle_player_move(row, col)
    if result["status"] == "error":
        messagebox.showinfo("提示", result["error"])
        return
    self._update_board()
    if result["game_over"]:
        self._show_game_over(result["winner"])
        return
    if result["need_ai_move"]:
        self.root.after(500, self._ai_move)

# app/gui_helper.py
def handle_player_move(self, row, col):
    result = self.session.make_player_move(row, col)
    return {
        "status": result["status"],
        "error": result.get("error"),
        "game_over": result["game_over"],
        "winner": result.get("winner"),
        "need_ai_move": result.get("need_ai_move", False)
    }
```

## 5. 主流程测试用例

### 5.1 GUIHelper 主流程测试

```python
def test_start_game_with_ai_first_creates_game_and_moves():
    """AI 先走时启动游戏应创建游戏并执行 AI 落子"""
    helper = GUIHelper(controller, session)
    result = helper.start_game_with_ai_first()
    assert result["status"] == "success"
    assert result["ai_moved"] is True
    assert controller.game is not None

def test_handle_player_move_success():
    """处理玩家落子成功"""
    helper = GUIHelper(controller, session)
    helper.start_game_with_ai_first()  # 创建游戏
    result = helper.handle_player_move(7, 7)
    assert result["status"] == "success"
    assert result["need_ai_move"] is True

def test_handle_ai_move_success():
    """处理 AI 落子成功"""
    helper = GUIHelper(controller, session)
    helper.start_game_with_ai_first()
    result = helper.handle_ai_move()
    assert result["status"] == "success"

def test_get_display_state_with_game():
    """获取有游戏时的显示状态"""
    helper = GUIHelper(controller, session)
    helper.start_game_with_ai_first()
    state = helper.get_display_state()
    assert state["has_game"] is True
    assert state["board"] is not None

def test_get_winner_text():
    """获取获胜方文本"""
    helper = GUIHelper(controller, session)
    assert "黑方" in helper.get_winner_text(1)
    assert "白方" in helper.get_winner_text(2)
    assert "平局" in helper.get_winner_text(3)
```

### 5.2 独立函数测试

```python
def test_get_ai_data_dir():
    """获取 ai_data 目录路径"""
    path = get_ai_data_dir()
    assert "ai_data" in path

def test_create_pve_config_ai_black():
    """创建 PvE 配置，AI 执黑"""
    selector_result = {"player_color": "white", "ai": "random"}
    config = create_pve_config(selector_result)
    assert config["game_mode"] == "pve"
    assert config["ai_player"] == 1  # AI 执黑
    assert config["ai_model"] == "random"

def test_create_pve_config_ai_white():
    """创建 PvE 配置，AI 执白"""
    selector_result = {"player_color": "black", "ai": "fc_v1"}
    config = create_pve_config(selector_result)
    assert config["ai_player"] == 2  # AI 执白
    assert config["ai_model"] == "fc_v1"
```

## 6. 边缘测试用例

```python
def test_start_game_with_ai_first_error():
    """AI 先走启动游戏失败"""
    # 模拟存档名已存在
    controller.new_game("test")
    helper = GUIHelper(controller, session)
    result = helper.start_game_with_ai_first()
    assert result["status"] == "error"

def test_handle_player_move_no_game():
    """无游戏时处理玩家落子"""
    helper = GUIHelper(controller, session)
    result = helper.handle_player_move(7, 7)
    assert result["status"] == "error"

def test_get_display_state_no_game():
    """获取无游戏时的显示状态"""
    helper = GUIHelper(controller, session)
    state = helper.get_display_state()
    assert state["has_game"] is False
```

## 7. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `app/gui_helper.py` | 新增 | GUI 辅助类和无界面依赖函数 |
| `app/__init__.py` | 修改 | 导出 GUIHelper |
| `gui/main.py` | 修改 | 使用 GUIHelper，移除业务逻辑 |
| `tests/app/test_gui_helper.py` | 新增 | GUIHelper 测试用例 |
| `design/gui_helper_refactor_design.md` | 新增 | 设计文档 |
