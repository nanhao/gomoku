# AI 数据目录路径修正设计文档

## 问题描述

当前代码中数据目录路径配置与实际数据存放位置不一致：
- 代码期望路径：`d:\projects\gomoku\ai_data` 和 `d:\projects\gomoku\train_data`
- 实际数据路径：`d:\projects\gomoku\ai\ai_data` 和 `d:\projects\gomoku\ai\train_data`

导致回放功能提示"暂无回放数据"。

## 解决方案

修改代码中的路径计算逻辑，使其指向 `ai/` 子目录下的数据目录。

## 涉及文件

1. `app/gui_helper.py` - `get_ai_data_dir()` 函数
2. `ai/battle.py` - DataRecorder 初始化路径
3. `ai/models/fc_v1/trainer.py` - 训练和评价数据目录路径
4. `ai/evaluator.py` - 默认数据目录路径

## 伪代码

### get_ai_data_dir() 修改
```python
def get_ai_data_dir() -> str:
    # 从 app/gui_helper.py 所在位置计算
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    # 修改为指向 ai/ai_data
    ai_dir = os.path.join(project_root, "ai")
    return os.path.join(ai_dir, "ai_data")
```

### battle.py 修改
```python
# 原代码
ai_data_dir = os.path.join(project_root, "ai_data")
# 修改为
ai_data_dir = os.path.join(project_root, "ai", "ai_data")
```

### trainer.py 修改
```python
# 原代码
ai_data_dir = os.path.join(project_root, "ai_data")
train_data_dir = os.path.join(project_root, "train_data")
# 修改为
ai_data_dir = os.path.join(project_root, "ai", "ai_data")
train_data_dir = os.path.join(project_root, "ai", "train_data")
```

### evaluator.py 修改
```python
# 原代码
def __init__(self, ai_data_dir: str = "ai_data", train_data_dir: str = "train_data", ...)
# 修改为
def __init__(self, ai_data_dir: str = None, train_data_dir: str = None, ...)
    if ai_data_dir is None:
        ai_data_dir = os.path.join("ai", "ai_data")
    if train_data_dir is None:
        train_data_dir = os.path.join("ai", "train_data")
```

## 主流程测试用例

1. **test_get_ai_data_dir_returns_correct_path**
   - 验证 `get_ai_data_dir()` 返回 `ai/ai_data` 路径
   - 验证路径存在性检查通过

2. **test_battle_creates_data_in_ai_subdir**
   - 运行 AI 对战
   - 验证数据创建在 `ai/ai_data/` 下

3. **test_trainer_uses_correct_paths**
   - 验证训练器使用 `ai/ai_data` 和 `ai/train_data`

4. **test_evaluator_default_paths**
   - 验证评价器默认路径指向 `ai/` 子目录

5. **test_replay_browser_finds_data**
   - 验证回放浏览器能正确找到 `ai/ai_data` 中的数据
