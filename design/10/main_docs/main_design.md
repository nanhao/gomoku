# 测试代码全面 Review 报告

> 意图 10：全面 review 测试代码
> 日期：2026-06-15
> 测试总数：1594 项，全部通过（✅ 0 失败）

---

## 一、超过 500 行的大文件（❌ 违反规范）

| 文件 | 行数 | 严重度 |
|------|------|--------|
| `tests/3/integration/test_sample_games_cli.py` | 605 | 🔴 高 |
| `tests/ai/unit/test_iterative_train.py` | 596 | 🔴 高 |
| `tests/ai/test_game_manager_temp.py` | 583 | 🔴 高 |

### 1.1 test_sample_games_cli.py (605 行) → 拆为 2 个文件

该文件按测试组 A~G 自然分组，边界清晰。拆分方案：

| 新文件 | 包含内容 | 预估行数 |
|--------|---------|--------|
| `test_sample_games_cli.py`（原地精简） | A 组（参数互斥）+ D 组（对弈循环）+ 公共 `_setup_common_mocks()` / `_create_record_json()` | ~340 |
| `test_sample_games_cli_progress.py`（新建） | E 组（进度输出）+ F 组（Ctrl+C 中断）+ G 组（向后兼容），从原文件 import `_setup_common_mocks` / `_create_record_json` | ~270 |

拆出文件的 import 依赖：
```python
from tests.3.integration.test_sample_games_cli import _setup_common_mocks, _create_record_json
```

同时将每个测试类上的 4 个 `@patch` 装饰器提升为类级别（见 §4.4）。

### 1.2 test_iterative_train.py (unit) (596 行) → 删除无效测试 + 拆出报告测试

- **删除 UT-5 全部 4 个测试**（L478-499）：这些测试只验证 `51.0 > 0.5` 这种 Python 内置比较逻辑，与 `IterativeTrainer` 的任何方法都无关。执行删除后文件减少 ~30 行。
- **将 UT-6 + UT-7（日志输出 + 最终统计报告）移入新文件** `test_iterative_train_report.py`：这两个测试组共约 170 行，只依赖 `_print_round_header`、`_print_round_result`、`_print_final_report` 三个输出方法，职责独立。移出后原文件约 396 行，符合规范。

### 1.3 test_game_manager_temp.py (583 行) → 拆为 2 个文件

文件中有两组完全独立的测试，拆分边界是 `_SpyRecorder` 及其子类 `_SpyRecorderFull`：

| 新文件 | 包含内容 | 预估行数 |
|--------|---------|----------|
| `test_game_manager_temp.py`（原地精简） | `_SpyModel` + `_SpyEngine` + `_TempSaveDirTestBase` + `_make_recorder` + 用例 23~28（温度采样编排） | ~360 |
| `test_game_manager_record.py`（新建） | `_SpyRecorder` + `_SpyRecorderFull` + `TestRecordMoveTempInfo` + `TestSamplingProbsOccupiedPositions` + `TestSingleGameResult` + `TestStartAiVsAiResultFormat` | ~230 |

---

## 二、测试代码 Bug

### B-1: conftest.py WRITE GUARD 对 `int` 类型路径产生误判

**位置**: `tests/conftest.py` L92-106 `_guarded_open` + L75-89 `_is_protected_path`

`builtins.open` 的第一个参数 `file` 可以是文件描述符（`int`），例如 `os.fdopen(3, 'w')`。当前 `_is_protected_path(filepath)` 对 `int` 调用 `str(3)` → `"3"`，然后 `os.path.abspath("3")` 返回当前工作目录下的 `{cwd}\3`，这个路径正常情况下不在 `PROTECTED_DIRS` 中，但语义上完全是错误的——文件描述符 3 指向的可能正好是受保护目录下的文件。

**修复代码**（在 `_is_protected_path` 函数开头增加类型检查）：
```python
def _is_protected_path(filepath):
    # 文件描述符（int）不可能是路径，直接放行
    if isinstance(filepath, int):
        return False
    try:
        abs_path = os.path.abspath(str(filepath))
        for protected in PROTECTED_DIRS:
            protected_prefix = protected + os.sep
            if abs_path == protected or abs_path.startswith(protected_prefix):
                return True
        return False
    except (TypeError, ValueError):
        return False
```

### B-2: test_engine.py 中 `_empty_board` 和 `_make_probs` 在同一个文件内重复定义

**位置**: `tests/ai/test_engine.py` L303-311（`TestGetMoveStrategy`） 和 L429-437（`TestGetMoveSamplingStrategy`）

两个类中定义了完全相同的两个辅助方法，这是同文件内重复。修复方案见 §3.3。

---

## 三、重复代码

### R-1: `_make_completed_process` 函数跨文件重复定义

**位置**:
- `tests/ai/unit/test_iterative_train.py` L65-71
- `tests/ai/integration/test_iterative_train.py` L54-60

**两处代码逐行完全相同**：
```python
def _make_completed_process(returncode=0, stdout="", stderr=""):
    result = Mock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result
```

**修复**：将此函数移入 `tests/ai/helpers.py`，两处改为 `from tests.ai.helpers import _make_completed_process`。

注意：`tests/ai/helpers.py` 已有的 `import` 中不含 `subprocess` 和 `unittest.mock.Mock`，需在 helpers.py 头部增加：
```python
import subprocess
from unittest.mock import Mock
```

### R-2: `temp_project` fixture 跨文件重复（两个文件各定义一份，仅差 3 行注释）

**位置**:
- `tests/ai/unit/test_iterative_train.py` L34-56
- `tests/ai/integration/test_iterative_train.py` L30-51

两处 fixture 的唯一差异是注释措辞不同，代码逻辑完全一致（创建 gen4 假权重 + models 目录）。

**修复**：创建 `tests/ai/conftest.py`（pytest 会自动从该目录加载），在其中定义：
```python
import os, shutil, tempfile
import pytest
from ai.iterative_train import MODELS_DIR, PARENT_GEN

@pytest.fixture
def temp_project():
    tmp = tempfile.mkdtemp()
    gen4_dir = os.path.join(tmp, MODELS_DIR, f"gen{PARENT_GEN}")
    os.makedirs(gen4_dir, exist_ok=True)
    with open(os.path.join(gen4_dir, "cnn_v3.pth"), 'w') as f:
        f.write("fake_model_weights")
    os.makedirs(os.path.join(tmp, MODELS_DIR), exist_ok=True)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)
```

然后从两个测试文件中删除各自的 `temp_project` fixture 定义。两个文件都在 `tests/ai/` 下，pytest 会自动发现 `tests/ai/conftest.py` 中的 fixture。

### R-3: `_empty_board` 和 `_make_probs` 同文件内重复

**位置**: `tests/ai/test_engine.py` L303-311 和 L429-437

**修复**：将这两个方法从 `TestGetMoveStrategy` 和 `TestGetMoveSamplingStrategy` 的类体中删除，提升为模块级函数（放在 `StubModel` 类定义之后，`TestAIEngine` 类之前，约 L50）：
```python
def _empty_board():
    return [[0] * 15 for _ in range(15)]

def _make_probs(pairs, total=225):
    probs = [0.0] * total
    for idx, p in pairs:
        probs[idx] = p
    return probs
```

两个测试类中所有 `self._empty_board()` 改为 `_empty_board()`，`self._make_probs(...)` 改为 `_make_probs(...)`。

### R-4: test_sample_games_cli.py 和 test_create_samples.py 的 4-`@patch` 模式重复

两个文件中每个测试方法都重复写 4 个 `@patch` 装饰器 + 4 个 mock 参数。`test_sample_games_cli.py` 已抽取了 `_setup_common_mocks()` 函数处理 mock 配置，但装饰器本身仍重复。

**修复**：将 4 个 `@patch` 提升到类级别。以 `test_sample_games_cli.py` 的 `TestArgMutualExclusion` 为例，当前每个方法：
```python
@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.AIEngine')
@patch('ai.create_samples.get_model')
def test_both_specified_exits(self, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls):
```

改为类级别（注意参数顺序反转，pytest 从下往上匹配）：
```python
@patch('ai.create_samples.get_model')
@patch('ai.create_samples.AIEngine')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.DataRecorder')
class TestArgMutualExclusion(unittest.TestCase):
    def test_both_specified_exits(self, mock_recorder_cls, mock_manager_cls,
                                   mock_engine_cls, mock_get_model):
```

此改动对 `test_sample_games_cli.py` 中 5 个测试类（约 16 个方法）减少约 60 行重复装饰器代码。`test_create_samples.py` 同理。

### R-5: 约 55 个文件手动 `sys.path.insert` 冗余

`tests/conftest.py` L28-35 已在 pytest 会话启动时将项目根目录插入 `sys.path[0]`，所有通过 pytest 运行的测试不需要再手动设置路径。

**修复规则**：
1. 对于**纯 pytest 运行**的文件（不含 `if __name__ == "__main__": unittest.main()`），直接删除文件顶部的 `sys.path.insert` 代码块。
2. 对于同时支持 `python -m unittest` 直接运行的文件，将路径设置移入 `if __name__ == "__main__"` 守卫内：
```python
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    unittest.main()
```

---

## 四、无效测试与冗余测试

### I-1: UT-5 的 4 个测试不测试任何被测代码

**位置**: `tests/ai/unit/test_iterative_train.py` L478-499

```python
def test_win_rate_above_50_promotes(self, trainer):
    assert (51.0 > WIN_RATE_THRESHOLD) is True       # 只验证 Python 的 > 运算符

def test_win_rate_50_not_promote(self, trainer):
    assert (50.0 > WIN_RATE_THRESHOLD) is False

def test_win_rate_below_50_not_promote(self, trainer):
    assert (49.0 > WIN_RATE_THRESHOLD) is False

def test_win_rate_50_1_promotes(self, trainer):
    assert (50.1 > WIN_RATE_THRESHOLD) is True
```

这四个测试不调用 `IterativeTrainer` 的任何方法，`trainer` 参数虽然传入但从未使用。它们测试的是 `float.__gt__` 的行为，与项目代码无关。

**修复**：直接删除整个 `TestIterationLogic` 类（L478-499），约 22 行。胜率阈值判定逻辑的正确性由集成测试 `IT-1.1`（单轮晋升）和 `IT-1.2`（单轮停止）覆盖，这些集成测试真实执行了 `trainer.run()` 并验证了晋升/停止行为。

### I-2: A-04 和 G-01 测试相同场景

**位置**: `tests/3/integration/test_sample_games_cli.py`

- `TestArgMutualExclusion.test_only_games_unchanged`（A-04, L155-167）：传 `--games 10 --output ...`，验证走 `start_ai_vs_ai` 路径
- `TestBackwardCompatibility.test_without_sample_games`（G-01, L676-688）：传 `--games 100 --output ...`，验证走 `start_ai_vs_ai` 路径

两个测试的 mock 条件完全相同，差异仅为 `--games 10` vs `--games 100`。它们验证的底层逻辑完全一致：不传 `--sample_games` 时应走 `start_ai_vs_ai`。

**修复**：删除 A-04，保留 G-01。理由：G-01 在"向后兼容"测试组中语义更明确，且使用了更典型的默认值 `--games 100`。A 组已有 6 个其他测试（A-01/A-02/A-03/A-05/A-06/A-07），删除 A-04 后仍完整。

### I-3: `test_create_engine_cnn_v2_not_random` 被 `test_create_engine_cnn_v2` 完全覆盖

**位置**: `tests/ai/test_engine.py` L276-294

```python
def test_create_engine_cnn_v2(self):                    # L276
    engine = create_engine("cnn_v2")
    self.assertIsInstance(engine.model, CNNV2Model, ...)  # 正向断言：必须是 CNNV2Model

def test_create_engine_cnn_v2_not_random(self):         # L290
    engine = create_engine("cnn_v2")
    self.assertNotIsInstance(engine.model, RandomModel, ...)  # 反向断言：不是 RandomModel
```

正向断言 `assertIsInstance(x, CNNV2Model)` 逻辑上蕴含了 `assertNotIsInstance(x, RandomModel)`（除非 `RandomModel` 是 `CNNV2Model` 的子类，但它不是）。因此 L290-294 是冗余测试。

**修复**：删除 `test_create_engine_cnn_v2_not_random` 方法（L290-294）。

### I-4: 空测试目录残留

| 目录 | 内容 |
|------|------|
| `tests/1/` | 只有 `__pycache__/`，无任何 `.py` 文件 |
| `tests/game/` | 只有 `save/` 子目录，无测试文件 |
| `tests/app/edge/_test_saves/` | 残留空目录 |
| `tests/app/primary/_test_battle_root/` | 残留空目录树（含 ai_data/train_data 子目录） |

**修复**：直接删除这 4 个目录。

### I-5: `TestRandomModel` 测试分散在 `test_engine.py` 和 `test_random_model.py`

**位置**:
- `tests/ai/test_engine.py` L143-252：`TestRandomModel` 类（8 个测试，含 predict 行为测试）
- `tests/ai/test_random_model.py`：`TestRandomModelContract` + `TestBaseModelInterface`（11 个测试，含接口契约测试）

两处有重叠覆盖：
- `test_predict_probability_sum` ≈ `test_predict_prob_sum_normal_board`（都验证 sum=1）
- `test_predict_all_occupied` ≈ `test_predict_prob_sum_full_board` + `test_predict_full_board_uniform`（都验证满棋盘）
- `test_name` ≈ `test_name_property`（都验证 name 属性）

而 `test_random_model.py` 额外测试了 `save`/`load`/`network` 接口契约，这些是 `test_engine.py` 没有的。

**修复**：
1. 将 `test_engine.py` 中 `TestRandomModel` 的 8 个测试合并到 `tests/ai/test_random_model.py`
2. 合并时去重，保留更精确的版本（`test_random_model.py` 中的实现更完整）
3. 从 `test_engine.py` 中删除整个 `TestRandomModel` 类（L143-252，约 110 行），同时删除不再需要的 `from ai.models.random.model import RandomModel`（L18）

执行后 `test_engine.py` 从 455 行降至约 345 行。

---

## 五、测试结构问题

### S-1: 目录组织方式混乱，5 种模式并存

当前存在的目录组织模式：

| 模式 | 位置 | 文件数 |
|------|------|--------|
| `unit/` / `edge/` / `primary/` | `tests/ai/unit/`, `tests/ai/edge/`, `tests/ai/primary/` | ~14 |
| `integration/` | `tests/ai/integration/`, `tests/integration/` | ~6 |
| 按意图编号 | `tests/3/unit/`, `tests/3/integration/` | ~3 |
| app 的 `primary/` / `edge/` | `tests/app/primary/`, `tests/app/edge/` | ~4 |
| 平铺（未分类） | `tests/ai/test_*.py`, `tests/gui/test_*.py` 等 | ~55 |

规范明确要求统一使用 **unit / integration / e2e** 三级结构。`primary`/`edge` 是设计文档的术语，不是测试目录规范。

**修复路径**（按阶段执行，每次迁移后跑全量测试确认通过）：

第一阶段：消除 `primary/` / `edge/` 命名
- `tests/ai/primary/test_evaluator_shape_detection.py` → `tests/ai/unit/`
- `tests/ai/primary/test_mini_batch_training.py` → `tests/ai/unit/`
- `tests/ai/edge/test_evaluator_shape_detection_edge.py` → `tests/ai/unit/`
- `tests/ai/edge/test_mini_batch_training_edge.py` → `tests/ai/unit/`
- `tests/app/primary/test_battle.py` → `tests/app/`（或 `tests/app/unit/`）
- `tests/app/primary/test_session.py` → `tests/app/`
- `tests/app/edge/test_session.py` → `tests/app/`
- 删除空目录 `tests/ai/primary/`, `tests/ai/edge/`, `tests/app/primary/`, `tests/app/edge/`

第二阶段：合并 `tests/3/` 到标准目录
- `tests/3/integration/test_sample_games_cli.py` → `tests/ai/integration/`
- `tests/3/unit/test_is_sample_game.py` → `tests/ai/unit/`
- `tests/3/unit/test_print_sample_progress.py` → `tests/ai/unit/`
- 删除目录 `tests/3/`

第三阶段：平铺文件分类（工作量最大，且依赖文件拆分先完成）：
- 将 `tests/ai/test_*.py` 中属于单元测试的文件迁移到 `tests/ai/unit/`
- 将属于集成测试的文件迁移到 `tests/ai/integration/`
- `tests/gui/` 目录维持不变（GUI 测试多为集成/端到端性质）

### S-2: 缺少 `e2e/` 目录

规范要求 `tests/e2e/` 用于端到端测试，但此目录不存在。`tests/integration/test_multi_gen_flow.py`（多代模型完整流程）本质是 e2e。

**修复**：创建 `tests/e2e/__init__.py`，将 `tests/integration/test_multi_gen_flow.py` 移入。

### S-3: 接近 500 行边界的文件（⚠️ 预警）

执行上述拆分、删除、迁移操作后，以下接近边界的文件将自然缩小：

| 文件 | 当前行数 | 修复后预估 | 触发修复 |
|------|---------|-----------|---------|
| `tests/ai/test_engine.py` | 455 | ~345 | I-5 删除 TestRandomModel |
| `tests/ai/test_create_samples.py` | 450 | ~410 | R-4 类级别 @patch |
| `tests/ai/integration/test_iterative_train.py` | 430 | ~410 | R-1 移出 _make_completed_process + R-2 移出 fixture |

`test_data_recorder.py`(469)、`test_evaluator_shape_detection.py`(468)、`test_loss_logger.py`(437)、`test_neural_network_model.py`(432) 需要后续主动拆分。

---

## 六、性能优化

### P-1: `TestCreateEngine` 每个测试真实加载 PyTorch 模型权重

**位置**: `tests/ai/test_engine.py` L258-282

`create_engine("fc_v1")`、`create_engine("cnn_v1")`、`create_engine("cnn_v2")` 三个测试每次都从磁盘加载 `.pth` 文件并初始化完整 PyTorch 网络。这三项测试的目的是验证模型注册逻辑（名称→类的映射），不需要真实加载权重。

**修复**：在 `TestCreateEngine` 类上添加 `@pytest.mark.slow`，并在 CI 中默认排除 slow 测试，仅在 nightly 构建中运行。同时 mock `torch.load` 可以额外减少 I/O，但鉴于 `conftest.py` 的 WRITE GUARD 已禁止写入模型目录，只读加载是安全的，添加 `slow` 标记即可。

### P-2: `test_game_manager_temp.py` 中真实对弈测试

`TestSingleGameResult` 和 `TestStartAiVsAiResultFormat` 的 10 个测试方法真实调用了 `play_single_game()` / `start_ai_vs_ai()`，每局对弈执行数十步完整棋盘操作。虽然使用轻量 `_SpyEngine`，但这组测试的耗时显著高于纯 mock 测试。

**修复**：这些测试在 §1.3 中会被拆出到 `test_game_manager_record.py`，拆分后在新文件头部添加 `@pytest.mark.slow` 注释说明。

---

## 七、其他问题

### O-1: `helpers.py` 未被充分利用

`tests/ai/helpers.py` 已提供 `create_output`、`create_sample_dir`、`create_minimal_sample_file` 三个辅助函数，但当前仅 `test_fc_v1_trainer.py` 和 `test_cnn_trainers.py` 等少数文件引用。大量测试文件内部重复实现了类似的造数据逻辑。

**修复**：在 R-1 中已将 `_make_completed_process` 移入 `helpers.py`。后续应继续将以下模式抽取到 `helpers.py`：
- `create_record_json`（`test_sample_games_cli.py` L65-85 和 `test_create_samples.py` 中的类似逻辑）
- 225 维零向量/概率向量的工厂函数

### O-2: `test_write_guard.py` 与 `conftest.py` 逻辑需同步

`tests/test_write_guard.py`（149 行）测试了 write guard 的行为，而 `tests/conftest.py`（119 行）实现了 write guard。当前 B-1 修复后，需同步检查 `test_write_guard.py` 中是否也需要增加 `int` 类型路径的测试用例。

---

## 八、执行计划与优先级

| 优先级 | # | 操作 | 影响文件 | 预期工作量 |
|--------|---|------|---------|-----------|
| 🔴 P0 | B-1 | 修复 conftest.py 文件描述符漏检 | 1 个文件 | 5 分钟 |
| 🔴 P0 | 1.1 | 拆分 test_sample_games_cli.py | 新建 1 个，修改 1 个 | 15 分钟 |
| 🔴 P0 | 1.2 | 删除 UT-5 + 拆出 report 测试 | 修改 1 个，新建 1 个 | 15 分钟 |
| 🔴 P0 | 1.3 | 拆分 test_game_manager_temp.py | 修改 1 个，新建 1 个 | 15 分钟 |
| 🟠 P1 | I-1 | 删除 TestIterationLogic（UT-5） | 同 1.2 | — |
| 🟠 P1 | R-1 | `_make_completed_process` 移入 helpers.py | 3 个文件 | 5 分钟 |
| 🟠 P1 | R-2 | 创建 tests/ai/conftest.py | 新建 1 个，修改 2 个 | 10 分钟 |
| 🟠 P1 | R-3 | `_empty_board`/`_make_probs` 提升为模块级 | 1 个文件 | 10 分钟 |
| 🟠 P1 | I-5 | 合并 TestRandomModel 到 test_random_model.py | 2 个文件 | 15 分钟 |
| 🟡 P2 | R-4 | 类级别 @patch（test_sample_games_cli.py） | 1 个文件 | 15 分钟 |
| 🟡 P2 | I-2 | 删除 A-04（与 G-01 重叠） | 1 个文件 | 2 分钟 |
| 🟡 P2 | I-3 | 删除 test_create_engine_cnn_v2_not_random | 1 个文件 | 2 分钟 |
| 🟡 P2 | I-4 | 删除 4 个空目录 | — | 2 分钟 |
| 🟡 P2 | R-4 | 类级别 @patch（test_create_samples.py） | 1 个文件 | 15 分钟 |
| 🟢 P3 | S-1 | 第一阶段：消除 primary/edge 命名 | 10 个文件 | 15 分钟 |
| 🟢 P3 | S-1 | 第二阶段：合并 tests/3/ | 3 个文件 | 10 分钟 |
| 🟢 P3 | S-2 | 创建 tests/e2e/ + 迁移 test_multi_gen_flow.py | 2 个文件 | 5 分钟 |
| 🟢 P3 | R-5 | 删除纯 pytest 文件的 sys.path.insert | ~50 个文件 | 20 分钟 |
| 🟢 P3 | P-1 | TestCreateEngine 加 slow 标记 | 1 个文件 | 2 分钟 |

**执行原则**：每完成一个 P0/P1 项，立即跑 `python -m pytest tests/ -x` 验证全量通过，再继续下一项。
