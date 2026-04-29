# 多代模型支持设计

## 1. 需求摘要

使一个模型支持多代（generation），目录结构从 `ai/models/模型名/pth文件` 变为 `ai/models/模型名/gen{i}/pth文件`。
所有脚本和 GUI 界面以 `模型名称.代数` 格式加载模型（如 `cnn_v3.2`）。

## 2. 核心决策

| 决策项 | 结论 |
|--------|------|
| 代数起始 | 从 1 开始（gen1, gen2, ...） |
| 多代判定 | MODEL_REGISTRY 中显式声明 `supports_multi_gen: bool`；声明为 True 即视为多代模型，不依赖文件系统状态推断 |
| 模型指定格式 | `模型名称.代数`（如 `cnn_v3.2`），不含 `.` 则视为非多代模型名 |
| 模型命名约束 | 模型名称禁止包含 `.`（点号），注册时硬校验，消除解析歧义 |
| 不带代数加载多代模型 | 报错 |
| 保存时指定代 | 同格式，手动指定，加载和保存目标为同一个代数 |
| 代语义 | 用户自定义，代码不做假设 → 修正见 §10.1 |
| random | 在 MODEL_REGISTRY 中声明 `supports_multi_gen=False`，走统一判定逻辑 |
| gen 目录不存在 | 自动创建，不检查连续性 |
| GUI 展示 | 下拉框显示 `cnn_v3.1`、`cnn_v3.2` 单独条目 |
| engine.py 回退 | 不回退：`create_engine` 遇 `get_model` 异常时直接上抛，不静默回退到 RandomModel |

## 3. 接口设计

### 3.1 ai/registry.py — MODEL_REGISTRY 结构变更（→ NamedTuple 改造见 §10.2）

**MODEL_REGISTRY 新增第三项 `supports_multi_gen`**，
**注册时硬校验模型名称不含 `.`**（点号）
（`if '.' in name: raise ValueError("模型名称禁止包含 '.'")`）：

```python
MODEL_REGISTRY: dict[str, tuple[str, str | None, bool]] = {
    # name: (model_class_path, trainer_class_path, supports_multi_gen)
    # 硬约束：模型名称禁止包含 "."（点号），确保 _parse_model_spec 的点号唯一语义为代数分隔符
    "random": ("ai.models.random.model.RandomModel", None, False),
    "fc_v1":  ("ai.models.fc_v1.model.FCv1Model", "ai.models.fc_v1.trainer.FCv1Trainer", False),
    "cnn_v1": ("ai.models.cnn_v1.model.CNNV1Model", "ai.models.cnn_v1.trainer.CNNV1Trainer", False),
    "cnn_v2": ("ai.models.cnn_v2.model.CNNV2Model", "ai.models.cnn_v2.trainer.CNNV2Trainer", False),
    "cnn_v3": ("ai.models.cnn_v3.model.CNNV3Model", "ai.models.cnn_v3.trainer.CNNV3Trainer", True),
}
```

> **当前仅 `cnn_v3` 声明为多代模型（`supports_multi_gen=True`），其余模型（`random`、`fc_v1`、`cnn_v1`、`cnn_v2`）均为非多代模型（`supports_multi_gen=False`）。**

### 3.2 ai/registry.py — 新增/变更的函数

**新增模块私有函数（不对外暴露）：**

```python
def _parse_model_spec(spec: str) -> ParsedModelSpec:
    """解析模型指定字符串，返回 ParsedModelSpec（→ NamedTuple 定义见 §10.2）

    使用右侧分割第一个 "."，将点号右侧部分作为代数解析。
    模型名称禁止包含 "."（由注册表硬校验保证），因此点号唯一语义为代数分隔符，
    不存在歧义，无需防御性回退。

    Args:
        spec: 模型指定字符串，如 "fc_v1" 或 "cnn_v3.2"

    Returns:
        ParsedModelSpec(name=model_name, gen=代数或None)

    Examples:
        _parse_model_spec("fc_v1")     → ParsedModelSpec(name="fc_v1", gen=None)
        _parse_model_spec("cnn_v3.2")  → ParsedModelSpec(name="cnn_v3", gen=2)
        _parse_model_spec("cnn_v3.0")  → ValueError（代数必须 ≥ 1）
        _parse_model_spec("cnn_v3.abc")→ ValueError（代数必须为正整数）
    """

def _is_multi_gen_model(model_name: str) -> bool:
    """判断模型是否为多代模型

    仅依据 MODEL_REGISTRY 中的 supports_multi_gen 声明，不扫描文件系统。
    声明为 True 即视为多代模型，无论文件系统上是否存在 gen{i}/ 目录。
    这确保 API 行为完全由注册表声明决定，不因文件系统状态变化而产生隐式状态机。

    Args:
        model_name: 注册表中的模型名称，如 "cnn_v3"

    Returns:
        True 表示多代模型，False 表示非多代模型
    """
    return MODEL_REGISTRY[model_name].supports_multi_gen
```

**新增公共函数：**

```python
def list_models_with_full_name() -> list[str]:
    """返回所有可加载的模型指定字符串列表（含代数的完整名称）

    对于非多代模型（registry 声明为 False），返回模型名称本身。
    对于多代模型（registry 声明为 True），扫描 gen{i}/ 子目录，返回每个已有代数的完整指定。
    若多代模型无已保存代数，则不展示该模型。
    扫描时若检测到 pth 文件与 gen{i}/ 目录共存，抛出 ValueError。

    Returns:
        可加载的模型指定字符串列表，如 ["random", "fc_v1", "cnn_v3.1", "cnn_v3.2"]

    Raises:
        ValueError: pth 文件与 gen{i}/ 子目录共存
    """
```

**变更的现有函数：**

```python
def get_model(spec: str):
    """根据模型指定字符串返回模型实例

    核心判定原则：用户显式指定 gen → 优先信任用户意图，按多代模型处理，
    无需查询 _is_multi_gen_model。仅在用户未指定 gen 时，通过注册表声明判定模型类型。

    Args:
        spec: 模型指定字符串，如 "fc_v1" 或 "cnn_v3.2"

    Returns:
        模型实例（BaseModel 子类），已加载对应代数的权重

    Raises:
        ValueError: 未知的模型名称
        ValueError: 多代模型未指定代数
        ValueError: 非多代模型指定了代数
    """

def get_trainer(spec: str, **kwargs):
    """根据模型指定字符串返回训练器实例

    与 get_model 相同的 spec 格式和校验逻辑。
    训练器创建时，模型的 DEFAULT_MODEL_PATH 指向对应的 gen{i}/ 路径。
    """
```

**降级的现有函数：**

`list_models()` 降级为模块私有函数 `_list_models()`，仅返回注册表中的基础模型名（如 `["random", "fc_v1", "cnn_v1", ...]`），
不包含代数信息。用于内部枚举，不用于用户选择或错误提示。
外部调用方应使用 `list_models_with_full_name()`，确保用户看到的模型列表与实际可接受的值一致。

### 3.3 ai/base_model.py — NeuralNetworkModel 变更

`NeuralNetworkModel.__init__` 新增 `gen` 参数：

```python
class NeuralNetworkModel(BaseModel):
    def __init__(self, model_path: str = None, fresh: bool = False, gen: int | None = None):
        """初始化神经网络模型

        Args:
            model_path: 模型文件路径，如果存在则加载
            fresh: 是否强制创建新模型（不加载已保存的模型）
            gen: 代数，指定时 DEFAULT_MODEL_PATH 指向 gen{i}/ 子目录
        """
```

当 `gen` 不为 None 时，`DEFAULT_MODEL_PATH` 变为 `ai/models/模型名/gen{i}/模型名.pth`。
RandomModel 不接受 gen 参数（若传入则报错）。

**NeuralNetworkModel 新增 `full_name` property**：

```python
class NeuralNetworkModel(BaseModel):
    @property
    def full_name(self) -> str:
        """含代数的完整模型名称

        多代模型返回 "模型名.代数"（如 "cnn_v3.2"），
        非多代模型返回模型名称本身（如 "fc_v1"），与 name 相同。

        用于代数敏感场景（训练数据目录、日志命名、DataRecorder 等），
        确保不同代数的模型产出数据相互隔离。
        """
        if self.gen is not None:
            return f"{self.name}.{self.gen}"
        return self.name
```

`name` 保持不变（如始终返回 `"cnn_v3"`），确保不关心代数的代码向后兼容。
代数敏感场景应改用 `full_name`，详见第 6 节受影响文件清单。

### 3.4 gui/components/mode_selector.py — _get_available_ais 变更

```python
def _get_available_ais(self) -> list[str]:
    """获取可用的 AI 列表

    返回格式：
    - 非多代模型：模型名称本身，如 "random"、"fc_v1"
    - 多代模型：每个代数一条，如 "cnn_v3.1"、"cnn_v3.2"

    委托给 registry.list_models_with_full_name() 实现。
    """
```

## 4. 主流程伪代码

### 4.1 加载模型（get_model）

```
输入: spec (如 "cnn_v3.2" 或 "fc_v1")

1. 解析 spec → parsed = _parse_model_spec(spec)；model_name = parsed.name，gen = parsed.gen
2. 校验 model_name 在 MODEL_REGISTRY 中存在
3. 若 gen is not None:
   a. 用户显式指定了代数 → 优先信任用户意图
   b. 校验 MODEL_REGISTRY[model_name].supports_multi_gen 为 True
      - 若为 False → 报错 "非多代模型不能指定代数"
   c. 将 gen 传递给模型构造函数
4. 若 gen is None:
   a. 用户未指定代数 → 查询注册表声明判定
   b. 判定 is_multi = _is_multi_gen_model(model_name)
   c. 若 is_multi → 报错 "多代模型必须指定代数，如 cnn_v3.1"
   d. 若 not is_multi → 按非多代模型处理
5. 导入模型类，创建实例：
   a. 若 gen is not None → model_cls(gen=gen)
   b. 若 gen is None → model_cls()（RandomModel 等不接受 gen 参数的模型走此分支）
6. 返回模型实例
```

### 4.2 保存模型（train_main.py --model cnn_v3.2 训练后保存）

```
1. get_model("cnn_v3.2") 创建模型 → DEFAULT_MODEL_PATH = ai/models/cnn_v3/gen2/cnn_v3.pth
2. 训练...
3. model.save() → 保存到 DEFAULT_MODEL_PATH
   a. 若 gen2/ 目录不存在 → os.makedirs 自动创建
   b. 写入 pth 文件
```

### 4.3 GUI 模型选择

```
1. _get_available_ais() 调用 registry.list_models_with_full_name()
   - 非多代模型 → 返回模型名本身
   - 多代模型 → 扫描 gen{i}/ 子目录，生成 "name.1", "name.2" ...
2. 用户选择 "cnn_v3.2" → 传入 create_engine("cnn_v3.2") → get_model("cnn_v3.2")
```

### 4.4 多代判定（_is_multi_gen_model）

```
输入: model_name (如 "cnn_v3")

1. 从 MODEL_REGISTRY 查询 supports_multi_gen 声明
   return MODEL_REGISTRY[model_name].supports_multi_gen
```

注：不扫描文件系统。声明 True 即为多代模型，行为完全由注册表决定，确保 API 确定性。
文件系统扫描（如列出现有代数、检测 pth/gen 共存）由 `list_models_with_full_name()` 负责。

### 4.5 _parse_model_spec

```
输入: spec (如 "cnn_v3.2" 或 "fc_v1")

1. 若 spec 不含 ".":
   return ParsedModelSpec(name=spec, gen=None)
2. 从右侧分割第一个 "." → (model_name, gen_str)
3. 将 gen_str 转为整数:
   a. 转换失败 → raise ValueError("代数必须为正整数")
   b. gen < 1 → raise ValueError("代数必须 ≥ 1")
4. return ParsedModelSpec(name=model_name, gen=gen)
```

注：模型名称禁止包含 `.`（由注册表硬校验保证），因此点号唯一语义为代数分隔符，
`_parse_model_spec` 无需防御性回退逻辑——含 `.` 的 spec 必然是 "模型名.代数" 格式。

## 5. 测试用例

### 5.1 _parse_model_spec

| 输入 | 期望输出 | 说明 |
|------|---------|------|
| `"fc_v1"` | `ParsedModelSpec(name="fc_v1", gen=None)` | 非多代模型名 |
| `"cnn_v3.2"` | `ParsedModelSpec(name="cnn_v3", gen=2)` | 多代指定 |
| `"cnn_v3.1"` | `ParsedModelSpec(name="cnn_v3", gen=1)` | 最小代数 |
| `"cnn_v3.0"` | ValueError | 代数 < 1 |
| `"cnn_v3.abc"` | ValueError | 非整数代数 |
| `"cnn_v3."` | ValueError | 空代数 |

> **注册校验测试**：`MODEL_REGISTRY` 注册时硬校验模型名不含 `.`，如 `"fc.v1"` → ValueError（"模型名称禁止包含 '.'")。此校验在 `_parse_model_spec` 之前执行，确保解析器永远不会遇到模型名含 `.` 的情况。

### 5.2 _is_multi_gen_model

| registry 声明 | 期望结果 | 说明 |
|--------------|---------|------|
| `random: False` | False | registry 声明为 False |
| `fc_v1: False` | False | registry 声明为 False |
| `cnn_v3: True` | True | 声明为 True 即为多代，无论文件系统状态 |
| `cnn_v3: False` | False | 声明为 False 即为非多代，无视文件系统 |

### 5.3 get_model

| 输入 spec | registry 声明 | 文件系统状态 | 期望结果 | 说明 |
|-----------|--------------|-------------|---------|------|
| `"random"` | `False` | - | RandomModel 实例 | 非多代正常 |
| `"fc_v1"` | `False` | fc_v1.pth 存在 | FCv1Model 实例 | 非多代正常 |
| `"cnn_v3.2"` | `True` | gen2 存在 | CNNV3Model(gen=2)，加载 gen2 权重 | 多代正常 |
| `"cnn_v3.2"` | `True` | gen2 不存在 | CNNV3Model(gen=2)，fresh 状态 | **用户显式指定 gen 优先**，无视目录状态 |
| `"cnn_v3.2"` | `False` | - | ValueError | 非多代不能指定代数 |
| `"cnn_v3"` | `True` | - | ValueError | 声明为 True 即多代，必须指定代数 |
| `"fc_v1.1"` | `False` | - | ValueError | 非多代不能指定代数 |
| `"unknown.1"` | 不存在 | - | ValueError | 未知模型名 |

### 5.4 get_trainer

| 输入 spec | registry 声明 | 文件系统状态 | 期望结果 | 说明 |
|-----------|--------------|-------------|---------|------|
| `"fc_v1"` | `False` | - | FCv1Trainer，路径指向原路径 | 非多代不变 |
| `"cnn_v3.2"` | `True` | - | CNNV3Trainer，路径指向 gen2/ | 多代路径正确 |
| `"cnn_v3"` | `True` | - | ValueError | 声明为 True 即多代，必须指定代数 |
| `"random"` | `False` | - | ValueError | random 无训练器 |

### 5.5 list_models_with_full_name

| registry 声明 | 文件系统状态 | 期望输出 | 说明 |
|-------------|-------------|---------|------|
| random=False, fc_v1=False, cnn_v3=True | cnn_v3 有 gen1, gen2 | `["random", "fc_v1", "cnn_v3.1", "cnn_v3.2"]` | 混合场景 |
| random=False, fc_v1=False, cnn_v3=True | cnn_v3 无 gen 目录 | `["random", "fc_v1"]` | 声明多代但无已保存代数 → 不展示该模型 |
| random=False, fc_v1=False, cnn_v3=True | cnn_v3 有 pth + gen1 共存 | ValueError（pth 与 gen 目录不能共存） | 数据不一致保护 |
| 全部 False | - | `["random", "fc_v1", "cnn_v1", "cnn_v2", "cnn_v3"]` | 当前状态 |

### 5.6 full_name property

| 模型实例 | 期望 full_name | 说明 |
|---------|---------------|------|
| `FCv1Model()` | `"fc_v1"` | 非多代，与 name 相同 |
| `CNNV3Model()` | `"cnn_v3"` | 多代但 gen=None，与 name 相同 |
| `CNNV3Model(gen=1)` | `"cnn_v3.1"` | 多代 gen=1 |
| `CNNV3Model(gen=2)` | `"cnn_v3.2"` | 多代 gen=2 |
| `RandomModel()` | `"random"` | 非多代，与 name 相同 |

### 5.7 model.save() 路径

| 场景 | 期望保存路径 | 说明 |
|------|-------------|------|
| `FCv1Model()` | `ai/models/fc_v1/fc_v1.pth` | 非多代不变 |
| `CNNV3Model(gen=2)` | `ai/models/cnn_v3/gen2/cnn_v3.pth` | 多代指向 gen 子目录 |
| gen2/ 不存在 | 自动创建目录后保存 | 自动创建 |

## 6. 受影响文件清单

### 需修改

| 文件 | 变更内容 |
|------|---------|
| `ai/registry.py` | MODEL_REGISTRY 结构变更（新增 `supports_multi_gen`，value type 改为 `RegistryEntry` NamedTuple）；新增 `_parse_model_spec`（返回 `ParsedModelSpec` NamedTuple）、`_is_multi_gen_model`、`list_models_with_full_name`；修改 `get_model`、`get_trainer`；`list_models` 降级为 `_list_models` |
| `ai/base_model.py` | NeuralNetworkModel.__init__ 新增 gen 参数，DEFAULT_MODEL_PATH 支持 gen 路径；新增 `full_name` property |
| `ai/base_trainer.py` | 新增 `model_full_name` property，委托 `self.model.full_name` |
| `ai/train_main.py` | help 文本、参数说明更新 |
| `ai/battle.py` | help 文本、参数说明更新 |
| `ai/create_samples.py` | help 文本、参数说明更新；`model.name` 改为 `model.full_name`，确保 DataRecorder 按代数隔离 |
| `ai/lr_range_test.py` | load_model 适配 registry 的 gen 支持 |
| `ai/self_play_cycle.py` | `model_name` 改为 `model_full_name`，确保训练会话目录按代数隔离 |
| `ai/loss_logger.py` | 支持通过 `model_full_name` 构造含代数的日志文件名 |
| `gui/components/mode_selector.py` | `_get_available_ais()` 委托给 registry.list_models_with_full_name() |
| `ai/engine.py` | `create_engine()` 移除 `except ValueError` 回退 RandomModel 逻辑，改为直接上抛异常。原回退逻辑在多代模型场景下会掩盖配置错误（如忘记指定代数时静默使用随机模型），比崩溃更危险 |
| `app/session.py` | 需适配 create_engine 不再回退 RandomModel 的行为，catch ValueError 转换为 ModelLoadError（详见 §9） |
| `app/battle.py` | 需在 run() 中 catch create_engine 的 ValueError，转换为 ModelLoadError（详见 §9） |
| `app/exceptions.py` | 新增文件，定义 `ModelLoadError` 领域异常（详见 §9.1） |
| `gui/ai_battle.py` | `AIBattleProgressWindow.start_battle()` 内部调用 `BattleSession.run()`，需 catch `ModelLoadError` 弹窗提示（详见 §9.4） |

### 无需修改（自动适配）

| 文件 | 原因 |
|------|------|
| `cli/main.py` | 纯人类对战，不涉及 AI |

## 7. 单代模型迁移到多代

若要将现有单代模型（如 `cnn_v3`）迁移到多代支持：

1. 将 `ai/models/cnn_v3/cnn_v3.pth` 移动到 `ai/models/cnn_v3/gen1/cnn_v3.pth`
2. 在 `ai/registry.py` 的 `MODEL_REGISTRY` 中将 `cnn_v3` 的 `supports_multi_gen` 改为 `True`
3. 验证：`get_model("cnn_v3.1")` 应成功加载迁移后的权重

**注意**：迁移后 `get_model("cnn_v3")` 将报错"多代模型必须指定代数"，因为 `supports_multi_gen=True` 的模型始终要求指定代数。需确保所有调用点更新为 `cnn_v3.1` 格式。

## 8. 已知限制

1. **模型保存非原子**：若进程在 `torch.save()` 执行期间被中断，可能导致 `.pth` 文件损坏。建议在生产训练中使用"写入临时文件→原子重命名"策略。
2. **目录扫描无缓存**：`list_models_with_full_name()` 每次调用都会扫描文件系统。当前模型数量（<10）下开销可忽略；若未来模型数量显著增长，建议引入惰性缓存。
3. **并发访问假设**：当前设计假设单进程串行访问模型目录。若未来引入多进程并行训练不同代数，需要增加文件锁或原子保存机制。
4. **模型命名约束**：模型名称禁止包含 `.`（点号），由 MODEL_REGISTRY 注册时硬校验强制执行（`if '.' in name: raise ValueError`），消除 `_parse_model_spec` 的解析歧义。

## 9. GUI 链路异常处理设计

`engine.py` 取消回退后，`create_engine()` 遇模型加载失败时直接上抛 ValueError。
app 层作为代理层，负责将底层 ValueError 转换为领域异常 `ModelLoadError`，
使 GUI 层只需 catch 一种异常类型即可统一处理模型加载失败场景。

### 9.1 新增 `app/exceptions.py`

```python
class ModelLoadError(Exception):
    """模型加载失败异常

    由 app 层在捕获 create_engine 的 ValueError 后抛出，
    向 GUI 层传达模型加载失败的详细信息。
    GUI 层捕获此异常后弹窗提示用户，回退到模式选择界面。
    """
```

### 9.2 app/session.py 变更

```python
class GameSession:
    def __init__(self, mode, controller, ai_name=None, ai_player=None):
        ...
        if mode == "pve" and ai_player is not None:
            try:
                engine = create_engine(ai_name or "random")
            except ValueError as e:
                raise ModelLoadError(f"模型加载失败：{e}") from e
            self._ai_manager = AIGameManager(controller, engine, ai_player)
```

**伪代码**：

```
1. try: engine = create_engine(ai_name)
2. except ValueError → raise ModelLoadError(包含原始错误信息)
3. ModelLoadError 携带用户可读的错误信息，如：
   - "模型加载失败：多代模型必须指定代数，如 cnn_v3.1"
   - "模型加载失败：未知的模型名称 'xxx'"
   - "模型加载失败：非多代模型不能指定代数"
```

### 9.3 app/battle.py 变更

```python
class BattleSession:
    def run(self, on_progress=None):
        try:
            engine1 = create_engine(self._ai1_name)
            engine2 = create_engine(self._ai2_name)
        except ValueError as e:
            raise ModelLoadError(f"模型加载失败：{e}") from e
        ...
```

**伪代码**：

```
1. try: engine1 = create_engine(ai1_name); engine2 = create_engine(ai2_name)
2. except ValueError → raise ModelLoadError(包含原始错误信息)
3. 任一引擎创建失败均中断对战流程
```

### 9.4 GUI 层处理

GUI 层在创建 `GameSession` 或调用 `BattleSession.run()` 时，
catch `ModelLoadError` → 弹 messagebox 提示用户具体错误信息 → 回退到模式选择界面。

```python
# 人机对战模式
try:
    session = GameSession(mode="pve", controller=controller, ai_name=ai_name, ai_player=ai_player)
except ModelLoadError as e:
    messagebox.showerror("模型加载失败", str(e))
    # 回退到模式选择界面

# AI 内战模式
try:
    results = battle_session.run(on_progress=callback)
except ModelLoadError as e:
    messagebox.showerror("模型加载失败", str(e))
    # 回退到模式选择界面
```

### 9.5 异常处理测试用例

| 场景 | 触发条件 | 期望结果 | 说明 |
|------|---------|---------|------|
| GameSession 模型不存在 | `ai_name="unknown"` | raise ModelLoadError("模型加载失败：未知的模型名称") | ValueError → ModelLoadError 转换 |
| GameSession 多代未指定代数 | `ai_name="cnn_v3"`（多代模型） | raise ModelLoadError("模型加载失败：多代模型必须指定代数") | 提示用户正确格式 |
| BattleSession 黑方模型失败 | `ai1_name="unknown"` | raise ModelLoadError | 引擎1创建失败 |
| BattleSession 白方模型失败 | `ai2_name="unknown"` | raise ModelLoadError | 引擎2创建失败 |
| GameSession 正常加载 | `ai_name="random"` | 无异常，session 正常创建 | 正常路径 |

## 10. 附录：辩论修正

### 10.1 代语义表述修正（来源：架构辩论报告）

**原表述**："代语义用户自定义，代码不做假设"

**问题**：该表述与代码行为矛盾——代码中 `_parse_model_spec` 将代数硬编码为正整数解析（`gen < 1` 报错、非整数报错），目录结构硬编码为 `gen{i}/` 格式。框架已在代码层面对"代"做出了明确的语义假设（正整数序号），文档声称"不做假设"会误导开发者。

**修正为**：**代数为正整数序号，其业务语义（训练配置、数据版本等）由用户在代码外自行管理。**

具体含义：
- **格式由框架约束**：代数必须是 ≥ 1 的整数，对应目录 `gen{i}/`，这是代码强制执行的
- **业务含义由用户赋予**：`gen1` 代表"基线训练"还是"旧数据训练"、`gen2` 代表"超参调整"还是"数据增强"，框架不记录、不验证、不约束，由用户在代码外（文档、提交信息、实验管理工具）自行管理
- **与现有代码一致**：`_parse_model_spec` 的整数解析、`gen < 1` 校验、`gen{i}/` 目录命名均与此表述对齐

### 10.2 NamedTuple 类型安全改造（来源：架构辩论报告）

**问题**：原方案中 `MODEL_REGISTRY` 的 value type 和 `_parse_model_spec` 的返回值均使用裸 `tuple`，字段含义仅靠位置索引和注释传达，存在错配风险且可读性差。

**改造方案**：将两处裸 tuple 替换为 `NamedTuple`，字段名自文档化，调用方通过属性访问替代位置索引。

#### 10.2.1 MODEL_REGISTRY — RegistryEntry

**改前**：`dict[str, tuple[str, str | None, bool]]`

**改后**：

```python
from typing import NamedTuple

class RegistryEntry(NamedTuple):
    model_class_path: str
    trainer_class_path: str | None
    supports_multi_gen: bool

MODEL_REGISTRY: dict[str, RegistryEntry] = {
    # name: RegistryEntry(model_class_path, trainer_class_path, supports_multi_gen)
    # 硬约束：模型名称禁止包含 "."（点号），确保 _parse_model_spec 的点号唯一语义为代数分隔符
    "random": RegistryEntry("ai.models.random.model.RandomModel", None, False),
    "fc_v1":  RegistryEntry("ai.models.fc_v1.model.FCv1Model", "ai.models.fc_v1.trainer.FCv1Trainer", False),
    "cnn_v1": RegistryEntry("ai.models.cnn_v1.model.CNNV1Model", "ai.models.cnn_v1.trainer.CNNV1Trainer", False),
    "cnn_v2": RegistryEntry("ai.models.cnn_v2.model.CNNV2Model", "ai.models.cnn_v2.trainer.CNNV2Trainer", False),
    "cnn_v3": RegistryEntry("ai.models.cnn_v3.model.CNNV3Model", "ai.models.cnn_v3.trainer.CNNV3Trainer", True),
}
```

**调用方变化**：`entry[0]` → `entry.model_class_path`，`entry[1]` → `entry.trainer_class_path`，`entry[2]` → `entry.supports_multi_gen`。字段顺序调整不再导致静默错配。

#### 10.2.2 _parse_model_spec 返回值 — ParsedModelSpec

**改前**：`tuple[str, int | None]`

**改后**：

```python
class ParsedModelSpec(NamedTuple):
    name: str
    gen: int | None

def _parse_model_spec(spec: str) -> ParsedModelSpec:
    ...
```

**调用方变化**：
- 解构语法完全兼容：`name, gen = _parse_model_spec(spec)` 仍然可用（NamedTuple 是 tuple 子类）
- 新增属性访问：`spec.name`、`spec.gen`，IDE 可自动补全
- Returns 文档从 `(model_name, gen)` 改为 `ParsedModelSpec(name=model_name, gen=代数或None)`
