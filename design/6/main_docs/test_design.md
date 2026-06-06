# 测试设计

---

## 1. 测试文件清单

| 文件 | 类别 | 说明 |
|------|------|------|
| `tests/ai/unit/test_iterative_train.py` | 单元测试 | IterativeTrainer 类各方法单元测试 |
| `tests/ai/integration/test_iterative_train.py` | 集成测试 | 完整迭代流程集成测试 |

---

## 2. 单元测试用例

### 2.1 构造器校验

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| UT-1.1 | gen4 权重存在时正常构造 | 临时目录含 gen4/cnn_v3.pth | 构造成功，不抛异常 |
| UT-1.2 | gen4 权重不存在时抛异常 | 临时目录无 gen4/ 目录 | 抛 `FileNotFoundError`，消息含 "gen4" |
| UT-1.3 | 自动推导 project_root | 不传 project_root | 正确推导到项目根目录 |
| UT-1.4 | 锁文件存在时拒绝启动 | 临时目录含 .iterative_lock | 抛 `RuntimeError`，消息含 "iterative" 或 "锁" |
| UT-1.5 | 锁文件不存在时正常构造 | 临时目录不含 .iterative_lock | 构造成功 |

### 2.2 模型文件操作

| 用例编号 | 用例描述 | 前置条件 | 预期输出 |
|----------|---------|---------|---------|
| UT-2.1 | copy gen4 → gen5（gen5 目录不存在） | gen4/cnn_v3.pth 存在，gen5/ 不存在 | 自动创建 gen5/，gen5/cnn_v3.pth 内容与 gen4 一致（md5 相同） |
| UT-2.2 | copy gen4 → gen5（gen5 目录已存在） | gen4/cnn_v3.pth 存在，gen5/cnn_v3.pth 已存在且内容不同 | gen5/cnn_v3.pth 被覆盖为 gen4 的副本 |
| UT-2.3 | 晋升操作正确更新 gen4 权重（copy2 + remove） | gen5/cnn_v3.pth 存在且内容 ≠ gen4/cnn_v3.pth | gen4/cnn_v3.pth 内容更新为晋升前 gen5 的内容（md5 校验一致），gen5/cnn_v3.pth 被删除 |
| UT-2.4 | 样本输出目录不存在时自动创建 | v4/ 目录不存在 | 调用前自动创建 v4/ 目录 |
| UT-2.5 | 晋升 copy2 成功后 remove 失败 | mock os.remove 抛 IOError | gen4 已更新（晋升语义正确），抛 IOError，gen5 副本残留 |

### 2.3 子进程调用参数构造

| 用例编号 | 用例描述 | 验证点 | 预期输出 |
|----------|---------|--------|---------|
| UT-3.1 | create_samples 调用参数正确 | 构造的 subprocess 命令行参数 | 包含 sys.executable、--model cnn_v3.4、--mode winner_path、--sample_games 500、--output 含 path1.json、--temp-side random |
| UT-3.2 | train_main 调用参数正确 | 构造的 subprocess 命令行参数 | 包含 sys.executable、--model cnn_v3.5、--train-from-dir 含 path1.json、--epochs 1、--batch-size 4096、--shuffle |
| UT-3.3 | battle 调用参数正确 | 构造的 subprocess 命令行参数 | 包含 sys.executable、--ai1 cnn_v3.4、--ai2 cnn_v3.5、--games 100 |
| UT-3.4 | 第 round_num 轨输出文件名为 path{round_num}.json | round_num=3 时构造的 output 参数 | 路径以 "path3.json" 结尾 |
| UT-3.5 | subprocess.run 传入 cwd=project_root | mock subprocess.run 的调用参数 | kwargs 含 cwd 且值为 self.project_root |
| UT-3.6 | subprocess.run 传入 PYTHONIOENCODING=utf-8 | mock subprocess.run 的调用参数 | env 字典含 PYTHONIOENCODING=utf-8 |
| UT-3.7 | subprocess.run 传入 encoding='utf-8' | mock subprocess.run 的调用参数（_generate_samples、_train_model、_run_battle 三处） | kwargs 含 encoding='utf-8' |

### 2.4 胜率解析

| 用例编号 | 用例描述 | 输入（battle stdout 片段） | 预期输出 |
|----------|---------|--------------------------|---------|
| UT-4.1 | 正常解析 cnn_v3.5 胜率 | `cnn_v3.5 胜: 48 局 (48.0%)` | 返回 48.0 |
| UT-4.2 | 解析含多行输出的完整 battle 输出 | 含逐局结果行 + 最终统计行 | 返回最终统计行中的胜率 |
| UT-4.3 | 胜率 100% | `cnn_v3.5 胜: 100 局 (100.0%)` | 返回 100.0 |
| UT-4.4 | 胜率 0% | `cnn_v3.5 胜: 0 局 (0.0%)` | 返回 0.0 |
| UT-4.5 | 无匹配行时抛异常 | `cnn_v3.5 胜: 无效` | 抛 `ValueError`，消息含 "胜率" |
| UT-4.6 | 胜率含多位小数 | `cnn_v3.5 胜: 48 局 (48.05%)` | 返回 48.05 |
| UT-4.7 | 含 cnn_v3.4 干扰行时正确匹配 cnn_v3.5 | `cnn_v3.4 胜: 52 局 (52.0%)` + `cnn_v3.5 胜: 48 局 (48.0%)` | 返回 48.0（匹配 cnn_v3.5 行，不误匹配 cnn_v3.4） |
| UT-4.8 | stdout 含进度等无关行时仍正确解析 | 含 "进度: 样本局 100/500" 等无关行 + 最终统计行 | 返回最终统计行中的胜率 |

### 2.5 迭代逻辑判定

| 用例编号 | 用例描述 | 给定胜率 | 预期行为 |
|----------|---------|---------|---------|
| UT-5.1 | 胜率 > 50.0% 应晋升 | 51.0 | 执行晋升操作（copy2 gen5→gen4 + remove gen5），round_num 自增，继续循环 |
| UT-5.2 | 胜率 = 50.0% 不应晋升 | 50.0 | 不执行晋升，停止循环，输出统计报告 |
| UT-5.3 | 胜率 < 50.0% 不应晋升 | 49.0 | 不执行晋升，停止循环，输出统计报告 |
| UT-5.4 | 胜率 = 50.1% 应晋升 | 50.1 | 执行晋升操作，继续循环 |

### 2.6 日志输出

| 用例编号 | 用例描述 | 验证点 | 预期输出 |
|----------|---------|--------|---------|
| UT-6.1 | 每轮开始时打印轮次标识 | 第 1 轮开始时的 stdout | 包含 "第 1 轨迭代开始" |
| UT-6.2 | 每轮结束时打印结果摘要（晋升） | 胜率 55.0% 时的 stdout | 匹配模式 `"第 1 轨结果: cnn_v3.5 胜率 = 55.0% → 晋升"` |
| UT-6.3 | 每轮结束时打印结果摘要（停止） | 胜率 48.0% 时的 stdout | 匹配模式 `"第 1 轨结果: cnn_v3.5 胜率 = 48.0% → 停止"` |
| UT-6.4 | 最终统计逐轮明细行格式（晋升） | 2 轨晋升后停止 | 匹配 `"第1轮: cnn_v3.4 胜 ... | cnn_v3.5 胜 ... | 平局 ... | cnn_v3.5 胜率 55.0% → 晋升"` |
| UT-6.5 | 最终统计逐轮明细行格式（停止） | 第 3 轨停止 | 匹配 `"第3轮: cnn_v3.4 胜 ... | cnn_v3.5 胜 ... | 平局 ... | cnn_v3.5 胜率 48.0% → 停止"` |

### 2.7 最终统计报告

| 用例编号 | 用例描述 | 前置条件 | 预期输出 |
|----------|---------|---------|---------|
| UT-7.1 | 报告含总迭代次数 | 完成 2 轮晋升后第 3 轮停止 | "总迭代次数: 2" |
| UT-7.2 | 报告含逐轮明细表 | 2 轮晋升后停止 | 表中有 3 行（第1轮晋升、第2轮晋升、第3轮停止），每行含胜场和平局 |
| UT-7.3 | 报告含停止原因 | 第 3 轮胜率 47% 停止 | 停止原因含 "第 3 轮" 和 "47.0%" |
| UT-7.4 | 报告含最终优胜模型 | 任意停止 | 含 "最终优胜模型" 和 gen4 路径 |
| UT-7.5 | 首轮即停止时统计正确 | 第 1 轮胜率 48% 停止 | 总迭代次数 0，逐轮明细 1 行，停止原因正确 |

### 2.8 边界条件

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| UT-8.1 | 首轮胜率 50.0% 停止 | 胜率 50.0 | 不晋升，总迭代次数 0 |
| UT-8.2 | 首轮胜率 50.1% 晋升 | 胜率 50.1 | 晋升，进入第 2 轮 |
| UT-8.3 | battle 输出仅有平局 | 0胜0负100平 | 胜率 0.0%，停止 |
| UT-8.4 | create_samples 生成空样本（0 条） | mock 子进程返回 exit_code=0，但样本文件为 `[]` | 在 train_main 调用前检测到空样本（读取 JSON 文件并校验长度），抛 RuntimeError，消息含 "空样本" |
| UT-8.5 | 样本文件不存在 | mock create_samples 成功但 path1.json 未实际生成 | 读取 JSON 文件失败，抛 RuntimeError |

---

## 3. 集成测试用例

### 3.1 完整迭代流程

| 用例编号 | 用例描述 | 前置条件 | 验证点 |
|----------|---------|---------|--------|
| IT-1.1 | 单轮完整流程（晋升） | mock 子进程：create_samples 返回 0、train_main 返回 0、battle 输出胜率 55% | ① gen4→gen5 被复制 ② 样本非空校验通过 ③ 三个子进程按序调用且 cwd=project_root ④ 晋升后 gen4 被更新（copy2+remove） ⑤ run() 返回 dict 含 1 轨晋升记录 |
| IT-1.2 | 单轮完整流程（停止） | mock 子进程均成功，battle 输出胜率 48% | ① 子进程按序调用 ② 不执行晋升 ③ gen4 未被修改 ④ run() 返回 dict 含 1 轨停止记录 |
| IT-1.3 | 多轮迭代（2 轨晋升后停止） | 第1轮胜率 55%、第2轮胜率 52%、第3轮胜率 48% | ① 共执行 3 轨 ② 前 2 轨晋升，第 3 轨停止 ③ 统计含 3 行明细 ④ total_iterations=2 |
| IT-1.4 | 子进程调用顺序正确 | 完整流程 | 验证 create_samples 在 train_main 之前调用，train_main 在 battle 之前调用 |

### 3.2 错误处理

| 用例编号 | 用例描述 | 前置条件 | 预期输出 |
|----------|---------|---------|---------|
| IT-2.1 | create_samples 失败 | mock create_samples 返回 exit_code=1 | 抛 RuntimeError，不执行 train_main 和 battle |
| IT-2.2 | train_main 失败 | mock train_main 返回 exit_code=1 | 抛 RuntimeError，不执行 battle |
| IT-2.3 | battle 失败 | mock battle 返回 exit_code=1 | 抛 RuntimeError |
| IT-2.4 | gen4 权重不存在 | 临时目录无 gen4/cnn_v3.pth | 构造时抛 FileNotFoundError |
| IT-2.5 | Ctrl+C 中断 | 第 2 轨样本生成中触发 KeyboardInterrupt | 打印 "用户中断"，输出已完成轮次的统计，正常退出 |
| IT-2.6 | 锁文件存在时构造失败 | 临时目录含 .iterative_lock | 构造时抛 RuntimeError |

### 3.3 文件系统副作用

| 用例编号 | 用例描述 | 验证点 | 预期输出 |
|----------|---------|--------|---------|
| IT-3.1 | 测试不修改真实模型文件 | 使用临时目录作为 project_root | 真实 `ai/models/cnn_v3/gen4/cnn_v3.pth` 不变 |
| IT-3.2 | 测试不修改真实样本文件 | 使用临时目录作为 project_root | 真实 `ai/merge_samples/v4/` 不变 |
| IT-3.3 | 测试结束后清理临时文件 | 临时目录在 tearDown 中删除 | 临时目录不存在 |
| IT-3.4 | 临时目录创建在 temp/ 下 | 使用 `temp/test_iterative_train/` 前缀 | 路径包含 temp/ |

---

## 4. 验收用例集

> 验收用例编号与 PRD §4 对齐：AC-1 至 AC-9。

### AC-1 验收：脚本文件存在且可执行（对应 PRD AC-1）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-1.1 | `ai/iterative_train.py` 文件存在 | 文件系统检查 |
| AC-1.2 | 脚本无语法错误 | `python -c "import ast; ast.parse(open('ai/iterative_train.py').read())"` 返回 0 |

### AC-2 验收：模型文件初始化（对应 PRD AC-2）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-2.1 | gen5 目录不存在时自动创建并复制 | 单元测试 UT-2.1 |
| AC-2.2 | gen5 目录已存在时直接覆盖 | 单元测试 UT-2.2 |

### AC-3 验收：样本生成子进程调用正确（对应 PRD AC-3）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-3.1 | create_samples 参数正确 | 单元测试 UT-3.1 |
| AC-3.2 | 样本文件输出到 path{round_num}.json | 集成测试 IT-1.1 中校验 mock 调用参数含正确路径 |
| AC-3.3 | v4 目录不存在时自动创建 | 单元测试 UT-2.4 |

### AC-4 验收：训练子进程调用正确（对应 PRD AC-4）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-4.1 | train_main 参数正确 | 单元测试 UT-3.2 |
| AC-4.2 | 训练后 gen5 权重已更新 | 集成测试 IT-1.1 中 mock 验证 train_main 在 gen5 上执行 |
| AC-4.3 | 训练使用正确的样本文件 | 单元测试 UT-3.2 中校验 --train-from-dir 参数 |

### AC-5 验收：对战评估与胜率判定（对应 PRD AC-5）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-5.1 | battle 参数正确 | 单元测试 UT-3.3 |
| AC-5.2 | 正确提取 cnn_v3.5 胜率 | 单元测试 UT-4.1~UT-4.4 |
| AC-5.3 | 胜率 > 50% 时继续循环 | 单元测试 UT-5.1、UT-5.4 |
| AC-5.4 | 胜率 ≤ 50% 时停止并报告 | 单元测试 UT-5.2、UT-5.3 |

### AC-6 验收：模型晋升（对应 PRD AC-6）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-6.1 | 胜率 > 50% 时 gen5 晋升为 gen4 | 单元测试 UT-2.3 |
| AC-6.2 | 晋升后 round_num 自增，下一轮输出文件名正确 | 单元测试 UT-3.4 |

### AC-7 验收：单轮日志（对应 PRD AC-7）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-7.1 | 每轮开始时打印轮次标识 | 单元测试 UT-6.1 |
| AC-7.2 | 每轮结束时打印结果摘要 | 单元测试 UT-6.2、UT-6.3 |

### AC-8 验收：最终统计报告（对应 PRD AC-8）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-8.1 | 停止后输出完整报告 | 单元测试 UT-7.1~UT-7.4 |
| AC-8.2 | 逐轮明细包含所有历史轮次 | 单元测试 UT-7.2 |
| AC-8.3 | 晋升标识正确 | 单元测试 UT-7.2 |

### AC-9 验收：错误处理（对应 PRD AC-9）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-9.1 | gen4 不存在时终止 | 单元测试 UT-1.2 |
| AC-9.2 | 子进程非零退出时终止 | 集成测试 IT-2.1~IT-2.3 |
| AC-9.3 | 子进程失败后不执行后续步骤 | 集成测试 IT-2.1 |

---

## 5. Mock 策略

### 5.1 子进程 Mock

所有子进程调用通过 `unittest.mock.patch` 替换 `subprocess.run` 实现。

**side_effect 策略**：由于一轮迭代内有 3 次不同的 subprocess.run 调用（create_samples、train_main、battle），mock 的 `side_effect` 设置为列表/函数，按调用顺序依次返回不同的 CompletedProcess：

- 第 1 次调用（create_samples）：模拟返回 `CompletedProcess(returncode=0, stdout="进度: ...", stderr="")`
- 第 2 次调用（train_main）：模拟返回 `CompletedProcess(returncode=0, stdout="...", stderr="")`
- 第 3 次调用（battle）：模拟返回 `CompletedProcess(returncode=0, stdout=含胜率文本, stderr="")`

多轮迭代场景下 side_effect 列表包含多轮的 mock 结果（3 × round_num 次）。失败场景使用特定位置的 returncode=1。

### 5.2 文件系统隔离

所有涉及文件读写的测试使用 `tempfile.TemporaryDirectory` 创建独立临时目录，作为 `project_root` 传入 IterativeTrainer。测试结束后自动清理。

### 5.3 不 Mock 的模块

- `os.path` / `shutil` / `os.remove`：文件操作逻辑通过临时目录真实执行，不 mock，以验证文件系统副作用（包括晋升的 copy2 + remove 两步操作）
- 日志输出：通过 `capsys` / `capsysbinary` fixture 捕获 stdout 进行验证

### 5.4 空样本 Mock

测试空样本场景（UT-8.4）时，mock subprocess.run 返回 returncode=0，同时在临时目录中预先写入 `[]` 作为 path{round_num}.json 内容，模拟 create_samples 成功但生成空样本文件的情况。文件不存在场景（UT-8.5）则不预先写入文件。