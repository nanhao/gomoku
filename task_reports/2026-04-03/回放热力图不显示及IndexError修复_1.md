# 回放热力图不显示及 IndexError 修复

## 1. 需求内容

用户反馈：使用回放功能时热力图没有显示，关闭窗口后终端报错：

```
File "replay_viewer.py", line 344, in _update_info_text
    if output_vec[i*3] == pos[0] and output_vec[i*3+1] == pos[1]:
IndexError: list index out of range
```

## 2. 解决思路

**根本原因**：`replay_viewer.py` 错误地假设 `output` 字段格式为 `(x, y, reward) * 225 = 675` 个元素，而项目实际规范中 `output` 是 **225 维列表**，每个元素按行优先顺序（索引 0~224）对应棋盘位置的 reward 值，不含坐标字段。

具体错误：
- `_draw_heatmaps`：用 `len(output_vector) >= 675` 和 `output_vector[i*3+2]` 读取 reward，实际 output 仅 225 元素，永远不满足条件 → 热力图始终不显示
- `_update_info_text`：用循环 `output_vec[i*3]` / `output_vec[i*3+1]` 比较坐标查找落子 reward，当 output 为 225 元素时索引越界 → IndexError

## 3. 执行过程

1. 阅读 `replay_viewer.py`，定位两处错误代码（第 278-288 行、第 335-347 行）
2. 查阅 save 文件和测试文件，确认项目 output 格式规范为 225 维
3. 修复 `_draw_heatmaps`：
   - 将判断条件 `>= 675` 改为 `>= 225`
   - 将 `[output_vector[i*3+2] for i in range(225)]` 改为 `list(output_vector[:225])`
   - 将 sample 热力图的条件 `step - 1 < len(sample_data)` 改为 `step > 0 and step - 1 < len(sample_data)`
4. 修复 `_update_info_text`：
   - 将遍历查找坐标改为直接用 `idx = pos[0] * 15 + pos[1]` 索引读取 reward
   - 将条件 `step <= len(self.sample_data)` 改为 `step > 0 and step <= len(self.sample_data)`
5. 更新测试文件 `test_replay_viewer.py`：将测试数据 output 从 675 格式改为 225 格式
6. 添加 4 个边缘测试用例覆盖 Bug 场景
7. 运行测试：19 passed, 1 error（error 为 Tkinter 环境问题，与代码无关）

## 4. 执行结果

- 热力图显示修复：output 为 225 维时能正确读取并绘制热力图
- IndexError 修复：跳转到结局时不再报错
- 新增 4 个边缘测试用例全部通过

## 5. 涉及文件

| 文件 | 操作 |
|------|------|
| `gui/components/replay_viewer.py` | 修复 `_draw_heatmaps` 和 `_update_info_text` |
| `tests/gui/test_replay_viewer.py` | 更新测试数据格式，添加边缘用例 |
