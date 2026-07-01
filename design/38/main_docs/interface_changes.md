# Interface Changes

## CLI Arguments

**无新增参数。** 保存间隔固定为 50 局，不可配置。

现有参数保持不变：
- `--model`, `--games`, `--simulations`, `--c-puct`, `--temp-steps`, `--temperature`, `--dirichlet-epsilon`, `--dirichlet-alpha`, `--output`, `--device`

## Output Files

### sample.json

格式不变：JSON 数组，每项为 `{"state": int[226], "pi": float[225], "z": int}`。

行为变更：
- **启动时**：若文件已存在，读取其内容作为初始数据
- **运行中**：每 50 局覆盖写入（含旧数据 + 新样本）
- **Ctrl+C 时**：覆盖写入当前累积样本
- **正常结束时**：覆盖写入全部样本（与旧行为一致）

### summary.json

新增字段 `completed_games` 记录当前已完成局数。

```json
{
    "model": "string",
    "games": 100,
    "total_samples": 12345,
    "completed_games": 75,
    "hyperparams": { ... }
}
```

## run_self_play() Function

签名不变，行为变更如 [main_design.md](main_design.md) §3 所述。
