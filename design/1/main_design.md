# create_samples 验收工具 — 整体架构设计

## 1. 元信息

| 项目 | 值 |
|------|-----|
| intention_id | 1 |
| 对应 PRD | requirements/1/PRD.md |
| 技术栈 | Python Flask + 原生 HTML/CSS/JS |
| 状态 | 待评审 |

## 2. 架构概览

### 2.1 系统边界

本工具为**单用户本地 Web 验收工具**，严格遵循零耦合原则：

- **不 import 任何 ai/ 包模块**
- 通过**子进程**调用 `ai/create_samples.py` 执行对弈
- 通过**读取中间产物**（record.json、样本输出文件）进行可视化
- 工具本身**不含业务逻辑**（对弈编排、样本生成、统计计算等）

### 2.2 架构分层

```
┌─────────────────────────────────────────────────┐
│                   浏览器 (Frontend)               │
│  参数填写区 │ 脚本输出区 │ 温度信息 │ 对局列表 │ 热力图 │
└──────────────────────┬──────────────────────────┘
                       │ HTTP
┌──────────────────────┴──────────────────────────┐
│                Flask 后端 (Backend)               │
│  API 路由 │ 子进程管理（同步阻塞） │ 数据解析 │
└──────┬───────────────────────────┬──────────────┘
       │ subprocess                 │ 读取文件
       ▼                           ▼
┌──────────────┐         ┌──────────────────────┐
│create_samples│──产出──▶│  ai/ai_data/{round}/ │
│    .py       │         │  ├─ 1/record.json    │
│              │         │  ├─ 2/record.json    │
│              │──产出──▶│  └─ ...              │
│              │         │  {output_path}       │
└──────────────┘         └──────────────────────┘
```

### 2.3 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | Flask | 轻量，项目已有 Python 环境，无需额外运行时 |
| 实时输出 | 无（同步阻塞等待） | POST /api/create-samples/execute 阻塞等待子进程结束，一次性返回全部输出 + 对局列表 |
| 前端 | 原生 HTML/CSS/JS + Canvas | 原型已用纯 HTML/CSS，热力图用 Canvas 渲染 |
| 子进程 | `subprocess.Popen` + `communicate()` | 同步阻塞等待，一次性收集输出 |
| 数据格式 | JSON | record.json 和样本输出均为 JSON |

## 3. 模块划分

```
tools/create_samples_tool/           ← 验收工具根目录
├── app.py                           ← Flask 应用入口 + 路由定义
├── executor.py                      ← 子进程管理（同步执行、输出收集）
├── data_parser.py                   ← 数据解析（record.json + 样本匹配）
├── static/
│   ├── index.html                   ← 主页面
│   ├── style.css                    ← 样式
│   └── app.js                       ← 前端逻辑
└── templates/                       ← (可选，如需 Jinja2)
```

### 3.1 模块职责

| 模块 | 职责 | 依赖 |
|------|------|------|
| `app.py` | Flask 路由、启动服务器 | executor, data_parser |
| `executor.py` | 同步执行子进程、收集输出、发现 round_dir | subprocess (标准库) |
| `data_parser.py` | 解析 record.json、样本文件匹配 | json, os (标准库) |

### 3.2 依赖约束

```
app.py → executor.py (同步执行子进程)
app.py → data_parser.py (解析数据)
executor.py → 无外部依赖（仅标准库 subprocess）
data_parser.py → 无外部依赖（仅标准库 json, os）
```

**禁止**：任何模块 import `ai.*` 包。

## 4. 关键流程概要

### 4.1 执行流程

1. 用户填写参数 → POST /api/create-samples/execute
2. 后端构造 CLI 命令，启动子进程，**同步阻塞等待**结束
3. 子进程结束后，POST 响应一次性返回完整输出 + 对局列表
4. 前端展示脚本输出和对局列表
5. 用户点击某局，传入 round_dir + sample_path → GET steps

→ 详细交互图见 [execute_flow.puml](execute_flow.puml)

### 4.2 对局查看流程

1. 用户点击对局列表中的某局
2. 前端请求该局的步数据
3. 后端读取对应 record.json + 匹配样本数据
4. 前端渲染三行热力图

→ 详细交互图见 [game_click_flow.puml](game_click_flow.puml)

## 5. 数据流向

```
用户 → [参数填写区] → POST /api/create-samples/execute → 后端构造 CLI 命令
                                            ↓ subprocess.Popen + communicate()
                                    create_samples.py 执行 (阻塞等待)
                                            ↓ 结束后
                        POST 响应 → [脚本输出区] + [对局列表区] + [温度信息区(前端本地)]
                               后端扫描 ai_data_dir → 发现 round_dir
                                            ↓ 用户点击某局
                 GET /api/create-samples/games/{id}/steps?round_dir=xxx&sample_path=xxx → [三行热力图区]
                                            ↓
                        record.json + 样本文件 → 解析匹配 → 渲染
```

## 6. 同步阻塞执行策略

### 6.1 核心机制

POST /api/create-samples/execute 调用 `process.communicate()` **同步阻塞**等待子进程结束，HTTP 响应在子进程完成后才返回。前端使用 `await fetch(...)` 等待响应。

- 无需轮询、无需 SSE、无需后台线程
- Flask 配置 `threaded=True` 确保不阻塞其他请求
- 执行期间前端显示"执行中..."提示

### 6.2 round_dir 发现

子进程完成后，后端通过以下方式定位数据目录：

1. 执行前：记录 `ai/ai_data/` 下已有目录集合 `S_before`
2. 执行后：扫描 `ai/ai_data/` 得到新集合 `S_after`
3. `S_after - S_before` 中的目录即为本次 round_dir
4. 若差集为空（极端情况）→ 返回错误

## 7. 零耦合守则

| 规则 | 说明 |
|------|------|
| 禁止 import ai.* | 验收工具不依赖任何业务模块 |
| 仅通过子进程调用 | create_samples.py 的唯一调用方式 |
| 仅读取中间产物 | record.json + 样本输出文件 |
| 不含业务逻辑 | 对弈编排、样本生成、统计计算等一律不实现 |
| 参数零校验 | 前端后端均不校验参数，透传给 argparse |

## 8. 关键设计决策

| # | 决策 | 理由 |
|---|------|------|
| D1 | 工具放在 tools/create_samples_tool/ | 与业务代码隔离，独立部署 |
| D2 | 同步阻塞而非异步+轮询 | communicate() 阻塞等待最简单，本地工具无需 SSE/轮询复杂度 |
| D3 | Flask 而非 FastAPI | 轻量，项目无需异步框架 |
| D4 | 前端原生 JS + Canvas | 原型已是纯 HTML，无需引入框架 |
| D5 | 参数零校验透传 | 项目决策：由 argparse 统一校验 |
| D6 | 无 run_id | 单用户本地工具，仅维护最近一次执行 |
| D7 | round_dir 差集发现 | 不修改 create_samples.py，通过文件系统差集定位 |
| D8 | step_id 匹配替代 input 向量比对 | 可靠、高性能、支持 D4 增强；详见 [step_id_matching_design.md](step_id_matching_design.md) |

## 9. 子文档索引

| 子文档 | 路径 | 说明 |
|--------|------|------|
| 后端 API 设计 | [backend_api_design.md](backend_api_design.md) | 端点、参数、响应、核心动作 |
| 数据解析设计 | [data_parser_design.md](data_parser_design.md) | record.json 解析、样本匹配、边界处理 |
| step_id 匹配设计 | [step_id_matching_design.md](step_id_matching_design.md) | step_id 格式、ai/ 包变更、匹配逻辑改造 |
| 执行交互图 | [execute_flow.puml](execute_flow.puml) | 点击执行按钮的交互流程 |
| 对局点击交互图 | [game_click_flow.puml](game_click_flow.puml) | 点击对局查看热力图的交互流程 |

## 10. 不纳入范围

| 项 | 排除原因 |
|----|---------|
| 热力图点击放大 | PRD v1.0 排除 |
| 多局同时展示 | PRD v1.0 仅展示单局 |
| 实时进度流式推送 | PRD v1.0 排除（执行完毕后一次性展示） |
| GUI 主程序改造 | 与本工具无关 |
| 修改 ai/ 包任何代码 | 零耦合原则 |
