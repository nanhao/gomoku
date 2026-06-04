/**
 * create_samples 验收工具 — 前端逻辑
 *
 * 功能：
 * 1. 页面初始化：fetch 模型列表填充下拉框
 * 2. 执行逻辑：POST /api/create-samples/execute + 按钮灰化
 * 3. 对局列表渲染：从 execute 响应的 games 渲染
 * 4. 温度信息渲染：前端表单本地持有
 * 5. 步数据请求：GET /api/create-samples/games/{id}/steps
 * 6. Canvas 渲染：复用热力图颜色方案 + 棋盘绘制
 * 7. 三行热力图构建：Row1/Row2/Row3 条件渲染
 * 8. Hover Tooltip：显示概率值
 */

// ===== 常量 =====
const BOARD_SIZE = 15;
const CANVAS_SIZE = 180;

// 色板: 白→黄→橙→红（与 heatmap_color.py DEFAULT_COLOR_STOPS 一致）
const COLOR_STOPS = [
    [1.0, 1.0, 1.0],   // 白
    [1.0, 0.94, 0.0],   // 黄
    [1.0, 0.55, 0.0],   // 橙
    [1.0, 0.0, 0.0],    // 红
];

// 星位点（标准五子棋 9 个星位）
const STAR_POINTS = [
    [3,3],[3,7],[3,11],[7,3],[7,7],[7,11],[11,3],[11,7],[11,11]
];

// ===== 全局状态 =====
let state = {
    roundDir: null,     // 当前 execute 返回的 round_dir
    outputPath: null,   // 当前 execute 返回的 output_path
    games: [],          // 当前对局列表
    selectedGameId: null, // 当前选中的对局 ID
    currentSteps: null, // 当前对局的步数据
    currentProbsData: null, // 当前步数据中的概率数据（用于 hover）
};

// ===== 页面初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    document.getElementById('btnRun').addEventListener('click', onExecute);
});

/**
 * 加载可用模型列表
 */
async function loadModels() {
    try {
        const resp = await fetch('/api/create-samples/models');
        const data = await resp.json();
        const select = document.getElementById('modelSelect');
        // 清空现有选项（保留占位符）
        select.innerHTML = '<option value="" disabled selected>— 请选择模型 —</option>';
        for (const model of data.models) {
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model;
            select.appendChild(opt);
        }
    } catch (e) {
        console.error('加载模型列表失败:', e);
    }
}

// ===== 执行逻辑 =====

/**
 * 收集表单参数
 */
function collectParams() {
    const tempSeedVal = document.getElementById('tempSeedInput').value;
    return {
        model: document.getElementById('modelSelect').value,
        mode: document.getElementById('modeSelect').value,
        games: parseInt(document.getElementById('gamesInput').value) || 100,
        output: document.getElementById('outputInput').value,
        no_augment: document.getElementById('noAugmentCheck').checked,
        no_temp_sampling: document.getElementById('noTempSamplingCheck').checked,
        temperature: parseFloat(document.getElementById('temperatureInput').value) || 1.0,
        temp_moves: parseInt(document.getElementById('tempMovesInput').value) || 12,
        temp_side: document.getElementById('tempSideSelect').value,
        temp_seed: tempSeedVal ? parseInt(tempSeedVal) : null,
    };
}

/**
 * 点击执行按钮
 */
async function onExecute() {
    const btn = document.getElementById('btnRun');
    const output = document.getElementById('scriptOutput');

    // 前端必填校验
    const params = collectParams();
    if (!params.model) {
        alert('请先选择模型');
        return;
    }
    if (!params.output || !params.output.trim()) {
        alert('output 为必填项，请填写输出路径（如 ai/train_data/samples.json）');
        return;
    }

    // 按钮灰化
    btn.disabled = true;
    btn.textContent = '执行中...';
    output.textContent = '执行中...\n';

    try {
        const resp = await fetch('/api/create-samples/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        const result = await resp.json();

        // 展示脚本输出
        output.textContent = result.output || '(无输出)';

        if (result.status === 'completed') {
            // 保存状态
            state.roundDir = result.round_dir;
            state.outputPath = result.output_path;
            state.games = result.games || [];
            state.selectedGameId = null;
            state.currentSteps = null;

            // 渲染温度信息（前端本地持有）
            renderTempInfo(params);

            // 渲染对局列表
            renderGameList(state.games);

            // 默认选中第一局
            if (state.games.length > 0) {
                selectGame(state.games[0].game_id);
            }
        } else {
            // 执行失败
            state.games = [];
            renderGameList([]);
            document.getElementById('heatmapTitle').textContent = '热力图对比 — 执行失败';
            document.getElementById('heatmapHint').textContent = '无数据';
            document.getElementById('heatmapHint').style.display = 'block';
            document.getElementById('heatmapContainer').innerHTML = '';
            document.getElementById('legendRow').style.display = 'none';
        }
    } catch (e) {
        output.textContent += '\n网络错误: ' + e.message;
    } finally {
        // 恢复按钮
        btn.disabled = false;
        btn.textContent = '▶ 执行';
    }
}

// ===== 温度信息渲染 =====

/**
 * 渲染温度信息区（前端本地持有，不随对局切换变化）
 */
function renderTempInfo(params) {
    const container = document.getElementById('tempInfoContent');
    // 判断是否禁用了温度采样
    const isDisabled = params.no_temp_sampling;
    const tempSideMap = {
        'both': '双方', 'black': '黑方', 'white': '白方', 'random': '随机'
    };
    container.innerHTML = `
        <div class="temp-info-grid">
            <span class="info-label">温度值</span>
            <span class="info-value ${isDisabled ? '' : 'highlight'}">${isDisabled ? '禁用' : 'T = ' + params.temperature}</span>
            <span class="info-label">采样步数阈值</span>
            <span class="info-value">${isDisabled ? '—' : '前 ' + params.temp_moves + ' 步'}</span>
            <span class="info-label">采样方</span>
            <span class="info-value">${isDisabled ? '—' : (tempSideMap[params.temp_side] || params.temp_side)}</span>
        </div>
    `;
}

// ===== 对局列表渲染 =====

/**
 * 渲染对局列表
 */
function renderGameList(games) {
    const container = document.getElementById('gameListContent');
    const countLabel = document.getElementById('gameCountLabel');
    countLabel.textContent = games.length > 0 ? `(${games.length}局)` : '';

    if (games.length === 0) {
        container.innerHTML = '<span class="game-list-hint">无对局数据</span>';
        return;
    }

    const ul = document.createElement('ul');
    ul.className = 'game-list';

    for (const game of games) {
        const li = document.createElement('li');
        li.className = 'game-item' + (game.is_draw ? ' draw' : '');
        li.dataset.gameId = game.game_id;

        // 胜负方 CSS 类
        let winnerClass = '';
        if (game.winner === 1) winnerClass = 'black-win';
        else if (game.winner === 2) winnerClass = 'white-win';
        else winnerClass = 'draw-result';

        li.innerHTML = `
            <span class="game-num">#${game.game_id}</span>
            <span class="game-winner ${winnerClass}">${game.winner_display}</span>
            <span class="game-temp-side">${game.temp_side_display}</span>
        `;

        li.addEventListener('click', () => selectGame(game.game_id));
        ul.appendChild(li);
    }

    container.innerHTML = '';
    container.appendChild(ul);
}

/**
 * 选中某局
 */
async function selectGame(gameId) {
    state.selectedGameId = gameId;

    // 更新列表高亮
    document.querySelectorAll('.game-item').forEach(item => {
        item.classList.toggle('selected', parseInt(item.dataset.gameId) === gameId);
    });

    // 请求步数据
    try {
        const url = `/api/create-samples/games/${gameId}/steps`
            + `?round_dir=${encodeURIComponent(state.roundDir)}`
            + `&sample_path=${encodeURIComponent(state.outputPath)}`;
        const resp = await fetch(url);
        const data = await resp.json();

        state.currentSteps = data;
        renderHeatmap(data);
    } catch (e) {
        console.error('获取步数据失败:', e);
    }
}

// ===== 热力图渲染 =====

/**
 * 渲染三行热力图
 */
function renderHeatmap(gameData) {
    const container = document.getElementById('heatmapContainer');
    const hint = document.getElementById('heatmapHint');
    const title = document.getElementById('heatmapTitle');
    const legend = document.getElementById('legendRow');

    container.innerHTML = '';
    hint.style.display = 'none';
    legend.style.display = 'flex';

    const steps = gameData.steps || [];
    if (steps.length === 0) {
        hint.textContent = '无步数据';
        hint.style.display = 'block';
        legend.style.display = 'none';
        return;
    }

    // 更新标题
    const drawSuffix = gameData.is_draw ? '（平局无样本）' : '';
    title.textContent = `热力图对比 — 第${gameData.game_id}局（${gameData.winner_display}${drawSuffix}）`;

    const grid = document.createElement('div');
    grid.className = 'heatmap-grid';

    // 步数标注行
    const stepRow = document.createElement('div');
    stepRow.className = 'step-labels-row';
    const spacer = document.createElement('div');
    spacer.className = 'row-label';
    stepRow.appendChild(spacer);
    const stepCells = document.createElement('div');
    stepCells.className = 'heatmap-cells-row';
    for (const step of steps) {
        const label = document.createElement('div');
        label.className = 'step-label';
        label.textContent = `Step ${step.step_index + 1}`;
        stepCells.appendChild(label);
    }
    stepRow.appendChild(stepCells);
    grid.appendChild(stepRow);

    // 三行数据
    const rowLabels = ['原始输出', '温度分布', '样本分布'];
    // 每行对应的数据获取函数
    const rowGetters = [
        (step) => step.raw_output,        // Row 1: 原始输出
        (step) => step.sampling_probs,     // Row 2: 温度分布
        (step) => step.sample_output,      // Row 3: 样本分布
    ];
    // 每行占位文字
    const placeholderTexts = [null, 'argmax', '无样本'];

    // 收集每步的概率数据用于 hover（在渲染循环之前收集，确保所有行可用）
    state.currentProbsData = steps.map(step => ({
        stepIndex: step.step_index,
        raw_output: step.raw_output,
        sampling_probs: step.sampling_probs,
        sample_output: step.sample_output,
    }));

    for (let row = 0; row < 3; row++) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'heatmap-row';
        const labelDiv = document.createElement('div');
        labelDiv.className = 'row-label';
        labelDiv.textContent = rowLabels[row];
        rowDiv.appendChild(labelDiv);
        const cellsDiv = document.createElement('div');
        cellsDiv.className = 'heatmap-cells-row';

        for (let s = 0; s < steps.length; s++) {
            const step = steps[s];
            const probs = rowGetters[row](step);

            const wrap = document.createElement('div');
            wrap.className = 'board-canvas-wrap';

            if (probs !== null && probs !== undefined) {
                // 正常热力图
                const canvas = document.createElement('canvas');
                canvas.width = CANVAS_SIZE;
                canvas.height = CANVAS_SIZE;
                wrap.appendChild(canvas);
                drawBoard(canvas, step.board, probs, step.move_position);
                // hover 事件
                setupHover(canvas, step.step_index, row);
            } else {
                // 占位图
                const placeholder = document.createElement('div');
                placeholder.className = 'placeholder-board';
                placeholder.textContent = placeholderTexts[row];
                wrap.appendChild(placeholder);
            }

            cellsDiv.appendChild(wrap);
        }

        rowDiv.appendChild(cellsDiv);
        grid.appendChild(rowDiv);
    }

    container.appendChild(grid);
}

// ===== Canvas 渲染函数 =====

/**
 * 将 [0,1] 值映射为 RGB 颜色
 */
function valueToRGB(v) {
    v = Math.max(0, Math.min(1, v));
    const n = COLOR_STOPS.length;
    const seg = 1.0 / (n - 1);
    let idx = Math.floor(v / seg);
    if (idx >= n - 1) idx = n - 2;
    const t = (v - idx * seg) / seg;
    const c1 = COLOR_STOPS[idx], c2 = COLOR_STOPS[idx + 1];
    return [
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    ];
}

/**
 * 在 canvas 上绘制棋盘热力图
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} board - 225 维棋盘状态 (0=空, 1=黑, 2=白)
 * @param {number[]} probs - 225 维概率分布
 * @param {number[]} movePos - 落子坐标 [row, col]
 */
function drawBoard(canvas, board, probs, movePos) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cellW = W / BOARD_SIZE, cellH = H / BOARD_SIZE;

    // 1) 棋盘底色
    ctx.fillStyle = '#DEB887';
    ctx.fillRect(0, 0, W, H);

    // 2) 热力图色块（空位上着色；落子位也着色但不画棋子）
    if (probs) {
        const maxProb = Math.max(...probs, 0.001);
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                const idx = r * 15 + c;
                // 已落子位不着色（落子位除外，需要着色）
                const isMovePos = movePos && r === movePos[0] && c === movePos[1];
                if (board[idx] !== 0 && !isMovePos) continue;
                const normalized = probs[idx] / maxProb;
                const rgb = valueToRGB(normalized);
                ctx.fillStyle = `rgba(${Math.round(rgb[0]*255)},${Math.round(rgb[1]*255)},${Math.round(rgb[2]*255)},0.65)`;
                ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
            }
        }
    }

    // 3) 网格线
    ctx.strokeStyle = '#8B7355';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= BOARD_SIZE; i++) {
        ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(W, i * cellH); ctx.stroke();
    }

    // 4) 星位点（跳过落子位）
    ctx.fillStyle = '#5D4037';
    for (const [sr, sc] of STAR_POINTS) {
        if (movePos && sr === movePos[0] && sc === movePos[1]) continue;
        ctx.beginPath();
        ctx.arc(sc * cellW + cellW/2, sr * cellH + cellH/2, 2, 0, Math.PI*2);
        ctx.fill();
    }

    // 5) 棋子（非落子位）
    const stoneR = cellW * 0.35;
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const idx = r * 15 + c;
            if (board[idx] === 0) continue;
            if (movePos && r === movePos[0] && c === movePos[1]) continue;
            const cx = c * cellW + cellW/2, cy = r * cellH + cellH/2;
            if (board[idx] === 1) {
                // 黑棋
                ctx.fillStyle = '#222';
                ctx.beginPath(); ctx.arc(cx, cy, stoneR, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = '#555';
                ctx.beginPath(); ctx.arc(cx - 1, cy - 1, stoneR*0.4, 0, Math.PI*2); ctx.fill();
            } else {
                // 白棋
                ctx.fillStyle = '#fff';
                ctx.beginPath(); ctx.arc(cx, cy, stoneR, 0, Math.PI*2); ctx.fill();
                ctx.strokeStyle = '#999';
                ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.arc(cx, cy, stoneR, 0, Math.PI*2); ctx.stroke();
            }
        }
    }

    // 6) 落子位绿色边框
    if (movePos) {
        const [mr, mc] = movePos;
        ctx.strokeStyle = '#0d6b2e';
        ctx.lineWidth = 2;
        ctx.strokeRect(mc * cellW, mr * cellH, cellW, cellH);
    }
}

// ===== Hover Tooltip =====

/**
 * 设置 canvas 的 hover 事件
 */
function setupHover(canvas, stepIndex, rowIndex) {
    const tooltip = document.getElementById('heatmapTooltip');

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const cellW = rect.width / BOARD_SIZE;
        const cellH = rect.height / BOARD_SIZE;
        const col = Math.floor(x / cellW);
        const row = Math.floor(y / cellH);

        if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
            tooltip.style.display = 'none';
            return;
        }

        // 获取对应行的概率值
        const probsInfo = state.currentProbsData.find(p => p.stepIndex === stepIndex);
        if (!probsInfo) { tooltip.style.display = 'none'; return; }

        let probs = null;
        let rowName = '';
        if (rowIndex === 0) { probs = probsInfo.raw_output; rowName = '原始'; }
        else if (rowIndex === 1) { probs = probsInfo.sampling_probs; rowName = '温度'; }
        else { probs = probsInfo.sample_output; rowName = '样本'; }

        if (!probs) { tooltip.style.display = 'none'; return; }

        const val = probs[row * 15 + col];
        tooltip.textContent = `${rowName} (${row},${col}): ${val.toFixed(4)}`;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 20) + 'px';
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}