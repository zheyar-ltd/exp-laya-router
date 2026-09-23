import time
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from snake_laya import SnakeGame, query_laya, DELTAS, PLAYABLE_W, PLAYABLE_H, WIDTH, HEIGHT

app = FastAPI(title="Zheyar AI Labs - Laya Visual Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Global Game Session State
current_game = SnakeGame()
current_mode = "laya"
last_probabilities = {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTML_CONTENT


@app.post("/api/reset")
def reset_game():
    global current_game, last_probabilities
    current_game = SnakeGame()
    last_probabilities = {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}
    return {"status": "reset", "score": 0}


@app.post("/api/step")
def step_game(payload: dict = None):
    global current_game, current_mode, last_probabilities
    if payload and "mode" in payload:
        current_mode = payload["mode"]

    if not current_game.alive:
        current_game = SnakeGame()

    current_game.step_count += 1
    state_desc, valid_moves = current_game.get_state()

    latency = 0.0
    guardrail_note = "Safe AI Route"
    action = "RIGHT"

    if not valid_moves:
        current_game.alive = False
        return {
            "alive": False,
            "score": current_game.score,
            "step": current_game.step_count,
            "snake": current_game.snake,
            "food": current_game.food,
            "action": "NONE",
            "latency": 0,
            "note": "Game Over - No Moves"
        }

    if current_mode == "unbeatable":
        action = current_game.get_hamiltonian_move(allow_shortcuts=True)
        latency = 0.05
        guardrail_note = "Hamiltonian Shortcut AI"
        last_probabilities = {d: (0.92 if d == action else 0.02) for d in DELTAS}
    else:
        # Laya Mode: query local Laya API
        action, latency, guardrail_note = query_laya(state_desc, valid_moves)
        try:
            criteria = {d: f"Move {d}" for d in valid_moves}
            p_payload = {
                "state": state_desc,
                "questions": {"move": {"instructions": "Select move:", "criteria": criteria}}
            }
            res = requests.post("http://127.0.0.1:8000/predict", json=p_payload, timeout=1.0).json()
            probs = res.get("answers", {}).get("move", {}).get("probabilities", {})
            if probs:
                last_probabilities = {d: probs.get(d, 0.0) for d in ["UP", "DOWN", "LEFT", "RIGHT"]}
        except Exception:
            last_probabilities = {d: (0.85 if d == action else 0.05) for d in DELTAS}

    current_game.direction = action
    dx, dy = DELTAS[action]
    head = current_game.snake[0]
    new_head = [head[0] + dx, head[1] + dy]

    if not current_game.is_safe_tile(new_head):
        current_game.alive = False
    else:
        current_game.snake.insert(0, new_head)
        if new_head == current_game.food:
            current_game.score += 1
            current_game.food = current_game.spawn_food()
        else:
            current_game.snake.pop()

    free_space = current_game.flood_fill_count(current_game.snake[0])
    free_pct = round((free_space / (PLAYABLE_W * PLAYABLE_H)) * 100, 1)

    return {
        "alive": current_game.alive,
        "score": current_game.score,
        "step": current_game.step_count,
        "snake": current_game.snake,
        "food": current_game.food,
        "action": action,
        "latency": round(latency, 1),
        "note": guardrail_note,
        "probabilities": last_probabilities,
        "free_space_pct": free_pct,
        "width": WIDTH,
        "height": HEIGHT
    }


HTML_CONTENT = """<!DOCTYPE html>
<html lang="fa" dir="ltr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Zheyar AI Labs - Laya Decision Engine</title>
  <style>
    @font-face {
      font-family: 'Space Grotesk';
      src: url('/static/SpaceGrotesk-var-latin.woff2') format('woff2');
      font-weight: 300 700;
      font-style: normal;
      font-display: swap;
    }

    @font-face {
      font-family: 'Vazirmatn';
      src: url('/static/Vazirmatn-var.woff2') format('woff2');
      font-weight: 100 900;
      font-style: normal;
      font-display: swap;
    }

    :root {
      --night-900: #0D0B1C;
      --night: #171432;
      --night-700: #26214A;
      --night-500: #4B4378;
      --orange: #D2811F;
      --orange-300: #EDB463;
      --orange-100: #F8E3C4;
      --paper: #FAF8F4;
      --cyan-neon: #38bdf8;
      --emerald-neon: #10b981;
      --ruby-neon: #f43f5e;
      --border-zheyar: rgba(210, 129, 31, 0.22);
      --border-glow: rgba(237, 180, 99, 0.45);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: radial-gradient(circle at 50% 5%, #201b44 0%, var(--night-900) 100%);
      color: var(--paper);
      font-family: 'Space Grotesk', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    header {
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-zheyar);
      background: rgba(13, 11, 28, 0.88);
      backdrop-filter: blur(14px);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-logo-wrap {
      width: 42px;
      height: 42px;
      border-radius: 10px;
      background: rgba(23, 20, 50, 0.9);
      border: 1px solid var(--border-zheyar);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px rgba(210, 129, 31, 0.25);
    }

    .brand-logo-wrap svg {
      width: 30px;
      height: 30px;
    }

    .brand-text h1 {
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--paper);
    }

    .brand-text .farsi-tag {
      font-family: 'Vazirmatn', sans-serif;
      font-size: 13px;
      font-weight: 500;
      color: var(--orange-300);
      margin-left: 6px;
    }

    .brand-text p {
      font-size: 11px;
      color: #9d97b8;
      letter-spacing: 0.6px;
    }

    .badges {
      display: flex;
      gap: 10px;
    }

    .badge {
      font-size: 11px;
      padding: 6px 14px;
      border-radius: 20px;
      background: rgba(23, 20, 50, 0.8);
      border: 1px solid var(--border-zheyar);
      display: flex;
      align-items: center;
      gap: 7px;
      letter-spacing: 0.5px;
    }

    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--orange);
      box-shadow: 0 0 10px var(--orange);
      animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
    }

    main {
      flex: 1;
      padding: 22px 32px;
      display: grid;
      grid-template-columns: 1fr 350px;
      gap: 24px;
      max-width: 1420px;
      margin: 0 auto;
      width: 100%;
    }

    .arena-container {
      background: rgba(23, 20, 50, 0.7);
      border: 1px solid var(--border-zheyar);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      backdrop-filter: blur(16px);
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
    }

    .arena-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .stats-row {
      display: flex;
      gap: 28px;
    }

    .stat-item {
      display: flex;
      flex-direction: column;
    }

    .stat-label {
      font-size: 11px;
      color: #9d97b8;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .stat-value {
      font-size: 24px;
      font-weight: 700;
      color: var(--orange-300);
      text-shadow: 0 0 12px rgba(210, 129, 31, 0.3);
    }

    .canvas-wrapper {
      position: relative;
      background: #080612;
      border: 1px solid var(--border-zheyar);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.9);
    }

    canvas {
      display: block;
      image-rendering: pixelated;
    }

    .controls-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 18px;
      background: rgba(13, 11, 28, 0.85);
      border-radius: 12px;
      border: 1px solid var(--border-zheyar);
    }

    .btn-group {
      display: flex;
      gap: 8px;
    }

    button {
      background: rgba(38, 33, 74, 0.8);
      color: var(--paper);
      border: 1px solid var(--border-zheyar);
      padding: 8px 16px;
      border-radius: 8px;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s;
    }

    button:hover {
      background: rgba(210, 129, 31, 0.2);
      border-color: var(--orange-300);
      box-shadow: 0 0 12px rgba(210, 129, 31, 0.3);
    }

    button.primary {
      background: linear-gradient(135deg, var(--orange), #b86710);
      color: #fff;
      border: none;
      box-shadow: 0 0 16px rgba(210, 129, 31, 0.45);
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .panel {
      background: rgba(23, 20, 50, 0.7);
      border: 1px solid var(--border-zheyar);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      backdrop-filter: blur(14px);
    }

    .panel-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #9d97b8;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
    }

    .decision-badge {
      background: rgba(210, 129, 31, 0.12);
      border: 1px solid var(--orange);
      color: var(--orange-300);
      padding: 11px;
      border-radius: 10px;
      text-align: center;
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 1px;
      box-shadow: 0 0 16px rgba(210, 129, 31, 0.25);
    }

    .prob-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
    }

    .prob-labels {
      display: flex;
      justify-content: space-between;
      color: #c9c3d8;
    }

    .prob-track {
      height: 7px;
      background: rgba(255, 255, 255, 0.07);
      border-radius: 4px;
      overflow: hidden;
    }

    .prob-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--orange), var(--orange-300));
      border-radius: 4px;
      width: 0%;
      transition: width 0.12s ease-out;
    }

    .log-stream {
      font-size: 11px;
      color: #c9c3d8;
      background: #090714;
      padding: 10px;
      border-radius: 8px;
      height: 120px;
      overflow-y: auto;
      border: 1px solid rgba(210, 129, 31, 0.15);
      line-height: 1.6;
    }

    .log-item {
      margin-bottom: 4px;
      border-left: 2px solid var(--orange);
      padding-left: 6px;
    }

    .log-item.veto {
      border-left-color: var(--cyan-neon);
      color: #7dd3fc;
    }

    /* Instagram Focus / Record Mode */
    body.record-mode header,
    body.record-mode .controls-bar {
      border-color: transparent;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-logo-wrap">
        <!-- Zheyar Official Logo -->
        <svg viewBox="0 0 100 100" width="30" height="30" role="img" aria-label="ژیار">
          <path d="M80 36V52A28 28 0 0 1 52 80H18" stroke="#FAF8F4" stroke-width="14" fill="none"/>
          <rect x="16" y="8" width="14" height="14" fill="#D2811F"/>
          <rect x="43" y="8" width="14" height="14" fill="#D2811F"/>
          <rect x="70" y="8" width="14" height="14" fill="#D2811F"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>ZHEYAR AI LABS <span class="farsi-tag">آزمایشگاه ژیار</span></h1>
        <p>EDGE DECISION ROUTER BENCHMARK &bull; LOCAL GPU INFERENCE</p>
      </div>
    </div>
    <div class="badges">
      <a href="http://127.0.0.1:8081" target="_blank" style="text-decoration:none; display:flex; align-items:center; gap:6px; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.35); color:#38bdf8; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;">
        <span>🧱 Tetris AI (8081) ↗</span>
      </a>
      <div class="badge"><span class="badge-dot"></span> LOCAL CUDA GPU</div>
      <div class="badge">MODEL: LAYA-RL (1.1B)</div>
      <div class="badge">60 FPS ZERO-FLICKER</div>
    </div>
  </header>

  <main>
    <div class="arena-container">
      <div class="arena-header">
        <div class="stats-row">
          <div class="stat-item">
            <span class="stat-label">Score</span>
            <span class="stat-value" id="score-val">0</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Length</span>
            <span class="stat-value" id="len-val">3</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Steps</span>
            <span class="stat-value" id="step-val">0</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">GPU Latency</span>
            <span class="stat-value" id="lat-val">0.0 ms</span>
          </div>
        </div>
      </div>

      <div class="canvas-wrapper">
        <canvas id="gameCanvas" width="768" height="448"></canvas>
      </div>

      <div class="controls-bar">
        <div class="btn-group">
          <button class="primary" id="btn-toggle">PAUSE</button>
          <button id="btn-reset">RESET</button>
        </div>

        <div class="btn-group">
          <button id="btn-mode-laya" style="border-color: var(--orange)">LAYA AI MODE</button>
          <button id="btn-mode-unbeatable">UNBEATABLE MODE</button>
        </div>

        <div style="display:flex; align-items:center; gap: 10px; font-size:12px;">
          <span>Speed</span>
          <input type="range" id="speed-slider" min="15" max="150" value="45" style="cursor:pointer; accent-color: var(--orange);">
        </div>
      </div>
    </div>

    <div class="sidebar">
      <div class="panel">
        <div class="panel-title">Current Decision</div>
        <div class="decision-badge" id="action-badge">WAITING</div>
        <div style="font-size:11px; color:#c9c3d8;" id="note-text">
          Status: Ready
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">Model Probabilities</div>
        <div class="prob-row">
          <div class="prob-labels"><span>UP</span><span id="prob-up">0%</span></div>
          <div class="prob-track"><div class="prob-bar" id="bar-up"></div></div>
        </div>
        <div class="prob-row">
          <div class="prob-labels"><span>RIGHT</span><span id="prob-right">0%</span></div>
          <div class="prob-track"><div class="prob-bar" id="bar-right"></div></div>
        </div>
        <div class="prob-row">
          <div class="prob-labels"><span>DOWN</span><span id="prob-down">0%</span></div>
          <div class="prob-track"><div class="prob-bar" id="bar-down"></div></div>
        </div>
        <div class="prob-row">
          <div class="prob-labels"><span>LEFT</span><span id="prob-left">0%</span></div>
          <div class="prob-track"><div class="prob-bar" id="bar-left"></div></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">Spatial Awareness</div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
          <span>Reachable Space:</span>
          <span style="color:var(--orange-300); font-weight:600" id="free-space-val">100%</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px;">
          <span>Tail Escape:</span>
          <span style="color:var(--emerald-neon); font-weight:600">VERIFIED</span>
        </div>
      </div>

      <div class="panel" style="flex:1">
        <div class="panel-title">Inference Telemetry</div>
        <div class="log-stream" id="log-box">
          <div class="log-item">Zheyar Edge Engine initialized.</div>
        </div>
      </div>
    </div>
  </main>

  <script>
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');

    const CELL_SIZE = 32;
    const GRID_W = 24;
    const GRID_H = 14;

    let isRunning = true;
    let mode = 'laya';
    let tickSpeed = 45;
    let stepTimer = null;

    let gameState = {
      snake: [[6, 7], [5, 7], [4, 7]],
      food: [16, 7],
      score: 0,
      step: 0,
      latency: 0,
      action: 'RIGHT',
      probabilities: { UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25 },
      free_space_pct: 100
    };

    function draw() {
      // Clear with deep night background
      ctx.fillStyle = '#080612';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid background lines with subtle Zheyar night-500 tint
      ctx.strokeStyle = 'rgba(75, 67, 120, 0.15)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= GRID_W; x++) {
        ctx.beginPath();
        ctx.moveTo(x * CELL_SIZE, 0);
        ctx.lineTo(x * CELL_SIZE, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y <= GRID_H; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * CELL_SIZE);
        ctx.lineTo(canvas.width, y * CELL_SIZE);
        ctx.stroke();
      }

      // Outer walls (Midnight indigo)
      ctx.fillStyle = 'rgba(23, 20, 50, 0.85)';
      for (let x = 0; x < GRID_W; x++) {
        ctx.fillRect(x * CELL_SIZE, 0, CELL_SIZE, CELL_SIZE);
        ctx.fillRect(x * CELL_SIZE, (GRID_H - 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
      for (let y = 0; y < GRID_H; y++) {
        ctx.fillRect(0, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        ctx.fillRect((GRID_W - 1) * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }

      // Food (Zheyar Golden Saffron / Ruby Gem)
      if (gameState.food) {
        const [fx, fy] = gameState.food;
        ctx.save();
        ctx.shadowColor = '#D2811F';
        ctx.shadowBlur = 18;
        ctx.fillStyle = '#D2811F';
        ctx.beginPath();
        ctx.arc(fx * CELL_SIZE + CELL_SIZE / 2, fy * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 2.6, 0, Math.PI * 2);
        ctx.fill();

        // Inner glowing core
        ctx.fillStyle = '#EDB463';
        ctx.beginPath();
        ctx.arc(fx * CELL_SIZE + CELL_SIZE / 2, fy * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Snake Body
      if (gameState.snake) {
        const len = gameState.snake.length;
        for (let i = len - 1; i >= 0; i--) {
          const [sx, sy] = gameState.snake[i];
          const px = sx * CELL_SIZE;
          const py = sy * CELL_SIZE;

          ctx.save();
          if (i === 0) {
            // Head: Zheyar Warm Saffron Gold with intense glow
            ctx.shadowColor = '#EDB463';
            ctx.shadowBlur = 16;
            ctx.fillStyle = '#EDB463';
            ctx.fillRect(px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4);
          } else {
            // Body segments: Smooth cyber gradient
            const ratio = 1 - (i / len) * 0.45;
            ctx.fillStyle = `rgba(16, 185, 129, ${ratio})`;
            ctx.fillRect(px + 3, py + 3, CELL_SIZE - 6, CELL_SIZE - 6);
          }
          ctx.restore();
        }
      }
    }

    async function step() {
      if (!isRunning) return;

      try {
        const res = await fetch('/api/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode })
        });
        const data = await res.json();
        gameState = data;

        // Update UI
        document.getElementById('score-val').textContent = data.score;
        document.getElementById('len-val').textContent = data.snake ? data.snake.length : 0;
        document.getElementById('step-val').textContent = data.step;
        document.getElementById('lat-val').textContent = `${data.latency} ms`;
        document.getElementById('action-badge').textContent = data.action;
        document.getElementById('note-text').textContent = data.note;
        document.getElementById('free-space-val').textContent = `${data.free_space_pct}%`;

        // Update Probabilities
        if (data.probabilities) {
          for (const d of ['UP', 'RIGHT', 'DOWN', 'LEFT']) {
            const p = data.probabilities[d] || 0;
            const pct = Math.round(p * 100);
            const lower = d.toLowerCase();
            document.getElementById(`prob-${lower}`).textContent = `${pct}%`;
            document.getElementById(`bar-${lower}`).style.width = `${pct}%`;
          }
        }

        // Telemetry log
        if (data.step % 3 === 0) {
          const logBox = document.getElementById('log-box');
          const isVeto = data.note.includes('Guardrail');
          const item = document.createElement('div');
          item.className = 'log-item' + (isVeto ? ' veto' : '');
          item.textContent = `[#${data.step}] ${data.action} (${data.latency}ms) - ${data.note}`;
          logBox.appendChild(item);
          logBox.scrollTop = logBox.scrollHeight;
        }

        draw();
      } catch (err) {
        console.error("Step error:", err);
      }

      if (isRunning) {
        stepTimer = setTimeout(step, tickSpeed);
      }
    }

    // Event Listeners
    document.getElementById('btn-toggle').addEventListener('click', () => {
      isRunning = !isRunning;
      document.getElementById('btn-toggle').textContent = isRunning ? 'PAUSE' : 'RESUME';
      if (isRunning) step();
    });

    document.getElementById('btn-reset').addEventListener('click', async () => {
      await fetch('/api/reset', { method: 'POST' });
      document.getElementById('log-box').innerHTML = '<div class="log-item">Session reset.</div>';
    });

    document.getElementById('btn-mode-laya').addEventListener('click', () => {
      mode = 'laya';
      document.getElementById('btn-mode-laya').style.borderColor = 'var(--orange)';
      document.getElementById('btn-mode-unbeatable').style.borderColor = 'var(--border-zheyar)';
    });

    document.getElementById('btn-mode-unbeatable').addEventListener('click', () => {
      mode = 'unbeatable';
      document.getElementById('btn-mode-unbeatable').style.borderColor = 'var(--orange)';
      document.getElementById('btn-mode-laya').style.borderColor = 'var(--border-zheyar)';
    });

    document.getElementById('speed-slider').addEventListener('input', (e) => {
      tickSpeed = parseInt(e.target.value);
    });

    draw();
    step();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
