# Zheyar AI Labs — Laya Real-Time Decision Hub

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Unit Tests](https://img.shields.io/badge/Tests-32%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance testbed and visual interactive dashboard for the **Laya RL Policy Router (1.1B)** on CUDA. Demonstrates sub-50ms discrete decision-making in real-time gaming environments with mathematical safety guardrails.

Developed at **Zheyar AI Labs**.

---

## 🎯 Architecture Overview

Large Language Models (LLMs) are often too slow for real-time control loops, typically requiring 500ms to 2000ms per inference. **Laya** is a 1.1B parameter edge policy router trained with reinforcement learning for discrete, sub-50ms decisions on local GPU hardware.

```
+-------------------------------------------------------------------------+
|                           GAME ENVIRONMENT                              |
|           (Snake 24x14 Grid  /  Tetris 10x20 Standard SRS)              |
+-------------------------------------------------------------------------+
                                     |
                         [State & Candidate Feature Extraction]
                                     |
                                     v
+-------------------------------------------------------------------------+
|                     LAYA DECISION ROUTER (CUDA)                         |
|   - Evaluates multi-candidate actions with dynamic natural criteria     |
|   - Emits calibrated probability distributions & strategic choices     |
|   - Sub-50ms local GPU inference latency                                |
+-------------------------------------------------------------------------+
                                     |
                        [Policy Output Validation]
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       SAFETY GUARDRAILS                                 |
|   - Snake: Hamiltonian cycle continuity + Flood-fill trap veto          |
|   - Tetris: Pierre Dellacherie hole penalty & anti-top-out safety net   |
+-------------------------------------------------------------------------+
                                     |
                           [Action Execution]
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  60 FPS HTML5 CANVAS WEB DASHBOARDS                     |
|        Snake Web UI (Port 8050)   <--->   Tetris Web UI (Port 8081)     |
+-------------------------------------------------------------------------+
```

---

## 🎮 Included Games

### 1. 🐍 Autonomous Snake AI
- **Playable Arena:** 24x14 grid (264 playable tiles).
- **Core Technology:** Laya RL Router guided by dynamic reachable-space flood fill and shortest BFS pathfinding.
- **Safety Fallback:** Precomputed 264-cell Hamiltonian cycle with shortcut jumps ensuring **0% collision rate** even at 99%+ board fill.
- **Web Interface:** Port `8050`, zero-flicker 60fps HTML5 Canvas, real-time probability meters, GPU latency monitor.

### 2. 🧱 Tetris Gravity & SRS Engine
- **Playable Arena:** Standard 10x20 grid, SRS (Super Rotation System) 7-tetrominoes (`I`, `O`, `T`, `S`, `Z`, `J`, `L`).
- **Core Technology:** Natural step-by-step gravity fall, Wall Kicks, line clearing, lock delay, and Hold slot.
- **Laya Integration:** Multi-candidate drop placement reasoning evaluated against Pierre Dellacherie feature space (Landing Height, Eroded Cells, Row/Col Transitions, Buried Holes, Cumulative Wells, Surface Bumpiness).
- **Modes:**
  - `🎮 MANUAL PLAY`: Human interactive play with responsive keyboard controls.
  - `⚡ LAYA AI BRAIN`: Autonomous real-time autopilot watching Laya navigate and drop pieces.
  - `🛡️ UNBEATABLE`: Pure mathematical Pierre Dellacherie optimizer (proven to survive >900k lines).
- **Web Interface:** Port `8081`, Web Audio synthesizer, cyberpunk dark theme, live probability bars.

---

## 📂 Repository Structure

```
.
├── main.py             # Shared FastAPI service wrapping laya.Router
│
├── snake_laya.py       # Snake game engine, Hamiltonian cycle generator, and CLI runner
├── web_snake.py        # Snake visual web dashboard (Port 8050)
├── web_dashboard.py    # Backward-compatible entry point for Snake dashboard
├── test_snake.py       # Automated Snake test suite (15 unit tests)
│
├── tetris_laya.py      # Standard Tetris gravity engine, Dellacherie optimizer, CLI runner
├── web_tetris.py       # Tetris visual web dashboard (Port 8081, Web Audio synth)
├── test_tetris.py      # Automated Tetris test suite (17 unit tests)
│
├── static/             # UI assets (Zheyar logo and local typography)
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA drivers
- `laya` package installed in your Python environment

### 2. Install Dependencies
```bash
git clone https://github.com/zheyar-ltd/exp-laya-router.git
cd exp-laya-router

# Create virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launching the Web Dashboards

#### 🧱 Start Tetris Dashboard:
```bash
python web_tetris.py --port 8081
```
Open **[http://localhost:8081](http://localhost:8081)** in your browser.
- **Controls (Manual):** `Left`/`Right` to move, `Up` or `Z` to rotate, `Down` for soft drop, `Space` for hard drop, `C` to hold.
- **Autopilot:** Click `⚡ LAYA AI BRAIN` to watch Laya play autonomously.
- **Switch Game:** Click the `🐍 Snake AI (8050)` button in the header.

#### 🐍 Start Snake Dashboard:
```bash
python web_snake.py --port 8050
```
Open **[http://localhost:8050](http://localhost:8050)** in your browser.
- Switch between **Laya AI** and **Unbeatable Hamiltonian** modes.
- Adjust tick speeds from 10ms to 200ms in real time.
- **Switch Game:** Click the `🧱 Tetris AI (8081)` button in the header.

#### 🌐 Shared Laya API Service (Optional for Multi-Agent Setup):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 🧪 Automated Test Suites

The repository includes **32 unit and integration tests** verifying core physics, collision detection, heuristic integrity, and safety guardrails:

```bash
# Run Snake test suite (15 tests)
python test_snake.py

# Run Tetris test suite (17 tests)
python test_tetris.py
```

### Test Coverage Highlights:
- **Snake (`test_snake.py`):** Pathfinding continuity, flood-fill safety, Hamiltonian cycle loops, trap avoidance, and virtual eating simulation.
- **Tetris (`test_tetris.py`):** 7-bag randomizer, SRS wall kick rotations, natural gravity tick fall, boundary and locked block collisions, 1-to-4 line clearing, buried hole detection, and 100-piece invincible survival.

---

## 📊 Technical Comparison

| Feature | 🐍 Snake AI | 🧱 Tetris AI |
| :--- | :--- | :--- |
| **Action Space** | Discrete 4 Directions (`UP`, `DOWN`, `LEFT`, `RIGHT`) | Discrete 2D Placements (`(Rotation, Column)`) |
| **Candidate Options** | 2 to 4 legal moves per tick | 20 to 34 legal placements per piece |
| **Inference Frequency** | Every grid step (50ms – 100ms) | Once per piece spawn (or per gravity tick) |
| **Evaluation Heuristic** | Hamiltonian distance + reachable flood fill | Pierre Dellacherie feature vector (9 metrics) |
| **Safety Guardrail** | Vetoes moves causing unreachable tail escape | Vetoes placements causing buried holes |
| **Dashboard Port** | `8050` | `8081` |
| **Audio Synthesis** | Canvas visualizer | Real-time Web Audio API synthesizer |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) &copy; Zheyar Labs.