# Laya Decision Router Benchmark

Benchmark and local testbed for the Laya RL decision router on CUDA. Includes an autonomous snake test environment with Hamiltonian cycle validation, flood-fill safety checks, and a local web dashboard.

Developed at Zheyar Labs.

## Overview

Large language models are too slow for real-time control loops, typically taking 500ms to 2s per step. Laya is a 1.1B parameter policy router tuned with reinforcement learning for discrete, low-latency decisions (under 50ms on a local GPU).

In this repository, Laya is tested as the decision engine for an autonomous agent navigating a 2D grid:
- State descriptions and candidate moves are evaluated each tick.
- The router returns calibrated probabilities and the selected action.
- A local safety guardrail (Hamiltonian cycle + flood-fill) verifies that moves do not trap the agent in dead ends.

## Features

- **Local GPU Inference:** Runs completely offline against the local Laya model via FastAPI.
- **Web Dashboard:** 60fps HTML5 canvas visualization on port 8050 with real-time latency and probability meters. Zero console flicker.
- **Unbeatable Fallback:** 264-cell precomputed Hamiltonian cycle with cycle-distance shortcuts to guarantee zero collisions.
- **Test Suite:** 15 unit and integration tests covering pathfinding, cycle continuity, virtual simulation, and guardrail overrides.

## Project Structure

`
.
|-- main.py             # FastAPI service wrapping laya.Router
|-- snake_laya.py       # Game logic, Hamiltonian cycle generator, and CLI runner
|-- web_dashboard.py    # Local web UI (FastAPI + HTML5 Canvas)
|-- test_snake.py       # Automated test suite (15 tests)
|-- benchmark_model.py  # Standalone GPU latency and throughput benchmark
|-- payload.json        # Example request payload
|-- static/             # UI assets (logo and local fonts)
-- requirements.txt    # Python dependencies
`

## Setup and Usage

### 1. Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- laya package installed in your Python environment

### 2. Install Dependencies

`ash
git clone https://github.com/zheyar-ltd/exp-laya-router.git
cd exp-laya-router

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
`

### 3. Start the Laya API

`ash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
`

The API docs are available at http://127.0.0.1:8000/docs.

### 4. Start the Web Dashboard

In a separate terminal:

`ash
python web_dashboard.py
`

Open http://127.0.0.1:8050 in your browser. You can toggle between Laya AI mode and pure Unbeatable mode, adjust tick speed, and watch real-time GPU inference times and probability distributions.

### 5. Run from Terminal (Headless / CLI)

`ash
# Terminal visual play (Laya AI):
python snake_laya.py --mode laya --speed 0.06

# Unbeatable Hamiltonian mode:
python snake_laya.py --mode unbeatable --speed 0.02

# Headless batch benchmark (5 games, 1000 steps each):
python snake_laya.py --mode benchmark --games 5 --max-steps 1000
`

### 6. Run Tests

`ash
python -m unittest test_snake.py -v
`

## Benchmark Numbers

Measured on local CUDA GPU:

- Average inference latency: 40-55 ms per call
- Throughput: 20-25 decisions per second
- Survival rate (Unbeatable mode): 100% over 1,000 steps

## License

MIT