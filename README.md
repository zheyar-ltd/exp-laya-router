# ⚡ Zheyar AI Labs — Laya Edge Decision Router Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![CUDA Accelerated](https://img.shields.io/badge/Hardware-CUDA%20GPU-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Tests](https://img.shields.io/badge/Tests-15%20Passing-success.svg)](#running-automated-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade benchmark and interactive real-time visual showcase for the **Laya RL Decision Engine** running 100% locally on custom GPU hardware. Built by **Zheyar AI Labs**.

---

## 🎯 Overview: Decision AI vs Generative AI

Most developers associate Artificial Intelligence exclusively with Large Language Models (LLMs) like GPT-4 or Claude. However, **generative text models are poorly suited for real-time autonomous systems**:
* **High Latency:** LLMs take 500ms – 3,000ms to produce tokens.
* **Non-Deterministic:** Generating text paragraphs introduces hallucinations and parser failures.
* **High Compute Overhead:** Unviable for edge robotics, autonomous vehicles, or real-time traffic routing.

`
       [ Human / High-Level Planner ]
                    │
                    ▼  (Goals / High-Level Context)
  ┌────────────────────────────────────────────────────────┐
  │         Laya Decision Engine (1.1B RL Router)          │
  │   • Sub-50ms Latency on Local GPU                      │
  │   • Calibrated Probabilities & Action Choice           │
  │   • Zero Cloud Dependency (Edge AI)                    │
  └────────────────────────────────────────────────────────┘
                    │
                    ▼  (Real-Time Action)
       [ Autonomous Agent / Hardware / Arena ]
                    ▲
                    │  (Guaranteed Safety Guardrail)
       [ Hamiltonian Cycle & Spatial Verifier ]
`

**Laya** acts as the **reflex system / cerebellum** of an autonomous agent: given a multi-dimensional state and structured criteria, it predicts the optimal discrete action in **under 50 milliseconds** with calibrated probabilities.

---

## ✨ Key Features

1. **60 FPS Zero-Flicker Visual Web Dashboard:**
   - Custom dark-mode UI styled with the official **Zheyar Design System** (#0D0B1C Midnight & #D2811F Warm Saffron).
   - High-performance HTML5 Canvas rendering without terminal screen-clearing flicker.
   - Real-time telemetry: Live GPU latency, model probability distributions, and spatial safety metrics.

2. **Graph-Theoretic Safety & Unbeatable Mode:**
   - **Hamiltonian Cycle:** Precomputed 264-cell closed cycle on the 22×12 playable grid ensuring zero self-collisions.
   - **Safe Shortcuts:** Takes direct greedy shortcuts toward food only when the cycle distance to the tail is safely preserved.
   - **Flood Fill Analysis:** Calculates connected free space to reject dead-end pocket moves.
   - **Virtual Path Lookahead:** Simulates reaching the target and verifies post-eat tail reachability.
   - **Safety Guardrail:** Intercepts and vetoes suicidal traps if an agent picks a hazardous route.

3. **15-Test Automated Verification Suite:**
   - Full coverage across geometry, pathfinding, virtual lookahead, Hamiltonian continuity, and Laya integration.

---

## 📁 Repository Structure

`	ext
exp-laya-router/
├── main.py               # FastAPI backend serving the Laya Router predict endpoint
├── snake_laya.py         # Autonomous snake engine (Hamiltonian AI + Guardrails)
├── web_dashboard.py      # Real-time visual dashboard (FastAPI + HTML5 Canvas)
├── test_snake.py         # 15-test automated unit & integration test suite
├── benchmark_model.py    # Raw CUDA GPU benchmark utility for Laya
├── payload.json          # Sample request payload for API testing
├── static/               # Zheyar brand assets (Logo SVG, Vazirmatn & Space Grotesk fonts)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
`

---

## 🚀 Step-by-Step Setup Guide

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for sub-50ms inference)
- Git

### 2. Clone and Setup Environment
`ash
git clone https://github.com/zheyar-ltd/exp-laya-router.git
cd exp-laya-router

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

> **Note on Laya:** Ensure the laya package is installed in your environment (pip install laya or via your local wheel).

---

### 3. Start the Laya Decision API Server
Run the local inference server on port 8000:
`ash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
`
*API docs available at: http://127.0.0.1:8000/docs*

---

### 4. Launch the Zheyar Visual Web Dashboard
In a new terminal window:
`ash
python web_dashboard.py
`
Open your browser at **http://127.0.0.1:8050** to view the live dashboard:
- Watch the agent make real-time decisions with live probability bars.
- Toggle between **LAYA AI Mode** and **UNBEATABLE Mode**.
- Adjust simulation speed or reset the game on the fly.
- Perfect for screen recording (60fps, zero flicker).

---

### 5. Run Terminal CLI Modes & Benchmarks
You can also run the agent directly inside your terminal:

`ash
# Run in Laya AI Mode
python snake_laya.py --mode laya --speed 0.06

# Run in Pure Unbeatable Mode (Hamiltonian Cycle)
python snake_laya.py --mode unbeatable --speed 0.02

# Run Fast Headless Benchmark (5 games, 1000 steps each)
python snake_laya.py --mode benchmark --games 5 --max-steps 1000
`

---

## 🧪 Running Automated Tests

Run the complete test suite using Python''s built-in unittest:

`ash
python -m unittest test_snake.py -v
`

### Test Coverage Highlights:
- TestHamiltonianCycle: Verifies 264-cell uniqueness, boundaries, Manhattan adjacency, and cycle distance math.
- TestGridAndMovement: Verifies collision detection on outer walls and body segments.
- TestPathfindingAndFloodFill: Tests BFS shortest paths, obstacle detours, and reachable pocket calculations.
- TestVirtualFoodSimulation: Validates virtual lookahead (safely detecting corner cul-de-sacs).
- TestHamiltonianShortcuts: Ensures shortcuts never jump past the tail.
- TestLayaIntegrationAndGuardrails: Confirms guardrail vetoes fatal moves.
- TestSimulationPerformance: Validates continuous survival for 500+ steps.

---

## 📊 Benchmark Results

| Metric | Measurement |
| :--- | :--- |
| **Inference Latency** | ~40 – 55 ms (Local CUDA GPU) |
| **Throughput** | ~20 – 25 decisions/sec |
| **Unbeatable Survival Rate** | 100% (Zero collisions over 1,000+ steps) |
| **Average Score (1000 steps)** | ~38 – 42 targets |
| **Cloud Dependency** | 0% (Fully Offline / On-Premise) |

---

## 🏢 About Zheyar

**Zheyar Ltd** specializes in Edge AI, autonomous agents, and industrial IoT solutions. We engineer hybrid architectures that combine high-level planning with deterministic, low-latency decision routers for real-world mission-critical applications.

- Website: [zheyar.com](https://zheyar.com)
- Organization: [github.com/zheyar-ltd](https://github.com/zheyar-ltd)

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
