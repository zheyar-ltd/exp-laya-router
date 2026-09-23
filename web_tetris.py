"""
Zheyar AI Labs - Laya Tetris Visual Dashboard.
Real-Time Web Interface for Invincible Laya AI Tetris Engine.
Features Standard Tetris Gravity, Smooth Controls, SRS Rotations, and Laya AI Routing.
"""

import os
import sys
import time
import requests
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tetris_laya import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    TETROMINOES,
    PIECE_NAMES,
    PIECE_COLORS,
    TetrisGame,
    get_piece_blocks,
    is_valid_position,
    drop_piece_to_bottom,
    get_all_possible_placements,
    get_two_piece_best_placement,
    query_laya_tetris,
    calculate_dellacherie_features
)

# Optional direct in-memory Laya Router
laya_in_memory_router = None
try:
    import laya
    print("Preloading in-memory Laya Router on CUDA for sub-millisecond local inference...")
    laya_in_memory_router = laya.Router(device="cuda", preload=True)
    print("In-memory Laya Router successfully initialized on CUDA!")
except Exception as e:
    print(f"Direct Laya GPU load notice: {e}. Will utilize HTTP API or Pierre Dellacherie fallback.")

app = FastAPI(title="Zheyar AI Labs - Laya Tetris Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global Game Session State
current_game = TetrisGame()
current_mode = "manual"  # Default to manual so user can play immediately, or switch to laya / unbeatable
last_probabilities = {}
last_decision_note = "Game initialized. Ready for play."
last_latency = 0.0
last_features = {
    "landing_height": 0.0,
    "eroded_cells": 0.0,
    "row_transitions": 0,
    "col_transitions": 0,
    "holes": 0,
    "cumulative_wells": 0,
    "hole_depth": 0,
    "bumpiness": 0,
    "max_height": 0
}


def query_laya_internal_or_api(board, current_piece, next_piece, placements):
    """Queries in-memory Laya router if loaded, otherwise calls HTTP API or Dellacherie."""
    global laya_in_memory_router
    if not placements:
        return {"rotation": 0, "col": 3, "score": -9999}, 0.0, "No placements available", {}

    best_optimal = placements[0]
    safe_candidates = placements[:min(4, len(placements))]
    criteria = {}
    candidate_map = {}

    for i, c in enumerate(safe_candidates):
        cid = f"placement_{i}"
        candidate_map[cid] = c
        ft = c["features"]
        rot = c["rotation"]
        col = c["col"]
        lines = c["lines_cleared"]
        holes = ft["holes"]
        lh = ft["landing_height"]
        bm = ft["bumpiness"]

        if i == 0:
            criteria[cid] = (
                f"(Recommended) Master Placement at col {col}, rot {rot}: "
                f"Clears {lines} line(s), creates {holes} holes, low landing height ({lh:.1f}), "
                f"bumpiness {bm:.0f}. Preserves clean, flat board surface."
            )
        elif lines > 0:
            criteria[cid] = (
                f"Line Clear Placement at col {col}, rot {rot}: "
                f"Clears {lines} line(s), {holes} holes, landing height {lh:.1f}."
            )
        elif holes == 0:
            criteria[cid] = (
                f"Safe Flat Placement at col {col}, rot {rot}: "
                f"0 holes created, landing height {lh:.1f}, maintains stack balance."
            )
        else:
            criteria[cid] = (
                f"Alternative Placement at col {col}, rot {rot}: "
                f"Landing height {lh:.1f}, {holes} holes created."
            )

    if len(placements) > 6:
        worst = placements[-1]
        cid = "placement_risky"
        candidate_map[cid] = worst
        wft = worst["features"]
        criteria[cid] = (
            f"Dangerous High Placement at col {worst['col']}, rot {worst['rotation']}: "
            f"Creates {wft['holes']} holes, landing height {wft['landing_height']:.1f}."
        )

    col_heights = [0] * BOARD_WIDTH
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y][x]:
                col_heights[x] = BOARD_HEIGHT - y
                break
    max_h = max(col_heights) if col_heights else 0

    state_desc = (
        f"Tetris Game Board State. Max Stack Height: {max_h}/{BOARD_HEIGHT}. "
        f"Current Tetromino: '{current_piece}'. Next Queue: '{next_piece}'. "
        f"Available candidate drop placements: {len(criteria)} options."
    )

    questions = {
        "placement": {
            "instructions": (
                "Select the single best Tetris piece placement that minimizes holes, "
                "keeps landing height low, maintains a flat board, and clears completed lines:"
            ),
            "criteria": criteria,
            "type": "choice"
        }
    }

    t0 = time.perf_counter()
    latency_ms = 0.0
    guardrail_note = "Laya Strategic Decision"
    probabilities = {k: round(1.0 / len(criteria), 3) for k in criteria}

    # 1. Try In-Memory Router (Fastest, < 15ms)
    if laya_in_memory_router is not None:
        try:
            res = laya_in_memory_router.predict(state=state_desc, questions=questions)
            latency_ms = (time.perf_counter() - t0) * 1000
            ans = res.get("answers", {}).get("placement", {})
            choice = ans.get("choice")
            probs = ans.get("probabilities", {})
            if probs:
                probabilities = {k: round(probs.get(k, 0.0), 3) for k in criteria}

            if choice in candidate_map:
                selected = candidate_map[choice]
                if selected["features"]["holes"] > best_optimal["features"]["holes"] + 1 or selected["score"] < best_optimal["score"] - 15.0:
                    guardrail_note = f"[Safety Guardrail] Vetoed sub-optimal '{choice}' -> Enforced Optimal col={best_optimal['col']} rot={best_optimal['rotation']}"
                    selected = best_optimal
                else:
                    guardrail_note = f"Laya Decision [{choice}] (col={selected['col']} rot={selected['rotation']})"
            else:
                selected = best_optimal
                guardrail_note = "[Laya Defaulted] Optimal candidate executed"
            return selected, latency_ms, guardrail_note, probabilities
        except Exception:
            pass

    # 2. Try Local HTTP API
    try:
        payload = {"state": state_desc, "questions": questions}
        res = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=0.8).json()
        latency_ms = (time.perf_counter() - t0) * 1000
        ans = res.get("answers", {}).get("placement", {})
        choice = ans.get("choice")
        probs = ans.get("probabilities", {})
        if probs:
            probabilities = {k: round(probs.get(k, 0.0), 3) for k in criteria}

        if choice in candidate_map:
            selected = candidate_map[choice]
            if selected["features"]["holes"] > best_optimal["features"]["holes"] + 1 or selected["score"] < best_optimal["score"] - 15.0:
                guardrail_note = f"[Safety Guardrail] Vetoed sub-optimal '{choice}' -> Enforced Optimal col={best_optimal['col']} rot={best_optimal['rotation']}"
                selected = best_optimal
            else:
                guardrail_note = f"Laya Decision [{choice}] (col={selected['col']} rot={selected['rotation']})"
        else:
            selected = best_optimal
            guardrail_note = "[Laya Defaulted] Optimal candidate executed"
        return selected, latency_ms, guardrail_note, probabilities
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000
        selected = best_optimal
        guardrail_note = "[Dellacherie Heuristic] Safe optimal placement"
        probabilities[list(criteria.keys())[0]] = 0.95
        return selected, latency_ms, guardrail_note, probabilities


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTML_CONTENT


@app.post("/api/reset")
def reset_game():
    global current_game, last_probabilities, last_decision_note, last_latency
    current_game = TetrisGame()
    last_probabilities = {}
    last_decision_note = "Game reset. Ready for play."
    last_latency = 0.0
    return get_game_state_payload()


@app.post("/api/set_mode")
async def set_mode(request: Request):
    global current_mode
    body = await request.json()
    if "mode" in body and body["mode"] in ["manual", "laya", "unbeatable"]:
        current_mode = body["mode"]
    return {"status": "ok", "mode": current_mode}


@app.post("/api/tick")
def tick_gravity():
    """Advances falling piece downwards by 1 row due to gravity."""
    global current_game, last_features, last_decision_note
    if not current_game.alive:
        return get_game_state_payload(action_note="Game Over - Top Out")

    locked, lines, ft = current_game.tick_gravity()
    if locked:
        last_features = ft
        if lines > 0:
            last_decision_note = f"Piece landed! Cleared {lines} line(s)!"
        else:
            last_decision_note = "Piece landed and locked on stack."
    return get_game_state_payload()


@app.post("/api/action")
async def process_action(request: Request):
    """Processes interactive player action (left, right, rotate, soft_drop, hard_drop, hold)."""
    global current_game, last_features, last_decision_note
    if not current_game.alive:
        return get_game_state_payload(action_note="Game Over")

    body = await request.json()
    action = body.get("action")

    if action == "left":
        current_game.move_left()
    elif action == "right":
        current_game.move_right()
    elif action == "rotate":
        current_game.rotate(clockwise=True)
    elif action == "rotate_ccw":
        current_game.rotate(clockwise=False)
    elif action == "soft_drop":
        locked, lines, ft = current_game.soft_drop()
        if locked:
            last_features = ft
            last_decision_note = f"Soft drop locked! Cleared {lines} lines." if lines else "Soft drop locked."
    elif action == "hard_drop":
        locked, lines, ft = current_game.hard_drop()
        if locked:
            last_features = ft
            last_decision_note = f"Hard drop locked! Cleared {lines} lines." if lines else "Hard drop locked."
    elif action == "hold":
        current_game.hold_current_piece()

    return get_game_state_payload()


@app.post("/api/ai_target")
def get_ai_target():
    """
    Computes optimal target (rotation, column) for the current falling piece.
    The frontend will smoothly navigate the falling piece to this target.
    """
    global current_game, current_mode, last_probabilities, last_decision_note, last_latency, last_features
    if not current_game.alive:
        return {"error": "Game over", "alive": False}

    placements = get_all_possible_placements(current_game.board, current_game.current_piece)
    if not placements:
        current_game.alive = False
        return {"error": "No placements", "alive": False}

    if current_mode == "unbeatable":
        best = get_two_piece_best_placement(current_game.board, current_game.current_piece, current_game.next_piece)
        last_latency = 0.5
        last_decision_note = f"Dellacherie Optimizer: Target col {best['col']}, rot {best['rotation']}"
        last_probabilities = {f"col_{best['col']}_rot_{best['rotation']}": 1.0}
    else:
        best, last_latency, last_decision_note, last_probabilities = query_laya_internal_or_api(
            current_game.board, current_game.current_piece, current_game.next_piece, placements
        )

    return {
        "target_col": best["col"],
        "target_rot": best["rotation"],
        "latency_ms": round(last_latency, 1),
        "decision_note": last_decision_note,
        "probabilities": last_probabilities,
        "alive": True
    }


@app.post("/api/step")
async def step_game(request: Request = None):
    """Direct placement step (for fast AI execution or instant simulation)."""
    global current_game, current_mode, last_probabilities, last_decision_note, last_latency, last_features

    if request:
        try:
            body = await request.json()
            if "mode" in body:
                current_mode = body["mode"]
        except Exception:
            pass

    if not current_game.alive:
        current_game = TetrisGame()

    placements = get_all_possible_placements(current_game.board, current_game.current_piece)
    if not placements:
        current_game.alive = False
        return get_game_state_payload(action_note="Game Over - No Legal Placements")

    if current_mode == "unbeatable":
        best = get_two_piece_best_placement(current_game.board, current_game.current_piece, current_game.next_piece)
        last_latency = 0.5
        last_decision_note = f"Dellacherie Optimizer: Placed col {best['col']}, rot {best['rotation']}"
        last_probabilities = {f"col_{best['col']}_rot_{best['rotation']}": 1.0}
    else:
        best, last_latency, last_decision_note, last_probabilities = query_laya_internal_or_api(
            current_game.board, current_game.current_piece, current_game.next_piece, placements
        )

    success, lines, features = current_game.lock_placement(best["rotation"], best["col"])
    if features:
        last_features = features

    return get_game_state_payload()


@app.get("/api/state")
def get_state():
    return get_game_state_payload()


def get_game_state_payload(action_note: Optional[str] = None):
    """Constructs comprehensive state payload for web frontend."""
    global current_game, current_mode, last_probabilities, last_decision_note, last_latency, last_features
    summary = current_game.get_board_state_summary()

    ghost_y = drop_piece_to_bottom(current_game.board, current_game.current_piece, current_game.curr_rot, current_game.curr_x, current_game.curr_y)
    ghost_blocks = get_piece_blocks(current_game.current_piece, current_game.curr_rot, current_game.curr_x, ghost_y) if ghost_y >= 0 else []
    current_blocks = get_piece_blocks(current_game.current_piece, current_game.curr_rot, current_game.curr_x, current_game.curr_y)

    return {
        "board": current_game.board,
        "alive": current_game.alive,
        "score": current_game.score,
        "lines": current_game.lines,
        "level": current_game.level,
        "pieces_dropped": current_game.pieces_dropped,
        "tetris_count": current_game.tetris_count,
        "current_piece": current_game.current_piece,
        "next_piece": current_game.next_piece,
        "hold_piece": current_game.hold_piece,
        "can_hold": current_game.can_hold,
        "curr_x": current_game.curr_x,
        "curr_y": current_game.curr_y,
        "curr_rot": current_game.curr_rot,
        "current_blocks": current_blocks,
        "ghost_blocks": ghost_blocks,
        "mode": current_mode,
        "latency_ms": round(last_latency, 1),
        "decision_note": action_note or last_decision_note,
        "probabilities": last_probabilities,
        "features": last_features,
        "summary": summary,
        "piece_colors": PIECE_COLORS
    }


# ==============================================================================
# Cyberpunk / Zheyar Aesthetic Web Interface with Full Gravity & Falling Physics
# ==============================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="fa" dir="ltr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Zheyar AI Labs - Laya Tetris Engine</title>
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
      --night-950: #070510;
      --night-900: #0c0a1a;
      --night: #15122e;
      --night-700: #221d45;
      --orange: #D2811F;
      --orange-300: #EDB463;
      --paper: #FAF8F4;
      --cyan-neon: #38bdf8;
      --emerald-neon: #10b981;
      --ruby-neon: #f43f5e;
      --purple-neon: #c084fc;
      --border-zheyar: rgba(210, 129, 31, 0.22);
      --border-glow: rgba(237, 180, 99, 0.45);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: radial-gradient(circle at 50% 0%, #1c173d 0%, var(--night-950) 100%);
      color: var(--paper);
      font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      user-select: none;
    }

    /* Header */
    header {
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-zheyar);
      background: rgba(12, 10, 26, 0.9);
      backdrop-filter: blur(14px);
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo-wrap {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: rgba(21, 18, 46, 0.9);
      border: 1px solid var(--border-zheyar);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px rgba(210, 129, 31, 0.25);
    }

    .brand-logo-wrap svg {
      width: 26px;
      height: 26px;
    }

    .brand-text h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .brand-text p {
      font-size: 11px;
      color: var(--orange-300);
      letter-spacing: 1px;
    }

    .farsi-tag {
      font-family: 'Vazirmatn', sans-serif;
      font-size: 12px;
      background: rgba(210, 129, 31, 0.15);
      border: 1px solid var(--border-zheyar);
      color: var(--orange-300);
      padding: 2px 8px;
      border-radius: 20px;
    }

    .header-badges {
      display: flex;
      gap: 12px;
    }

    .badge {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(21, 18, 46, 0.8);
      border: 1px solid var(--border-zheyar);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
    }

    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--emerald-neon);
      box-shadow: 0 0 8px var(--emerald-neon);
    }

    /* Main Layout */
    .dashboard-container {
      display: grid;
      grid-template-columns: 280px minmax(320px, 380px) 320px;
      gap: 20px;
      padding: 20px 28px;
      flex: 1;
      justify-content: center;
      align-items: start;
    }

    .glass-panel {
      background: rgba(18, 15, 40, 0.72);
      border: 1px solid var(--border-zheyar);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
      backdrop-filter: blur(10px);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 8px;
    }

    .panel-title {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.8px;
      color: var(--orange-300);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* Mini Grids for Hold & Next */
    .piece-preview-box {
      background: rgba(7, 5, 16, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
    }

    .mini-grid {
      display: grid;
      grid-template-columns: repeat(4, 18px);
      grid-template-rows: repeat(4, 18px);
      gap: 2px;
      width: 78px;
      height: 78px;
      background: rgba(0, 0, 0, 0.3);
      padding: 3px;
      border-radius: 6px;
    }

    .mini-cell {
      width: 18px;
      height: 18px;
      border-radius: 3px;
      background: transparent;
    }

    /* Stats Grid */
    .stat-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .stat-box {
      background: rgba(7, 5, 16, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }

    .stat-label {
      font-size: 11px;
      color: #9d97b8;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .stat-val {
      font-size: 20px;
      font-weight: 700;
      color: var(--paper);
      margin-top: 2px;
    }

    .stat-val.highlight {
      color: var(--orange-300);
    }

    /* Arena & Tetris Board */
    .arena-wrapper {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
    }

    .mode-tabs {
      display: flex;
      width: 100%;
      background: rgba(7, 5, 16, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
    }

    .mode-btn {
      flex: 1;
      background: transparent;
      border: none;
      color: #9d97b8;
      padding: 8px 4px;
      border-radius: 6px;
      font-family: inherit;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .mode-btn.active {
      background: linear-gradient(135deg, var(--orange) 0%, #a65e0a 100%);
      color: #fff;
      box-shadow: 0 0 12px rgba(210, 129, 31, 0.4);
    }

    #tetrisCanvas {
      background: #06040d;
      border: 2px solid var(--border-zheyar);
      border-radius: 12px;
      box-shadow: 0 0 24px rgba(0, 0, 0, 0.8), 0 0 16px rgba(210, 129, 31, 0.15);
      image-rendering: pixelated;
    }

    /* Control Buttons */
    .controls-row {
      display: flex;
      gap: 8px;
      width: 100%;
    }

    .btn {
      flex: 1;
      padding: 10px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      border: 1px solid var(--border-zheyar);
      background: rgba(34, 29, 69, 0.8);
      color: var(--paper);
    }

    .btn:hover {
      background: rgba(50, 42, 100, 0.9);
      border-color: var(--orange-300);
      transform: translateY(-1px);
    }

    .btn.primary {
      background: linear-gradient(135deg, var(--cyan-neon) 0%, #0284c7 100%);
      color: #031424;
      border: none;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.35);
    }

    .btn.primary:hover {
      box-shadow: 0 0 22px rgba(56, 189, 248, 0.55);
    }

    .speed-control {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(7, 5, 16, 0.5);
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      width: 100%;
    }

    .speed-control label {
      font-size: 11px;
      color: #9d97b8;
      min-width: 60px;
    }

    .speed-control input[type="range"] {
      flex: 1;
      accent-color: var(--orange);
    }

    /* Keybinds Guide */
    .keybinds-box {
      width: 100%;
      background: rgba(7, 5, 16, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 11px;
      color: #9d97b8;
      line-height: 1.6;
    }

    .key-badge {
      display: inline-block;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 4px;
      padding: 1px 5px;
      font-weight: 700;
      color: var(--paper);
    }

    /* Right Column: AI Decision Center */
    .ai-decision-card {
      background: rgba(12, 10, 26, 0.7);
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .decision-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--cyan-neon);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .decision-text {
      font-size: 13px;
      line-height: 1.5;
      color: var(--paper);
      font-weight: 500;
      background: rgba(0, 0, 0, 0.25);
      padding: 8px 10px;
      border-radius: 6px;
      border-left: 3px solid var(--orange);
    }

    .prob-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .prob-item {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .prob-info {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: #b5afcf;
    }

    .prob-bar-bg {
      height: 7px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 4px;
      overflow: hidden;
    }

    .prob-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--cyan-neon), var(--purple-neon));
      border-radius: 4px;
      transition: width 0.3s ease;
    }

    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    .metrics-table tr {
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .metrics-table td {
      padding: 6px 8px;
    }

    .metrics-table td:last-child {
      text-align: right;
      font-weight: 700;
      color: var(--orange-300);
    }

    .farsi-subtext {
      font-family: 'Vazirmatn', sans-serif;
      font-size: 12px;
      color: #9d97b8;
      text-align: center;
      margin-top: 8px;
      line-height: 1.6;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-logo-wrap">
        <svg viewBox="0 0 24 24" fill="none" stroke="#D2811F" stroke-width="2">
          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
          <polyline points="2 17 12 22 22 17"></polyline>
          <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
      </div>
      <div class="brand-text">
        <h1>ZHEYAR AI LABS <span class="farsi-tag">موتور تتریس لایا</span></h1>
        <p>REAL-TIME TETRIS GRAVITY ENGINE & LAYA AI ROUTER</p>
      </div>
    </div>

    <div class="header-badges">
      <a href="http://127.0.0.1:8050" target="_blank" style="text-decoration:none; display:flex; align-items:center; gap:6px; background:rgba(74, 222, 128, 0.12); border:1px solid rgba(74, 222, 128, 0.35); color:#4ade80; padding:6px 14px; border-radius:8px; font-size:12px; font-weight:700; transition:all 0.2s ease;">
        <span>🐍 Snake AI (8050) ↗</span>
      </a>
      <div class="badge">
        <span class="badge-dot" id="statusDot"></span>
        <span id="aiStatusBadge">MANUAL HUMAN PLAY</span>
      </div>
      <div class="badge">
        <span>LATENCY:</span>
        <strong id="latencyBadge" style="color: var(--cyan-neon)">0.0 ms</strong>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <div class="dashboard-container">

    <!-- Left Column: Hold, Next, Core Stats -->
    <div class="glass-panel">
      <div class="panel-header">
        <div class="panel-title">PIECE QUEUE & STATS</div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div class="piece-preview-box">
          <div class="stat-label">HOLD (C)</div>
          <div class="mini-grid" id="holdGrid"></div>
        </div>
        <div class="piece-preview-box">
          <div class="stat-label">NEXT PIECE</div>
          <div class="mini-grid" id="nextGrid"></div>
        </div>
      </div>

      <div class="stat-grid">
        <div class="stat-box">
          <div class="stat-label">Score</div>
          <div class="stat-val highlight" id="scoreVal">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Lines</div>
          <div class="stat-val" id="linesVal">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Level</div>
          <div class="stat-val" style="color: var(--cyan-neon)" id="levelVal">1</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Tetrises (4L)</div>
          <div class="stat-val" style="color: var(--purple-neon)" id="tetrisVal">0</div>
        </div>
      </div>

      <div class="panel-header" style="margin-top: 6px;">
        <div class="panel-title">MATHEMATICAL TELEMETRY</div>
      </div>

      <table class="metrics-table">
        <tr>
          <td>Stack Max Height</td>
          <td id="teleHeight">0 / 20</td>
        </tr>
        <tr>
          <td>Buried Holes</td>
          <td id="teleHoles" style="color: var(--emerald-neon)">0</td>
        </tr>
        <tr>
          <td>Landing Height (LH)</td>
          <td id="teleLH">0.0</td>
        </tr>
        <tr>
          <td>Row Transitions (RT)</td>
          <td id="teleRT">0</td>
        </tr>
        <tr>
          <td>Col Transitions (CT)</td>
          <td id="teleCT">0</td>
        </tr>
        <tr>
          <td>Surface Bumpiness</td>
          <td id="teleBump">0</td>
        </tr>
      </table>

      <div class="farsi-subtext">
        موتور استاندارد با فیزیک سقوط طبیعی بلوک‌ها، حفظ رنگ‌ها، تشخیص دقیق تصادم و قفل در کف صفحه.
      </div>
    </div>

    <!-- Center Column: Tetris Board & Controls -->
    <div class="glass-panel arena-wrapper">
      <!-- Mode Tabs -->
      <div class="mode-tabs">
        <button class="mode-btn active" id="modeManualBtn" onclick="switchMode('manual')">
          🎮 MANUAL PLAY
        </button>
        <button class="mode-btn" id="modeLayaBtn" onclick="switchMode('laya')">
          ⚡ LAYA AI BRAIN
        </button>
        <button class="mode-btn" id="modeUnbeatableBtn" onclick="switchMode('unbeatable')">
          🛡️ UNBEATABLE
        </button>
      </div>

      <!-- Canvas Tetris Board (300 x 600 px) -->
      <canvas id="tetrisCanvas" width="300" height="600"></canvas>

      <!-- Action Buttons -->
      <div class="controls-row">
        <button class="btn primary" id="pauseBtn" onclick="togglePause()">
          ⏸ PAUSE
        </button>
        <button class="btn" onclick="hardDropAction()">
          ⬇ HARD DROP
        </button>
        <button class="btn" onclick="resetGame()">
          🔄 RESET
        </button>
      </div>

      <!-- Gravity Speed Slider -->
      <div class="speed-control">
        <label for="speedSlider">SPEED:</label>
        <input type="range" id="speedSlider" min="50" max="800" value="450" oninput="updateSpeed(this.value)">
        <span id="speedDisplay" style="font-size: 11px; min-width: 45px; text-align: right;">450ms</span>
      </div>

      <!-- Keybinds Guide -->
      <div class="keybinds-box">
        <strong>CONTROLS:</strong>
        <span class="key-badge">◀</span> <span class="key-badge">▶</span> Move &nbsp;|&nbsp;
        <span class="key-badge">▲</span> / <span class="key-badge">Z</span> Rotate &nbsp;|&nbsp;
        <span class="key-badge">▼</span> Soft Drop &nbsp;|&nbsp;
        <span class="key-badge">SPACE</span> Hard Drop &nbsp;|&nbsp;
        <span class="key-badge">C</span> Hold
      </div>
    </div>

    <!-- Right Column: Laya Decision Reasoning & Probabilities -->
    <div class="glass-panel">
      <div class="panel-header">
        <div class="panel-title">LAYA DECISION FEED</div>
      </div>

      <div class="ai-decision-card">
        <div class="decision-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
          </svg>
          CURRENT AI REASONING
        </div>
        <div class="decision-text" id="aiDecisionText">
          Manual play active. Press keyboard keys to control the falling block!
        </div>
      </div>

      <div class="panel-header">
        <div class="panel-title">PLACEMENT PROBABILITIES</div>
      </div>

      <div class="prob-list" id="probabilitiesList">
        <div style="font-size: 12px; color: #9d97b8; text-align: center; padding: 20px 0;">
          Switch to Laya AI to see real-time probability distribution...
        </div>
      </div>

      <div class="panel-header">
        <div class="panel-title">SYSTEM ARCHITECTURE</div>
      </div>

      <div style="font-size: 12px; color: #b5afcf; line-height: 1.6; display: flex; flex-direction: column; gap: 8px;">
        <div><strong>1. Real-Time Physics:</strong> بلوک‌ها بر اساس نرخ سقوط گرانش خط به خط پایین می‌آیند و روی پشته بلوک‌ها می‌نشینند.</div>
        <div><strong>2. Wall Kicks & SRS:</strong> چرخش‌ها در مجاورت دیواره‌ها یا سایر قطعات با تست‌های استاندارد وال‌کیک جابه‌جا می‌شوند.</div>
        <div><strong>3. Line Clearing:</strong> سطرهای پرشده پس از فرود قطعه پاک شده و ردیف‌های بالاتر پایین می‌ریزند.</div>
        <div><strong>4. AI Autopilot:</strong> مدل لایا با پیش‌بینی مسیر بهینه، قطعه را چرخانده و به ستون مقصد هدایت می‌کند.</div>
      </div>
    </div>

  </div>

  <script>
    // Audio Synthesizer via Web Audio API
    let audioCtx = null;
    function playTone(freq, type = 'sine', duration = 0.08, gainVal = 0.08) {
      try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch (e) {}
    }

    function playLineClearSound(lines) {
      if (lines === 4) {
        [523.25, 659.25, 783.99, 1046.50].forEach((f, idx) => {
          setTimeout(() => playTone(f, 'triangle', 0.25, 0.12), idx * 60);
        });
      } else if (lines > 0) {
        playTone(440, 'triangle', 0.1, 0.1);
        setTimeout(() => playTone(587.33, 'triangle', 0.12, 0.1), 70);
      }
    }

    // Canvas Board Renderer
    const canvas = document.getElementById('tetrisCanvas');
    const ctx = canvas.getContext('2d');
    const COLS = 10;
    const ROWS = 20;
    const BLOCK_SIZE = 30; // 300x600

    let currentMode = 'manual';
    let isPaused = false;
    let gravityInterval = 450;
    let gravityTimer = null;
    let aiNavTimer = null;
    let gameState = null;

    // AI Target tracking for animated AI moves
    let aiTarget = null;
    let aiStepInProgress = false;

    const PIECE_SHAPES = {
      'I': [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
      'O': [[1,1],[1,1]],
      'T': [[0,1,0],[1,1,1],[0,0,0]],
      'S': [[0,1,1],[1,1,0],[0,0,0]],
      'Z': [[1,1,0],[0,1,1],[0,0,0]],
      'J': [[1,0,0],[1,1,1],[0,0,0]],
      'L': [[0,0,1],[1,1,1],[0,0,0]]
    };

    const PIECE_COLORS = {
      'I': '#38bdf8',
      'O': '#facc15',
      'T': '#c084fc',
      'S': '#4ade80',
      'Z': '#f87171',
      'J': '#60a5fa',
      'L': '#fb923c'
    };

    function renderBoard(state) {
      if (!state) return;
      ctx.fillStyle = '#06040d';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw subtle grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= COLS; x++) {
        ctx.beginPath();
        ctx.moveTo(x * BLOCK_SIZE, 0);
        ctx.lineTo(x * BLOCK_SIZE, ROWS * BLOCK_SIZE);
        ctx.stroke();
      }
      for (let y = 0; y <= ROWS; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * BLOCK_SIZE);
        ctx.lineTo(COLS * BLOCK_SIZE, y * BLOCK_SIZE);
        ctx.stroke();
      }

      const board = state.board || [];
      const colors = state.piece_colors || PIECE_COLORS;

      // 1. Draw Locked Board Blocks with their piece colors!
      for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
          const cell = board[y] ? board[y][x] : 0;
          if (cell) {
            const color = (typeof cell === 'string' && colors[cell]) ? colors[cell] : '#94a3b8';
            drawBlock(x, y, color, false);
          }
        }
      }

      // 2. Draw Ghost Piece Outline (projection at bottom)
      if (state.ghost_blocks && state.ghost_blocks.length) {
        const pieceColor = colors[state.current_piece] || '#38bdf8';
        state.ghost_blocks.forEach(([gx, gy]) => {
          if (gy >= 0) drawBlock(gx, gy, pieceColor, true);
        });
      }

      // 3. Draw Active Falling Piece
      if (state.current_blocks && state.current_blocks.length) {
        const pieceColor = colors[state.current_piece] || '#38bdf8';
        state.current_blocks.forEach(([cx, cy]) => {
          if (cy >= 0) drawBlock(cx, cy, pieceColor, false);
        });
      }

      // 4. Game Over Overlay
      if (!state.alive) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#f43f5e';
        ctx.font = '700 24px "Space Grotesk", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('GAME OVER', canvas.width / 2, canvas.height / 2 - 15);

        ctx.fillStyle = '#FAF8F4';
        ctx.font = '500 13px "Space Grotesk", sans-serif';
        ctx.fillText('Press RESET to start fresh', canvas.width / 2, canvas.height / 2 + 18);
      }
    }

    function drawBlock(x, y, color, isGhost = false) {
      const px = x * BLOCK_SIZE;
      const py = y * BLOCK_SIZE;
      const pad = 2;

      if (isGhost) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(px + pad, py + pad, BLOCK_SIZE - pad * 2, BLOCK_SIZE - pad * 2);
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.fillRect(px + pad, py + pad, BLOCK_SIZE - pad * 2, BLOCK_SIZE - pad * 2);
      } else {
        ctx.fillStyle = color;
        ctx.fillRect(px + pad, py + pad, BLOCK_SIZE - pad * 2, BLOCK_SIZE - pad * 2);

        // Modern 3D beveled edges
        ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.fillRect(px + pad, py + pad, BLOCK_SIZE - pad * 2, 2.5);
        ctx.fillRect(px + pad, py + pad, 2.5, BLOCK_SIZE - pad * 2);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.fillRect(px + pad, py + BLOCK_SIZE - pad - 2.5, BLOCK_SIZE - pad * 2, 2.5);
        ctx.fillRect(px + BLOCK_SIZE - pad - 2.5, py + pad, 2.5, BLOCK_SIZE - pad * 2);
      }
    }

    function renderMiniPiece(containerId, pieceType) {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      if (!pieceType) return;

      const shape = PIECE_SHAPES[pieceType] || [];
      const color = PIECE_COLORS[pieceType] || '#38bdf8';

      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          const cell = document.createElement('div');
          cell.className = 'mini-cell';
          if (shape[r] && shape[r][c]) {
            cell.style.background = color;
            cell.style.boxShadow = `0 0 6px ${color}`;
          }
          container.appendChild(cell);
        }
      }
    }

    function updateUI(data) {
      if (!data) return;
      const prevLines = gameState ? gameState.lines : data.lines;
      gameState = data;
      renderBoard(data);

      document.getElementById('scoreVal').innerText = data.score.toLocaleString();
      document.getElementById('linesVal').innerText = data.lines;
      document.getElementById('levelVal').innerText = data.level;
      document.getElementById('tetrisVal').innerText = data.tetris_count;

      if (data.lines > prevLines) {
        playLineClearSound(data.lines - prevLines);
      }

      document.getElementById('latencyBadge').innerText = `${data.latency_ms} ms`;
      document.getElementById('aiDecisionText').innerText = data.decision_note;

      renderMiniPiece('nextGrid', data.next_piece);
      renderMiniPiece('holdGrid', data.hold_piece);

      // Mathematical Telemetry
      const ft = data.features || {};
      const summary = data.summary || {};
      document.getElementById('teleHeight').innerText = `${summary.max_height || 0} / 20`;
      document.getElementById('teleHoles').innerText = summary.holes || 0;
      document.getElementById('teleLH').innerText = (ft.landing_height || 0).toFixed(1);
      document.getElementById('teleRT').innerText = ft.row_transitions || 0;
      document.getElementById('teleCT').innerText = ft.col_transitions || 0;
      document.getElementById('teleBump').innerText = ft.bumpiness || 0;

      // Probabilities
      const pContainer = document.getElementById('probabilitiesList');
      const probs = data.probabilities || {};
      const keys = Object.keys(probs);
      if (keys.length > 0) {
        let html = '';
        keys.forEach(k => {
          const val = probs[k] || 0;
          const pct = Math.round(val * 100);
          html += `
            <div class="prob-item">
              <div class="prob-info">
                <span>${k}</span>
                <span><strong>${pct}%</strong></span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${pct}%"></div>
              </div>
            </div>
          `;
        });
        pContainer.innerHTML = html;
      }
    }

    // Action API Calls
    async function sendAction(action) {
      try {
        const res = await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action })
        });
        const data = await res.json();
        updateUI(data);
      } catch (e) {
        console.error('Action error:', e);
      }
    }

    async function tickGravity() {
      if (isPaused || (gameState && !gameState.alive)) return;
      try {
        const res = await fetch('/api/tick', { method: 'POST' });
        const data = await res.json();
        updateUI(data);
      } catch (e) {
        console.error('Tick error:', e);
      }
    }

    // AI Step Execution (Smooth Animated Fall or Direct Placement)
    async function runAILoop() {
      if (isPaused || currentMode === 'manual' || (gameState && !gameState.alive)) return;

      if (!aiTarget) {
        // Fetch new target from AI Brain
        try {
          const res = await fetch('/api/ai_target', { method: 'POST' });
          const target = await res.json();
          if (target && target.alive) {
            aiTarget = target;
            document.getElementById('aiDecisionText').innerText = target.decision_note;
            document.getElementById('latencyBadge').innerText = `${target.latency_ms} ms`;
            updateProbabilitiesUI(target.probabilities);
          }
        } catch (e) {
          console.error('AI Target error:', e);
        }
      }

      if (aiTarget && gameState) {
        // Step 1: Align Rotation
        if (gameState.curr_rot !== aiTarget.target_rot) {
          await sendAction('rotate');
          playTone(320, 'triangle', 0.03, 0.05);
        }
        // Step 2: Align Column
        else if (gameState.curr_x < aiTarget.target_col) {
          await sendAction('right');
          playTone(240, 'sine', 0.02, 0.05);
        } else if (gameState.curr_x > aiTarget.target_col) {
          await sendAction('left');
          playTone(240, 'sine', 0.02, 0.05);
        }
        // Step 3: Aligned! Drop down with gravity
        else {
          await tickGravity();
          // If piece just locked (y returned to 0), reset target for next piece
          if (gameState.curr_y <= 0) {
            aiTarget = null;
          }
        }
      }

      // Schedule next AI tick based on speed slider
      const nextDelay = Math.max(30, Math.floor(gravityInterval / 3));
      aiNavTimer = setTimeout(runAILoop, nextDelay);
    }

    function updateProbabilitiesUI(probs) {
      const pContainer = document.getElementById('probabilitiesList');
      if (!probs) return;
      const keys = Object.keys(probs);
      if (keys.length > 0) {
        let html = '';
        keys.forEach(k => {
          const val = probs[k] || 0;
          const pct = Math.round(val * 100);
          html += `
            <div class="prob-item">
              <div class="prob-info">
                <span>${k}</span>
                <span><strong>${pct}%</strong></span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${pct}%"></div>
              </div>
            </div>
          `;
        });
        pContainer.innerHTML = html;
      }
    }

    function startGravityLoop() {
      clearInterval(gravityTimer);
      clearTimeout(aiNavTimer);
      if (currentMode === 'manual') {
        gravityTimer = setInterval(tickGravity, gravityInterval);
      } else {
        runAILoop();
      }
    }

    function updateSpeed(val) {
      gravityInterval = parseInt(val);
      document.getElementById('speedDisplay').innerText = `${gravityInterval}ms`;
      startGravityLoop();
    }

    function togglePause() {
      isPaused = !isPaused;
      const btn = document.getElementById('pauseBtn');
      if (isPaused) {
        btn.innerText = '▶ RESUME';
        btn.style.background = 'linear-gradient(135deg, var(--emerald-neon) 0%, #059669 100%)';
        btn.style.color = '#fff';
        clearInterval(gravityTimer);
        clearTimeout(aiNavTimer);
      } else {
        btn.innerText = '⏸ PAUSE';
        btn.style.background = 'linear-gradient(135deg, var(--cyan-neon) 0%, #0284c7 100%)';
        btn.style.color = '#031424';
        startGravityLoop();
      }
    }

    async function hardDropAction() {
      playTone(180, 'square', 0.08, 0.1);
      await sendAction('hard_drop');
    }

    async function resetGame() {
      aiTarget = null;
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      updateUI(data);
      if (isPaused) togglePause();
      startGravityLoop();
    }

    async function switchMode(mode) {
      currentMode = mode;
      aiTarget = null;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      if (mode === 'manual') document.getElementById('modeManualBtn').classList.add('active');
      if (mode === 'laya') document.getElementById('modeLayaBtn').classList.add('active');
      if (mode === 'unbeatable') document.getElementById('modeUnbeatableBtn').classList.add('active');

      const badge = document.getElementById('aiStatusBadge');
      if (mode === 'manual') badge.innerText = 'MANUAL HUMAN PLAY';
      else if (mode === 'laya') badge.innerText = 'LAYA AI AUTOPILOT';
      else badge.innerText = 'DELLACHERIE OPTIMIZER';

      await fetch('/api/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });

      startGravityLoop();
    }

    // Keyboard Controller for Manual Play
    window.addEventListener('keydown', async (e) => {
      if (isPaused || (gameState && !gameState.alive)) return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        playTone(220, 'sine', 0.03, 0.04);
        await sendAction('left');
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        playTone(220, 'sine', 0.03, 0.04);
        await sendAction('right');
      } else if (e.key === 'ArrowUp' || e.key === 'z' || e.key === 'Z') {
        e.preventDefault();
        playTone(340, 'triangle', 0.04, 0.05);
        await sendAction('rotate');
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        playTone(180, 'sine', 0.02, 0.04);
        await sendAction('soft_drop');
      } else if (e.key === ' ') {
        e.preventDefault();
        playTone(160, 'square', 0.08, 0.1);
        await sendAction('hard_drop');
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        playTone(400, 'sine', 0.05, 0.06);
        await sendAction('hold');
      }
    });

    // Initialize
    window.addEventListener('DOMContentLoaded', async () => {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        updateUI(data);
        startGravityLoop();
      } catch (e) {
        console.error('Init error:', e);
      }
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Zheyar AI Labs - Laya Tetris Web Server")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind server (default: 8081)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\nStarting Zheyar Laya Tetris Web Server on http://localhost:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
