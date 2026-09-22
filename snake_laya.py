import os
import sys
import time
import random
import argparse
import requests
from collections import deque

API_URL = "http://127.0.0.1:8000/predict"
WIDTH, HEIGHT = 24, 14
PLAYABLE_W, PLAYABLE_H = WIDTH - 2, HEIGHT - 2  # 22, 12 -> 264 tiles

OPPOSITES = {
    "UP": "DOWN", "DOWN": "UP",
    "LEFT": "RIGHT", "RIGHT": "LEFT"
}

DELTAS = {
    "UP": (0, -1), "DOWN": (0, 1),
    "LEFT": (-1, 0), "RIGHT": (1, 0)
}

DIR_NAMES = {v: k for k, v in DELTAS.items()}


def generate_hamiltonian_cycle(w=PLAYABLE_W, h=PLAYABLE_H):
    """
    Generates a closed Hamiltonian cycle covering all w*h playable tiles.
    Coordinates run from x in 1..w and y in 1..h.
    Requires w and h to be even.
    """
    assert w % 2 == 0 and h % 2 == 0, "Width and height must be even"
    cycle = []

    # 1. Top row from x=1 to w
    for x in range(1, w + 1):
        cycle.append((x, 1))

    # 2. Rightmost column from y=2 to h
    for y in range(2, h + 1):
        cycle.append((w, y))

    # 3. Serpentine vertical columns from x=w-1 down to 2
    # Number of columns is w-2 (even)
    for col_idx, x in enumerate(range(w - 1, 1, -1)):
        if col_idx % 2 == 0:
            for y in range(h, 1, -1):
                cycle.append((x, y))
        else:
            for y in range(2, h + 1):
                cycle.append((x, y))

    # 4. Leftmost column from y=h down to 2 (connects back to (1, 1))
    for y in range(h, 1, -1):
        cycle.append((1, y))

    return cycle


HAMILTONIAN_CYCLE = generate_hamiltonian_cycle(PLAYABLE_W, PLAYABLE_H)
CYCLE_INDEX = {pos: i for i, pos in enumerate(HAMILTONIAN_CYCLE)}
CYCLE_LEN = len(HAMILTONIAN_CYCLE)


def cycle_dist(pos_a, pos_b):
    """Distance from pos_a to pos_b along the Hamiltonian cycle."""
    idx_a = CYCLE_INDEX[tuple(pos_a)]
    idx_b = CYCLE_INDEX[tuple(pos_b)]
    return (idx_b - idx_a) % CYCLE_LEN


class SnakeGame:
    def __init__(self, start_pos=None, initial_food=None):
        if start_pos is not None:
            self.snake = [list(p) for p in start_pos]
        else:
            self.snake = [[6, 7], [5, 7], [4, 7]]
        self.direction = "RIGHT"
        self.food = list(initial_food) if initial_food else [16, 7]
        self.score = 0
        self.step_count = 0
        self.alive = True

    def spawn_food(self):
        snake_set = {tuple(p) for p in self.snake}
        available = [
            [x, y] for x in range(1, WIDTH - 1)
            for y in range(1, HEIGHT - 1)
            if (x, y) not in snake_set
        ]
        if not available:
            return None  # Board is completely filled! Max score win!
        return random.choice(available)

    def is_safe_tile(self, pos, custom_body=None):
        """Returns True if pos is within bounds and not in the snake body."""
        body = custom_body if custom_body is not None else self.snake[:-1]
        x, y = pos
        return 0 < x < WIDTH - 1 and 0 < y < HEIGHT - 1 and pos not in body

    def flood_fill_count(self, start, custom_body=None):
        """Calculates the number of reachable tiles from start."""
        body_set = {tuple(p) for p in (custom_body if custom_body is not None else self.snake[:-1])}
        queue = deque([tuple(start)])
        visited = {tuple(start)}

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in DELTAS.values():
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if 0 < nx < WIDTH - 1 and 0 < ny < HEIGHT - 1:
                    if nxt not in body_set and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        return len(visited)

    def find_path_bfs(self, start, target, obstacle_body=None):
        """Finds the shortest BFS path from start to target avoiding obstacles."""
        body_set = {tuple(p) for p in (obstacle_body if obstacle_body is not None else self.snake[:-1])}
        target_tuple = tuple(target)
        start_tuple = tuple(start)

        if start_tuple == target_tuple:
            return [start]

        queue = deque([[start]])
        visited = {start_tuple}

        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == target:
                return path

            for dx, dy in DELTAS.values():
                nxt = [curr[0] + dx, curr[1] + dy]
                nxt_tuple = tuple(nxt)
                if nxt == target or (0 < nxt[0] < WIDTH - 1 and 0 < nxt[1] < HEIGHT - 1 and nxt_tuple not in body_set):
                    if nxt_tuple not in visited:
                        visited.add(nxt_tuple)
                        queue.append(path + [nxt])
        return None

    def simulate_eating_food(self, path_to_food):
        """
        Simulates the snake following path_to_food until it eats the food.
        Returns True if, after eating the food, the virtual head can still reach the virtual tail.
        """
        if not path_to_food or len(path_to_food) < 2:
            return False

        virtual_snake = [list(p) for p in self.snake]
        for step in path_to_food[1:]:
            virtual_snake.insert(0, list(step))
            if step == self.food:
                # Eating food: snake grows, tail is not popped on this final step
                pass
            else:
                virtual_snake.pop()

        v_head = virtual_snake[0]
        v_tail = virtual_snake[-1]
        v_body_without_tail = virtual_snake[:-1]

        # Check if virtual head can reach virtual tail
        path_to_tail = self.find_path_bfs(v_head, v_tail, obstacle_body=v_body_without_tail)
        return path_to_tail is not None

    def find_longest_path_to_tail(self, head, tail, custom_body=None):
        """
        Finds a safe wandering path towards tail that maximizes space and avoids traps.
        Falls back to Hamiltonian cycle step if needed.
        """
        best_move = None
        max_dist = -1
        max_space = -1

        for d, (dx, dy) in DELTAS.items():
            if d == OPPOSITES[self.direction]:
                continue
            nxt = [head[0] + dx, head[1] + dy]
            if not self.is_safe_tile(nxt, custom_body):
                continue

            sim_body = [nxt] + (custom_body if custom_body is not None else self.snake[:-1])
            path = self.find_path_bfs(nxt, tail, sim_body)
            space = self.flood_fill_count(nxt, sim_body)

            if path is not None:
                dist = len(path)
                if dist > max_dist or (dist == max_dist and space > max_space):
                    max_dist = dist
                    max_space = space
                    best_move = d

        return best_move

    def get_hamiltonian_move(self, allow_shortcuts=True):
        """
        Computes the next move based on the Hamiltonian cycle.
        If allow_shortcuts is True, safely takes shortcuts toward the food
        as long as the head remains strictly behind the tail.
        """
        head = self.snake[0]
        tail = self.snake[-1]
        head_idx = CYCLE_INDEX[tuple(head)]
        tail_idx = CYCLE_INDEX[tuple(tail)]
        dist_to_tail = (tail_idx - head_idx) % CYCLE_LEN
        snake_len = len(self.snake)

        # Default move: strictly follow the Hamiltonian cycle
        next_cycle_tile = list(HAMILTONIAN_CYCLE[(head_idx + 1) % CYCLE_LEN])
        dx = next_cycle_tile[0] - head[0]
        dy = next_cycle_tile[1] - head[1]
        default_dir = DIR_NAMES.get((dx, dy), "RIGHT")

        if not allow_shortcuts or self.food is None:
            return default_dir

        food_idx = CYCLE_INDEX[tuple(self.food)]
        dist_to_food = (food_idx - head_idx) % CYCLE_LEN

        best_dir = default_dir
        min_dist_to_food = dist_to_food

        for d, (ddx, ddy) in DELTAS.items():
            if d == OPPOSITES[self.direction]:
                continue
            nxt = [head[0] + ddx, head[1] + ddy]
            if not self.is_safe_tile(nxt):
                continue

            nxt_idx = CYCLE_INDEX[tuple(nxt)]
            nxt_dist_to_tail = (tail_idx - nxt_idx) % CYCLE_LEN
            nxt_dist_to_food = (food_idx - nxt_idx) % CYCLE_LEN

            # Safe shortcut condition:
            # 1. nxt must not skip past or cut off the tail (must preserve buffer > snake_len + 3)
            # 2. nxt must move strictly closer to food along cycle
            safe_buffer = snake_len + 3
            if nxt_dist_to_tail > safe_buffer and nxt_dist_to_tail <= dist_to_tail:
                if nxt_dist_to_food < min_dist_to_food:
                    min_dist_to_food = nxt_dist_to_food
                    best_dir = d

        return best_dir

    def evaluate_moves(self):
        """
        Thoroughly analyzes all adjacent legal moves, computing safety,
        free space, distance to food, tail escape routes, and Hamiltonian metrics.
        """
        head = self.snake[0]
        tail = self.snake[-1]
        fx, fy = self.food if self.food else (head[0], head[1])

        moves_info = {}

        # 1. Shortest path to food
        shortest_path_to_food = self.find_path_bfs(head, [fx, fy])
        safe_food_first_step = None
        if shortest_path_to_food and len(shortest_path_to_food) >= 2:
            if self.simulate_eating_food(shortest_path_to_food):
                step = shortest_path_to_food[1]
                safe_food_first_step = DIR_NAMES.get((step[0] - head[0], step[1] - head[1]))

        # 2. Hamiltonian shortcut move
        hamiltonian_best = self.get_hamiltonian_move(allow_shortcuts=True)

        for d, (dx, dy) in DELTAS.items():
            if d == OPPOSITES[self.direction]:
                continue
            nxt = [head[0] + dx, head[1] + dy]
            if not self.is_safe_tile(nxt):
                continue

            sim_body = [nxt] + self.snake[:-1]
            space = self.flood_fill_count(nxt, sim_body)
            path_to_tail = self.find_path_bfs(nxt, tail, sim_body)
            has_tail_escape = path_to_tail is not None

            curr_dist = abs(head[0] - fx) + abs(head[1] - fy)
            nxt_dist = abs(nxt[0] - fx) + abs(nxt[1] - fy)
            closer_to_apple = nxt_dist < curr_dist

            is_safe_food_move = (d == safe_food_first_step)
            is_hamiltonian_move = (d == hamiltonian_best)

            # A move is classified as a lethal trap if reachable area is too small and no tail escape
            is_trap = (space < len(self.snake) and not has_tail_escape)

            moves_info[d] = {
                "closer_to_apple": closer_to_apple,
                "free_space": space,
                "has_tail_escape": has_tail_escape,
                "is_safe_food_move": is_safe_food_move,
                "is_hamiltonian_move": is_hamiltonian_move,
                "is_trap": is_trap,
                "dist_to_apple": nxt_dist
            }

        return moves_info

    def get_state(self):
        head = self.snake[0]
        tail = self.snake[-1]
        fx, fy = self.food if self.food else (-1, -1)
        moves_info = self.evaluate_moves()

        state_desc = (
            f"Step: {self.step_count}. Snake Length: {len(self.snake)}/{PLAYABLE_W * PLAYABLE_H}. "
            f"Head: {head}, Apple: [{fx}, {fy}], Tail: {tail}. "
            f"Available legal moves: {list(moves_info.keys())}."
        )
        return state_desc, moves_info

    def draw(self, action, latency_ms, mode="LAYA AI", guardrail_note=""):
        os.system("cls" if os.name == "nt" else "clear")
        grid = [["  " for _ in range(WIDTH)] for _ in range(HEIGHT)]

        for x in range(WIDTH):
            grid[0][x] = "⬛"
            grid[HEIGHT - 1][x] = "⬛"
        for y in range(HEIGHT):
            grid[y][0] = "⬛"
            grid[y][WIDTH - 1] = "⬛"

        if self.food:
            grid[self.food[1]][self.food[0]] = "🍎"

        for i, seg in enumerate(self.snake):
            if 0 < seg[0] < WIDTH - 1 and 0 < seg[1] < HEIGHT - 1:
                grid[seg[1]][seg[0]] = "🟢" if i == 0 else "🟩"

        fill_pct = (len(self.snake) / (PLAYABLE_W * PLAYABLE_H)) * 100
        print("=" * (WIDTH * 2))
        print(f"     SNAKE AI ENGINE: [{mode.upper()}]       ")
        print("=" * (WIDTH * 2))
        for row in grid:
            print("".join(row))
        print("-" * (WIDTH * 2))
        print(f"Score: {self.score} | Length: {len(self.snake)} ({fill_pct:.1f}%) | Steps: {self.step_count}")
        status_line = f"Action: [{action}] | Latency: {latency_ms:.1f}ms"
        if guardrail_note:
            status_line += f" | {guardrail_note}"
        print(status_line)
        print("=" * (WIDTH * 2))


def query_laya(state_desc: str, valid_moves: dict):
    """
    Sends the state and properly structured criteria for ALL valid moves to Laya Router API.
    Uses a Safety Guardrail to ensure Laya never takes a fatal/trapping move.
    """
    if not valid_moves:
        return "RIGHT", 0.0, "No moves available"

    criteria = {}
    safe_moves = []

    for d, info in valid_moves.items():
        if info["is_trap"]:
            criteria[d] = f"Dangerous move {d}: Trap warning! Only {info['free_space']} tiles free. Dead-end collision risk."
        elif info["is_safe_food_move"]:
            criteria[d] = f"(Recommended) Optimal move {d}: Direct safe route to apple (dist={info['dist_to_apple']}), tail escape verified, {info['free_space']} free tiles."
            safe_moves.append(d)
        elif info["is_hamiltonian_move"]:
            criteria[d] = f"Safe strategic move {d}: Follows Hamiltonian safety path, preserving {info['free_space']} free tiles."
            safe_moves.append(d)
        elif info["has_tail_escape"]:
            criteria[d] = f"Cautious move {d}: Circles around obstacles towards tail escape with {info['free_space']} free tiles."
            safe_moves.append(d)
        else:
            criteria[d] = f"Alternative move {d}: Navigates open area with {info['free_space']} tiles."
            safe_moves.append(d)

    # If all moves had trap warnings, allow any move that maximizes space
    if not safe_moves:
        best_space_move = max(valid_moves.keys(), key=lambda m: valid_moves[m]["free_space"])
        safe_moves = [best_space_move]

    payload = {
        "state": state_desc,
        "questions": {
            "move": {
                "instructions": "Select the single safest and most efficient direction to eat the apple without trapping the snake:",
                "criteria": criteria
            }
        }
    }

    t0 = time.perf_counter()
    guardrail_note = "Laya Choice"
    try:
        res = requests.post(API_URL, json=payload, timeout=2.5).json()
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = res.get("answers", {}).get("move", {}).get("choice")

        # Safety Guardrail: If Laya picked an invalid or fatal trap move when safe alternatives exist
        if choice not in valid_moves or valid_moves[choice]["is_trap"]:
            fallback = safe_moves[0]
            guardrail_note = f"[Guardrail] Vetoed dangerous '{choice}' -> Selected '{fallback}'"
            choice = fallback
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = safe_moves[0]
        guardrail_note = "[Offline Fallback] Safe move selected"

    return choice, latency_ms, guardrail_note


def play(mode="laya", speed=0.06, max_steps=10000, silent=False):
    """
    Main game loop.
    Modes:
      - 'laya': Laya Router AI with Safety Guardrail
      - 'unbeatable': Pure Hamiltonian Cycle with Safe Shortcuts (100% win rate)
    """
    game = SnakeGame()

    while game.alive and game.step_count < max_steps:
        game.step_count += 1
        state_text, valid_moves = game.get_state()

        if not valid_moves:
            game.alive = False
            if not silent:
                print("\nNo valid moves left. Game over!")
            break

        guardrail_note = ""
        latency = 0.0

        if mode == "unbeatable":
            action = game.get_hamiltonian_move(allow_shortcuts=True)
            latency = 0.05
        else:  # laya mode
            action, latency, guardrail_note = query_laya(state_text, valid_moves)

        game.direction = action
        dx, dy = DELTAS[action]
        head = game.snake[0]
        new_head = [head[0] + dx, head[1] + dy]

        # Verify legality
        if not game.is_safe_tile(new_head):
            game.alive = False
            if not silent:
                print(f"\nCollision detected at {new_head}! Game over.")
            break

        game.snake.insert(0, new_head)

        if new_head == game.food:
            game.score += 1
            game.food = game.spawn_food()
            if game.food is None:
                # Board completely filled!
                if not silent:
                    game.draw(action, latency, mode=mode, guardrail_note="PERFECT BOARD CLEAR! VICTORY!")
                    print("\nCONGRATULATIONS! Snake completely filled the 264-tile board!")
                return game.score, game.step_count, True
        else:
            game.snake.pop()

        if not silent:
            game.draw(action, latency, mode=mode, guardrail_note=guardrail_note)
            if speed > 0:
                time.sleep(speed)

    return game.score, game.step_count, False


def run_benchmark(num_games=5, mode="unbeatable", max_steps=2000):
    """Runs automated benchmark games and outputs statistical summary."""
    print(f"\n========================================================")
    print(f"       RUNNING BENCHMARK: {num_games} GAMES ({mode.upper()})")
    print(f"========================================================")

    scores = []
    steps_list = []
    t_start = time.perf_counter()

    for i in range(1, num_games + 1):
        score, steps, won = play(mode=mode, speed=0, max_steps=max_steps, silent=True)
        scores.append(score)
        steps_list.append(steps)
        status = "WON" if won else "ALIVE" if steps >= max_steps else "DIED"
        print(f"  Game {i:02d}: Score = {score:3d} | Steps = {steps:5d} | Status: {status}")

    t_total = time.perf_counter() - t_start
    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps_list) / len(steps_list)

    print("-" * 56)
    print(f"Benchmark Results:")
    print(f"  Average Score: {avg_score:.1f} / {PLAYABLE_W * PLAYABLE_H}")
    print(f"  Max Score:     {max(scores)}")
    print(f"  Min Score:     {min(scores)}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Total Time:    {t_total:.2f}s ({(t_total / num_games):.2f}s/game)")
    print("=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laya Intelligent & Unbeatable Snake Game")
    parser.add_argument("--mode", choices=["laya", "unbeatable", "benchmark"], default="laya",
                        help="Game mode: 'laya' (AI with guardrail), 'unbeatable' (Hamiltonian shortcuts), or 'benchmark'")
    parser.add_argument("--speed", type=float, default=0.06, help="Delay between frames in seconds")
    parser.add_argument("--games", type=int, default=5, help="Number of benchmark games to run")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps before stopping game")
    args = parser.parse_args()

    if args.mode == "benchmark":
        run_benchmark(num_games=args.games, mode="unbeatable", max_steps=args.max_steps)
    else:
        play(mode=args.mode, speed=args.speed, max_steps=args.max_steps)