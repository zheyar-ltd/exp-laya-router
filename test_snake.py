import unittest
import requests
from snake_laya import (
    SnakeGame,
    HAMILTONIAN_CYCLE,
    CYCLE_INDEX,
    CYCLE_LEN,
    PLAYABLE_W,
    PLAYABLE_H,
    WIDTH,
    HEIGHT,
    DELTAS,
    cycle_dist,
    generate_hamiltonian_cycle,
    query_laya,
    play
)


class TestHamiltonianCycle(unittest.TestCase):
    def test_cycle_length_and_uniqueness(self):
        expected_len = PLAYABLE_W * PLAYABLE_H  # 22 * 12 = 264
        self.assertEqual(len(HAMILTONIAN_CYCLE), expected_len)
        self.assertEqual(len(set(HAMILTONIAN_CYCLE)), expected_len)

    def test_cycle_within_playable_bounds(self):
        for x, y in HAMILTONIAN_CYCLE:
            self.assertTrue(1 <= x <= PLAYABLE_W, f"x={x} out of bounds")
            self.assertTrue(1 <= y <= PLAYABLE_H, f"y={y} out of bounds")

    def test_cycle_continuity_and_closure(self):
        for i in range(CYCLE_LEN):
            curr_pos = HAMILTONIAN_CYCLE[i]
            next_pos = HAMILTONIAN_CYCLE[(i + 1) % CYCLE_LEN]
            dist = abs(curr_pos[0] - next_pos[0]) + abs(curr_pos[1] - next_pos[1])
            self.assertEqual(dist, 1, f"Gap in cycle at step {i}: {curr_pos} -> {next_pos}")

    def test_cycle_distance_calculation(self):
        p0 = HAMILTONIAN_CYCLE[0]
        p1 = HAMILTONIAN_CYCLE[1]
        p_last = HAMILTONIAN_CYCLE[-1]

        self.assertEqual(cycle_dist(p0, p0), 0)
        self.assertEqual(cycle_dist(p0, p1), 1)
        self.assertEqual(cycle_dist(p1, p0), CYCLE_LEN - 1)
        self.assertEqual(cycle_dist(p_last, p0), 1)


class TestGridAndMovement(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGame(start_pos=[[6, 7], [5, 7], [4, 7]])

    def test_safe_tile_bounds(self):
        # Walls
        self.assertFalse(self.game.is_safe_tile([0, 5]))
        self.assertFalse(self.game.is_safe_tile([WIDTH - 1, 5]))
        self.assertFalse(self.game.is_safe_tile([5, 0]))
        self.assertFalse(self.game.is_safe_tile([5, HEIGHT - 1]))

        # Open tile
        self.assertTrue(self.game.is_safe_tile([6, 8]))

    def test_body_collision_detection(self):
        # Body segments (except the tail tip which vacates upon non-eating step)
        self.assertFalse(self.game.is_safe_tile([5, 7]))


class TestPathfindingAndFloodFill(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGame(start_pos=[[5, 5], [4, 5], [3, 5]])

    def test_bfs_direct_path(self):
        path = self.game.find_path_bfs([5, 5], [5, 8])
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 4)
        self.assertEqual(path[0], [5, 5])
        self.assertEqual(path[-1], [5, 8])

    def test_bfs_wall_detour(self):
        # Put obstacles in between
        obstacles = [[5, 6], [4, 6], [6, 6]]
        path = self.game.find_path_bfs([5, 5], [5, 7], obstacle_body=obstacles)
        self.assertIsNotNone(path)
        for tile in path:
            self.assertNotIn(tile, obstacles)

    def test_bfs_blocked(self):
        # Fully encircle target [2, 2]
        obstacles = [[1, 2], [3, 2], [2, 1], [2, 3]]
        path = self.game.find_path_bfs([5, 5], [2, 2], obstacle_body=obstacles)
        self.assertIsNone(path)

    def test_flood_fill_accuracy(self):
        # Complete open grid with 3-segment snake
        space = self.game.flood_fill_count([5, 6])
        # Total playable is 264, minus 2 body segments = 262
        self.assertEqual(space, 262)


class TestVirtualFoodSimulation(unittest.TestCase):
    def test_safe_food_simulation(self):
        game = SnakeGame(start_pos=[[5, 5], [4, 5], [3, 5]], initial_food=[7, 5])
        path = game.find_path_bfs(game.snake[0], game.food)
        self.assertTrue(game.simulate_eating_food(path))

    def test_trapped_food_simulation(self):
        # Head at [1, 2], Food at [1, 1] (corner).
        # [2, 1] is occupied by body, and tail is far away at [5, 5].
        # Stepping into [1, 1] leaves head with walls on UP/LEFT and body on DOWN/RIGHT.
        body = [
            [1, 3], [2, 3], [2, 2], [2, 1], [3, 1],
            [4, 1], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
        ]
        game = SnakeGame(start_pos=[[1, 2]] + body, initial_food=[1, 1])
        path = game.find_path_bfs(game.snake[0], game.food)
        self.assertIsNotNone(path)
        can_escape = game.simulate_eating_food(path)
        self.assertFalse(can_escape)


class TestHamiltonianShortcuts(unittest.TestCase):
    def test_hamiltonian_move_safety(self):
        game = SnakeGame()
        for _ in range(50):
            move = game.get_hamiltonian_move(allow_shortcuts=True)
            self.assertIn(move, ["UP", "DOWN", "LEFT", "RIGHT"])
            dx, dy = DELTAS[move]
            head = game.snake[0]
            nxt = [head[0] + dx, head[1] + dy]
            self.assertTrue(game.is_safe_tile(nxt))
            game.snake.insert(0, nxt)
            game.snake.pop()


class TestLayaIntegrationAndGuardrails(unittest.TestCase):
    def test_guardrail_vetoes_trap(self):
        # Create state where UP is a death trap with only 2 free tiles,
        # but RIGHT is completely open with 250 free tiles
        valid_moves = {
            "UP": {
                "closer_to_apple": True,
                "free_space": 2,
                "has_tail_escape": False,
                "is_safe_food_move": False,
                "is_hamiltonian_move": False,
                "is_trap": True,
                "dist_to_apple": 1
            },
            "RIGHT": {
                "closer_to_apple": False,
                "free_space": 250,
                "has_tail_escape": True,
                "is_safe_food_move": False,
                "is_hamiltonian_move": True,
                "is_trap": False,
                "dist_to_apple": 5
            }
        }
        choice, latency, note = query_laya("Snake state test", valid_moves)
        # The guardrail MUST never select UP
        self.assertEqual(choice, "RIGHT")
        self.assertTrue("Guardrail" in note or "Fallback" in note or choice == "RIGHT")


class TestSimulationPerformance(unittest.TestCase):
    def test_unbeatable_mode_survives_500_steps(self):
        score, steps, won = play(mode="unbeatable", speed=0, max_steps=500, silent=True)
        # In 500 steps, snake should eat plenty of apples and never die
        self.assertGreaterEqual(steps, 500)
        self.assertGreater(score, 15)


if __name__ == "__main__":
    unittest.main()
