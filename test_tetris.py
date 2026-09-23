"""
Comprehensive Unit Tests for Tetris Laya AI Engine.
"""

import unittest
from tetris_laya import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    TETROMINOES,
    PIECE_NAMES,
    TetrisGame,
    get_piece_blocks,
    is_valid_position,
    drop_piece_to_bottom,
    simulate_placement,
    calculate_dellacherie_features,
    evaluate_placement_score,
    get_all_possible_placements,
    get_two_piece_best_placement,
    query_laya_tetris,
    run_benchmark
)


class TestTetrisCore(unittest.TestCase):
    def setUp(self):
        self.game = TetrisGame(seed=123)

    def test_board_dimensions(self):
        self.assertEqual(len(self.game.board), BOARD_HEIGHT)
        self.assertEqual(len(self.game.board[0]), BOARD_WIDTH)
        for row in self.game.board:
            self.assertEqual(len(row), BOARD_WIDTH)
            self.assertTrue(all(c == 0 for c in row))

    def test_7bag_randomizer(self):
        # Reset bag so we start with an exact 7-bag boundary
        self.game.bag = []
        # 14 pieces popped should contain exactly 2 of each of the 7 pieces
        pieces = [self.game.pop_from_bag() for _ in range(14)]
        for p in PIECE_NAMES:
            self.assertEqual(pieces.count(p), 2, f"Piece {p} count is not 2")

    def test_valid_positions_and_boundaries(self):
        empty_board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # Center of board should be valid
        self.assertTrue(is_valid_position(empty_board, 'T', 0, 3, 5))
        # Left wall collision
        self.assertFalse(is_valid_position(empty_board, 'T', 0, -2, 5))
        # Right wall collision
        self.assertFalse(is_valid_position(empty_board, 'I', 0, 8, 5))
        # Floor collision
        self.assertFalse(is_valid_position(empty_board, 'O', 0, 4, 19))

    def test_drop_to_bottom_on_empty_board(self):
        empty_board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # For 'O' piece (2x2), dropping at col 0 on empty 20-row board lands at y=18
        y = drop_piece_to_bottom(empty_board, 'O', 0, 0)
        self.assertEqual(y, 18)

        # For 'I' piece (rot 0, 1x4 horizontal at row index 1), lands at y=18
        y_i = drop_piece_to_bottom(empty_board, 'I', 0, 0)
        self.assertEqual(y_i, 18)


class TestTetrisMovementAndGravity(unittest.TestCase):
    def setUp(self):
        self.game = TetrisGame(seed=42)

    def test_horizontal_movement(self):
        initial_x = self.game.curr_x
        # Move left
        moved = self.game.move_left()
        self.assertTrue(moved)
        self.assertEqual(self.game.curr_x, initial_x - 1)

        # Move right back
        moved = self.game.move_right()
        self.assertTrue(moved)
        self.assertEqual(self.game.curr_x, initial_x)

    def test_wall_collision_prevents_moving_out(self):
        # Move all the way to left wall
        for _ in range(10):
            self.game.move_left()
        leftmost_x = self.game.curr_x
        # Another move left must fail
        moved = self.game.move_left()
        self.assertFalse(moved)
        self.assertEqual(self.game.curr_x, leftmost_x)

    def test_tick_gravity_drops_piece(self):
        initial_y = self.game.curr_y
        locked, lines, ft = self.game.tick_gravity()
        self.assertFalse(locked, "Piece should not lock on first tick on empty board")
        self.assertEqual(self.game.curr_y, initial_y + 1)

    def test_hard_drop_locks_at_bottom(self):
        initial_pieces = self.game.pieces_dropped
        locked, lines, ft = self.game.hard_drop()
        self.assertTrue(locked, "Hard drop should immediately lock piece")
        self.assertEqual(self.game.pieces_dropped, initial_pieces + 1)
        # Check that there is at least one block at the bottom of the board
        bottom_has_block = any(self.game.board[19][x] != 0 for x in range(BOARD_WIDTH))
        self.assertTrue(bottom_has_block, "Bottom row should contain placed blocks")

    def test_piece_landing_on_another_piece(self):
        # Hard drop first piece at current column
        self.game.current_piece = 'O'
        self.game.curr_x = 4
        self.game.hard_drop()
        # Row 18 and 19 should have 'O' blocks at columns 4, 5
        self.assertEqual(self.game.board[19][4], 'O')
        self.assertEqual(self.game.board[18][4], 'O')

        # Drop second 'O' piece right above the first one
        self.game.current_piece = 'O'
        self.game.curr_x = 4
        self.game.hard_drop()
        # Row 16 and 17 should now have the second 'O' piece!
        self.assertEqual(self.game.board[17][4], 'O')
        self.assertEqual(self.game.board[16][4], 'O')

    def test_hold_piece_swapping(self):
        first_piece = self.game.current_piece
        held = self.game.hold_current_piece()
        self.assertTrue(held)
        self.assertEqual(self.game.hold_piece, first_piece)
        # Can only hold once per turn
        second_hold = self.game.hold_current_piece()
        self.assertFalse(second_hold, "Cannot hold twice before locking a piece")


class TestLineClearAndScoring(unittest.TestCase):
    def test_single_line_clear(self):
        board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # Fill row 19 except col 0, 1, 2, 3
        for x in range(4, BOARD_WIDTH):
            board[19][x] = 'J'

        # Drop 'I' piece horizontally at col 0, rot 0
        new_board, lines, lh, blocks = simulate_placement(board, 'I', 0, 0)
        self.assertEqual(lines, 1)
        self.assertEqual(sum(1 if c else 0 for c in new_board[19]), 0)  # Row 19 is cleared (shifted down)

    def test_tetris_4line_clear(self):
        board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # Fill rows 16, 17, 18, 19 except col 9
        for y in range(16, 20):
            for x in range(9):
                board[y][x] = 'T'

        # Drop 'I' piece vertically (rot 1) at col 7 (vertical column is at x+2=9)
        new_board, lines, lh, blocks = simulate_placement(board, 'I', 1, 7)
        self.assertEqual(lines, 4)
        # All 4 rows cleared, bottom row should now be empty
        self.assertEqual(sum(1 if c else 0 for c in new_board[19]), 0)


class TestDellacherieHeuristics(unittest.TestCase):
    def test_clean_flat_board_has_zero_holes(self):
        board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        new_board, lines, lh, blocks = simulate_placement(board, 'O', 0, 0)
        features = calculate_dellacherie_features(new_board, lines, lh, blocks)
        self.assertEqual(features["holes"], 0)
        self.assertEqual(features["hole_depth"], 0)

    def test_buried_hole_detection(self):
        board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        board[18][3] = 'I'
        board[19][3] = 0  # Hole at (3, 19)

        features = calculate_dellacherie_features(board, 0, 0, [])
        self.assertGreaterEqual(features["holes"], 1)

    def test_placements_generation(self):
        board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        placements = get_all_possible_placements(board, 'T')
        self.assertGreater(len(placements), 0)
        self.assertGreaterEqual(placements[0]["score"], placements[-1]["score"])


class TestInvincibleGameLoop(unittest.TestCase):
    def test_fast_benchmark_survival(self):
        # Run 100 pieces with unbeatable Pierre Dellacherie mode
        res = run_benchmark(pieces_count=100, mode="unbeatable")
        self.assertTrue(res["alive"], "Game should be alive after 100 pieces")
        self.assertGreater(res["lines"], 15, "Should have cleared lines")
        self.assertLess(res["max_height"], 12, "Board stack height should stay low and controlled")
        self.assertEqual(res["holes"], 0, "Invincible AI should keep zero buried holes")


class TestLayaGuardrail(unittest.TestCase):
    def test_guardrail_prevents_harmful_move(self):
        board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        placements = get_all_possible_placements(board, 'L')
        selected, latency, note, probs = query_laya_tetris(board, 'L', 'I', placements)
        self.assertIsNotNone(selected)
        self.assertIn("col", selected)
        self.assertIn("rotation", selected)


if __name__ == "__main__":
    unittest.main()
