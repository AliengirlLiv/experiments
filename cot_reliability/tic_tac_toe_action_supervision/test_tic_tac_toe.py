import unittest
from tic_tac_toe import *


class TicTacToeTests(unittest.TestCase):
    def test_action_str_to_idxs(self):
        self.assertEqual(action_str_to_idxs('a1'), (0, 0))
        self.assertEqual(action_str_to_idxs('a2'), (0, 1))
        self.assertEqual(action_str_to_idxs('a3'), (0, 2))

    def test_idxs_to_action_str(self):
        self.assertEqual(idxs_to_action_str((0, 0)), 'a1')
        self.assertEqual(idxs_to_action_str((0, 1)), 'a2')
        self.assertEqual(idxs_to_action_str((0, 2)), 'a3')

    def test_get_value(self):
        board = [[X, O, EMPTY], [EMPTY, X, O], [X, O, X]]
        self.assertEqual(get_value(board, (0, 0)), X)
        self.assertEqual(get_value(board, (1, 1)), X)
        self.assertEqual(get_value(board, (2, 2)), X)

    def test_is_open(self):
        board = [[X, O, EMPTY], [EMPTY, X, O], [X, O, X]]
        self.assertFalse(is_open(board, (0, 0)))
        self.assertTrue(is_open(board, (0, 2)))
        self.assertFalse(is_open(board, (1, 1)))


if __name__ == '__main__':
    unittest.main()