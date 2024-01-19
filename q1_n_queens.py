import numpy as np
import random
import matplotlib.pyplot as plt


class NQueens:

    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.col = set()
        self.diag = set()
        self.anti_diag = set()

    def is_safe(self, r, c):
        return not (c in self.col or
                    (r + c) in self.diag or
                    (r - c) in self.anti_diag)

    def place_queen(self, r, c):
        self.col.add(c)
        self.diag.add(r + c)
        self.anti_diag.add(r - c)
        self.board[r, c] = 1

    def remove_queen(self, r, c):
        self.col.remove(c)
        self.diag.remove(r + c)
        self.anti_diag.remove(r - c)
        self.board[r, c] = 0

    def backtrack(self, r=0):
        if r == self.n:
            return self.board

        for c in range(self.n):
            if self.is_safe(r, c):
                self.place_queen(r, c)

                res = self.backtrack(r + 1)
                if not res is None:
                    return res

                self.remove_queen(r, c)

        return None

    def las_vegas_n_queens(self, limit=10):
        total_runs = 0

        while total_runs < limit:
            total_runs += 1

            for r in range(self.n):
                available_cols = [c for c in range(
                    self.n) if self.is_safe(r, c)]
                if len(available_cols) == 0:
                    self.reset()
                    break
                self.place_queen(r, random.choice(available_cols))
            else:
                return self.board

        return None

    def solve(self, method):
        if method == "b":
            self.board = self.backtrack()
        elif method == "l":
            self.board = self.las_vegas_n_queens()

        return self.board

    def draw_board(self):
        if self.board is None:
            print("No solution found")
            return

        _, ax = plt.subplots()
        text_size = max(10, 250 // self.n)

        for (i, j), value in np.ndenumerate(self.board):
            color = '#EEEED2' if (i + j) % 2 == 0 else '#769656'
            rect = plt.Rectangle([j, i], 1, 1, facecolor=color)
            ax.add_patch(rect)

            if value == 1:
                plt.text(j + 0.5, i + 0.5, '♛', size=text_size, color='black',
                         ha='center', va='center')

        plt.xlim(0, self.n)
        plt.ylim(0, self.n)
        plt.show()

    def test_eight_queens_las_vegas(self, runs=10000):
        self.n = 8
        self.reset()
        success = sum(
            [1 for _ in range(runs) if not self.las_vegas_n_queens() is None])

        print(f"Success rate: {success/runs}")


n = int(input("""
Enter n: 
- n represents the number of queens and the size of the n x n board.
"""))

method = input("""
Enter the solution method (b/l):
- b: Backtracking.
- l: Las Vegas
""").lower()

board = None
n_queens = NQueens(n)

if method == "b" or method == "l":
    board = n_queens.solve(method)
    print("Board:", board)
    n_queens.draw_board()
else:
    print("Invalid solution method")

n_queens.test_eight_queens_las_vegas()
