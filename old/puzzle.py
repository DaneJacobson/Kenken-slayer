import copy
import functools
import itertools
import random

import numpy as np
import pandas as pd

CAGE_SIZES = [1, 2, 2, 2, 2, 2, 3, 3, 4, 4] #TODO: non-jank prob distributions, see division as well
OP1 = ['#']
OP2 = ['+', '*', '-', '-', '-', '-',  '-',  '-', '/', '/', '/', '/', '/', '/', '/', '/', '/'] 
OP2NODIV = ['+', '*', '-']
OPELSE = ['+', '*']

class KenKen:

    def __init__(self, n, please_print: bool=False):
        self._n = n
        self._please_print = please_print
        self._idxs = list(range(self._n))
        self._nums = list(map(lambda i: i + 1, self._idxs))
        self._answers = None
        self._answers_machine = None
        self._cages = None
        self._operators = None
        self._totals = None
        self._dict_rep = {}

        # Generate unique KenKen puzzle
        self._attempts = 1
        while True:
            if self._please_print: print('Attempt: ' + str(self._attempts))
            self.genPuzzle(please_print=self._please_print)
            if self.uniquePuzzle(): break
            else: self._attempts += 1
        self.convertOps()
        self.convertAns()

    def convertOps(self):
        new_operators = np.zeros((self._n, self._n), dtype=int)
        for x in self._idxs:
            for y in self._idxs:
                if self._operators[x][y] == '#': new_operators[x][y] = 0
                elif self._operators[x][y] == '+': new_operators[x][y] = 1
                elif self._operators[x][y] == '*': new_operators[x][y] = 2
                elif self._operators[x][y] == '-': new_operators[x][y] = 3
                elif self._operators[x][y] == '/': new_operators[x][y] = 4
        self._operators = new_operators

    def convertAns(self):
        self._answers_machine = np.copy(self._answers)
        for x in self._idxs:
            for y in self._idxs:
                self._answers_machine[x][y] -= 1

    def _add_adj_cells(self, cell, unmarked, adj_set):
        x, y = cell[0], cell[1]
        up, down, left, right = (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)
        if up in unmarked: adj_set.add(up)
        if down in unmarked: adj_set.add(down)
        if left in unmarked: adj_set.add(left)
        if right in unmarked: adj_set.add(right)
        adj_set.discard(cell)
        unmarked.discard(cell)

    def _operate(self, op, vals):
        if op == '#':
            return vals[0]
        elif op == '+':
            return functools.reduce(lambda a, b: a + b, vals)
        elif op == '-':
            return functools.reduce(lambda a, b: a - b, vals)
        elif op == '*':
            return functools.reduce(lambda a, b: a * b, vals)
        elif op == '/':
            return int(functools.reduce(lambda a, b: a / b, vals))
        else:
            print('%s is not an operator' % (op))

    # def dlxUniquePuzzle(self): #DLX Method
    #     # construct DLX matrix
    #     dlx_matrix = np.zeros(shape=(self._n ** 3, 3 * self._n + len(self._dict_rep))) # RxCy#z, Row-Col+Row-Num+Col-Num+Cage-Num

    #     # Setting Constraints
    #     for x in self._idxs:
    #         for y in self._idxs:
    #             for z in self._nums:
    #                 dlx_matrix[(x * (self._n ** 2)) + (y * self._n) + z][(x * self._n) + y] # Row-Col Constraints
    #                 dlx_matrix[(x * (self._n ** 2)) + (y * self._n) + z][(self._n ** 2) + (self._n * x) + z] # Row-Num Constraints
    #                 dlx_matrix[(x * (self._n ** 2)) + (y * self._n) + z][(2 * (self._n ** 2)) + (self._n * y) + z] # Col-Num Constraints
    #                 # Cage Constraints

    #     opt_board = [[None for _ in self._idxs] for _ in self._idxs]
    #     for cage_info in self._dict_rep.values():
    #         cells, op_type, total = cage_info['cells'], cage_info['op_type'], cage_info['total']
    #         options = set()

    #         if op_type == '#':
    #             options.add(total)
    #         elif op_type == '+':
    #             for i in self._nums:
    #                 if i < total:
    #                     options.add(i)
    #         elif op_type == '-':
    #             for i, j in itertools.permutations(self._nums, 2): # doesn't check (1,1), e.g.
    #                 big, small = (i, j) if i > j else (j, i)
    #                 if big - small == total:
    #                     options.add(big)
    #                     options.add(small)
    #         elif op_type == '*':
    #             for i in self._nums:
    #                 if total % i == 0:
    #                     options.add(i)
    #         elif op_type == '/':
    #             for i, j in itertools.permutations(self._nums, 2): # doesn't check (1,1), e.g.
    #                 big, small = (i, j) if i > j else (j, i)
    #                 if big / small == total:
    #                     options.add(big)
    #                     options.add(small)

    #         for x, y in cells:
    #             opt_board[x][y] = copy.copy(options)      

    #     # run exhaustive DLX, if 2 answers found, return None
    #     # reconstruct final answer from unique answer

    def uniquePuzzle(self): # TODO: replace with DLX
        # construct initial possibilities, using each total and operator to eliminate outright
        opt_board = [[None for _ in self._idxs] for _ in self._idxs]
        for cage_info in self._dict_rep.values():
            cells, op_type, total = cage_info['cells'], cage_info['op_type'], cage_info['total']
            options = set()

            if op_type == '#':
                options.add(total)
            elif op_type == '+':
                for i in self._nums:
                    if i < total:
                        options.add(i)
            elif op_type == '-':
                for i, j in itertools.permutations(self._nums, 2): # doesn't check (1,1), e.g.
                    big, small = (i, j) if i > j else (j, i)
                    if big - small == total:
                        options.add(big)
                        options.add(small)
            elif op_type == '*':
                for i in self._nums:
                    if total % i == 0:
                        options.add(i)
            elif op_type == '/':
                for i, j in itertools.permutations(self._nums, 2): # doesn't check (1,1), e.g.
                    big, small = (i, j) if i > j else (j, i)
                    if big / small == total:
                        options.add(big)
                        options.add(small)

            for x, y in cells:
                opt_board[x][y] = copy.copy(options)

        # use board and cage_info to backtrack, checking rows and cols
        cells = [(0, 0)]
        options = [opt_board]
        solutions = []
        solution_found = False
        while len(options) != 0:
            new_cells = []
            new_options = []
            for (x, y), option in list(zip(cells, options)):
                new_cell = (
                    x if y < self._n - 1 else x + 1,
                    y + 1 if y < self._n - 1 else 0
                )
                for val in list(option[x][y]):
                    new_option = copy.deepcopy(option)
                    new_option[x][y] = val

                    # if the cage is completed, the total needs to be correct
                    cage_id = self._cages[x][y]
                    op_type = self._operators[x][y]
                    total = self._totals[x][y]
                    cage_cells = self._dict_rep[cage_id]['cells']
                    complete_cage = True
                    nums = []
                    for i, j in cage_cells:
                        if isinstance(new_option[i][j], set): complete_cage = False
                        else: nums.append(new_option[i][j])
                    if complete_cage:
                        if self._operate(op_type, sorted(nums, reverse=True)) != total: continue
                        
                    # eliminate row and column clashes
                    complete_sol = True
                    empty_set = False

                    for i in self._idxs:
                        if isinstance(new_option[i][y], set): 
                            complete_sol = False
                            new_option[i][y].discard(val)
                            if len(new_option[i][y]) == 0: empty_set = True
                    for j in self._idxs:
                        if isinstance(new_option[x][j], set): 
                            complete_sol = False
                            new_option[x][j].discard(val)
                            if len(new_option[x][j]) == 0: empty_set = True
                    
                    if empty_set: # if an empty set was created, try next value
                        continue
                    elif complete_sol: # if solution, add to set!s
                        if solution_found: return False
                        else: solution_found = True
                        tuplelized_sol = tuple(tuple(row) for row in new_option)
                        solutions.append(tuplelized_sol)
                    else:
                        new_cells.append(new_cell)
                        new_options.append(new_option)
            cells = new_cells
            options = new_options
        return True

    def genPuzzle(self, please_print):
        n = self._n
        idxs = self._idxs

        # Generate Answers TODO: doesn't find all possible Latin Squares
        nums = [[(i + j) % n + 1 for i in idxs] for j in idxs]
        for _ in idxs: random.shuffle(nums)
        for i, j in itertools.permutations(idxs, 2):
            if random.random() > 0.5:
                for r in idxs:
                    nums[r][i], nums[r][j] = nums[r][j], nums[r][i]
        nums = np.array(nums)

        # Generate Cages, Operators, Totals
        cage_idx, data = 0, {}
        cages = [[None for _ in idxs] for _ in idxs]
        operators = [[None for _ in idxs] for _ in idxs]
        totals = [[None for _ in idxs] for _ in idxs]
        unmarked = set([(i, j) for i in idxs for j in idxs])

        while len(unmarked) > 0:

            cage_size = random.sample(CAGE_SIZES, k=1)[0]
            adj_set = set(random.sample(list(unmarked), k=1))

            for _ in range(cage_size):
                if len(adj_set) == 0: break
                cell = random.sample(list(adj_set), k=1)[0]
                cages[cell[0]][cell[1]] = cage_idx
                if cage_idx not in data:
                    data[cage_idx] = {'cells': [cell]}
                else:
                    data[cage_idx]['cells'].append(cell)
                self._add_adj_cells(cell, unmarked, adj_set)

            info = data[cage_idx]
            vals = list(map(lambda c: nums[c[0]][c[1]], info['cells']))
            if len(vals) == 1:
                op_type = OP1[0]
                total = self._operate(op=op_type, vals=vals)
            elif len(vals) == 2:
                op_type = random.sample(OP2, k=1)[0]
                big, small = vals if vals[0] > vals[1] else vals[::-1]
                if not (op_type == '/' and big % small == 0):
                    op_type = random.sample(OP2NODIV, k=1)[0]
                total = self._operate(op=op_type, vals=[big, small])
            else:
                op_type = random.sample(OPELSE, k=1)[0]
                total = self._operate(op=op_type, vals=vals)

            info['op_type'] = op_type
            info['total'] = total
            for c in info['cells']: 
                operators[c[0]][c[1]] = op_type
                totals[c[0]][c[1]] = total

            cage_idx += 1

        if please_print:
            print("NUMS")
            print(pd.DataFrame(nums), '\n')
            print("CAGES")
            print(pd.DataFrame(cages), '\n')
            print("OPERATORS")
            print(pd.DataFrame(operators), '\n')
            print("TOTALS")
            print(pd.DataFrame(totals), '\n')

        self._answers = np.array(nums, dtype=int)
        self._cages = np.array(cages, dtype=int)
        self._operators = np.array(operators, dtype=str)
        self._totals = np.array(totals, dtype=int)
        self._dict_rep = data