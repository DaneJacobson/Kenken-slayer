import copy
import functools
import itertools
import random

import pandas as pd

CAGE_SIZES = [1, 2, 2, 2, 2, 2, 3, 3, 4, 4] #TODO: non-jank prob distributions, see division as well
OP1 = ['#']
OP2 = ['+', '*', '-', '-', '-', '-',  '-',  '-', '/', '/', '/', '/', '/', '/', '/', '/', '/'] 
OP2NODIV = ['+', '*', '-']
OPELSE = ['+', '*']

class KenKenPuzzle:

    def __init__(self, n):
        self._n = n
        self._idxs = list(range(self._n))
        self._nums = list(map(lambda i: i + 1, self._idxs))
        self._grid_rep = {}

        # Generate unique KenKen puzzle
        self._attempts = 1
        while True:
            print('Attempt: ' + str(self._attempts))
            self.genPuzzle(please_print=True)
            if self.uniquePuzzle(): break
            else: self._attempts += 1

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

    def uniquePuzzle(self):
        # construct initial possibilities, using each total and operator to eliminate outright
        opt_board = [[None for _ in self._idxs] for _ in self._idxs]
        for cage_info in self._grid_rep['dict_rep'].values():
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
                    cage_id = self._grid_rep['cages'][x][y]
                    op_type = self._grid_rep['ops'][x][y]
                    total = self._grid_rep['totals'][x][y]
                    cage_cells = self._grid_rep['dict_rep'][cage_id]['cells']
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

        # Generate Answers
        nums = [[(i + j) % n + 1 for i in idxs] for j in idxs]
        for _ in idxs: random.shuffle(nums)
        for i, j in itertools.permutations(idxs, 2):
            if random.random() > 0.5:
                for r in idxs:
                    nums[r][i], nums[r][j] = nums[r][j], nums[r][i]

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

        self._grid_rep['size'] = self._n
        self._grid_rep['nums'] = nums
        self._grid_rep['cages'] = cages
        self._grid_rep['ops'] = operators
        self._grid_rep['totals'] = totals
        self._grid_rep['dict_rep'] = data

        return self._grid_rep

puzzle = KenKenPuzzle(5)
print('Number of attempts: ' + str(puzzle._attempts))
print('done')