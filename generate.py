import os
import subprocess
import sys

def generate_test_set(puzzle_n: int, n: int) -> None:
    if not os.path.exists('data/puzzles'):
        os.makedirs('data/puzzles')
    if not os.path.exists('data/solutions'):
        os.makedirs('data/solutions')

    for puzzle_num in range(puzzle_n):
        puzzle_string = 'data/puzzles/kk' + str(puzzle_num) + '.txt'
        solution_string = 'data/solutions/kksol' + str(puzzle_num) + '.txt'
        subprocess.Popen(['./cdok/cdok', '-o', puzzle_string, 'generate'])
        subprocess.Popen(['./cdok/cdok', '-i', puzzle_string, '-o', solution_string, 'solve'])

if __name__ == "__main__":
    generate_test_set(int(sys.argv[1]), int(sys.argv[2]))
