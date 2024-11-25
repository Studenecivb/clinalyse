from cmath import inf
import logging

import numpy as np
from decimal import Decimal

logging.basicConfig(level=logging.INFO)


class InputData:
    def __init__(self, input_data: np.array, names_of_loci: list, ploidy: np.array, geno_pos=0, test=None):
        self.input_data = input_data
        self.geno_pos = geno_pos
        self.names_of_loci = names_of_loci
        self.data_labelled_ready = None
        self.test = test
        self.ploidy = ploidy
        self.ploidy_ready = None

    # Loading the input data
    def load_data(self):
        self.data_labelled_ready = self.input_data[self.geno_pos:]
        self.ploidy_ready = self.ploidy[self.geno_pos:]
        logging.info('Data loaded successfully')


class FibGrid:
    def __init__(self, start: float, end: float, grain: float):
        self.fib = None
        self.grid = None
        self.start = start
        self.end = end
        self.grain = grain
        self.create_grid()
        self.create_fib()

    # Setting up Fibonacci search
    @staticmethod
    def _fibonacci_number(n: int):
        return np.round(((1 + 5**0.5) / 2) ** n / (5**0.5)).astype(int)

    def create_grid(self):
        if self.start < 0 < self.end:
            if Decimal(f'{self.end}') % Decimal(f'{self.grain}') == 0:
                left_interval = np.arange(self.start, 0, self.grain)
                right_interval = np.arange(0, self.end, self.grain)
                self.grid = np.concatenate((left_interval, right_interval))
            else:
                left_interval = np.arange(self.start, 0, self.grain)
                right_interval = np.arange(left_interval[-1]+self.grain, self.end, self.grain)
                self.grid = np.concatenate((left_interval, right_interval))
        else:
            self.grid = np.arange(self.start, self.end, self.grain)


    def create_fib(self):
        i = 2
        while FibGrid._fibonacci_number(i) < len(self.grid):
            i += 1
        self.fib = FibGrid._fibonacci_number(i - np.arange(0, 3, 1))


class _Fibmax:
    def __init__(self, list_of_fibgrids: list):
        self.list_of_fibgrids = list_of_fibgrids

    def fibmax(self, function_to_max, fix_axis=None):
        fibonacci_numbers = []
        values = []
        for axis in self.list_of_fibgrids:
            fibonacci_numbers.append(axis.fib)
            values.append(axis.grid)
        n = list(map(lambda x: len(x) - 1, values))
        number_of_axes = len(self.list_of_fibgrids)
        offset = np.full(number_of_axes, -1)
        evals = []  # will expand
        cpoint = np.full(
            number_of_axes, 0
        )  # here one corner - leave open to user choice of startpoint
        optimizing = np.full(number_of_axes, True)
        i2 = None
        if fix_axis is not None:
            optimizing[fix_axis[0]] = False
            cpoint[fix_axis[0]] = fix_axis[1]
        cpointvals = []
        axis = 0
        best = [-inf] * (len(self.list_of_fibgrids) + 1)

        for c in cpoint:
            cpointvals.append(values[axis][c])
            axis += 1

        def evaluate_n_store(i):
            cpoint[axis] = i
            cpointvals[axis] = values[axis][i]
            f = function_to_max(cpointvals)
            evals.append([cpoint.copy(), f])
            return f

        axis = 0
        while np.any(optimizing):
            while not optimizing[axis]:
                if axis == number_of_axes - 1:
                    axis = 0
                else:
                    axis += 1
            i1 = min(offset[axis] + fibonacci_numbers[axis][2], n[axis])
            eval1 = evaluate_n_store(i1)
            if eval1 > best[-1]:
                best = [cpoint, eval1]
            # fibshrink
            fibonacci_numbers[axis] = np.roll(fibonacci_numbers[axis], -1)
            fibonacci_numbers[axis][2] = (
                fibonacci_numbers[axis][0] - fibonacci_numbers[axis][1]
            )

            i2 = min(i1 + fibonacci_numbers[axis][2], n[axis])
            eval2 = evaluate_n_store(i2)
            if eval2 > best[-1]:
                best = [cpoint, eval2]
            if eval1 < eval2:
                offset[axis] = i1
            if fibonacci_numbers[axis][0] <= 1:
                optimizing[axis] = False
            if axis == number_of_axes - 1:
                axis = 0
            else:
                axis += 1

        if i2 == 1:
            evaluate_n_store(0)
        return evals, best
