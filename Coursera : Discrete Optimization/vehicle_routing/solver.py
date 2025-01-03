#!/usr/bin/python

import math
from collections import namedtuple

from functions import fake_solver

Customer = namedtuple("Customer", ["index", "demand", "x", "y"])


def length(customer1, customer2):
    return math.sqrt(
        (customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2
    )


def solve_it(input_data):
    return fake_solver(input_data)


import sys

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location) as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)"
        )
