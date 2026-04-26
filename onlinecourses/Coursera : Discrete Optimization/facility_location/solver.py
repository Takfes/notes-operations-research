#!/usr/bin/python

import math
from collections import namedtuple

from functions import (
    calculate_data_footprint,
    load_footprints_from_disk,
    load_solution_from_disk,
    parse_input_data,
)

Point = namedtuple("Point", ["x", "y"])
Facility = namedtuple(
    "Facility", ["index", "setup_cost", "capacity", "location"]
)
Customer = namedtuple("Customer", ["index", "demand", "location"])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_it(input_data):
    data = parse_input_data(input_data, input_type="string")
    footprints = load_footprints_from_disk()
    footprint = calculate_data_footprint(data)
    file_name = footprints[str(footprint)]
    output_data = load_solution_from_disk(file_name)
    return output_data


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
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)"
        )
