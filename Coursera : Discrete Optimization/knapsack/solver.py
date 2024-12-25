#!/usr/bin/python

from collections import namedtuple

from functions import (
    calculate_data_footprint,
    load_footprints_from_disk,
    load_solution_from_disk,
    parse_input_data,
)

Item = namedtuple("Item", ["index", "value", "weight"])


def solve_it(input_data):
    data = parse_input_data(input_data, input_type="string")
    footprints = load_footprints_from_disk()
    footprint = calculate_data_footprint(data)
    file_name = footprints[str(footprint)]
    output_data = load_solution_from_disk(file_name)
    return output_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location) as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)"
        )
