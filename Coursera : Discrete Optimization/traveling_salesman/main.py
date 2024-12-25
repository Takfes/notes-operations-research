import time
from pathlib import Path

from functions import (
    DATA_DIR,
    calculate_distance_matrix,
    calculate_sequence_length_from_distance_matrix,
    generate_dummy_output,
    generate_output,
    parse_input_data,
    plot_tsp_trip,
    tsp_greedy_subtrip_insertion,
    tsp_heuristic_wrapper,
    tsp_nearest_point_insertion,
    tsp_or_constraint_solver,
    tsp_two_opt_improvement,
    write_solution,
)

"""
tsp_51_1     Traveling Salesman Problem 1 -> 428.87
tsp_100_3    Traveling Salesman Problem 2 -> 20750.76
tsp_200_2    Traveling Salesman Problem 3 -> 29448.20
tsp_574_1    Traveling Salesman Problem 4 -> 36943.51
tsp_1889_1   Traveling Salesman Problem 5 -> 320220.18
tsp_33810_1  Traveling Salesman Problem 6 -> 67619859.63
"""


# * DEFINE CONSTANTS
RUN_OR_SOLVER = False
OR_CONSTRAINT_SOLVER_STRATEGY = "guided"  # "first"
OR_CONSTRAINT_SOLVER_LOGGING = False
OR_CONSTRAINT_SOLVER_TIME_LIMIT = 60 * 3
RUN_HEURISTIC = True  # ! tsp_33810_1 run heuristic, not the OR Solver
HEURISTIC_START_POINT_OPTIMIZATION = False
HEURISTIC_HILL_CLIMB = False
HEURISTIC_CONSTRUCTIVE_FUNCTION = tsp_nearest_point_insertion
HEURISTIC_ENABLE_TWO_OPT_IMPROVEMENT = False
HEURISTIC_ENABLE_SWAP_IMPROVEMENT = False
file_name = "tsp_33810_1"
# Initialize variables to prevent errors if not defined
seqhr_length = None
seqor_length = None


# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_locations = data["n_locations"]
locations = data["locations"]
dm = calculate_distance_matrix(locations)


# * TSP OR TOOLS SOLVER
if RUN_OR_SOLVER:
    or_start = time.perf_counter()
    print("Running OR Solver...")
    seqor = tsp_or_constraint_solver(
        dm,
        search_strategy=OR_CONSTRAINT_SOLVER_STRATEGY,
        time_limit=OR_CONSTRAINT_SOLVER_TIME_LIMIT,
        logging=OR_CONSTRAINT_SOLVER_LOGGING,
    )
    or_end = time.perf_counter()
    seqor_length = calculate_sequence_length_from_distance_matrix(
        dm, seqor
    ).item()
    print(f"OR Solver took {or_end - or_start:.2f} seconds")
    print(f"OR Solver Solution: {seqor_length:.2f}")
    if len(seqor) < 1000:
        plot_tsp_trip(locations, sequence=seqor, title=file_name)
    else:
        print("Skipping plot due to large number of locations")


# * TSP CUSTOM HEURISTICS
if RUN_HEURISTIC:
    hr_start = time.perf_counter()
    print("Running Heuristic Solver...")
    seqhr = tsp_heuristic_wrapper(
        dm,
        start_point_optimization=HEURISTIC_START_POINT_OPTIMIZATION,
        hill_climb=HEURISTIC_HILL_CLIMB,
        constructive_heuristic=HEURISTIC_CONSTRUCTIVE_FUNCTION,
        enable_two_opt_improvement=HEURISTIC_ENABLE_TWO_OPT_IMPROVEMENT,
        enable_swap_improvement=HEURISTIC_ENABLE_SWAP_IMPROVEMENT,
    )
    hr_end = time.perf_counter()
    seqhr_length = calculate_sequence_length_from_distance_matrix(
        dm, seqhr
    ).item()
    print(f"Heuristic Solver took {hr_end - hr_start:.2f} seconds")
    print(f"Heuristic Solver Solution: {seqhr_length:.2f}")
    if len(seqhr) < 1000:
        plot_tsp_trip(locations, sequence=seqhr, title=file_name)
    else:
        print("Skipping plot due to large number of locations")

# * COMPARE RESULTS AND COMPILE FINAL ANSWER
if seqhr_length and seqor_length:
    sequence = seqhr if seqhr_length < seqor_length else seqor
    print(
        f"Best solution: {'Heuristic' if seqhr_length < seqor_length else 'OR Solver'}"
    )
elif seqhr_length is not None:
    sequence = seqhr
elif seqor_length is not None:
    sequence = seqor
else:
    raise ValueError("No solution found")


# * WRITE SOLUTION
assert len(sequence[:-1]) == n_locations, "Mismatch in sequence length"
output_data = generate_output(sequence, dm)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)
