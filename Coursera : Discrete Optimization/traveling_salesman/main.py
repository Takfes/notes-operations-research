from pathlib import Path

from functions import (
    DATA_DIR,
    calculate_distance_matrix,
    calculate_trip_sequence_from_distance_matrix,
    calculate_trip_sequence_from_locations,
    generate_output,
    hill_climb_corrective,
    list_files_in_dir,
    parse_input_data,
    plot_tsp_trip,
    start_point_optimizer,
    tsp_greedy_subtrip_insertion,
    tsp_nearest_point_insertion,
    tsp_or_constraint_solver,
    tsp_swap_improvement,
    tsp_two_opt_improvement,
)
from matplotlib.pyplot import plot

"""
tsp_51_1     Traveling Salesman Problem 1
tsp_100_3    Traveling Salesman Problem 2
tsp_200_2    Traveling Salesman Problem 3
tsp_574_1    Traveling Salesman Problem 4
tsp_1889_1   Traveling Salesman Problem 5
tsp_33810_1  Traveling Salesman Problem 6
"""

# * DEFINE CONSTANTS
START_POINT_OPTIMIZATION = False
file_name = "tsp_200_2"

# * LOAD DATA
list_files_in_dir(full_path=False)
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
locations = data["locations"]
dm = calculate_distance_matrix(locations)

# * TSP OR TOOLS SOLVER
seqor = tsp_or_constraint_solver(dm, "guided", time_limit=30, logging=True)
calculate_trip_sequence_from_distance_matrix(dm, seqor)
plot_tsp_trip(locations, sequence=seqor, title=file_name)
output_data = generate_output(seqor, dm)
print(output_data)

# * TSP CUSTOM HEURISTICS
if START_POINT_OPTIMIZATION:
    seq_spo, _, _ = start_point_optimizer(dm, tsp_greedy_subtrip_insertion)
else:
    seq_spo = tsp_greedy_subtrip_insertion(dm)

seq_spo_ht = hill_climb_corrective(dm, seq_spo, tsp_two_opt_improvement)
seq_spo_ht_hs = hill_climb_corrective(dm, seq_spo_ht, tsp_swap_improvement)

calculate_trip_sequence_from_distance_matrix(dm, seq_spo_ht_hs)
plot_tsp_trip(locations, sequence=seq_spo_ht_hs, title=file_name)
output_data = generate_output(seq_spo_ht_hs, dm)
print(output_data)
