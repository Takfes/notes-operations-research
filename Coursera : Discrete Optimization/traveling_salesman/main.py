from pathlib import Path

from functions import (
    DATA_DIR,
    calculate_distance_matrix,
    euclidean_distance,
    list_files_in_dir,
    parse_input_data,
    plot_tsp_trip,
)

"""
tsp_51_1     Traveling Salesman Problem 1
tsp_100_3    Traveling Salesman Problem 2
tsp_200_2    Traveling Salesman Problem 3
tsp_574_1    Traveling Salesman Problem 4
tsp_1889_1   Traveling Salesman Problem 5
tsp_33810_1  Traveling Salesman Problem 6
"""

# * DEFINE CONSTANTS
file_name = "tsp_5_1"

# * LOAD DATA
list_files_in_dir(full_path=False)
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")

n_locations = data["n_locations"]
locations = data["locations"]

dm = calculate_distance_matrix(locations)
dm.head()

plot_tsp_trip(locations, title=file_name)
