import json
import os
from pathlib import Path


def list_files_in_dir(dir_path, full_path=True):
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    if not full_path:
        files = [os.path.basename(file) for file in files]
    return files


def parse_input_data(input_data, input_type="string"):
    """Parse a facility location problem from either a file or directly from a string."""

    if input_type == "string":
        lines = input_data.split("\n")[:-1]
    elif input_type == "file":
        with open(input_data) as file:
            lines = file.readlines()
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Parse problem parameters
    n_facilities, n_customers = map(int, lines[0].split())

    # Parse facilities data
    facilities_columns = ["cost", "capacity", "fx", "fy"]
    facilities_lines = lines[1 : n_facilities + 1]
    assert len(facilities_lines) == n_facilities, (
        f"Expected {n_facilities} facilities, got {len(facilities_lines)}"
    )
    facilities = pd.DataFrame(
        [x.split() for x in facilities_lines],
        columns=facilities_columns,
        dtype=float,
    )

    # Parse customers data
    customerers_columns = ["demand", "cx", "cy"]
    customers_lines = lines[n_facilities + 1 :]
    assert len(customers_lines) == n_customers, (
        f"Expected {n_customers} customers, got {len(customers_lines)}"
    )
    customers = pd.DataFrame(
        [x.split() for x in customers_lines],
        columns=customerers_columns,
        dtype=float,
    )

    return {
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        "facilities": facilities,
        "customers": customers,
    }


def calculate_data_footprint(data):
    data_id = (
        data["n_facilities"]
        + data["n_customers"]
        + data["facilities"].sum().sum().item()
        + data["customers"].sum().sum().item()
    )
    return data_id


def save_dict_to_disk(dictionary, file_path):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


DATA_DIR = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data"

# Get all data files
data_files = list_files_in_dir(DATA_DIR, full_path=False)
data_ids = {}

for idx, file_name in enumerate(data_files, start=1):
    file_path = Path(Path(DATA_DIR) / Path(file_name))

    data = parse_input_data(file_path, input_type="file")

    data_id = calculate_data_footprint(data)

    data_ids[data_id] = file_name

    print(f"File Name: {file_name} | Data Id: {data_id=}")

assert idx == len(data_ids), f"Expected {len(data_files)} data files, got {idx}"

# Save data_ids dictionary to disk
save_dict_to_disk(data_ids, "footprints.json")
