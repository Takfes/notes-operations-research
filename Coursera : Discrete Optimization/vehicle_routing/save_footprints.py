from pathlib import Path

import pandas as pd
from functions import (
    DATA_DIR,
    footprint_function,
    list_files_in_dir,
    parse_input_data,
    save_footprints_to_disk,
)

# Get all data files
datafiles = list_files_in_dir(full_path=False)
footprints = {}

for idx, file_name in enumerate(datafiles, start=1):
    input_data = DATA_DIR / Path(file_name)
    data = parse_input_data(input_data, input_type="file")
    footprint = footprint_function(data)
    if footprint in footprints:
        print(
            f"Duplicate data id found between {footprints[footprint]} and {file_name}"
        )
        continue
    footprints[footprint] = file_name
    # print(f"File Name: {file_name} | Data Id: {footprint=}")

assert idx == len(footprints), (
    f"Expected {len(datafiles)} data files, got {len(footprints)}"
)

# Save footprints dictionary to disk
save_footprints_to_disk(footprints)
