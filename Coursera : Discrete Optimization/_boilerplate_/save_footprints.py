from pathlib import Path

from functions import (
    DATA_DIR,
    calculate_data_footprint,
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
    footprint = calculate_data_footprint(data)
    if footprint in footprints:
        print(
            f"Duplicate data id found between {footprints[footprint]} and {file_name}"  # tsp_318_1 and tsp_318_2
        )
        continue
    footprints[footprint] = file_name
    print(f"File Name: {file_name} | Data Id: {footprint=}")

assert idx == len(footprints), (
    f"Expected {len(datafiles)} data files, got {idx}"
)

# Save footprints dictionary to disk
save_footprints_to_disk(footprints)
