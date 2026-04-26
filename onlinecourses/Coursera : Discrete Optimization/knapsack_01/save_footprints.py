from pathlib import Path

from functions import (
    DATA_DIR,
    calculate_data_footprint,
    list_files_in_dir,
    parse_input_data,
    save_footprints_to_disk,
)

# Get all data files
data_files = list_files_in_dir(full_path=False)
data_ids = {}

for idx, file_name in enumerate(data_files, start=1):
    # print(idx, file_name)
    input_data = DATA_DIR / Path(file_name)
    data = parse_input_data(input_data, input_type="file")
    data_id = calculate_data_footprint(data)
    if data_id in data_ids:
        print(
            f"Duplicate data id found between {data_ids[data_id]} and {file_name}"  # tsp_318_1 and tsp_318_2
        )
        continue
    data_ids[data_id] = file_name
    print(f"File Name: {file_name} | Data Id: {data_id=}")

assert idx == len(data_ids), f"Expected {len(data_files)} data files, got {idx}"

# Save data_ids dictionary to disk
save_footprints_to_disk(data_ids)
