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
    file_path = DATA_DIR / Path(file_name)

    data = parse_input_data(file_path, input_type="file")

    data_id = calculate_data_footprint(data)

    data_ids[data_id] = file_name

    print(f"File Name: {file_name} | Data Id: {data_id=}")

assert idx == len(data_ids), f"Expected {len(data_files)} data files, got {idx}"

# Save data_ids dictionary to disk
save_footprints_to_disk(data_ids)
