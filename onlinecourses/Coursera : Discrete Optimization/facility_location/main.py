import time
from pathlib import Path

from functions import (
    DATA_DIR,
    build_model,
    calculate_data_footprint,
    generate_output,
    load_footprints_from_disk,
    parse_input_data,
    write_solution,
)
from pyomo.opt import SolverFactory

"""
# ==============================================================
# MAIN APPLICATION
# ==============================================================
"""

# input_data = """3 4
# 100 100 1065.0 1065.0
# 100 100 1062.0 1062.0
# 100 500 0.0 0.0
# 50 1397.0 1397.0
# 50 1398.0 1398.0
# 75 1399.0 1399.0
# 75 586.0 586.0
# """

"""
"fl_25_2"   # Facility Location Problem 1 # 0.0361 seconds | optimality gap: 0.00%
"fl_50_6"   # Facility Location Problem 2 # 0.3735 seconds | optimality gap: 0.01%
"fl_100_7"  # Facility Location Problem 3 # 0.3905 seconds | optimality gap: 0.00%
"fl_100_1"  # Facility Location Problem 4 # 1803.6453 seconds | optimality gap: 8.00%
                                          # 2192.1589 seconds | optimality gap: 7.51%
"fl_200_7"  # Facility Location Problem 5 # 252.3118 seconds | optimality gap: 0.01%
"fl_500_7"  # Facility Location Problem 6 # 1276.3987 seconds | optimality gap: 5.28%
                                          # 2525.2982 seconds | optimality gap: 5.27%
"fl_1000_2" # Facility Location Problem 7 # 1860.0783 seconds | optimality gap: 2.07%
"fl_2000_2" # Facility Location Problem 8 # 2029.6632 seconds | optimality gap: 86.39%
"""

# * DEFINE CONSTANTS
EXPLORATION_ENABLED = False
STORE_SOLUTION = True
TIME_LIMIT = 60 * 30  # None
SOLVER_TEE = False
SOLVER = "appsi_highs"  # "glpk" "ipopt" "appsi_highs" "cbc"
file_name = "fl_2000_2"

# * READ DATA
footprints = load_footprints_from_disk()
input_data = Path(Path(DATA_DIR) / Path(file_name))
data = parse_input_data(input_data, input_type="file")
# data = parse_input_data(input_data, input_type="string")

print(f"File Name: {file_name}")
print(f"Problem Size : {data["n_facilities"]=}, {data["n_customers"]=}")
print(f"total={data['n_facilities'] * data['n_customers']:,}")

footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Data Footprint : {footprint}")

# * SETUP OPTIMIZATION
model = build_model(data)
solver = SolverFactory(SOLVER)
if TIME_LIMIT is not None:
    solver.options["time_limit"] = TIME_LIMIT

print(f"Solving model with {SOLVER=}...")
start = time.perf_counter()
results = solver.solve(model, tee=SOLVER_TEE)
end = time.perf_counter()
print(f"Execution time : {end - start:.4f} seconds")

# * OUTPUT RESULTS
# solver report
solver_status = results.solver.status.value
solver_termination_condition = results.solver.termination_condition.value
lower_bound = results.problem.lower_bound
upper_bound = results.problem.upper_bound
optimality_gap = 1 - (lower_bound / upper_bound)

print("> Solver Report:")
print(f"* Solver name : {SOLVER}")
print(f"* Solver execution time : {end - start:.4f} seconds")
print(f"* Solver status: {solver_status}")
print(f"* Solver termination condition: {solver_termination_condition}")
print(f"* Solver optimality gap: {optimality_gap:.2%}")

# output data
output_data = generate_output(model, results)
print("> Solution:")
print(output_data)

# * STORE SOLUTION
if STORE_SOLUTION:
    print("Storing solution to disk...")
    write_solution(contents=output_data, file_name=file_name)
