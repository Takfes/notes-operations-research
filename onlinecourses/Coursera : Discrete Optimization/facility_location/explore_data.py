import matplotlib.pyplot as plt
import pandas as pd
from functions import (
    DATA_DIR,
    calculate_distance_matrix,
    parse_input_data,
    plot_facilities_and_customers,
)

# * READ DATA
file_name = "fl_500_7"
file_path = DATA_DIR / file_name
data = parse_input_data(file_path, input_type="file")
n_facilities = data["n_facilities"]
n_customers = data["n_customers"]
facilities = data["facilities"]
customers = data["customers"]

# * EXPLORE DATA

# Explore facility costs
facility_cost_coefficient_of_variation = (
    facilities.cost.std() / facilities.cost.mean()
).item()
print(
    f"Facility cost coefficient of variation: {facility_cost_coefficient_of_variation:.4f}"
)
facilities.cost.hist()
plt.show()

facilities.cost.describe()
facilities.capacity.describe()

# Explore customer demand
customer_demand_coefficient_of_variation = (
    customers.demand.std() / customers.demand.mean()
).item()
print(
    f"Customer demand coefficient of variation: {customer_demand_coefficient_of_variation:.4f}"
)
customers.demand.hist()
plt.show()

customers.demand.describe()
customers.demand.sum()

# Explore geographical distribution of facilities and customers
plot_facilities_and_customers(facilities, customers)

# Explore distances
dm_wide = calculate_distance_matrix(
    facilities, customers
)  # facilities x customers

dm_wide.sum(axis=1).sort_values(ascending=True)

dm = (
    dm_wide.T.reset_index()
    .melt(id_vars="index", var_name="facility", value_name="distance")
    .rename(columns={"index": "customer"})
)

distance_threshold = dm["distance"].mean()
distance_threshold = dm.groupby("customer")["distance"].describe()["25%"].max()

filtered_distances = (
    dm[dm.distance <= distance_threshold]
    .groupby("customer")["distance"]
    .count()
    .sort_values(ascending=True)
)
print(filtered_distances)

# Calculate the average of the X lowest distance values per customer
x_closest_facilities = int(n_facilities / 2)
x_closest_facilities = 25
average_lowest_distances = (
    dm.groupby("customer")["distance"]
    .apply(lambda x: x.nsmallest(x_closest_facilities).min())
    .reset_index(name="avg_5_lowest_distances")
    .sort_values("avg_5_lowest_distances", ascending=False)
)

print(average_lowest_distances)
