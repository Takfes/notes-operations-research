# https://www.gurobi.com/jupyter_models/vehicle-rental-optimization/
from itertools import product

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def generate_data():
    data = {}
    # demand
    demand = {
        "Glasgow": [100, 150, 135, 83, 120, 230],
        "Manchester": [250, 143, 80, 225, 210, 98],
        "Birmingham": [95, 195, 242, 111, 70, 124],
        "Plymouth": [160, 99, 55, 96, 115, 80],
    }
    data["demand"] = pd.DataFrame(
        demand, index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    )
    # interactions
    interactions = {
        "Glasgow": [60, 15, 15, 8],
        "Manchester": [20, 55, 20, 12],
        "Birmingham": [10, 25, 54, 27],
        "Plymouth": [10, 5, 11, 53],
    }
    data["interactions"] = pd.DataFrame(
        interactions, index=["Glasgow", "Manchester", "Birmingham", "Plymouth"]
    )
    # transfers
    transfers = {
        "Glasgow": [None, 20, 30, 50],
        "Manchester": [20, None, 15, 35],
        "Birmingham": [30, 15, None, 25],
        "Plymouth": [50, 35, 25, None],
    }
    data["transfers"] = pd.DataFrame(
        transfers, index=["Glasgow", "Manchester", "Birmingham", "Plymouth"]
    )
    # repairs
    repairs = {
        "Capacity": [12, 20],
    }
    data["repairs"] = pd.DataFrame(repairs, index=["Manchester", "Birmingham"])
    # rates
    rates = {"Return to Same Depot": [50, 70, 120], "Return to Another Depot": [70, 100, 150]}
    data["rates"] = pd.DataFrame(rates, index=["1-Day hire", "2-Day hire", "3-Day hire"])
    # costs
    costs = {
        "Marginal Cost": [20, 25, 30],
    }
    data["costs"] = pd.DataFrame(costs, index=["1-day", "2-day", "3-day"])
    # distribution
    distribution = {
        "Rental Days Distribution": [0.55, 0.2, 0.25],
    }
    data["distribution"] = pd.DataFrame(distribution, index=["1-day", "2-day", "3-day"])
    return data


# extract data
data = generate_data()
demand = data["demand"]
interactions = data["interactions"]
transfers = data["transfers"]
repairs = data["repairs"]
rates = data["rates"]
costs = data["costs"]
distribution = data["distribution"]

# create model
model = pyo.ConcreteModel()

# define parameters
CAR_OWNING_COST_A_WEEK = 15
CAR_RETURN_DAMAGE_RATE = 0.1
CAR_RETURN_DAMAGE_CHARGE = 100

# define sets
set_depots_d = demand.columns.tolist()
set_dates_t = range(1, demand.shape[0] + 1)
set_depots_repair_dr = repairs.index.tolist()
set_cars_move_cm = [x for x in product(set_depots_d, set_depots_d) if x[0] != x[1]]

# define variables

# define objective function

# define constraint : DEMAND

# define constraints : STEADY STATE
