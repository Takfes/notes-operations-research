from itertools import product

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def generate_data():
    data = {}
    # technician_data
    technician_data = {
        "Albert": [480, "Heidelberg"],
        "Bob": [480, "Heidelberg"],
        "Carlos": [480, "Freiburg im Breisgau"],
        "Doris": [480, "Freiburg im Breisgau"],
        "Ed": [480, "Heidelberg"],
        "Flor": [360, "Freiburg im Breisgau"],
        "Gina": [360, "Heidelberg"],
    }
    data["technician_data"] = pd.DataFrame(technician_data, index=["Minutes", "Depot"])
    # job_data
    job_data = {"Priority": [2, 3, 1, 1, 2, 3, 4], "Duration (min)": [60, 30, 60, 60, 120, 90, 60]}
    data["job_data"] = pd.DataFrame(
        job_data,
        index=[
            "Equipment Installation",
            "Equipment Setup",
            "Inspect/Service Equipment",
            "Repair - Regular",
            "Repair - Important",
            "Repair - Urgent",
            "Repair - Critical",
        ],
    )
    # qualification_data
    qualification_data = {
        "Albert": [0, 1, 1, 0, 0, 0, 0],
        "Bob": [0, 1, 0, 1, 0, 1, 0],
        "Carlos": [0, 1, 0, 0, 1, 1, 0],
        "Doris": [1, 1, 0, 0, 1, 1, 1],
        "Ed": [1, 0, 0, 1, 1, 1, 1],
        "Flor": [1, 1, 0, 1, 1, 1, 1],
        "Gina": [1, 0, 0, 0, 1, 0, 1],
    }
    data["qualification_data"] = pd.DataFrame(
        qualification_data,
        index=[
            "Equipment Installation",
            "Equipment Setup",
            "Inspect/Service Equipment",
            "Repair - Regular",
            "Repair - Important",
            "Repair - Urgent",
            "Repair - Critical",
        ],
    )
    # customer_data
    customer_data = {
        "Job type": [
            "Equipment Setup",
            "Equipment Setup",
            "Repair - Regular",
            "Equipment Installation",
            "Equipment Installation",
            "Repair - Critical",
            "Inspect/Service Equipment",
        ],
        "Due time": [480, 600, 660, 720, 840, 900, 960],  # Converted to minutes from 7:00
        "Time Window Start": [
            30,
            90,
            180,
            240,
            360,
            420,
            480,
        ],  # Converted start times of time windows to minutes from 7:00
        "Time Window End": [
            150,
            210,
            300,
            360,
            480,
            540,
            600,
        ],  # Converted end times of time windows to minutes from 7:00
    }
    data["customer_data"] = pd.DataFrame(
        customer_data,
        index=[
            "C1:Mannheim",
            "C2:Karlsruhe",
            "C3:Baden-Baden",
            "C4:Bühl",
            "C5:Offenburg",
            "C6:Lahr/Schwarzwald",
            "C7:Lörrach",
        ],
    )
    # travel_time_data
    travel_time_data = {
        "Heidelberg": [0, 120, 24, 50, 67, 71, 88, 98, 150],
        "Freiburg im Breisgau": [120, 0, 125, 85, 68, 62, 45, 39, 48],
        "Mannheim": [24, 125, 0, 53, 74, 77, 95, 106, 160],
        "Karlsruhe": [50, 85, 53, 0, 31, 35, 51, 61, 115],
        "Baden-Baden": [67, 68, 74, 31, 0, 16, 36, 46, 98],
        "Bühl": [71, 62, 77, 35, 16, 0, 30, 40, 92],
        "Offenburg": [88, 45, 95, 51, 36, 30, 0, 26, 80],
        "Lahr/Schwarzwald": [98, 39, 106, 61, 46, 40, 26, 0, 70],
        "Lörrach": [150, 48, 160, 115, 98, 92, 80, 70, 0],
    }
    data["travel_time_data"] = pd.DataFrame(
        travel_time_data,
        index=[
            "Heidelberg",
            "Freiburg im Breisgau",
            "Mannheim",
            "Karlsruhe",
            "Baden-Baden",
            "Bühl",
            "Offenburg",
            "Lahr/Schwarzwald",
            "Lörrach",
        ],
    )
    return data


# extract data
data = generate_data()
technician_data = data["technician_data"]
job_data = data["job_data"]
qualification_data = data["qualification_data"]
travel_time_data = data["travel_time_data"]
customer_data = data["customer_data"]
customer_data["Priority"] = customer_data["Job type"].map(job_data["Priority"])
customer_data["Duration"] = customer_data["Job type"].map(job_data["Duration (min)"])
customer_data["City"] = customer_data.index.str.split(":").str[1]

# create model
SOLVER = "glpk"
model = pyo.ConcreteModel()

# define parameters
SHIFT_HOURS = 10 * 60  # 10 hours in minutes

# define sets
set_tech_t = technician_data.columns.tolist()
set_time_d = range(1, SHIFT_HOURS + 1)
set_orders_o = customer_data.index.tolist()

# define data
availability = technician_data.loc["Minutes", :].to_dict()
depots = technician_data.loc["Depot", :].to_dict()
qualifications = qualification_data.stack().to_dict()
distances = travel_time_data.stack().to_dict()
order_priority = customer_data["Priority"].to_dict()
order_duration = customer_data["Duration"].to_dict()
order_tw_start = customer_data["Time Window Start"].to_dict()
order_tw_end = customer_data["Time Window End"].to_dict()
order_deadline = customer_data["Due time"].to_dict()
order_city = customer_data["City"].to_dict()
order_type = customer_data["Job type"].to_dict()

# define technician-job-travel-time
aux = pd.DataFrame(data=product(set_tech_t, set_orders_o), columns=["tech", "order"])
aux["tech_city"] = aux["tech"].map(depots)
aux["order_city"] = aux["order"].map(order_city)
aux["travel_time"] = aux.apply(lambda x: distances[(x["tech_city"], x["order_city"])], axis=1)
t_o_times = (
    aux[["tech", "order", "travel_time"]].set_index(["tech", "order"]).to_dict()["travel_time"]
)

# define techinicial-job-qualification
qua = pd.DataFrame(data=product(set_tech_t, set_orders_o), columns=["tech", "order"])
qua["type"] = qua["order"].map(order_type)
qua["qualification"] = qua.apply(lambda x: qualifications[(x["type"], x["tech"])], axis=1)
t_o_qualifications = [
    (row["tech"], row["order"])
    for _, row in qua.loc[qua["qualification"] == 1, ["tech", "order"]].iterrows()
]

# define variables
model.dv = pyo.Var(t_o_qualifications, set_time_d, domain=pyo.Binary)

# define objective function
# * The firm’s goal is to minimize the total weighted lateness of all jobs, with their priority being the weights.
model.obj = pyo.Objective(
    expr=sum(
        model.dv[t, o, d] * order_priority[o] * max(d + order_duration[o] - order_deadline[o], 0)
        for t in set_tech_t
        for o in set_orders_o
        for d in set_time_d
        if (t, o) in t_o_qualifications
    ),
    sense=pyo.minimize,
)

"""
A technician, if utilized, departs from the service center where he/she is based and returns to the same service center after his/her assigned jobs are completed.
A technician’s available capacity during the scheduling horizon cannot be exceeded.
A job, if selected, is assigned to at most one technician who possesses the required skills.
A technician must arrive at a customer location during an interval (time window) specified by the customer, and must complete a job before the deadline required by the customer. This an important constraint for guaranteeing customer satisfaction.
"""
# define constraint : TECHNICIAN CAPACITY
model.tech_capacity = pyo.ConstraintList()
for t in set_tech_t:
    tech_capacity = sum(
        model.dv[t, o, d] * (order_duration[o] + 2 * t_o_times[t, o])
        for o in set_orders_o
        for d in set_time_d
        if (t, o) in t_o_qualifications
    )
    model.tech_capacity.add(tech_capacity <= availability[t])

# define constraint : JOB ASSIGNMENT
model.job_assignment = pyo.ConstraintList()
for o in set_orders_o:
    job_assignment = sum(
        model.dv[t, o, d] for t in set_tech_t for d in set_time_d if (t, o) in t_o_qualifications
    )
    model.job_assignment.add(job_assignment == 1)

# define constraint : TIME WINDOW
model.time_window = pyo.ConstraintList()
for o in set_orders_o:
    time_window_max = (
        sum(
            model.dv[t, o, d] * d
            for t in set_tech_t
            for d in set_time_d
            if (t, o) in t_o_qualifications
        )
        <= order_tw_end[o]
        # <= order_deadline[o] - order_duration[o]
    )
    time_window_min = (
        sum(
            model.dv[t, o, d] * d
            for t in set_tech_t
            for d in set_time_d
            if (t, o) in t_o_qualifications
        )
        >= order_tw_start[o]
    )
    model.time_window.add(time_window_max)
    model.time_window.add(time_window_min)

# solve model
opt = SolverFactory(SOLVER)
results = opt.solve(model)

if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    # Access other solution information as needed
    print(f"Weighted lateness ${pyo.value(model.obj):=.2f}")
    print(75 * "-")

    for x in model.dv:
        if model.dv[x]() > 0:
            print(f"{x} = {model.dv[x]():.0f}")
