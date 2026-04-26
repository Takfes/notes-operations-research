
# This code attempts to model a given problem in order to determine the feasibility of a certain lowest bid tender for a project. The problem is as follows:

# 4 different tasks : supervision, cleaning, organization, and maintenance
# 7 days a week
# 40 hours per week
# 5 days per week

# supervision : 3 shifts per day, covering 10 hours
# cleaning : 2 shifts per day, covering 10 hours
# organization : 2 shift per day, covering 10 hours
# maintenance : 1 shift per day, covering 10 hours

# 5 supervisors, 1,2,3,4,5
# 6 is supervisor for 10 hours + maintenance for 30 hours
# 3 cleaners : 7,8,9
# 10 is maintenance for 20 hours organizer + cleaner for 20 hours
# 3 organizers, 11,12,13
# 1 maintenance : 14

from itertools import product

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# ============================================================================
# DATA
# ============================================================================
# We test the candidate 14-worker solution described in the problem.
# The skill matrix q[w, t] = 1 iff worker w is qualified for task t.
# Tasks: 'sup' = supervision, 'cln' = cleaning, 'crd' = coordination, 'mnt' = maintenance.
def generate_data():
    data = {}

    # ---- skills matrix (the candidate solution to test) ----
    # Workers 1-5  : supervisors only
    # Worker  6    : supervisor + maintenance (multi-skilled)
    # Workers 7-9  : cleaners only
    # Worker  10   : maintenance + cleaning (multi-skilled)
    # Workers 11-13: coordinators only
    # Worker  14   : maintenance only
    tasks = ["sup", "cln", "crd", "mnt"]
    skills_rows = {
        1:  [1, 0, 0, 0],
        2:  [1, 0, 0, 0],
        3:  [1, 0, 0, 0],
        4:  [1, 0, 0, 0],
        5:  [1, 0, 0, 0],
        6:  [1, 0, 0, 1],
        7:  [0, 1, 0, 0],
        8:  [0, 1, 0, 0],
        9:  [0, 1, 0, 0],
        10: [0, 1, 1, 0],
        11: [0, 0, 1, 0],
        12: [0, 0, 1, 0],
        13: [0, 0, 1, 0],
        14: [0, 0, 0, 1],
    }
    data["skills"] = pd.DataFrame.from_dict(skills_rows, orient="index", columns=tasks)

    # ---- coverage requirements (simultaneous headcount) per task ----
    data["coverage"] = pd.Series(
        {"sup": 3, "cln": 2, "crd": 2, "mnt": 1}, name="required_headcount"
    )

    return data


data = generate_data()
skills = data["skills"]
coverage = data["coverage"]

# ============================================================================
# MODEL
# ============================================================================
SOLVER = "glpk"
model = pyo.ConcreteModel()


# ============================================================================
# SETS
# ============================================================================
# W : workers, T : tasks, D : days of the week, H : hour-slots in the operating window
# Operating window is 09:00-19:00 -> 10 hourly slots indexed 1..10.
set_workers = skills.index.tolist()        # [1, ..., 14]
set_tasks = skills.columns.tolist()        # ['sup', 'cln', 'crd', 'mnt']
set_days = list(range(1, 8))               # 1..7
set_hours = list(range(1, 11))             # 1..10  (10 hour-slots per day)


# ============================================================================
# PARAMETERS
# ============================================================================
HOURS_PER_WEEK = 40              # exactly 40 hours per worker per week
DAYS_PER_WEEK = 5                # exactly 5 working days per worker per week
MAX_HOURS_PER_DAY = len(set_hours)  # 10 -> tight upper bound used in the y-linking big-M

# qualification q[w, t] : 1 if worker w can perform task t
qualification = {(w, t): int(skills.loc[w, t]) for w in set_workers for t in set_tasks}

# required simultaneous headcount per task
required_headcount = coverage.to_dict()    # {'sup': 3, 'cln': 2, 'crd': 2, 'mnt': 1}

# ============================================================================
# DECISION VARIABLES
# ============================================================================
# x[w, t, d, h] = 1 iff worker w performs task t during hour h of day d.
# Defined only over qualified (w, t) pairs to keep the model lean — workers not
# qualified for a task simply have no variable for it (equivalent to forcing 0).
qualified_pairs = [(w, t) for w in set_workers for t in set_tasks if qualification[w, t] == 1]

model.x = pyo.Var(
    [(w, t, d, h) for (w, t) in qualified_pairs for d in set_days for h in set_hours],
    domain=pyo.Binary,
)

# y[w, d] = 1 iff worker w works at all on day d (any task, any hour).
# Auxiliary variable used to express the "exactly 5 working days" rule linearly.
model.y = pyo.Var(set_workers, set_days, domain=pyo.Binary)

# ============================================================================
# OBJECTIVE
# ============================================================================
# Pure feasibility check: any constant objective works. We use 0 so the solver
# stops as soon as it finds a feasible schedule (or proves infeasibility).
model.obj = pyo.Objective(expr=0, sense=pyo.minimize)


# ============================================================================
# HELPER : safely sum x over (t, h) for a given (w, d), skipping unqualified tasks
# ============================================================================
def hours_worked_on_day(w, d):
    """Total hours worker w spends working on day d, across all qualified tasks and hours."""
    return sum(
        model.x[w, t, d, h]
        for t in set_tasks if qualification[w, t] == 1
        for h in set_hours
    )

# ============================================================================
# CONSTRAINT : ONE TASK PER HOUR
# ============================================================================
# At any (day, hour), a worker can be assigned to at most one task.
# (A worker may also be idle that hour — sum can be 0.)
model.cnstr_one_task_per_hour = pyo.ConstraintList()
for w in set_workers:
    for d in set_days:
        for h in set_hours:
            qualified_tasks = [t for t in set_tasks if qualification[w, t] == 1]
            if qualified_tasks:  # only meaningful if worker has at least one skill
                model.cnstr_one_task_per_hour.add(
                    sum(model.x[w, t, d, h] for t in qualified_tasks) <= 1
                )

# model.cnstr_one_task_per_hour.pprint()

# ============================================================================
# CONSTRAINT : COVERAGE  (the heart of the problem)
# ============================================================================
# For every task, at every hour-slot of every day in the operating window,
# the number of workers assigned to that task must meet the required headcount.
# This is what enforces "at least N workers on duty simultaneously".
model.cnstr_coverage = pyo.ConstraintList()
for t in set_tasks:
    qualified_workers_for_t = [w for w in set_workers if qualification[w, t] == 1]
    for d in set_days:
        for h in set_hours:
            model.cnstr_coverage.add(
                sum(model.x[w, t, d, h] for w in qualified_workers_for_t)
                >= required_headcount[t]
            )

# model.cnstr_coverage.pprint()

# ============================================================================
# CONSTRAINT : LINK y[w, d] TO x  (big-M indicator)
# ============================================================================
# y[w, d] should equal 1 iff worker w does any work on day d.
# Two linear inequalities encode this:
#   (a) y <= sum(x)    -> if no work that day, forces y = 0
#   (b) sum(x) <= M*y  -> if any work that day, forces y = 1   (M = 10 is tight)
model.cnstr_y_link_lower = pyo.ConstraintList()
model.cnstr_y_link_upper = pyo.ConstraintList()
for w in set_workers:
    for d in set_days:
        daily_hours = hours_worked_on_day(w, d)
        model.cnstr_y_link_lower.add(model.y[w, d] <= daily_hours)
        model.cnstr_y_link_upper.add(daily_hours <= MAX_HOURS_PER_DAY * model.y[w, d])

# model.cnstr_y_link_lower.pprint()
# model.cnstr_y_link_upper.pprint()

# ============================================================================
# CONSTRAINT : EXACTLY 5 WORKING DAYS PER WEEK
# ============================================================================
model.cnstr_five_days = pyo.ConstraintList()
for w in set_workers:
    model.cnstr_five_days.add(sum(model.y[w, d] for d in set_days) == DAYS_PER_WEEK)

# model.cnstr_five_days.pprint()

# ============================================================================
# CONSTRAINT : EXACTLY 40 WORKING HOURS PER WEEK
# ============================================================================
model.cnstr_forty_hours = pyo.ConstraintList()
for w in set_workers:
    weekly_hours = sum(hours_worked_on_day(w, d) for d in set_days)
    model.cnstr_forty_hours.add(weekly_hours == HOURS_PER_WEEK)


# ============================================================================
# REVIEW MODEL COMPONENTS
# ============================================================================
# from pyomo.environ import *

# for c in model.component_objects(Constraint, active=True):
#     print(f"\nConstraint: {c.name}")
#     for index in c:
#         expr = c[index].expr
#         print(f"  {index}: {expr}")

# ============================================================================
# SOLVE
# ============================================================================
opt = SolverFactory(SOLVER)
results = opt.solve(model)


# ============================================================================
# OUTPUT
# ============================================================================
if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    print("FEASIBLE: the problem satisfies all constraints.")
    print(75 * "-")

    # weekly hours per worker per task
    rows = []
    for (w, t, d, h), var in model.x.items():
        if pyo.value(var) > 0.5:
            rows.append((w, t, d, h))
    schedule = pd.DataFrame(rows, columns=["worker", "task", "day", "hour"])

    print("Weekly hours per worker per task:")
    print(
        schedule.groupby(["worker", "task"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=set_workers, columns=set_tasks, fill_value=0)
    )
    print(75 * "-")

    print("Working days per worker (should all be 5):")
    print(schedule.groupby("worker")["day"].nunique().reindex(set_workers, fill_value=0))
    print(75 * "-")

    print("Total hours per worker (should all be 40):")
    print(schedule.groupby("worker").size().reindex(set_workers, fill_value=0))

elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print("INFEASIBLE: the problem CANNOT satisfy all constraints.")
else:
    print(f"Solver ended with status: {results.solver.termination_condition}")