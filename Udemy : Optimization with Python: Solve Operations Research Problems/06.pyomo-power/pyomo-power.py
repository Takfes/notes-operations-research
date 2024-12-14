import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

SOLVER = "glpk"

# read data
datagen = pd.read_excel("src/06.pyomo-power/pyomo-power.xlsx", sheet_name="datagen")
dataload = pd.read_excel("src/06.pyomo-power/pyomo-power.xlsx", sheet_name="dataload")
npg = datagen.shape[0]

# ============================================
# instantiate model
# ============================================
model = pyo.ConcreteModel()

# ============================================
# define variables
# ============================================
model.Pg = pyo.Var(range(npg),bounds=(0,None))
Pg = model.Pg # for easier access of the variables
model.Pg.pprint() # examine the variables

# ============================================
# define constraints
# ============================================
# balance demand
pg_sum = sum([Pg[x] for x in Pg])
model.balance = pyo.Constraint(expr= pg_sum == dataload.demand.sum())
model.balance.pprint()

# load points conditional constraint
model.cond = pyo.Constraint(expr = Pg[0] + Pg[3] >= dataload.demand[0])
model.cond.pprint()

# limit constraints
for i,x in enumerate(datagen.limit):
    Pg[i].upper = float(x)
Pg.pprint()

# # limit constraints alternative way
# model.limits = pyo.ConstraintList()
# for x in range(npg):
#     model.limits.add(expr = Pg[x] <= datagen.limit[x])
# model.limits.pprint()

# ============================================
# define objective function
# ============================================
cost_sum = sum([Pg[x]*datagen.cost[x] for x in range(npg)])
model.obj = pyo.Objective(expr=cost_sum, sense=minimize)

# ============================================
# define solver and solve
# ============================================
opt = SolverFactory(SOLVER)
results = opt.solve(model)

sol = [pyo.value(Pg[x]) for x in range(npg)]
datagen['pg'] = sol
