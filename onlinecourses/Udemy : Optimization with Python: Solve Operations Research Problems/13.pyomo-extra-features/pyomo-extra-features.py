import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

SOLVER = "cplex_direct"

# ============================================
# instantiate model
# ============================================
model = pyo.ConcreteModel()

# ============================================
# define variables
# ============================================
nb_mach = 4
nb_time = 10
model.x = pyo.Var(range(1,nb_mach+1),range(1,nb_time+1),within=Integers,bounds=(0,10))
model.x.pprint()
x = model.x

# ============================================
# define constraints
# ============================================

# contraint 1
model.c1 = pyo.ConstraintList()
for t in range(1,nb_time+1):
    model.c1.add(expr = 2*x[2,t] - 8*x[3,t]<=0)
model.c1.pprint()

# contraint 2
model.c2 = pyo.ConstraintList()
for t in range(3,nb_time+1):
    model.c2.add(expr= x[2,t] - 2*x[3,t-2] + x[4,t] >=1)
model.c2.pprint()

# contraint 3
model.c3 = pyo.ConstraintList()
for t in range(1,nb_time+1):
    model.c3.add(expr = sum([x[m,t] for m in range(1,nb_mach+1)])<=50)
model.c3.pprint()

# constraint 4
model.c4 = pyo.ConstraintList()
for t in range(2,nb_time+1):
    model.c4.add(expr = x[1,t] + x[2,t-1] + x[3,t] + x[4,t] <= 10)
model.c4.pprint()

# ============================================
# define objective
# ============================================
model.obj = pyo.Objective(expr=sum([x[m,t] for m in range(1,nb_mach+1) for t in range(1,nb_time+1)]), sense=maximize)
model.obj.pprint()

# ============================================
# define solver and solve
# ============================================
opt = SolverFactory(SOLVER)
# opt.options['MIPgaps'] = 0.05 # 5% works for gurobi
results = opt.solve(model,tee=True) # tee = True to observe solver progress

model.x.pprint()

print(f'{pyo.value(model.obj)=}')

for m in range(1,nb_mach+1):
    for t in range(1,nb_time+1):    
        print(f'{pyo.value(x[m,t])=}')

# # Duality
# model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
# model.dual[model.c2]
