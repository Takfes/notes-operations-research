
# Infeasibilities

- two different ways to tackle infeasibilities are discussed below, `penalty terms` vs `hierarchical objectives`
- further reading [Pyomo Infeasibility Diagnostics](https://pyomo.readthedocs.io/en/6.8.0/contributed_packages/iis.html) and [Pyomo Repeated Solves](https://pyomo.readthedocs.io/en/stable/howto/manipulating.html)

## Penalty Terms

- introduce slack variables, analogous to the domain of the decision variable that is being violated
- the main objective function is modified - introduce penalty coefficients in the objective function

```python
import pyomo.environ as pyo
# Initialize the model
model = pyo.ConcreteModel()
# Define variables
model.x = pyo.Var(within=pyo.NonNegativeReals)
# Define artificial variable for constraint relaxation
model.slack = pyo.Var(within=pyo.NonNegativeReals)
# Original constraint: x <= 10
# Relaxed constraint: x <= 10 + slack
model.constraint = pyo.Constraint(expr=model.x <= 10 + model.slack)
# Objective function with penalty on slack variable
penalty_weight = 1000  # Large penalty to discourage violations
model.objective = pyo.Objective(expr=model.x + penalty_weight * model.slack, sense=pyo.minimize)
# Solve the model
solver = pyo.SolverFactory('glpk')
solver.solve(model)
# Display results
print(f"x = {pyo.value(model.x)}")
print(f"Slack = {pyo.value(model.slack)}")
```

## Hierarchy of objectives

- adopt a two step approach:
- introduce artificial variables; these must be defined in the domains of the variables that need to be relaxed
- solve the auxiliary problem first; construct an objective targeted to minimize the artificial variables
- once the auxiliary problem is solved, grab the value of the artificial variable and introduce this as a new constraint
- then solve the actual problem, including the additional constraint
- following is a step-approach how to do that...

```python
import pyomo.environ as pyo
# Initialize the model
model = pyo.ConcreteModel()
# Sets for suppliers I and customers J
model.I = pyo.Set(initialize=availabilities.keys())
model.J = pyo.Set(initialize=demands.keys())
# Parameters
model.b = pyo.Param(model.I, initialize=availabilities)
model.d = pyo.Param(model.J, initialize=demands)
model.c = pyo.Param(model.I, model.J, initialize=costs)
# Decision variables
model.x = pyo.Var(model.I, model.J, within=pyo.NonNegativeReals)

# STEP 1.
# Define artificial variable for constraint relaxation
# artificial variable are defined in the domain of the variables they are supposed to be relaxing; in this instance assuming the model.J is the variable being violated and hence needs relaxation
model.z = pyo.Var(model.J, within=pyo.NonNegativeReals)

# Supplier availablity constraints
def av_cstr(model, i):
    return sum(model.x[i,j] for j in model.J) <= model.b[i]
model.av_cstr = pyo.Constraint(model.I, rule=av_cstr)

# STEP 2.
# Demand equality constraints
# Notice we now include the artificial variable in the constraint below
def dem_cstr(model, j):
    return sum(model.x[i,j] + model.z[j] for i in model.I) == model.d[j]
model.dem_cstr = pyo.Constraint(model.J, rule=dem_cstr)

# STEP 3.
# Artificial/Auxiliary Objective Function
def art_obj(model):
    return sum(model.z[:])
model.art_obj = pyo.Objective(rule=art_obj, sense=pyo.minimize)

# Objective function
def obj(model):
    total_cost = sum(
        model.x[i,j] * model.c[i,j]
        for i in model.I
        for j in model.J
    )
    return total_cost
model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

# Define solver
solver = pyo.SolverFactory('glpk')

# STEP 4.
# Solve for the auxiliary/articifial objective first
model.obj.deactivate()
model.art_obj.activate()
solver.solve(model)

# STEP 5.
# Access the results of the optimized auxiliary objective
K = model.art_obj()

# STEP 6.
# Introduce artificial-variable related constraint
def art_cstr(model):
    return sum(model.z[:]) <= K
model.art_cstr = pyo.Constraint(model.J, rule=art_cstr)

# STEP 7.
# Now solve for the actual problem
model.obj.activate()
model.art_obj.deactivate()
solver.solve(model)

# Display results
# Use objective as a callable to see its value
model.obj()
sol = [
    {"from": i, "to": j, "value": val}
    for (i, j), val in model.x.extract_values().items()
]
```
