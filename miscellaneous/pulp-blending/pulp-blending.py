import pandas as pd
from pulp import *
# import operator

df = pd.read_excel('whiskas.xlsx')

requirements = [100,8.0,6.0,2.0,0.4]

ingredients = df.Stuff.tolist()
costs_ = [0.013, 0.008, 0.010, 0.002, 0.005, 0.001]
costs = {k:v for k,v in zip(ingredients,costs_)}

'''
dv : pct of ingredients
obj : minimize sumprod dv, costs
s.t. :
1) sumprod dv == 100
2) sumprod dv, protein >= 8.0
3) sumprod dv, fat >= 6.0
4) sumprod dv, fibre <= 2.0
5) sumprod dv, salt <= 0.4
'''

# instantiate problem
prob = LpProblem("The_Whiskas_Problem", LpMinimize)

# decision variables
decision_vars = LpVariable.dicts("ingredients", ingredients, 0)

# objective function
prob += (
    lpSum([decision_vars[x] * costs[x] for x in ingredients]),
    'Objective Function : Minimize total cost'
)
# constraint 1
prob += (
    lpSum([decision_vars[x] for x in ingredients]) == 100,
    'Total Quantity'
)
# constraint 2
proteins = {k:v for k,v in zip(ingredients,df['Protein'])}
prob += (
    lpSum([decision_vars[x] * proteins[x] for x in ingredients]) >= 8.0,
    'Protein Requirements'
)
# constraint 3
fat = {k:v for k,v in zip(ingredients,df['Fat'])}
prob += (
    lpSum([decision_vars[x] * fat[x] for x in ingredients]) >= 6.0,
    'Fat Requirements'
)
# constraint 4
fibre = {k:v for k,v in zip(ingredients,df['Fibre'])}
prob += (
    lpSum([decision_vars[x] * fibre[x] for x in ingredients]) <= 2.0,
    'Fibre Requirements'
)
# constraint 5
salt = {k:v for k,v in zip(ingredients,df['Salt'])}
prob += (
    lpSum([decision_vars[x] * salt[x] for x in ingredients]) <= 0.4,
    'Salt Requirements'
)

# The problem data is written to an .lp file
prob.writeLP("whiskas_model.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)
    
# The optimised objective function value is printed to the screen
print("Total Cost of Ingredients per can = ", value(prob.objective))
