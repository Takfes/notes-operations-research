import pandas as pd
from pulp import *
import itertools

df = pd.read_excel('beers.xlsx')

costs = df.\
    assign(Warehouse_to_Bar = lambda x: x.Warehouse_to_Bar.astype(str)).\
        set_index('Warehouse_to_Bar').\
            to_dict()

warehouses = ['A','B']
bars = [str(x) for x in range(1,6)]

supply_ = [1000,4000]
demand_ = [500,900,1800,200,700]

supply = dict(zip(warehouses,supply_))
demand = dict(zip(bars, demand_))

routes = list(itertools.product(warehouses,bars))

'''
dv : how many beers for every warehouse - bar pair
obj : minimize dv * cost
1) supply : sum of beers leaving a warehouse <= supply
2) demand : sum of beers reaching a bar == demand
'''

# instantiate minimization problem
prob = LpProblem('Beer_Transportation', LpMinimize)

# define decision variables
decision_vars = LpVariable.dicts('routes',(warehouses,bars),0,None,LpInteger)

# define objective function
prob += lpSum([decision_vars[w][b] * costs[w][b] for w,b in routes])

# define supply constrains
for w in warehouses:
    prob += lpSum([decision_vars[w][b] for b in bars]) <= supply[w]

# define demand constrains
for b in bars:
    prob += lpSum([decision_vars[w][b] for w in warehouses]) >= demand[b]

# solve problem
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# print out solution
for w in warehouses:
    for b in bars:
        if value(decision_vars[w][b]):
            print(f'from {w} to {b} - {value(decision_vars[w][b])}')
