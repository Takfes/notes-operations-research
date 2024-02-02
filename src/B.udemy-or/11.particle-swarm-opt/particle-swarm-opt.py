import numpy as np
from pyswarm import pso


def objective(x):
    pen = 0
    x[0] = np.round(x[0],0)
    if not -x[0]+2*x[1]*x[0]<=8: pen = np.inf
    if not 2*x[0]+x[1]<=14: pen = np.inf
    if not 2*x[0]-x[1]<=10: pen = np.inf
    return -(x[0]+x[1]*x[0]) + pen

def constraints(x):
    return []

lb = [0,0]
ub = [10,10]
x0 = [0,0]

xopt, fopt = pso(objective,lb,ub,x0,constraints)

print(f'{xopt[0]=} & {xopt[1]=}')