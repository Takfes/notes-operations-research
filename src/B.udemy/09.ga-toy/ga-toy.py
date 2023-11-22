import numpy as np
from geneticalgorithm import geneticalgorithm as ga


def fitness(x):
    pen=0
    if not -x[0]+2*x[1]*x[0]<=8: pen = np.inf # constraints
    if not 2 * x[0] + x[1] <=14: pen = np.inf # constraints
    if not 2 * x[0] - x[1] <=10: pen = np.inf # constraints
    return -(x[0]+x[0]*x[1]) + pen # objective function = maximize x + x*y

varbounds = np.array([[0,10],[0,10]])
vartypes = np.array([['int'],['real']])

algorithm_param = {'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model = ga(
    function = fitness,
    dimension = 2,
    variable_type_mixed=vartypes,
    variable_boundaries=varbounds,
    algorithm_parameters=algorithm_param)

model.run()
