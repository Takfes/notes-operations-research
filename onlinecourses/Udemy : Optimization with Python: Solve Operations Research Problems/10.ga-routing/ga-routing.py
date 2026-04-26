import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga

paths = pd.read_excel('src/10.ga-routing/ga-routing.xlsx',sheet_name='paths')
# nodes = pd.read_excel('src/10.ga-routing/ga-routing.xlsx',sheet_name='nodes')
dims = paths.shape[0]

varbounds = np.array([[0,1] for _ in range(dims)]) 
vartypes = np.array([['int'] for _ in range(dims)])

def fitness(x):
    pen = 0
    penalty_value = 100_000
    xb = [bool(x) for x in x]
    temp = paths.loc[xb,:].copy()
    temp['shifted'] = temp.node_from.shift(-1).astype('Int32').fillna(paths.node_to.max())
    temp2 = temp.dropna().query('node_to==shifted')
    check_min = temp2.node_from.min()
    check_max = temp2.node_to.max()
    if check_min!=paths.node_from.min(): pen = penalty_value
    if check_max!=paths.node_to.max(): pen = penalty_value
    if temp2.shape[0] != sum(x): pen = penalty_value
    value = penalty_value if temp2.empty else temp2.distance.sum()
    return value + pen

algorithm_param = {'max_num_iteration': 500,\
                   'population_size':100,\
                   'mutation_probability':0.30,\
                   'elit_ratio': 0.10,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':100}

model = ga(
    function = fitness,
    dimension = dims,
    variable_type_mixed=vartypes,
    variable_boundaries=varbounds,
    algorithm_parameters=algorithm_param)
model.run()