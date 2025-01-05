```python
solver = SolverFactory('gurobi')
solver.options['mipgap'] = 0.01
solver.options['MIPFocus'] = 1
solver.options['Heuristics'] = 1
solver.options['cuts']=3
```
source : [github repo](https://github.com/PEESEgroup/Offshore_Wind_to_H2/blob/71302e43e78b92d3ef68769d20df160ac5aefe7c/Statewise/Case_Study_CGH2_New.ipynb#L424)