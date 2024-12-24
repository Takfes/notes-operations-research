```python
def solve_model(model):
        solver_name = "appsi_highs"
        solver = pyo.SolverFactory(solver_name)
        solver.options['parallel'] = 'on'
        solver.options['time_limit'] = 3600/2  # 30 minutes time limit
        solver.options['presolve'] = 'on'
        solver.options['mip_rel_gap'] = 0.01  # 1% relative gap
        solver.options['simplex_strategy'] = 1  # Dual simplex
        solver.options['simplex_max_concurrency'] = 8  # Max concurrency
        solver.options['mip_min_logging_interval'] = 10  # Log every 10 seconds
        solver.options['mip_heuristic_effort'] = 0.2  # Increase heuristic effort

        result = solver.solve(model, tee=True)

        # Check solver status
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Optimal solution found.")
        elif result.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
            print("Time limit reached, solution may be suboptimal.")
        else:
            print(f"Solver terminated with condition: {result.solver.termination_condition}")

        print(result)

solve_model(model)
```

source : [github repo](https://github.com/JuanManuelJaureguiRozo/Proyecto_MOS/blob/047b6ff2cbbb6e4dd22e14d442cf14b113716cbb/Entrega%202/results_case_1/case_1_base.ipynb#L470)


```python
else:
        Solver = pyo.SolverFactory(pyo.value(Model.Engine))   # Local solver installed
        if pyo.value(Model.Engine) == 'couenne':   # Non-linear
            print('No options for Couenne') # Couenne doesn't accept command line options, use couenne.opt instead
        elif pyo.value(Model.Engine) == 'cbc':   # Linear
            Solver.options['seconds'] = pyo.value(Model.TimeLimit)
            Solver.options['log'] = 1   # Default is 1. 2 or 3 provides more detail. Also have slog, which provides much more detail
        elif pyo.value(Model.Engine) == 'appsi_highs':   # Linear
            Solver.options['time_limit'] = pyo.value(Model.TimeLimit)
            Solver.options['log_file'] = 'highs.log'   # Sometimes HiGHS doesn't update the console as it solves, so write log file too
            Solver.options['mip_rel_gap'] = MipGap   # Relative gap. 10 = stop at first feasible solution. 0 = find optimum. 0.1 = within 10% of optimum
            #Solver.options['parallel'] = 'on'
            #Solver.options['mip_heuristic_effort'] = 0.2   # default = 0.05, range = 0..1
        elif pyo.value(Model.Engine) == 'gurobi_direct':
            Solver.options['timelimit'] = pyo.value(Model.TimeLimit)
            Solver.options['logfile'] = 'gurobi.log'
            Solver.options['mipgap'] = MipGap
            Solver.options['solutionlimit'] = SolutionLimit
            #Solver.options['heuristics'] = 0.25   # 0..1 default = 0.05
        else:
            print('Unknown local solver when setting options')
```

source : [github repo](https://github.com/prajeshshrestha/Crossword-Generator/blob/48707af4cb7087704c46c76d19db7f0c497b11ac/MILP%20-%20Approach/Crossword%20-%20Model%202%20Details.ipynb#L58)

```python
# mip_feasibility_tolerance: This determines how much a solution can violate constraints while still being considered feasible.

# mip_pool_age_limit: Controls how long solutions stay in the solution pool.

# mip_pool_soft_limit: Sets a target for the maximum number of solutions to keep in the pool.

solver.options['mip_feasibility_tolerance'] = 1e-5  # More relaxed vs 1e-8 -> Very strict
solver.options['mip_pool_age_limit'] = 0     # Only keep best solution vs -1  -> Keep all solutions
solver.options['mip_pool_soft_limit'] = 1    # Minimal solution storage vs 50 -> Keep up to 50 solutions
```
source : [claude]