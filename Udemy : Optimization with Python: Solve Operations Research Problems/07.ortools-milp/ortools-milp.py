from ortools.linear_solver import pywraplp

SOLVER = 'cplex_direct' # 'GLOP'
solver = pywraplp.Solver.CreateSolver(SOLVER)

x = solver.NumVar(0,10,'x')
# x = solver.IntVar(0,10,'x')
y = solver.NumVar(0,10,'y')

solver.Add(-x+2*y<=8)
# solver.Add(-x+2*y<=7)
solver.Add(2*x+y<=14)
solver.Add(2*x-y<=10)

solver.Maximize(x+y)

results = solver.Solve()

if results==pywraplp.Solver.OPTIMAL:
    print('Optimal Solution Found')
    print(f'{x.solution_value()=}')
    print(f'{y.solution_value()=}')