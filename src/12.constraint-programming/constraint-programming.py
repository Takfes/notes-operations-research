from ortools.sat.python import cp_model


# https://developers.google.com/optimization/cp/cp_solver
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count

model = cp_model.CpModel()

x = model.NewIntVar(0,1000,'x')
y = model.NewIntVar(0,1000,'y')
z = model.NewIntVar(0,1000,'z')

model.Add(2*x+7*y+3*z<=50)
model.Add(3*x-5*y+7*z<=45)
model.Add(5*x+2*y-6*z<=37)
model.Add(x+y+z>=10)

# objective function is optional in the CP context
# model.Maximize(2*x+2*y+3*z)

solver = cp_model.CpSolver()
status = solver.Solve(model)

print(f'Status : {solver.StatusName(status)}')
print(f'Objective Value : {solver.ObjectiveValue()}')
print(f'x : {solver.Value(x)}')
print(f'y : {solver.Value(y)}')
print(f'z : {solver.Value(z)}')

solution_printer = VarArraySolutionPrinter([x,y,z])
# in a non-optimization context, enumerate all possible combinations
status = solver.SearchForAllSolutions(model, solution_printer)