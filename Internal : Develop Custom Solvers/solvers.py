"""
This script provides a framework for solving linear and integer optimization problems using Linear Programming (LP) and Branch-and-Bound techniques. It includes classes and methods for defining optimization problems, solving them, and visualizing the results.

- `solve_lp_function` function solves standalone LP problems.
- `OptimizationProblem` class encapsulates the definition of an optimization problem.
- `OPTIMIZATION_PROBLEMS` list contains sample optimization problems for testing.
- `update_problem_bounds` function creates a new problem with updated variable bounds.
- `Solver` class provides static methods for solving and plotting LP problems.
- `LinearSolver` class extends `Solver` for solving linear relaxation problems.
- `IntegerSolver` class extends `Solver` for solving integer optimization problems using the Branch-and-Bound method.
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from pyvis.network import Network
import networkx as nx


class OptimizationProblem:
    def __init__(
        self,
        objective_coeffs,
        constraint_matrix,
        constraint_bounds,
        variable_bounds,
        objective_direction="max",
        variable_types=None,
    ):
        self.objective_coeffs = objective_coeffs
        self.constraint_matrix = constraint_matrix
        self.constraint_bounds = constraint_bounds
        self.variable_bounds = variable_bounds
        self.objective_direction = objective_direction
        self.variable_types = (
            variable_types if variable_types else ["continuous"] * len(objective_coeffs)
        )

    def __str__(self):
        return (
            f"Objective Coefficients: {self.objective_coeffs}\n"
            f"Constraint Matrix: {self.constraint_matrix}\n"
            f"Constraint Bounds: {self.constraint_bounds}\n"
            f"Variable Bounds: {self.variable_bounds}\n"
            f"Objective Direction: {self.objective_direction}\n"
            f"Variable Types: {self.variable_types}"
        )

    def __repr__(self):
        return (
            f"OptimizationProblem(objective_coeffs={self.objective_coeffs},\n"
            f"constraint_matrix={self.constraint_matrix},\n"
            f"constraint_bounds={self.constraint_bounds},\n"
            f"variable_bounds={self.variable_bounds},\n"
            f"objective_direction={self.objective_direction},\n"
            f"variable_types={self.variable_types})\n"
        )

    def is_mip(self):
        """Check if the problem contains any integer or binary variables."""
        return any(var_type != "continuous" for var_type in self.variable_types)


OPTIMIZATION_PROBLEMS = [
    {
        "name": "Problem 0",
        "url": "https://www.youtube.com/watch?v=upcsrgqdeNQ",
        "object": OptimizationProblem(
            objective_coeffs=[5, 6],
            constraint_matrix=[[1, 1], [4, 7]],
            constraint_bounds=[5, 28],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="max",
        ),
        "z": 27.0,
        "solution": [3.0, 2.0],
    },
    {
        "name": "Problem 1",
        "url": "https://www.youtube.com/watch?v=BzKUhT20wDc",
        "object": OptimizationProblem(
            objective_coeffs=[5, 4],
            constraint_matrix=[[1, 1], [10, 6]],
            constraint_bounds=[5, 45],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="max",
        ),
        "z": 23.0,
        "solution": [3.0, 2.0],
    },
    {
        "name": "Problem 2",
        "url": "https://www.youtube.com/watch?v=BzKUhT20wDc",
        "object": OptimizationProblem(
            objective_coeffs=[5, 4],
            constraint_matrix=[[-3, -2], [-2, -3]],
            constraint_bounds=[-5, -7],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="min",
        ),
        "z": 12.0,
        "solution": [0.0, 3.0],
    },
    {
        "name": "Problem 3",
        "url": "https://pub.towardsai.net/branch-and-bound-introduction-prior-to-coding-the-algorithm-from-scratch-cc265f2909e7",
        "object": OptimizationProblem(
            objective_coeffs=[1, 1],
            constraint_matrix=[[-1, 1], [8, 2]],
            constraint_bounds=[2, 19],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="max",
        ),
        "z": 4.0,
        "solution": [1.0, 3.0],
    },
    {
        "name": "Problem 4",
        "url": "https://medium.com/walmartglobaltech/understanding-branch-and-bound-in-optimization-problems-d8117da0e2c5",
        "object": OptimizationProblem(
            objective_coeffs=[5, 8],
            constraint_matrix=[[1, 1], [5, 9]],
            constraint_bounds=[6, 45],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="max",
        ),
        "z": 40.0,
        "solution": [0.0, 5.0],
    },
]


def update_problem_bounds(problem, var_index, bound_type, bound_value):
    """
    Create a new OptimizationProblem with an updated bound for a specific variable.
    """
    new_bounds = problem.variable_bounds[:]
    if bound_type == "lower":
        new_bounds[var_index] = (bound_value, new_bounds[var_index][1])
    elif bound_type == "upper":
        new_bounds[var_index] = (new_bounds[var_index][0], bound_value)

    return OptimizationProblem(
        problem.objective_coeffs,
        problem.constraint_matrix,
        problem.constraint_bounds,
        new_bounds,
        problem.objective_direction,
        problem.variable_types,
    )


# Standalone LP Solver Function
def solve_lp_function(
    objective_coeffs,
    constraint_matrix,
    constraint_bounds,
    variable_bounds,
    objective_direction="max",
    verbose=False,
):
    """Standalone LP solver function."""
    if objective_direction == "max":
        objective_sign = -1
    elif objective_direction == "min":
        objective_sign = 1
    else:
        raise ValueError("Objective sign must be 'max' or 'min'.")

    result = linprog(
        c=objective_sign * np.array(objective_coeffs),  # Negate for maximization
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=variable_bounds,
        method="highs",
    )
    if result.success:
        if verbose:
            print(75 * "=")
            print(result)
            print(75 * "=")
            print()
        return result.fun * objective_sign, result.x
    else:
        return None, None  # Infeasible


class Solver:
    @staticmethod
    def solve_lp(problem: OptimizationProblem, verbose: bool = False):
        """Static method for solving LP problems."""
        return solve_lp_function(
            problem.objective_coeffs,
            problem.constraint_matrix,
            problem.constraint_bounds,
            problem.variable_bounds,
            problem.objective_direction,
            verbose,
        )

    @staticmethod
    def plot(problem: OptimizationProblem, figsize=(6, 4), scale=1):
        """Plot the feasible region and objective function for 2D problems."""
        if len(problem.objective_coeffs) != 2:
            raise ValueError("Graphical plotting is only supported for 2D problems.")

        fig, ax = plt.subplots(figsize=figsize)

        # Constraint lines
        x = np.linspace(0, 100, 400)
        for i, row in enumerate(problem.constraint_matrix):
            if row[1] != 0:
                ax.plot(
                    x,
                    (problem.constraint_bounds[i] - row[0] * x) / row[1],
                    label=f"Constraint {i+1}",
                )

        # Feasible region shading
        y = np.linspace(0, 100, 400)
        X, Y = np.meshgrid(x, y)
        feasible = np.ones_like(X, dtype=bool)
        for i, row in enumerate(problem.constraint_matrix):
            feasible &= row[0] * X + row[1] * Y <= problem.constraint_bounds[i]
        ax.contourf(X, Y, feasible, levels=1, colors=["lightgray"], alpha=0.5)

        # Plot objective function direction
        c = problem.objective_coeffs
        ax.quiver(
            0,
            0,
            c[0],
            c[1],
            angles="xy",
            scale_units="xy",
            scale=scale,
            color="red",
            label="Objective",
        )

        # Plot intersection points
        intersections = Solver._find_intersections(problem)
        for point in intersections:
            ax.plot(point[0], point[1], "bo")  # Blue dot for intersections
            ax.annotate(
                f"({point[0]:.2f}, {point[1]:.2f})",
                (point[0], point[1]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="center",
            )

        # Automatically adjust x and y limits based on the problem
        x_min, x_max = Solver._get_bounds(problem, 0)
        y_min, y_max = Solver._get_bounds(problem, 1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def _find_intersections(problem: OptimizationProblem):
        """Find all intersection points of the constraints and axes."""
        intersections = []

        # Check intersections between each pair of constraints
        for i, row1 in enumerate(problem.constraint_matrix):
            for j, row2 in enumerate(problem.constraint_matrix):
                if i >= j:
                    continue
                A = np.array([row1, row2])
                b = np.array(
                    [
                        problem.constraint_bounds[i],
                        problem.constraint_bounds[j],
                    ]
                )
                if np.linalg.matrix_rank(A) == 2:  # Ensure the lines are not parallel
                    intersection = np.linalg.solve(A, b)
                    if all(
                        intersection >= 0
                    ):  # Only consider non-negative intersections
                        intersections.append(intersection)

        # Check intersections with the axes
        for i, row in enumerate(problem.constraint_matrix):
            if row[1] != 0:  # Intersection with y-axis (x=0)
                y_intercept = problem.constraint_bounds[i] / row[1]
                if y_intercept >= 0:
                    intersections.append([0, y_intercept])
            if row[0] != 0:  # Intersection with x-axis (y=0)
                x_intercept = problem.constraint_bounds[i] / row[0]
                if x_intercept >= 0:
                    intersections.append([x_intercept, 0])

        return intersections

    @staticmethod
    def _get_bounds(problem: OptimizationProblem, index):
        """Get the bounds for the x or y axis based on the problem constraints."""
        bounds = [0, 10]  # Default bounds
        for i, row in enumerate(problem.constraint_matrix):
            if row[index] != 0:
                bound = problem.constraint_bounds[i] / row[index]
                bounds.append(bound)
        return min(bounds), max(bounds)


class LinearSolver(Solver):
    def __init__(self):
        pass  # Decoupled from a specific problem

    # TODO : Decide on abstraction vs inheritance
    def solve(self, problem: OptimizationProblem, verbose: bool = False):
        """Solve the linear relaxation of a given problem."""
        return Solver.solve_lp(problem, verbose)

    # def plot(self, problem: OptimizationProblem):
    #     """Plot the feasible region and objective function for 2D problems."""
    #     Solver.plot(problem)


class IntegerSolver(Solver):
    """
    - If there exist more than one non integer variables, choose the variable with the largest fractional part to branch on.
    - In a maximization problem, choose the node with the largest objective value (z) to branch first.
    - In a minimization problem, choose the node with the smallest objective value (z) to branch first.
    - Stop branching when an integer solution is found and branching cannot further improve the objective value.
    - Prune the branch if the objective value of the node is less than or equal to the best integer solution found so far.
    - If the LP relaxation is infeasible, prune the branch.
    """

    def __init__(self):
        pass

    @staticmethod
    def _pop_index(is_fifo):
        return 0 if is_fifo else -1

    @staticmethod
    def _is_integer(solution):
        """Check if all variables in the solution are integers."""
        return all(abs(x - round(x)) < 1e-6 for x in solution)

    @staticmethod
    def _find_branch_variable(solution):
        """Find the variable with the largest fractional part to branch on."""
        variable_index = int(np.argmax([x - int(x) for x in solution]))
        variable_value = solution[variable_index].item()
        return variable_index, variable_value

    def solve(
        self,
        problem: OptimizationProblem,
        verbose: bool = False,
        is_fifo: bool = True,
        graph_path: str = "pyvis_networkx.html",
    ):
        """Solve the Integer version of the problem."""
        # initialize the problem
        verbose = verbose
        is_fifo = is_fifo
        pop_index = self._pop_index(is_fifo)
        G = nx.Graph()
        problem_index = 0
        has_solution = False
        best_solution = None
        best_objective = None
        best_node = None
        results = []
        solver = Solver()
        objective_direction = problem.objective_direction
        best_objective = float("-inf") if objective_direction == "max" else float("inf")
        root_node_name = "LP0-ROOT"
        branches = [(root_node_name, problem)]

        while branches:
            # extract problem from the branches list
            current_node, current_problem = branches.pop(pop_index)
            # print(f"Solving {current_node}...")

            # add current node to the graph
            G.add_node(current_node)

            # set the root node color and position - this will be adjusted only once
            if problem_index == 0:
                G.nodes[current_node]["color"] = "black"

            # solve the problem
            z, solution = solver.solve_lp(current_problem, verbose=verbose)

            # if the problem is infeasible, continue to the next problem
            if z is None:
                results.append((current_node, z, solution, "infeasible"))
                G.nodes[current_node]["color"] = "red"
                continue

            # if the problem is integer, update the best solution
            if self._is_integer(solution):
                if (objective_direction == "max" and z > best_objective) or (
                    objective_direction == "min" and z < best_objective
                ):
                    best_objective = z
                    best_solution = solution
                    best_node = current_node
                    has_solution = True
                    results.append(
                        (current_node, z, solution, "integer")
                    )  # record integer nodes
                elif (objective_direction == "max" and z < best_objective) or (
                    objective_direction == "min" and z > best_objective
                ):
                    results.append(
                        (current_node, z, solution, "fathomed")
                    )  # record fathomed nodes
                    G.nodes[current_node]["color"] = (
                        "orange"  # add color for fathomed nodes
                    )
                continue

            # if success and not integer, record results and continue
            results.append((current_node, z, solution, "fractional"))

            # find the variable with the largest fractional part to branch on
            variable_index, variable_value = self._find_branch_variable(solution)

            # create left subproblem with the lower bound
            left = update_problem_bounds(
                current_problem,
                variable_index,
                "upper",
                np.floor(variable_value).item(),
            )
            left_problem_name = f"LP{problem_index+1}-x{variable_index}-left"  # create a new problem name
            branches.append(
                (left_problem_name, left)
            )  # add the new problem to the branches list
            G.add_node(left_problem_name)  # add the new problem to the graph
            left_edge_name = f"x{variable_index} <= {np.floor(variable_value).item():.0f}"  # create the edge name
            G.add_edge(
                current_node, left_problem_name, label=left_edge_name
            )  # add the edge to the graph

            # create right subproblem with the upper bound
            right = update_problem_bounds(
                current_problem, variable_index, "lower", np.ceil(variable_value).item()
            )
            right_problem_name = f"LP{problem_index+2}-x{variable_index}-right"  # create a new problem name
            branches.append(
                (right_problem_name, right)
            )  # add the new problem to the branches list
            G.add_node(right_problem_name)  # add the new problem to the graph
            right_edge_name = f"x{variable_index} >= {np.ceil(variable_value).item():.0f}"  # create the edge name
            G.add_edge(
                current_node, right_problem_name, label=right_edge_name
            )  # add the edge to the graph

            # update the problem index
            problem_index += 2

        G.nodes[best_node]["color"] = "green"  # add color for integer nodes
        net = Network(notebook=True)
        net.from_nx(G)
        net.show(graph_path)

        if verbose:
            print(f"Best integer solution found at node {best_node}:")
            print(f"Objective Value: {best_objective}")
            print(f"Solution: {best_solution}")

        if has_solution:
            return best_objective, best_solution
        else:
            return None, None


# problem = OPTIMIZATION_PROBLEMS[4]["object"]
# solver = IntegerSolver()
# solver.solve(problem)


# def _is_integer(solution):
#     """Check if all variables in the solution are integers."""
#     return all(abs(x - round(x)) < 1e-6 for x in solution)


# def _find_branch_variable(solution):
#     """Find the variable with the largest fractional part to branch on."""
#     variable_index = int(np.argmax([x - int(x) for x in solution]))
#     variable_value = solution[variable_index].item()
#     return variable_index, variable_value


# def _pop_index(is_fifo):
#     return 0 if is_fifo else -1


# problem = OPTIMIZATION_PROBLEMS[4]["object"]

# G = nx.Graph()
# is_fifo = False
# pop_index = _pop_index(is_fifo)

# # initialize the problem
# problem_index = 0
# has_solution = False
# best_solution = None
# best_objective = None
# best_node = None
# results = []
# solver = Solver()
# objective_direction = problem.objective_direction
# best_objective = float("-inf") if objective_direction == "max" else float("inf")
# root_node_name = "LP0-ROOT"
# branches = [(root_node_name, problem)]

# while branches:
#     # extract problem from the branches list
#     current_node, current_problem = branches.pop(pop_index)
#     print(f"Solving {current_node}...")

#     # add current node to the graph
#     G.add_node(current_node)

#     # set the root node color and position - this will be adjusted only once
#     if problem_index == 0:
#         G.nodes[current_node]["color"] = "black"

#     # solve the problem
#     z, solution = solver.solve_lp(current_problem, verbose=False)

#     # if the problem is infeasible, continue to the next problem
#     if z is None:
#         results.append((current_node, z, solution, "infeasible"))
#         G.nodes[current_node]["color"] = "red"
#         continue

#     # if the problem is integer, update the best solution
#     if _is_integer(solution):
#         if (objective_direction == "max" and z > best_objective) or (
#             objective_direction == "min" and z < best_objective
#         ):
#             best_objective = z
#             best_solution = solution
#             best_node = current_node
#             has_solution = True
#             results.append(
#                 (current_node, z, solution, "integer")
#             )  # record integer nodes
#         elif (objective_direction == "max" and z < best_objective) or (
#             objective_direction == "min" and z > best_objective
#         ):
#             results.append(
#                 (current_node, z, solution, "fathomed")
#             )  # record fathomed nodes
#             G.nodes[current_node]["color"] = "orange"  # add color for fathomed nodes
#         continue

#     # if success and not integer, record results and continue
#     results.append((current_node, z, solution, "fractional"))

#     # find the variable with the largest fractional part to branch on
#     variable_index, variable_value = _find_branch_variable(solution)

#     # create left subproblem with the lower bound
#     left = update_problem_bounds(
#         current_problem, variable_index, "upper", np.floor(variable_value).item()
#     )
#     left_problem_name = (
#         f"LP{problem_index+1}-x{variable_index}-left"  # create a new problem name
#     )
#     branches.append(
#         (left_problem_name, left)
#     )  # add the new problem to the branches list
#     G.add_node(left_problem_name)  # add the new problem to the graph
#     left_edge_name = f"x{variable_index} <= {np.floor(variable_value).item():.0f}"  # create the edge name
#     G.add_edge(
#         current_node, left_problem_name, label=left_edge_name
#     )  # add the edge to the graph

#     # create right subproblem with the upper bound
#     right = update_problem_bounds(
#         current_problem, variable_index, "lower", np.ceil(variable_value).item()
#     )
#     right_problem_name = (
#         f"LP{problem_index+2}-x{variable_index}-right"  # create a new problem name
#     )
#     branches.append(
#         (right_problem_name, right)
#     )  # add the new problem to the branches list
#     G.add_node(right_problem_name)  # add the new problem to the graph
#     right_edge_name = f"x{variable_index} >= {np.ceil(variable_value).item():.0f}"  # create the edge name
#     G.add_edge(
#         current_node, right_problem_name, label=right_edge_name
#     )  # add the edge to the graph

#     # update the problem index
#     problem_index += 2

# G.nodes[best_node]["color"] = "green"  # add color for integer nodes
# net = Network(notebook=True)
# net.from_nx(G)
# net.show("pyvis_networkx.html")

# if has_solution:
#     print(f"Best integer solution found at node {best_node}:")
#     print(f"Objective Value: {best_objective}")
#     print(f"Solution: {best_solution}")
