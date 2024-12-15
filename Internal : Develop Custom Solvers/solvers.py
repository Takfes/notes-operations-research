import numpy as np
from scipy.optimize import linprog
from graphviz import Digraph
from typing import Union
import matplotlib.pyplot as plt


class OptimizationProblem:
    def __init__(
        self,
        objective_coeffs,
        constraint_matrix,
        constraint_bounds,
        variable_bounds,
        variable_types=None,
    ):
        self.objective_coeffs = objective_coeffs
        self.constraint_matrix = constraint_matrix
        self.constraint_bounds = constraint_bounds
        self.variable_bounds = variable_bounds
        self.variable_types = (
            variable_types if variable_types else ["continuous"] * len(objective_coeffs)
        )

    def __str__(self):
        return (
            f"Objective Coefficients: {self.objective_coeffs}\n"
            f"Constraint Matrix: {self.constraint_matrix}\n"
            f"Constraint Bounds: {self.constraint_bounds}\n"
            f"Variable Bounds: {self.variable_bounds}\n"
            f"Variable Types: {self.variable_types}"
        )

    def __repr__(self):
        return (
            f"OptimizationProblem(objective_coeffs={self.objective_coeffs},\n"
            f"constraint_matrix={self.constraint_matrix},\n"
            f"constraint_bounds={self.constraint_bounds},\n"
            f"variable_bounds={self.variable_bounds},\n"
            f"variable_types={self.variable_types})\n"
        )

    def is_mip(self):
        """Check if the problem contains any integer or binary variables."""
        return any(var_type != "continuous" for var_type in self.variable_types)

    def create_branch(self, var_index, bound_type, bound_value):
        """
        Create a new OptimizationProblem with an updated bound for a specific variable.
        """
        new_bounds = self.variable_bounds[:]
        if bound_type == "lower":
            new_bounds[var_index] = (bound_value, new_bounds[var_index][1])
        elif bound_type == "upper":
            new_bounds[var_index] = (new_bounds[var_index][0], bound_value)

        return OptimizationProblem(
            self.objective_coeffs,
            self.constraint_matrix,
            self.constraint_bounds,
            new_bounds,
            self.variable_types,
        )


def solve_lp(
    objective_coeffs,
    constraint_matrix,
    constraint_bounds,
    variable_bounds,
    verbose=False,
):
    """
    Solves an LP problem.
    """
    result = linprog(
        c=-np.array(objective_coeffs),  # Negate for maximization
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=variable_bounds,
        method="highs",  # Use the HiGHS solver for better performance
    )

    if result.success:
        if verbose:
            print(75 * "=")
            print(result)
            print(75 * "=")
            print()
        return result.fun, result.x  # Return objective value and solution
    else:
        return None, None  # Infeasible


class Solver:
    def __init__(self, problem: OptimizationProblem):
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("The problem must be an instance of OptimizationProblem")
        self.problem = problem
        self.objective_coeffs = problem.objective_coeffs
        self.constraint_matrix = problem.constraint_matrix
        self.constraint_bounds = problem.constraint_bounds
        self.variable_bounds = problem.variable_bounds
        self.variable_types = problem.variable_types

    def validate_input(self):
        if len(self.objective_coeffs) != len(self.variable_bounds):
            raise ValueError(
                "Length of objective coefficients must match variable bounds"
            )
        if len(self.constraint_matrix) != len(self.constraint_bounds):
            raise ValueError("Number of constraints must match constraint bounds")
        for var_type in self.variable_types:
            if var_type not in ["continuous", "integer", "binary"]:
                raise ValueError(f"Invalid variable type: {var_type}")


class LinearSolver(Solver):
    def solve(
        self, verbose: bool = False
    ) -> Union[tuple[float, np.ndarray], tuple[None, None]]:
        """Solve the linear relaxation of the problem."""
        self.validate_input()
        return solve_lp(
            self.objective_coeffs,
            self.constraint_matrix,
            self.constraint_bounds,
            self.variable_bounds,
            verbose,
        )


class SimplexSolver(Solver):
    pass


class GraphicalSolver(Solver):
    def _find_intersections(self):
        """Find all intersection points of the constraints and axes."""
        intersections = []

        # Check intersections between each pair of constraints
        for i, row1 in enumerate(self.problem.constraint_matrix):
            for j, row2 in enumerate(self.problem.constraint_matrix):
                if i >= j:
                    continue
                A = np.array([row1, row2])
                b = np.array(
                    [
                        self.problem.constraint_bounds[i],
                        self.problem.constraint_bounds[j],
                    ]
                )
                if np.linalg.matrix_rank(A) == 2:  # Ensure the lines are not parallel
                    intersection = np.linalg.solve(A, b)
                    if all(
                        intersection >= 0
                    ):  # Only consider non-negative intersections
                        intersections.append(intersection)

        # Check intersections with the axes
        for i, row in enumerate(self.problem.constraint_matrix):
            if row[1] != 0:  # Intersection with y-axis (x=0)
                y_intercept = self.problem.constraint_bounds[i] / row[1]
                if y_intercept >= 0:
                    intersections.append([0, y_intercept])
            if row[0] != 0:  # Intersection with x-axis (y=0)
                x_intercept = self.problem.constraint_bounds[i] / row[0]
                if x_intercept >= 0:
                    intersections.append([x_intercept, 0])

        return intersections

    def _get_bounds(self, index):
        """Get the bounds for the x or y axis based on the problem constraints."""
        bounds = [0, 10]  # Default bounds
        for i, row in enumerate(self.problem.constraint_matrix):
            if row[index] != 0:
                bound = self.problem.constraint_bounds[i] / row[index]
                bounds.append(bound)
        return min(bounds), max(bounds)

    def plot(self):
        """Plot the feasible region and objective function for 2D problems."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Constraint lines
        x = np.linspace(0, 100, 400)
        for i, row in enumerate(self.problem.constraint_matrix):
            if row[1] != 0:
                ax.plot(
                    x,
                    (self.problem.constraint_bounds[i] - row[0] * x) / row[1],
                    label=f"Constraint {i+1}",
                )

        # Feasible region shading
        y = np.linspace(0, 100, 400)
        X, Y = np.meshgrid(x, y)
        feasible = np.ones_like(X, dtype=bool)
        for i, row in enumerate(self.problem.constraint_matrix):
            feasible &= row[0] * X + row[1] * Y <= self.problem.constraint_bounds[i]
        ax.contourf(X, Y, feasible, levels=1, colors=["lightgray"], alpha=0.5)

        # Plot objective function direction
        c = self.problem.objective_coeffs
        ax.quiver(
            0,
            0,
            c[0],
            c[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            label="Objective",
        )

        # Plot intersection points
        intersections = self._find_intersections()
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
        x_min, x_max = self._get_bounds(0)
        y_min, y_max = self._get_bounds(1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        plt.grid()
        plt.show()


class BNB_Solver(Solver):
    def __init__(self, problem, verbose=False):
        super().__init__(problem)
        self.verbose = verbose
        self.best_solution = None
        self.best_objective = float("-inf")
        self.graph = Digraph()
        self.node_counter = 0

    def is_integer(self, solution):
        """Check if all variables in the solution are integers."""
        return all(abs(x - round(x)) < 1e-6 for x in solution)

    def branch_and_bound(self):
        """Branch-and-Bound implementation."""
        queue = [(self.problem, 0)]  # Start with the root problem
        self.graph.node("0", label="Root", shape="circle")

        while queue:
            current_problem, parent_id = queue.pop(0)
            obj, sol = self.solve(verbose=self.verbose)

            if obj is None or obj <= self.best_objective:
                # Prune branch
                continue

            if self.is_integer(sol):
                # Update the best known integer solution
                if obj > self.best_objective:
                    self.best_solution = sol
                    self.best_objective = obj
                    if self.verbose:
                        print(f"New best solution: {sol} with objective {obj}")
                continue

            # Branch on the first fractional variable
            fractional_var = next(
                i for i, x in enumerate(sol) if abs(x - round(x)) >= 1e-6
            )
            frac_value = sol[fractional_var]

            # Create two branches
            lower_branch = self.problem.create_branch(
                fractional_var, "upper", np.floor(frac_value)
            )
            upper_branch = self.problem.create_branch(
                fractional_var, "lower", np.ceil(frac_value)
            )

            # Add branches to the queue with unique node IDs
            left_id = str(self.node_counter + 1)
            right_id = str(self.node_counter + 2)
            self.node_counter += 2

            queue.append((lower_branch, left_id))
            queue.append((upper_branch, right_id))

            # Update graph
            self.graph.node(
                left_id,
                label=f"x{fractional_var} ≤ {np.floor(frac_value)}",
                shape="circle",
            )
            self.graph.edge(str(parent_id), left_id)

            self.graph.node(
                right_id,
                label=f"x{fractional_var} ≥ {np.ceil(frac_value)}",
                shape="circle",
            )
            self.graph.edge(str(parent_id), right_id)

    def draw_graph(self, filename="branch_and_bound"):
        """Save the B&B tree as a graph image."""
        self.graph.render(filename, format="png", cleanup=True)
