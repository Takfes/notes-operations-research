import numpy as np
from scipy.optimize import linprog
from graphviz import Digraph
import matplotlib.pyplot as plt


# Standalone LP Solver Function
def solve_lp_function(
    objective_coeffs,
    constraint_matrix,
    constraint_bounds,
    variable_bounds,
    verbose=False,
):
    """Standalone LP solver function."""
    result = linprog(
        c=-np.array(objective_coeffs),  # Negate for maximization
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
        return result.fun, result.x
    else:
        return None, None  # Infeasible


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


class Solver:
    @staticmethod
    def solve_lp(problem: OptimizationProblem, verbose: bool = False):
        """Static method for solving LP problems."""
        return solve_lp_function(
            problem.objective_coeffs,
            problem.constraint_matrix,
            problem.constraint_bounds,
            problem.variable_bounds,
            verbose,
        )

    @staticmethod
    def plot(problem: OptimizationProblem):
        """Plot the feasible region and objective function for 2D problems."""
        if len(problem.objective_coeffs) != 2:
            raise ValueError("Graphical plotting is only supported for 2D problems.")

        fig, ax = plt.subplots(figsize=(8, 8))

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
            scale=1,
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


class LinearSolver:
    def __init__(self):
        pass  # Decoupled from a specific problem

    def solve(self, problem: OptimizationProblem, verbose: bool = False):
        """Solve the linear relaxation of a given problem."""
        return Solver.solve_lp(problem, verbose)

    def plot(self, problem: OptimizationProblem):
        """Plot the feasible region and objective function for 2D problems."""
        Solver.plot(problem)


class MIP_BNB_Solver:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.best_solution = None
        self.best_objective = float("-inf")
        self.graph = Digraph()
        self.node_counter = 0

    def is_integer(self, solution):
        """Check if all variables in the solution are integers."""
        return all(abs(x - round(x)) < 1e-6 for x in solution)

    def _find_branch_variable(self, sol):
        branch_index = np.argmax([x - int(x) for x in sol])
        branch_value = sol[branch_index]
        return branch_value, branch_index

    def _find_branch_upper_bound(self, branch_value):
        return np.floor(branch_value)

    def _find_branch_lower_bound(self, branch_value):
        return np.ceil(branch_value)

    def solve(self, problem: OptimizationProblem):
        """Branch-and-Bound solver for the given problem."""
        queue = [(problem, 0)]  # Start with the root problem
        self.graph.node("0", label="Root", shape="circle")

        while queue:
            current_problem, parent_id = queue.pop(0)
            obj, sol = Solver.solve_lp(current_problem, verbose=self.verbose)

            if obj is None or obj <= self.best_objective:
                # Prune branch
                continue

            if self.is_integer(sol):
                # Update the best known integer solution
                if obj > self.best_objective:
                    self.best_solution = sol
                    self.best_objective = obj
                continue

            # Branch on the first fractional variable
            fractional_var = next(
                i for i, x in enumerate(sol) if abs(x - round(x)) >= 1e-6
            )
            frac_value = sol[fractional_var]

            # Create two branches
            lower_branch = current_problem.create_branch(
                fractional_var, "upper", np.floor(frac_value)
            )
            upper_branch = current_problem.create_branch(
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

        return self.best_objective, self.best_solution

        # while queue:
        #     current_problem, parent_id = queue.pop(0)
        #     obj, sol = Solver.solve_lp(current_problem, verbose=self.verbose)

        #     if obj is None or obj <= self.best_objective:
        #         # Prune branch
        #         continue

        #     if self.is_integer(sol):
        #         # Update the best known integer solution
        #         if obj > self.best_objective:
        #             self.best_solution = sol
        #             self.best_objective = obj
        #         continue

        #     # Find branch variable
        #     branch_value, branch_index = self._find_branch_variable(sol)
        #     branch_upper_bound = self._find_branch_upper_bound(branch_value)
        #     branch_lower_bound = self._find_branch_lower_bound(branch_value)

        #     lower_branch = current_problem.create_branch(
        #         branch_index, "upper", branch_upper_bound
        #     )
        #     upper_branch = current_problem.create_branch(
        #         branch_index, "lower", branch_lower_bound
        #     )

        #     # Add branches to the queue with unique node IDs
        #     left_id = str(self.node_counter + 1)
        #     right_id = str(self.node_counter + 2)
        #     self.node_counter += 2

        #     queue.append((lower_branch, left_id))
        #     queue.append((upper_branch, right_id))

        #     # Update graph
        #     self.graph.node(
        #         left_id,
        #         label=f"x{branch_index} ≤ {branch_upper_bound}",
        #         shape="circle",
        #     )
        #     self.graph.edge(str(parent_id), left_id)

        #     self.graph.node(
        #         right_id,
        #         label=f"x{branch_index} ≥ {branch_lower_bound}",
        #         shape="circle",
        #     )
        #     self.graph.edge(str(parent_id), right_id)

        # return self.best_objective, self.best_solution

    def draw_graph(self, filename="branch_and_bound"):
        """Save the B&B tree as a graph image."""
        self.graph.render(filename, format="png", cleanup=True)

    def plot(self, problem: OptimizationProblem):
        """Plot the feasible region and objective function for 2D problems."""
        Solver.plot(problem)
