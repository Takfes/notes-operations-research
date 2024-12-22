import unittest
import numpy as np
from solvers import (
    OptimizationProblem,
    LinearSolver,
    IntegerSolver,
    OPTIMIZATION_PROBLEMS,
    solve_lp_function,
)


class TestSolvers(unittest.TestCase):
    def setUp(self):
        """Set up test cases using OPTIMIZATION_PROBLEMS"""
        self.test_problems = OPTIMIZATION_PROBLEMS
        self.linear_solver = LinearSolver()
        self.integer_solver = IntegerSolver()

    def test_solve_lp_function(self):
        """Test the standalone LP solver function"""
        for problem_info in self.test_problems:
            problem = problem_info["object"]
            expected_z = problem_info["z"]
            expected_solution = problem_info["solution"]

            z, solution = solve_lp_function(
                problem.objective_coeffs,
                problem.constraint_matrix,
                problem.constraint_bounds,
                problem.variable_bounds,
                problem.objective_direction,
            )

            # Check if objective value matches expected
            self.assertIsNotNone(z)
            self.assertAlmostEqual(z, expected_z, places=4)

            # Check if solution matches expected
            self.assertIsNotNone(solution)
            np.testing.assert_array_almost_equal(solution, expected_solution, decimal=4)

    def test_linear_solver(self):
        """Test the LinearSolver class"""
        for problem_info in self.test_problems:
            problem = problem_info["object"]
            expected_z = problem_info["z"]
            expected_solution = problem_info["solution"]

            z, solution = self.linear_solver.solve(problem)

            # Check if objective value matches expected
            self.assertIsNotNone(z)
            self.assertAlmostEqual(z, expected_z, places=4)

            # Check if solution matches expected
            self.assertIsNotNone(solution)
            np.testing.assert_array_almost_equal(solution, expected_solution, decimal=4)

    def test_integer_solver(self):
        """Test the IntegerSolver class"""
        # Create a specific test case for integer optimization
        integer_problem = OptimizationProblem(
            objective_coeffs=[5, 4],
            constraint_matrix=[[2, 3], [2, 1]],
            constraint_bounds=[12, 6],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="max",
            variable_types=["integer", "integer"],  # Specify integer variables
        )

        expected_z = 18.0
        expected_solution = np.array([2.0, 2.0])

        z, solution = self.integer_solver.solve(integer_problem)

        # Check if objective value matches expected
        self.assertIsNotNone(z)
        self.assertAlmostEqual(z, expected_z, places=4)

        # Check if solution matches expected
        self.assertIsNotNone(solution)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=4)

        # Verify solution contains only integers
        for val in solution:
            self.assertAlmostEqual(val, round(val), places=4)

    def test_infeasible_problem(self):
        """Test solver behavior with infeasible problems"""
        infeasible_problem = OptimizationProblem(
            objective_coeffs=[1, 1],
            constraint_matrix=[[1, 0], [-1, 0]],  # Contradictory constraints
            constraint_bounds=[1, -2],  # x ≤ 1 and x ≥ 2 (infeasible)
            variable_bounds=[(None, None), (None, None)],
            objective_direction="max",
        )

        # Test with LinearSolver
        z, solution = self.linear_solver.solve(infeasible_problem)
        self.assertIsNone(z)
        self.assertIsNone(solution)

        # Test with IntegerSolver
        z, solution = self.integer_solver.solve(infeasible_problem)
        self.assertIsNone(z)
        self.assertIsNone(solution)

    def test_unbounded_problem(self):
        """Test solver behavior with unbounded problems"""
        unbounded_problem = OptimizationProblem(
            objective_coeffs=[1, 1],
            constraint_matrix=[[1, 1]],  # Only one constraint
            constraint_bounds=[1],
            variable_bounds=[(0, None), (0, None)],  # No upper bounds
            objective_direction="max",
        )

        # Test with LinearSolver
        z, solution = self.linear_solver.solve(unbounded_problem)
        self.assertIsNone(z)
        self.assertIsNone(solution)

    def test_optimization_direction(self):
        """Test both maximization and minimization problems"""
        # Test maximization
        max_problem = OptimizationProblem(
            objective_coeffs=[5, 4],
            constraint_matrix=[[2, 3], [2, 1]],
            constraint_bounds=[12, 6],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="max",
        )

        z_max, solution_max = self.linear_solver.solve(max_problem)
        self.assertIsNotNone(z_max)
        self.assertGreater(z_max, 0)  # Should be positive for this specific problem

        # Test minimization
        min_problem = OptimizationProblem(
            objective_coeffs=[5, 4],
            constraint_matrix=[[2, 3], [2, 1]],
            constraint_bounds=[12, 6],
            variable_bounds=[(0, None), (0, None)],
            objective_direction="min",
        )

        z_min, solution_min = self.linear_solver.solve(min_problem)
        self.assertIsNotNone(z_min)
        self.assertLess(z_min, z_max)  # Minimum should be less than maximum


if __name__ == "__main__":
    unittest.main()
