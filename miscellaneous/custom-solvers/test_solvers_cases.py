import pytest
import numpy as np
from solvers import IntegerSolver, OPTIMIZATION_PROBLEMS


@pytest.mark.parametrize("problem_info", OPTIMIZATION_PROBLEMS)
def test_integer_solver_results(problem_info):
    """Test that IntegerSolver produces expected results for all optimization problems"""
    problem = problem_info["object"]
    expected_z = problem_info["z"]
    expected_solution = problem_info["solution"]

    # Convert problem to integer optimization by setting variable types
    problem.variable_types = ["integer"] * len(problem.objective_coeffs)

    solver = IntegerSolver()
    z, solution = solver.solve(problem)

    assert z is not None, "Solver returned None for objective value"
    assert solution is not None, "Solver returned None for solution"
    assert np.isclose(z, expected_z, rtol=1e-4), f"Expected z={expected_z}, got z={z}"
    np.testing.assert_allclose(
        solution,
        expected_solution,
        rtol=1e-4,
        err_msg=f"Expected solution={expected_solution}, got solution={solution}",
    )

    # Verify solutions are integers
    assert all(
        np.isclose(x, round(x), rtol=1e-4) for x in solution
    ), f"Not all values in solution are integers: {solution}"
