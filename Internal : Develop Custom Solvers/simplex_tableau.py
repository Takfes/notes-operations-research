import numpy as np

np.set_printoptions(suppress=True)


def simplex(c, A, b, verbose=False):
    """
    Simplex algorithm to solve linear programming problems.
    Maximize c^T x subject to Ax <= b, x >= 0.

    Parameters:
        c (np.array): Coefficients of the objective function.
        A (np.array): Constraint matrix.
        b (np.array): Constraint bounds.

    Returns:
        dict: Solution containing the optimal value and decision variables.
    """

    # TODO : check if this works for both maximization and minimization

    # Number of variables and constraints
    n_vars = len(c)
    n_constraints = len(b)

    # Add slack variables
    A = np.hstack([A, np.eye(n_constraints)])
    c = np.hstack([c, np.zeros(n_constraints)])

    # Initial tableau
    tableau = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b
    tableau[-1, :-1] = c

    horizontals = (
        [f"X{i+1}" for i in range(n_vars)]
        + [f"S{i+1}" for i in range(n_constraints)]
        + ["Z"]
    )
    verticals = [f"S{i+1}" for i in range(n_constraints)] + ["Z"]
    basics = [f"S{i+1}" for i in range(n_constraints)] + ["Z"]

    if verbose:
        print(f"Initial tableau:\n{tableau}")

    # Simplex iterations
    while True:
        # Check if optimal (no negative coefficients in the bottom row)
        if all(tableau[-1, :-1] >= 0):
            if verbose:
                print("Non-negative coefficients in the bottom row")
                print("Optimal solution found!")
            break

        # Pivoting: Find entering and leaving variables
        entering = np.argmin(tableau[-1, :-1])  # entering variable index on z row
        ratios = np.divide(
            tableau[:-1, -1],
            tableau[:-1, entering],
            out=np.full_like(tableau[:-1, -1], np.inf),
            where=tableau[:-1, entering] > 0,
        )
        leaving = np.argmin(ratios)  # leaving variable index on columns
        pivot = tableau[leaving, entering]

        # Print entering and leaving variables
        if verbose:
            print(
                f"Pivot row : {entering} Pivot column : {leaving} Pivot value : {pivot}"
            )
            print(f"Entering: {horizontals[entering]}, Leaving: {verticals[leaving]}")
            basics = (
                " ".join(basics)
                .replace(verticals[leaving], horizontals[entering])
                .split()
            )
            print(f"Basics: {basics}")

        # Update tableau
        # Divide pivot row by pivot value - therefore ensuring a 1 in the pivot position
        # Iterate all rows and subtract the above after multiplied by the value in the pivot column of the row - therefore ensuring a 0 in the pivot column
        tableau[leaving, :] /= pivot
        for i in range(tableau.shape[0]):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]

        if verbose:
            print("-" * 50)
            print(tableau)

    # Extract solution
    # TODO : Check if this works for all cases
    solution = np.zeros(n_vars + n_constraints)
    for j, x in enumerate(horizontals[:-1]):
        if x.startswith("X") and x in basics:
            index = basics.index(x)
            # print(f"{x} = {tableau[index, -1]}")
            solution[j] = tableau[index, -1]

    results = {
        "optimal_value": tableau[-1, -1],
        "decision_variables": solution[:n_vars],
        "tableau": tableau,
    }
    return results


# Example usage
c = np.array([-6, -5, -4])  # Objective function coefficients
A = np.array([[2, 1, 1], [1, 3, 2], [2, 1, 2]])  # Constraint matrix
b = np.array([240, 360, 300])  # Constraint bounds

result = simplex(c, A, b, verbose=True)
print("Optimal Value:", result["optimal_value"])
print("Decision Variables:", result["decision_variables"])
