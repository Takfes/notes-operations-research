## Simplex Method

**Standard Form**: Before applying the simplex method, the linear programming problem should be put into the canonical or standard form. In standard form, a linear programming problem has the following structure:

Objective Function: The problem seeks to maximize the objective function. If the original problem is a minimization problem, it can be converted to a maximization problem by multiplying the objective function by -1.

Constraints: All constraints are expressed as linear equations (i.e., they're equalities). If the original problem has inequality constraints, slack or surplus variables are added to convert them into equalities.

Non-negativity Condition: All decision variables are non-negative.

---

**Simplex Method Steps**: Here's a high-level overview of how the simplex method works:

Step 1 - Initialization: Begin with a feasible solution. This is usually obtained by setting all decision variables to zero, and the value of the slack/surplus variables is equal to the right-hand side of the corresponding constraint.

Step 2 - Optimality Check: Check if the current solution is optimal. This is done by examining the coefficients of the non-basic variables in the objective function row of the simplex table. If all coefficients are non-positive, then the current solution is optimal.

Step 3 - Pivot Operation: If the current solution isn't optimal, choose a non-basic variable with a positive coefficient in the objective function row to become a basic variable (entering variable). This is usually done using the largest coefficient rule or the first positive rule.

Step 4 - Feasibility Check: Choose a basic variable to become non-basic (leaving variable). This is done by performing the minimum ratio test.

Step 5 - Iteration: Update the simplex table by performing row operations to pivot on the element at the intersection of the entering variable column and the leaving variable row. Repeat Steps 2-5 until an optimal solution is found.

---

**Simplex Method Key Notions**

- **Slack and Excess Variables**: These are auxiliary variables added to linear programming problems to convert inequalities into equalities, which are easier to work with.

- **Slack Variables**: When you have a less-than-or-equal-to constraint (`≤`), you add a slack variable to the left-hand side of the constraint to make it an equality. Slack variables can be thought of as the unused resources in an inequality constraint. For example, if you have a constraint like `2x + 3y ≤ 5`, you would add a slack variable `s` to get `2x + 3y + s = 5`, where `s ≥ 0`.

- **Excess Variables**: These are similar to slack variables but are used for greater-than-or-equal-to constraints (`≥`). In this case, you subtract an excess variable from the left-hand side to make the constraint an equality. Excess variables can be thought of as the amount by which a constraint is exceeded.

- **Artificial Variables**: These are variables added to a problem to obtain an initial feasible solution, which is necessary for starting the simplex method. Artificial variables are introduced into constraints where there's no obvious way to create an initial feasible solution. They're usually removed from the problem after obtaining an initial solution.

- **Big M Method**: This is a method used in linear programming to handle constraints of the type `≥` or `=` in the simplex method, which requires an initial feasible solution. The Big M Method introduces artificial variables with a very large penalty (M) to the objective function. This penalty ensures that these variables will be removed from the final solution, as the goal of an optimization problem is to minimize (or maximize) the objective function.

The basic steps in the Big M method are:

- For each constraint where an artificial variable is needed, add the artificial variable to the constraint to make it an equality.

- In the objective function, add a term `-M * artificial_variable` (for a maximization problem) or `+M * artificial_variable` (for a minimization problem).

- Use the simplex method to solve the modified problem. The large penalty associated with the artificial variables will drive their values to zero if a feasible solution exists.

---

**Simplex Method Initialization**

**Maximize** Z = 2X1 + 3X2

**Subject to**

X1 + X2 ≤ 4 (Constraint 1)

X1 - X2 ≥ 2 (Constraint 2)

X1 + 2X2 = 3 (Constraint 3)

X1, X2 ≥ 0

First, we need to convert this problem to standard form. We introduce a slack variable S1 for the first constraint, a surplus variable S2 for the second constraint, and an artificial variable A1 for the third constraint:

**Maximize** Z = 2X1 + 3X2

**Subject to**

X1 + X2 + S1 = 4 (Constraint 1)

X1 - X2 - S2 = 2 (Constraint 2)

X1 + 2X2 + A1 = 3 (Constraint 3)

X1, X2, S1, S2, A1 ≥ 0

We also modify the objective function to account for the artificial variable using the Big M method. We penalize the objective function for the presence of the artificial variable by subtracting a large value M times the artificial variable:

**Maximize** Z = 2X1 + 3X2 - MA1

The resulting simplex tableau is as follows:

|     | X1  | X2  | S1  | S2  | A1  | RHS |
| --- | --- | --- | --- | --- | --- | --- |
| Z   | -2  | -3  | 0   | 0   | -M  | 0   |
| S1  | 1   | 1   | 1   | 0   | 0   | 4   |
| S2  | 1   | -1  | 0   | -1  | 0   | 2   |
| A1  | 1   | 2   | 0   | 0   | 1   | 3   |

Now, we would proceed with the simplex method performing pivot operations to improve the solution and maintain feasibility, until an optimal solution is found.

---

**Simplex Method Walkthrough**
Sure, let's consider a simple linear programming problem to illustrate the steps of the simplex method:

**Maximize** Z = 3X1 + 2X2

**Subject to**

X1 + 2X2 <= 14 (Constraint 1)

3X1 + 2X2 <= 18 (Constraint 2)

X1, X2 >= 0

First, we convert this problem to standard form:

**Maximize** Z = 3X1 + 2X2

**Subject to**

X1 + 2X2 + S1 = 14 (Constraint 1)

3X1 + 2X2 + S2 = 18 (Constraint 2)

X1, X2, S1, S2 >= 0

where S1 and S2 are slack variables that have been added to turn the inequalities into equalities. This problem can be represented in a simplex tableau as follows:

|     | X1  | X2  | S1  | S2  | RHS |
| --- | --- | --- | --- | --- | --- |
| Z   | 3   | 2   | 0   | 0   | 0   |
| S1  | 1   | 2   | 1   | 0   | 14  |
| S2  | 3   | 2   | 0   | 1   | 18  |

Now, let's proceed with the simplex method:

**Step 1 - Initialization**: The initial basic feasible solution is (X1, X2) = (0, 0) with Z = 0.

**Step 2 - Optimality Check**: The coefficients of X1 and X2 in the objective function row are positive, so the current solution isn't optimal.

**Step 3 - Pivot Operation**: Choose X1 as the entering variable (since it has the largest coefficient in the objective function row).

**Step 4 - Feasibility Check**: The minimum ratio test gives 14/1 = 14 for S1 and 18/3 = 6 for S2. So, S2 is the leaving variable.

**Step 5 - Iteration**: Perform row operations to pivot on the element at the intersection of the X1 column and the S2 row. We divide the second row by 3, then subtract the new second row from the first row multiplied by 3, and the third row multiplied by 1. This yields the following simplex tableau:

|     | X1  | X2  | S1  | S2   | RHS |
| --- | --- | --- | --- | ---- | --- |
| Z   | 0   | -1  | 0   | 1    | 18  |
| S1  | 0   | 1   | 1   | -1/3 | 8   |
| X1  | 1   | 2/3 | 0   | 1/3  | 6   |

**Step 2 - Optimality Check**: The coefficient of X2 in the objective function row is negative, so the current solution isn't optimal.

**Step 3 - Pivot Operation**: Choose X2 as the entering variable.

**Step 4 - Feasibility Check**: The minimum ratio test gives 8/1 = 8 for S1 and 6/(2/3) = 9 for X1. So, S1 is the leaving variable.

**Step 5 - Iteration**: Perform row operations to pivot on the element at the intersection of the X2 column and the S1 row. We multiply the second row by -1, then add the new second row to the first

row multiplied by 1, and the third row multiplied by -2/3. This yields the following simplex tableau:

|     | X1  | X2  | S1  | S2  | RHS |
| --- | --- | --- | --- | --- | --- |
| Z   | 0   | 0   | 1   | 1/3 | 20  |
| X2  | 0   | 1   | -1  | 1/3 | 8   |
| X1  | 1   | 0   | 2/3 | 1/9 | 2   |

**Step 2 - Optimality Check**: All coefficients in the objective function row are non-positive, so the current solution is optimal.

Therefore, the optimal solution is (X1, X2) = (2, 8) with Z = 20.

As you can see from this example, the simplex method iteratively improves the solution by moving along the edges of the feasible region in the direction that most increases (for a maximization problem) or decreases (for a minimization problem) the objective function, until an optimal solution is found.
