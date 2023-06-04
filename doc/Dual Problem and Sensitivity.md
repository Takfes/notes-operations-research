## Duality and Sensitivity Analysis

Duality is a fundamental concept in mathematical optimization, and in particular, linear programming. It provides the framework for understanding the relationships between two types of optimization problems known as the "primal problem" and its "dual problem".

**Primal Problem and Dual Problem:**

The primal problem is the original optimization problem that we want to solve. The dual problem is a related optimization problem derived from the primal problem.

- If the primal problem is a minimization problem, then the dual problem is a maximization problem, and vice versa.
- The objective function coefficients of the primal problem become the right-hand-side constants of the dual problem
- The right-hand-side constants of the primal problem become the objective function coefficients of the dual problem.
- Constraints in the primal problem correspond to variables in the dual problem, and vice versa.

**Connection between Dual Solution and Shadow Prices:**

The solution to the dual problem gives the shadow prices for the primal problem. This is one of the main reasons why the dual problem is important: it provides crucial information for sensitivity analysis in the primal problem.

Sensitivity analysis is a technique used in linear programming to understand how changes in the problem's parameters affect the optimal solution. It is particularly useful when we're interested in understanding how sensitive the solution is to changes in the problem data.

The dual problem is particularly useful for sensitivity analysis with respect to changes in the right-hand side values of the constraints. The reason is that the optimal value of the dual problem gives the shadow prices, which tell us how much the objective function value will change with a one-unit change in the right-hand side of a constraint, assuming all other problem data remain the same.

**Example and its Dual:**

**Primal Problem (P)**:
Maximize Z = 2X1 + 3X2
Subject to:
X1 + X2 ≤ 4
2X1 + X2 ≤ 5
X1, X2 ≥ 0

**Dual Problem (D)**:
Minimize W = 4Y1 + 5Y2
Subject to:
Y1 + 2Y2 ≥ 2
Y1 + Y2 ≥ 3
Y1, Y2 ≥ 0

**Shadow Prices:**

In the context of linear programming, shadow prices (also known as dual values or marginal costs) are the values of the dual variables at the optimal solution. They provide information about how much the objective function value would change with a one-unit change in the right-hand-side of a constraint.

Let's assume that the optimal solution of the dual problem is (Y1*, Y2*) = (2, 1), which gives W* = 4*2 + 5\*1 = 13.

This means that Y1* = 2 is the shadow price of the first constraint in the primal problem, and Y2* = 1 is the shadow price of the second constraint. So if the right-hand-side of the first constraint increases by one unit (from 4 to 5), the optimal value of the primal problem (Z*) would increase by Y1* = 2 units, assuming all other problem data remain the same. Similarly, if the right-hand-side of the second constraint increases by one unit (from 5 to 6), Z* would increase by Y2* = 1 units.

**Further Sensitivity Analysis**

For sensitivity analysis with respect to changes in the coefficients of the objective function or the constraint matrix, the situation is more complicated, and we generally need to use different techniques. For instance, we might use something called the "range of optimality" for changes in the objective function coefficients, or the "range of feasibility" for changes in the constraint matrix coefficients.

**Usefulness of Duality:**

The principle of duality is useful for several reasons:

1. **Lower Bounds**: In a maximization problem, the optimal value of the dual provides a lower bound to the optimal value of the primal. Similarly, in a minimization problem, the optimal value of the dual provides an upper bound to the optimal value of the primal.

2. **Sensitivity Analysis**: The dual problem can be used for sensitivity analysis. In particular, the dual variables (also known as shadow prices or dual values) can be used to analyze how the optimal solution changes with changes in the problem data.

3. **Computational Efficiency**: Sometimes, the dual problem is easier to solve than the primal problem.
