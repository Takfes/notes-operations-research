*Branch and Bound:*

- Solve the relaxed problem (ignore integer constraints).
- Branch on fractional variables, dividing the problem into subproblems.
- Use bounds from the relaxed solution to prune nodes in the search tree.

*Gomory Cuts:*

- Cuts are derived from the fractional rows in the simplex tableau of the relaxed problem.
- You always solve the relaxed problem first before generating Gomory cuts or other cuts.Ignore integer constraints temporarily to solve the LP relaxation. Then identify fractional variables
- Add constraints (cuts) to exclude fractional solutions without eliminating integer-feasible solutions.
- Solve the relaxed problem again with the new cuts applied.Branch on variables as needed, solving subproblems iteratively.
- Tightens the relaxed feasible region, reducing the need for excessive branching.

*Branch and Cut:*

- Combines Branch and Bound with Gomory cuts (and other cutting planes).
- At each node, solve the relaxed problem, generate cuts to improve bounds, then re-solve before deciding to branch.
- Specialized cuts (Cover Inequalities, Clique Cuts, Lifted Inequalities) leverage problem-specific properties for more efficient pruning.
- The process alternates between solving, cutting, and branching, leveraging tighter bounds to minimize the search tree.

*Overall efficiency*
- Adding cuts & re-solving the relaxed problem, does take additional computation time. However, this re-solving is beneficial for a few reasons:
- Revisiting the relaxed problem with additional cuts tightens bounds, saving time in the overall tree exploration.
- Tighter Bounds: Cuts restrict the solution space, eliminating fractional solutions and making the relaxed problem closer to the integer feasible region. Subsequent branching becomes more effective. Nodes can be pruned earlier due to tighter bounds.
- Reduced Tree Exploration: Adding cuts reduces the number of nodes explored because. The feasible region shrinks.Fewer branches are needed to converge to an integer solution.