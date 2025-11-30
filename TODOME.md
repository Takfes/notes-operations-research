
## Explore topics in more depth

- [coursera local search](./docs/local_search_notes.md) summarise local search course
- [udemy meta-heuristics](./Udemy%20:%20AI%20and%20Meta-Heuristics%20(Combinatorial%20Optimization)%20Python/) recap genetic algorithms and simulated annealing

- [OneTab Custom Collection](https://www.one-tab.com/page/tdDthPY_RHupORllzBV4qg)
    
    - big-M method
        - typical usecases for the big-M method? In some cases, although a certain condition can be expressed using the big-M method, it is not advised, since there are better alternatives, can you elaborate more on that?
        - can big-M method refer to something different rather than the switch-on/off logic? there are references suggesting that big-M method is a variation of simplex. what's this about?
    - how does warmstart work in MILP solvers?
        - This is vital in rolling horizon planning and re-optimization.
    - linearization tricks
        - piecewise linear functions
    - KKT conditions
    - cutting planes
        - Extra linear constraints added to eliminate fractional LP solutions without removing any integer feasible solutions — thus tightening the LP relaxation.
    - gomory cuts
        - A specific type of cutting plane derived directly from the simplex tableau. They are general-purpose cuts for integer programs (the “classical” cutting plane)
    - lagrangian relaxation
    - branch-and-bound vs branch-and-cut vs branch-and-price
        - branch-and-cut : Combination of Branch-and-Bound + Cutting Planes
        - branch-and-price : Combination of Branch-and-Bound + Column Generation
    - column generation
        - can you help me understand what's column generation, how does it work, where is it applicable, what problem does it solve, ideally provide toy example
        - [TDS - Column Generation](https://towardsdatascience.com/solving-a-resource-planning-problem-with-mathematical-programming-and-column-generation-07fc7dd21ca6/)
        - [Jump Blog - Julia - CG](https://jump.dev/JuMP.jl/stable/tutorials/algorithms/cutting_stock_column_generation/)
        - [Youtube Video - Branch & Price - Cutting Stock](https://www.youtube.com/watch?v=XvYlZGJpp2o&t=4s)
        - [Youtube Video - CG Cutting Stock](https://www.youtube.com/watch?v=O918V86Grhc&t=11s)
    - benders decomposition
    - dantzig-wolfe decomposition
    - duality and pricing problem
        - shadow prices (yi) : how much the objective function will improve if we relax a constraint (i) by one unit
        - reduced costs : for each variable (xj), how much its objective coefficient should improve before it becomes part of the optimal solution 
    - hungarian method
    - max clique problems
    - VNS (variable neighborhood search)
    - heuristics vs meta-heuristics