## Multi-Objective Optimization Intuition

First, you only have a multi-objective problem if you NEED to create more than one objective function for your problem.

Multi-Objective problems have many concepts, I introduce some of them in this article. Also, I recommend additional reading in the topic.

Introduction to multi-objective problems

In the context of multi-objective problems, always note if the objective functions that you are modeling have the same nature or not?

This is important, **because if you have more than one objective function but they have the same nature, for example money, you may just sum them and have a single objective function.**

For example, suppose that you wish to minimize the investment during the purchase of a car and also you wish to minimize the maintenance cost of the car in the future. **You could create a single objective function as minimize investment + maintenance_cost, which is NOT a multi-objective problem.**

Now, suppose that you wish to minimize the investment in a car but at same time you wish to maximize the comfort level of the car. **In this case, the evaluation of a solution is not simple, because comfort and price have different natures. In this case, you would have two objective functions:**

Objective function 1: minimize investment

Objective function 2: maximize comfort

**You could create a single objective function, such as, minimize (investment - comfort), however you would not be able to study the possible different solutions that the combination of these two objective function can provide**. Therefore, since you have two objective functions and they have different natures, you should use a multi-objective approach.

For multi-objective problems we have specific algorithms and concepts. And an important concept is the pareto front. Suppose a problem where you have two objective functions (f1 and f2) both with minimization senses. Thus, we can plot all possible solutions and the Pareto Front as follow.

All possible solutions are defined as the "objective function space" or "solution space" and the Pareto Front is the set of non-dominated solutions. In this context, non-dominated solutions are solutions that you do not have other better solutions. Note in the pareto front we cannot tell if one blue solution is better then another, we can just tell that all the blue solutions are better then the red solutions.

Please, note that our objective is NOT minimize f1 + f2, but it is minimize f1 AND minimize f2. **Since f1 and f2 does not have the same nature is unfair if you try to sum them.**

To get the best solution you can define a strategy based on the pareto front, such as the closest point to the coordinate (0, 0), the middle point in the pareto front, among others.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a very well-known algorithm to solve multi-objective problems. [pymoo python package](https://pymoo.org/)
