# Logic to Constraints

## Logical Operations Overview

### Common Logical Operations
| Logic      | Expression | Interpretation        |
| ---------- | ---------- | --------------------  |
| NOT        | ¬x         | 1 if x = 0            |
| AND        | x ∧ y      | 1 if both true        |
| OR         | x ∨ y      | 1 if either true      |
| IMPLIES    | x → y      | “if x then y”         |
| EQUIVALENT | x ↔ y      | “x if and only if y”  |
| XOR        | x ⊕ y      | 1 if exactly one true |
| NAND       | ¬(x ∧ y)   | 1 if not both true    |
| NOR        | ¬(x ∨ y)   | 1 if neither true     |

### Truth Table for Common Logical Operations
| x | y | ¬x | x ∧ y | x ∨ y | x → y | x ↔ y | x ⊕ y | ¬(x ∧ y) | ¬(x ∨ y) |
|---|---|----|-------|-------|-------|-------|-------|----------|----------|
| 0 | 0 |  1 |   0   |   0   |   1   |   1   |   0   |    1     |    1     |
| 0 | 1 |  1 |   0   |   1   |   1   |   0   |   1   |    1     |    0     |
| 1 | 0 |  0 |   0   |   1   |   0   |   0   |   1   |    1     |    0     |
| 1 | 1 |  0 |   1   |   1   |   1   |   1   |   0   |    0     |    0     |


### Note on Implication (→)

- x→y can be interpreted as "if x is true, then y must also be true". 
- Alternatively, this reads the condition is false **or** the consequence holds, i.e., ¬x ∨ y.
- From a truth table perspective, the implication x→y is true whenever it does not violate its only requirement, which is that whenever x is true, y must also be true. In all other cases (x=0), the implication imposes no condition and is **vacuously true**.
- ¬x ∨ y (CNF format) gives the same truth values as x→y (Logical format).
- x ≤ y (Linear format) gives the same truth values as x→y (Logical format).
- Therefore, x→y ≡ ¬x ∨ y ≡ x ≤ y

## Resources

- [ChatGPT Discussion on Logic to Constraints Translation](https://chatgpt.com/g/g-p-6907c32186348191bf3c0ce867dc87ee-operations-research/c/69060466-dea4-832f-b00a-9faed25952d9)
- [Youtube Playlist - Modelling Theory](https://www.youtube.com/watch?v=JVjv0CCzHok&list=PLCip3d1iHEMX5JKm6ggM6mA9KTruV2w62)
- [Youtube Video - 0-1 Binary Constraints 1](https://www.youtube.com/watch?v=B3biWsBLeCw&list=PLCip3d1iHEMX5JKm6ggM6mA9KTruV2w62&index=1)
- [Youtube Video - 0-1 Binary Constraints 2](https://www.youtube.com/watch?v=MO8uQnIch6I&list=PLCip3d1iHEMX5JKm6ggM6mA9KTruV2w62&index=2)
- [Youtube Video - Either-Or, If-Then](https://www.youtube.com/watch?v=iQ3PlKKorXA&list=PLCip3d1iHEMX5JKm6ggM6mA9KTruV2w62&index=4)
- [Youtube Video - Conditional Constraints](https://www.youtube.com/watch?v=TuEX9vzMG5M&list=PLCip3d1iHEMX5JKm6ggM6mA9KTruV2w62&index=5)
- [Medium - A Comprehensive Guide to Modeling Techniques in Mixed-Integer Linear Programming](https://medium.com/data-science/a-comprehensive-guide-to-modeling-techniques-in-mixed-integer-linear-programming-3e96cc1bc03d)
- [Linearization Handbook for MILP Optimization](https://www.amazon.com/Linearization-Handbook-MILP-Optimization-Practitioners/dp/B0FLXMGZMJ)

---

## Translate Logical Expressions to Linear Inequalities

### **CNF** : Logical Expressions to CNF and then to Linear Constraints
- [Youtube Video - Convert logic into CNF](https://www.youtube.com/watch?v=Jf2T8RdCYfA)
- [Youtube Video - Conjunctive Normal Form (CNF) and Disjunctive Normal Form (DNF)](https://www.youtube.com/watch?v=2cgHa02s_SA)
- [Logic Conditions as constraints (CNF) - Part 1](https://www.solvermax.com/blog/logic-conditions-as-constraints-part-1)
- [Paper Logic Expressions to Linear Inequalities](https://www.solvermax.com/downloads/2008-Yunes-logical-expressions.pdf)
- [Wolfram Alpha Solver - Logic](https://www.wolframalpha.com/)

> *how to express "if and only if" (IFF) (x ↔ y) using Wolfram Alpha* <br>
> say I need to express `Z = (X AND Y)`, which is equivalent to `(X AND Y) IFF Z`<br>
> in Wolfram Alpha this can be expressed in any of the following ways: <br>
> * `((x and y) implies z) AND (z implies (x and y))`
> * `(x && y) \[Equivalent] z`
> * `(x and y) XNOR z`

### **Decomposition and Translation** for Logical Expressions to Linear Constraints
- [Logic conditions as constraints (D&T) - Part 2](https://www.solvermax.com/blog/logic-conditions-as-constraints-part-2)
- [Teaching Use of Binary Variables in Integer Linear Programs: Formulating Logical Conditions](https://pubsonline.informs.org/doi/epdf/10.1287/ited.2017.0177)
