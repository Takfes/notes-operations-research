# Special Constraints

## Summary
- if-then (implication) constraints
- logical operations
- activate/deactivate constraints

<br>

---

<br>


## Implications

### Summary

- `y ⇒ x` · `if y then x` → $y \le x$
- `y ⇒ ¬ x` · `if y then not x`→ $y \le 1 - x$
- `¬ y ⇒ x` · `if not y then x`→ $1- y \le x$
- `¬ y ⇒ ¬ x` · `if not y then not x` → $x \le y$


### Process

- Rule : **y ⇒ x**
    - Logical form: $ \neg y \vee x $ · `If 𝑦 is false or 𝑥 is true, the statement holds`
    - Rewrite: $ (1 - y) + x \geq 1 $ · `The ≥ part suggests the statement holds`
    - Simplify: $ y \leq x $

- Rule : **y ⇒ ¬ x**
    - Logical form: $ \neg y \vee \neg x $
    - Rewrite: $ (1 - y) + (1 - x) \geq 1 $
    - Simplify: $ y \leq 1 - x $

- Rule : **¬ y ⇒ x**
    - Logical form: $ y \vee x $
    - Rewrite: $ y + x \geq 1 $
    - Rearrange: $ 1 - y \leq x $

- Rule: **¬ y ⇒ ¬ x**
    - Logical form: $ y \vee \neg x $
    - Rewrite: $ y + (1 - x) \geq 1 $
    - Simplify: $ x \leq y $

### Clarifications
The rule $ y \implies x $ only specifies behavior **when $ y = 1 $**:
- If $ y = 1 $: $ x $ must be $ 1 $ (to satisfy the implication).
- If $ y = 0 $: The rule **does not impose any requirement** on $ x $, so $ x $ can be either $ 0 $ or $ 1 $. This is because the implication $ y \implies x $ is always true when $ y = 0 $ (this is a quirk of logical implication).
- Thus, if $ y $ is false, $ x $'s value doesn’t matter because the 
#### Intuition:
- Think of $ y \implies x $ as a **promise**:
- If $ y $ happens, then $ x $ must also happen.
- If $ y $ doesn’t happen, you don’t break the promise, no matter what $ x $ does.


<br>

---

<br>

## Logical Operations

- **AND ($ z = x \wedge y $)**:
    - $ z $ is the **minimum** of $ x $ and $ y $.  
    - Derived by enforcing $ z \leq x $, $ z \leq y $, and $ z \geq x + y - 1 $.

- **OR ($ z = x \vee y $)**:
    - $ z $ is the **maximum** of $ x $ and $ y $.  
    - Derived by enforcing $ z \geq x $, $ z \geq y $, and $ z \leq x + y $.


<br>

---

<br>

## Big-M method

- used to turn constraints on and off
- $ ax \le b + M (1-y)$
<br>
```mermaid
graph TD
    A[y binary] -->|y=0| D[True]
    A[y binary] -->|y=1| E[a*x<=b]
```