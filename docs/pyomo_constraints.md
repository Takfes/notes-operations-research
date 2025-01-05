# How to define constraints in `Pyomo`

## `pyo.Constraint`
### `product mix`

$$
\begin{align}
    \text{max} \quad & \sum_{j \in J} c_j x_j \\
    \text{s.t.} \quad & \sum_{j \in J} a_{i, j} x_{j} \leq b_{i} & \forall \; i \in I \\
    & x_{j} \geq 0 & \forall \; j \in J \\
\end{align}
$$

```python
# Resource availablity constraints
def av_cstr(model, i):
    return sum(model.a[i,j] * model.x[j] for j in model.J) <= model.b[i]

model.av_cstr = pyo.Constraint(model.I, rule=av_cstr)
```

- constrainted applied on resources, saying that :
- for every resource (i),
- the quantity of that resource used in the production of all (that's why we are summing over j) products
- should be less than or equal to the availability of that resource (indexed by i -> b[i])
- the av_cstr rule `rule=av_cstr` is applied over all resources in the set I `model.I`

### `transporation`

$$
\begin{align}
    \text{min} \quad & \sum_{i \in I}\sum_{j \in J} c_{i, j} x_{i, j} \\
    \text{s.t.} \quad & \sum_{j \in J} x_{i, j} \leq b_{i} & \forall \; i \in I \\
    \quad & \sum_{i \in I} x_{i, j} = d_{j} & \forall \; j \in J \\
    & x_{i, j} \geq 0 & \forall \;i \in I, j \in J \\
\end{align}
$$

```python
# Demand equality constraints
def dem_cstr(model, j):
    return sum(model.x[i,j] for i in model.I) == model.d[j]
    # return sum(model.x[:,j]) == model.d[j]

model.dem_cstr = pyo.Constraint(model.J, rule=dem_cstr)

# Supplier availablity constraints
def av_cstr(model, i):
    return sum(model.x[i,j] for j in model.J) <= model.b[i]
    # return sum(model.x[i,:]) <= model.b[i]

model.av_cstr = pyo.Constraint(model.I, rule=av_cstr)
```

## `pyo.ConstraintList`

### `factory-planning`

```python
# create constraint : INVENTORY TARGET
model.cnstr_final_stock = pyo.ConstraintList()
for p in set_products:  # ! just for last month
    model.cnstr_final_stock.add(model.store[p, demand.shape[0]] == FINAL_STOCK)

# create constraint : MACHINE CAPACITY
model.cnstr_monthly_machine_hours = pyo.ConstraintList()
for m in set_machines:
    for t in set_time:
        machine_hours_expr = sum(
            model.make[p, t] * hours_per_machine_type_per_product[m, p]
            for p in set_products
            if hours_per_machine_type_per_product[m, p] > 0
        )
        # model.cnstr_monthly_machine_hours.add(machine_hours_expr >= 0)
        model.cnstr_monthly_machine_hours.add(
            machine_hours_expr
            <= operating_machines_per_machine_type_per_month[m, t] * HOURS_A_MONTH
        )
```