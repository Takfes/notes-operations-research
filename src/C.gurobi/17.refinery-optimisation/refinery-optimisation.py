import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def generate_data():
    data = {}
    # crude oils
    crude = {
        "Light naphtha": [0.1, 0.15],
        "Medium naphtha": [0.2, 0.25],
        "Heavy naphtha": [0.2, 0.18],
        "Light oil": [0.12, 0.08],
        "Heavy oil": [0.2, 0.19],
        "Residuum": [0.13, 0.12],
    }
    data["crude"] = pd.DataFrame(crude, index=["Crude 1", "Crude 2"])
    # profit contributions (in pence per barrel)
    profit = {
        "Product": ["Premium Gasoline", "Regular Gasoline", "Jet fuel", "Fuel oil", "Lube oil"],
        "Value": [700, 600, 400, 350, 150],
    }
    data["profit"] = pd.DataFrame(profit).set_index("Product")
    # return data
    return data


# extract data
data = generate_data()

crude_oils_df = data["crude"]
crude_oils = crude_oils_df.stack().to_dict()

profit_df = data["profit"]
profit = {k: (v / 100) for k, v in profit_df["Value"].to_dict().items()}

# define model
M = pyo.ConcreteModel()
SOLVER = "glpk"

# define parameters
# REFOMING
RT_NL_RG = 0.60  # RATIO NAFTHA LIGHT TO REFORMED GASOLINE
RT_NM_RG = 0.52  # RATIO NAFTHA MEDIUM TO REFORMED GASOLINE
RT_NH_RG = 0.45  # RATIO NAFTHA HEAVY TO REFORMED GASOLINE
# CRACKING
RT_OL_CO = 0.68  # RATIO OIL LIGHT TO CRACKED OIL
RT_OL_CG = 0.28  # RATIO OIL LIGHT TO CRACKED GASOLINE
RT_OH_CO = 0.75  # RATIO OIL HEAVY TO CRACKED OIL
RT_OH_CG = 0.20  # RATIO OIL HEAVY TO CRACKED GASOLINE
RT_RS_LO = 0.50  # RATIO RESIDUUM TO LUBE OIL

# BLENDING GASOLINE
# OCTANES
OCT_NL = 90
OCT_NM = 80
OCT_NH = 70
OCT_RG = 115
OCT_CG = 105
# OCTANE TARGETS
OCT_REGULAR_TARGET = 84
OCT_PREMIUM_TARGET = 94

# BLENDING JET FUEL
# VAPOR PRESSURE
VP_LO = 1.00
VP_HO = 0.60
VP_CO = 1.50
VP_RS = 0.05
# VAPOR TARGET
VP_JET_FUEL_TARGET = 1.0

# BLENDING FUEL OIL
# BLENDING RATIOS
BLRT_LO = 10
BLRT_HO = 3
BLRT_CO = 4
BLRT_RS = 1
# BLENDING RATIOS NORMALIZED
BLRT_TTL = BLRT_LO + BLRT_HO + BLRT_CO + BLRT_RS
RT_LO_FO = BLRT_LO / BLRT_TTL
RT_HO_FO = BLRT_HO / BLRT_TTL
RT_CO_FO = BLRT_CO / BLRT_TTL
RT_RS_FO = BLRT_RS / BLRT_TTL

# CONSTRAINT PARAMETERS
MAX_CRUD1 = 20_000
MAX_CRUD2 = 30_000
MAX_DISTL = 45_000
MAX_REFOR = 10_000
MAX_CRACK = 8_000
MAX_LUBEO = 500
MIN_LUBEO = 1_000
PREMvREGL = 0.4

"""
-----------------------------------------------
REFORMING :
Naphtha Heavy -> Reformed Gasoline
Naphtha Medium -> Reformed Gasoline
Naphtha Light -> Reformed Gasoline
-----------------------------------------------
CRACKING :
Oil Heavy -> Cracked Oil + Cracked Gasoline
Oil Light -> Cracked Oil + Cracked Gasoline
Residuum -> Lube Oil
-----------------------------------------------
BLENDING :
Regular Gasoline <- naphtha, reformed gasoline, and cracked gasoline
Premium Gasiline <- naphtha, reformed gasoline, and cracked gasoline
Jet Fuel <- light, heavy, cracked oils and residuum
Fuel Oil <- light, heavy, cracked oils and residuum
Lube Oil <- residuum
-----------------------------------------------
The daily availability of crude 1 is 20 000 barrels.
The daily availability of crude 2 is 30 000 barrels.
At most 45 000 barrels of crude can be distilled per day.
At most 10 000 barrels of naphtha can be reformed per day.
At most 8000 barrels of oil can be cracked per day.
The daily production of lube oil must be between 500 and 1000 barrels.
Premium gasoline production must be at least 40% of regular gasoline production.
-----------------------------------------------
"""
# define decision variables

# Quantity of barrels coming from each crude oil resource
M.CR1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, MAX_CRUD1))
M.CR2 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, MAX_CRUD2))

# Quantity of barrels of each product coming from Distillation
M.LN = pyo.Var(domain=pyo.NonNegativeReals)
M.MN = pyo.Var(domain=pyo.NonNegativeReals)
M.HN = pyo.Var(domain=pyo.NonNegativeReals)
M.LO = pyo.Var(domain=pyo.NonNegativeReals)
M.HO = pyo.Var(domain=pyo.NonNegativeReals)
M.R = pyo.Var(domain=pyo.NonNegativeReals)

# Quantity of barrels of each product intended for Reforming
M.LNRG = pyo.Var(domain=pyo.NonNegativeReals)
M.MNRG = pyo.Var(domain=pyo.NonNegativeReals)
M.HNRG = pyo.Var(domain=pyo.NonNegativeReals)
# Quantity of barrels of each product coming from Reforming
M.RG = pyo.Var(domain=pyo.NonNegativeReals)

# Quantity of barrels of each product intended for Cracking
M.LOCGO = pyo.Var(domain=pyo.NonNegativeReals)
M.HOCGO = pyo.Var(domain=pyo.NonNegativeReals)
M.RLBO = pyo.Var(domain=pyo.NonNegativeReals)
# Quantity of barrels of each product coming from Cracking
M.CG = pyo.Var(domain=pyo.NonNegativeReals)
M.CO = pyo.Var(domain=pyo.NonNegativeReals)
M.LBO = pyo.Var(domain=pyo.NonNegativeReals, bounds=(MIN_LUBEO, MAX_LUBEO))

# Quantity of barrels of each product intended for Blending Premium Gasoline
M.LNPMF = pyo.Var(domain=pyo.NonNegativeReals)
M.MNPMF = pyo.Var(domain=pyo.NonNegativeReals)
M.HNPMF = pyo.Var(domain=pyo.NonNegativeReals)
M.RGPMF = pyo.Var(domain=pyo.NonNegativeReals)
M.CGPMF = pyo.Var(domain=pyo.NonNegativeReals)
# Quantity of barrels of each product coming from Blending Premium Gasoline
M.PMF = pyo.Var(domain=pyo.NonNegativeReals)

# Quantity of barrels of each product intended for Blending Regural Gasoline
M.LNRMF = pyo.Var(domain=pyo.NonNegativeReals)
M.MNRMF = pyo.Var(domain=pyo.NonNegativeReals)
M.HNRMF = pyo.Var(domain=pyo.NonNegativeReals)
M.RGRMF = pyo.Var(domain=pyo.NonNegativeReals)
M.CGRMF = pyo.Var(domain=pyo.NonNegativeReals)
# Quantity of barrels of each product coming from Blending Regural Gasoline
M.RMF = pyo.Var(domain=pyo.NonNegativeReals)

# Quantity of barrels of each product intended for Blending Jet Fuel
M.LOJF = pyo.Var(domain=pyo.NonNegativeReals)
M.HOJF = pyo.Var(domain=pyo.NonNegativeReals)
M.RJF = pyo.Var(domain=pyo.NonNegativeReals)
M.COJF = pyo.Var(domain=pyo.NonNegativeReals)
# Quantity of barrels of each product coming from Blending Jet Fuel
M.JF = pyo.Var(domain=pyo.NonNegativeReals)

# Quantity of barrels of each product coming from Blending Fuel Oil
M.FO = pyo.Var(domain=pyo.NonNegativeReals)

# define objective function
M.obj = pyo.Objective(
    expr=M.RMF * profit["Regular Gasoline"]
    + M.PMF * profit["Premium Gasoline"]
    + M.JF * profit["Jet fuel"]
    + M.FO * profit["Fuel oil"]
    + M.LBO * profit["Lube oil"],
    sense=pyo.maximize,
)

# define constraints

# ------------------------------------------------
# CAP CONSTRAINTS
# ------------------------------------------------
# MAX DISTILLATION CAP
M.CNS_DISTIL_CAP = pyo.Constraint(expr=M.CR1 + M.CR2 <= MAX_DISTL)
# MAX REFORMING CAP
M.CNS_REFORM_CAP = pyo.Constraint(expr=M.LNRG + M.MNRG + M.HNRG <= MAX_REFOR)
# MAX CRACKING CAP
M.CNS_CRACK_CAP = pyo.Constraint(expr=M.LOCGO + M.HOCGO <= MAX_CRACK)

# ------------------------------------------------
# YIELD CONSTRAINTS
# ------------------------------------------------
# DISTILLATION YIELD
M.CNS_YIELD_LN = pyo.Constraint(
    expr=M.CR1 * crude_oils["Crude 1", "Light naphtha"]
    + M.CR2 * crude_oils["Crude 2", "Light naphtha"]
    == M.LN
)
M.CNS_YIELD_MN = pyo.Constraint(
    expr=M.CR1 * crude_oils["Crude 1", "Medium naphtha"]
    + M.CR2 * crude_oils["Crude 2", "Medium naphtha"]
    == M.MN
)
M.CNS_YIELD_HN = pyo.Constraint(
    expr=M.CR1 * crude_oils["Crude 1", "Heavy naphtha"]
    + M.CR2 * crude_oils["Crude 2", "Heavy naphtha"]
    == M.HN
)
M.CNS_YIELD_LOI = pyo.Constraint(
    expr=M.CR1 * crude_oils["Crude 1", "Light oil"] + M.CR2 * crude_oils["Crude 2", "Light oil"]
    == M.LO
)
M.CNS_YIELD_HOI = pyo.Constraint(
    expr=M.CR1 * crude_oils["Crude 1", "Heavy oil"] + M.CR2 * crude_oils["Crude 2", "Heavy oil"]
    == M.HO
)
M.CNS_YIELD_RS = pyo.Constraint(
    expr=M.CR1 * crude_oils["Crude 1", "Residuum"] + M.CR2 * crude_oils["Crude 2", "Residuum"]
    == M.R
)
# REFORMING YIELD
M.CNS_YIELD_LNRG = pyo.Constraint(
    expr=M.LNRG * RT_NL_RG + M.MNRG * RT_NM_RG + M.HNRG * RT_NH_RG == M.RG
)
# CRACKING YIELD
M.CNS_YIELD_CO = pyo.Constraint(expr=M.LOCGO * RT_OL_CO + M.HOCGO * RT_OH_CO == M.CO)
M.CNS_YIELD_CG = pyo.Constraint(expr=M.LOCGO * RT_OL_CG + M.HOCGO * RT_OH_CG == M.CG)
M.CNS_YIELD_LUBEO = pyo.Constraint(expr=M.RLBO * RT_RS_LO == M.LBO)
# BLENDING YIELD
M.CNS_YIELD_RG = pyo.Constraint(expr=M.LNRMF + M.MNRMF + M.HNRMF + M.RGRMF + M.CGRMF == M.RMF)
M.CNS_YIELD_PG = pyo.Constraint(expr=M.LNPMF + M.MNPMF + M.HNPMF + M.RGPMF + M.CGPMF == M.PMF)
M.CNS_YIELD_JF = pyo.Constraint(expr=M.LOJF + M.HOJF + M.COJF + M.RJF == M.JF)

# ------------------------------------------------
# BALANCE CONSTRAINTS
# ------------------------------------------------
M.CNS_BAL_LN = pyo.Constraint(expr=M.LNRMF + M.LNPMF + M.LNRG == M.LN)
M.CNS_BAL_MN = pyo.Constraint(expr=M.MNRMF + M.MNPMF + M.MNRG == M.MN)
M.CNS_BAL_HN = pyo.Constraint(expr=M.HNRMF + M.HNPMF + M.HNRG == M.HN)

M.CNS_BAL_LO = pyo.Constraint(expr=M.LOCGO + M.LOJF + M.FO * RT_LO_FO == M.LO)
M.CNS_BAL_HO = pyo.Constraint(expr=M.HOCGO + M.HOJF + M.FO * RT_HO_FO == M.HO)
M.CNS_BAL_CO = pyo.Constraint(expr=M.COJF + M.FO * RT_CO_FO == M.CO)
M.CNS_BAL_RS = pyo.Constraint(expr=M.RLBO + M.RJF + M.FO * RT_RS_FO == M.R)

M.CNS_BAL_RG = pyo.Constraint(expr=M.RGRMF + M.RGPMF == M.RG)
M.CNS_BAL_CG = pyo.Constraint(expr=M.CGRMF + M.CGPMF == M.CG)

# ------------------------------------------------
# QUALITY CONSTRAINTS
# ------------------------------------------------
# OCTANES REGULAR GASOLINE
M.CNS_OCT_RG = pyo.Constraint(
    expr=M.LNRMF * OCT_NL
    + M.MNRMF * OCT_NM
    + M.HNRMF * OCT_NH
    + M.RGRMF * OCT_RG
    + M.CGPMF * OCT_CG
    >= OCT_REGULAR_TARGET * M.RMF
)
# OCTANES PREMIUM GASOLINE
M.CNS_OCT_PG = pyo.Constraint(
    expr=M.LNPMF * OCT_NL
    + M.MNPMF * OCT_NM
    + M.HNPMF * OCT_NH
    + M.RGPMF * OCT_RG
    + M.CGPMF * OCT_CG
    >= OCT_PREMIUM_TARGET * M.PMF
)
# VAPOUR PRESSURE JET FUEL
M.CNS_VP_JF = pyo.Constraint(
    expr=M.LOJF * VP_LO + M.HOJF * VP_HO + M.COJF * VP_CO + M.RJF * VP_RS
    <= VP_JET_FUEL_TARGET * M.JF
)

# ------------------------------------------------
# REGULAR TO PREMIUM RATIO
# ------------------------------------------------
M.CNS_REG_TO_PREM = pyo.Constraint(expr=M.PMF >= PREMvREGL * M.RMF)

# solve model
# M.pprint()
opt = SolverFactory(SOLVER)
results = opt.solve(M)
results["Solver"][0]["Termination condition"].value

# output results
if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    # Access other solution information as needed
    print(f"Profit ${pyo.value(M.obj):=.2f}")
    print(75 * "-")
