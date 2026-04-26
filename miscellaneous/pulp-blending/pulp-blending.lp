\* The_Whiskas_Problem *\
Minimize
Objective_Function_:_Minimize_total_cost: 0.008 ingredients_Beef
 + 0.013 ingredients_Chicken + 0.001 ingredients_Gel + 0.01 ingredients_Mutton
 + 0.002 ingredients_Rice + 0.005 ingredients_Wheat_bran
Subject To
Fat_Requirements: 0.1 ingredients_Beef + 0.08 ingredients_Chicken
 + 0.11 ingredients_Mutton + 0.01 ingredients_Rice
 + 0.01 ingredients_Wheat_bran >= 6
Fibre_Requirements: 0.005 ingredients_Beef + 0.001 ingredients_Chicken
 + 0.003 ingredients_Mutton + 0.1 ingredients_Rice
 + 0.15 ingredients_Wheat_bran <= 2
Protein_Requirements: 0.2 ingredients_Beef + 0.1 ingredients_Chicken
 + 0.15 ingredients_Mutton + 0.04 ingredients_Wheat_bran >= 8
Salt_Requirements: 0.005 ingredients_Beef + 0.002 ingredients_Chicken
 + 0.007 ingredients_Mutton + 0.002 ingredients_Rice
 + 0.008 ingredients_Wheat_bran <= 0.4
Total_Quantity: ingredients_Beef + ingredients_Chicken + ingredients_Gel
 + ingredients_Mutton + ingredients_Rice + ingredients_Wheat_bran = 100
End
