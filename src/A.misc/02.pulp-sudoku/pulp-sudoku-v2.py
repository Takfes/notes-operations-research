import pandas as pd
from pulp import *

SHEETNAME = 'example2'

# read in sudoku - prefilled values
df = pd.read_excel('sudoku.xlsx',usecols='B:J',
                   header=None,skiprows=1,sheet_name=SHEETNAME)\
                       .fillna(0)

# convert to integers
fcols = df.select_dtypes('float').columns
df[fcols] = df[fcols].apply(pd.to_numeric, downcast='integer')

# track prefilled values to be used as constrains
side = range(df.shape[0])

input_data = []

for col in side:
    for row in side:
        tempval = df.values[row,col]
        if tempval:
            input_data.append((tempval,row+1,col+1))

# define dimensions for a typical sudoku
VALS = ROWS = COLS = range(1,10)

# define BOXES
BOXES = [
    [(3 * i + k + 1, 3 * j + l + 1) for k in range(3) for l in range(3)]
    for i in range(3)
    for j in range(3)
]

# The prob variable is created to contain the problem data
prob = LpProblem("Sudoku_Problem")

# The decision variables are created
choices = LpVariable.dicts("Choice", (VALS, ROWS, COLS), cat="Binary")

# ensure that a single variable is used per cell
for row in ROWS:
    for col in COLS:
        prob += lpSum([choices[val][row][col] for val in VALS]) == 1

# sudoku constrains 
for val in VALS:
    for row in ROWS:
        prob += lpSum([choices[val][row][col] for col in COLS]) == 1
    for col in COLS:
        prob += lpSum([choices[val][row][col] for row in ROWS]) == 1
    for box in BOXES:
        prob += lpSum([choices[val][row][col] for (row,col) in box]) == 1

# prefilled data constrains
for (val,row,col) in input_data:
    prob += choices[val][row][col] == 1

# The problem data is written to an .lp file
prob.writeLP("sudoku2.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# A file called sudokuout.txt is created/overwritten for writing to
sudokuout = open("sudokuout2.txt", "w")

# The solution is written to the sudokuout.txt file
for r in ROWS:
    if r in [1, 4, 7]:
        sudokuout.write("+-------+-------+-------+\n")
    for c in COLS:
        for v in VALS:
            if value(choices[v][r][c]) == 1:
                if c in [1, 4, 7]:
                    sudokuout.write("| ")
                sudokuout.write(str(v) + " ")
                if c == 9:
                    sudokuout.write("|\n")
sudokuout.write("+-------+-------+-------+")
sudokuout.close()



