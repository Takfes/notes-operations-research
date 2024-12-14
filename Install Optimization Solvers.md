## Installations

### glpk

- http://www.osemosys.org/uploads/1/8/5/0/18504136/glpk_installation_guide_for_windows10_-_201702.pdf

### highs

- https://groups.google.com/g/pyomo-forum/c/tApZf3i2cso

```bash
pip install highspy
```

### cbc

- https://stackoverflow.com/questions/60888032/how-to-install-cbc-for-pyomo-locally-on-windows-machine
- Visit the [Bintray repository](https://bintray.com/coin-or/download/Cbc) for COIN-OR.
- Download the latest CBC binary for Windows.
- Extract the zip file and ensure that the directory containing the `cbc.exe` file is added to your system's PATH.

```bash
brew install coin-or-tools/coinor/cbc
sudo apt-get install coinor-cbc
```

```python
solver = SolverFactory('cbc')
```

### ipopt

- https://stackoverflow.com/questions/59951763/install-ipopt-solver-to-use-with-pyomo-in-windows

```bash
brew install ipopt
pip install cyipopt
```

### cplex

- [Download CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
- `python /Applications/CPLEX_Studio_Community2211/python/setup.py install`
- `pyomo help --solvers`