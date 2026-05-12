"""
Verification-only helper (NOT used by the BnB solver).

The project requires that OR-Tools only be used as an LP solver, but the
handout's hint explicitly permits using its IP solver to *verify* the
optimum value found by Branch-and-Bound. This script does exactly that.

Usage:
    venv/bin/python3 src/verify_with_ip.py input/<file>.ip
"""
import sys
from pathlib import Path

import numpy as np

from ortools.linear_solver import pywraplp
from ipinstance import IPInstance


def solve_with_ip(inst: IPInstance) -> int:
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP IP solver not available")

    x = [solver.IntVar(0, 1, f"x_{i}") for i in range(inst.numTests)]

    for p in range(inst.numPairs):
        ct = solver.Constraint(1.0, solver.infinity())
        for i in inst.tests_for_pair[p]:
            ct.SetCoefficient(x[i], 1.0)

    obj = solver.Objective()
    for i in range(inst.numTests):
        obj.SetCoefficient(x[i], float(inst.costOfTest[i]))
    obj.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"IP solver did not find optimum (status={status})")
    return int(round(solver.Objective().Value()))


if __name__ == "__main__":
    path = sys.argv[1]
    inst = IPInstance(path)
    optimum = solve_with_ip(inst)
    print(f"{Path(path).name}: IP optimum = {optimum}")
