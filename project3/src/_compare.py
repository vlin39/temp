"""Throwaway experiment: compare LB strengths on every input/*.scm instance.

Variants:
  A. Current LP                                  (baseline = results.log)
  B. LP + valid inequality v[f] >= x[c,f]
  C. MIP with v[f] integer (y, x continuous)
  D. Full IP: y, x in {0,1}, v integer
"""
from __future__ import annotations

import glob, math, sys, time
from typing import Tuple

import numpy as np
from ortools.linear_solver.python import model_builder

sys.path.insert(0, "src")
from lpinstance import LPInstance


def build_and_solve(inst: LPInstance, variant: str, time_limit_s: float = 60.0) -> Tuple[float, str, float]:
    model = model_builder.Model()
    solver = model_builder.Solver("SCIP")

    C, F = inst.numCustomers, inst.numFacilities
    V = inst.numMaxVehiclePerFacility

    if variant == "A":
        y = [model.new_num_var(0.0, 1.0, f"y_{f}") for f in range(F)]
        x = [[model.new_num_var(0.0, 1.0, f"x_{c}_{f}") for f in range(F)] for c in range(C)]
        v = [model.new_num_var(0.0, float(V), f"v_{f}") for f in range(F)]
    elif variant == "B":
        y = [model.new_num_var(0.0, 1.0, f"y_{f}") for f in range(F)]
        x = [[model.new_num_var(0.0, 1.0, f"x_{c}_{f}") for f in range(F)] for c in range(C)]
        v = [model.new_num_var(0.0, float(V), f"v_{f}") for f in range(F)]
    elif variant == "C":
        y = [model.new_num_var(0.0, 1.0, f"y_{f}") for f in range(F)]
        x = [[model.new_num_var(0.0, 1.0, f"x_{c}_{f}") for f in range(F)] for c in range(C)]
        v = [model.new_int_var(0, V, f"v_{f}") for f in range(F)]
    elif variant == "D":
        y = [model.new_int_var(0, 1, f"y_{f}") for f in range(F)]
        x = [[model.new_int_var(0, 1, f"x_{c}_{f}") for f in range(F)] for c in range(C)]
        v = [model.new_int_var(0, V, f"v_{f}") for f in range(F)]
    else:
        raise ValueError(variant)

    for c in range(C):
        model.add(sum(x[c][f] for f in range(F)) == 1)
    for c in range(C):
        for f in range(F):
            model.add(x[c][f] <= y[f])
    for f in range(F):
        model.add(sum(float(inst.demandC[c]) * x[c][f] for c in range(C)) <= float(inst.capacityF[f]) * y[f])
    for f in range(F):
        model.add(sum(float(inst.distanceCF[c, f]) * x[c][f] for c in range(C)) <= float(inst.truckDistLimit) * v[f])
    for f in range(F):
        model.add(v[f] <= float(V) * y[f])

    if variant in ("B",):
        for c in range(C):
            for f in range(F):
                model.add(v[f] >= x[c][f])

    opening = sum(float(inst.openingCostF[f]) * y[f] for f in range(F))
    service = sum(float(inst.allocCostCF[c, f]) * x[c][f] for c in range(C) for f in range(F))
    truck = float(inst.truckUsageCost) * sum(v[f] for f in range(F))
    model.minimize(opening + service + truck)

    solver.set_time_limit_in_seconds(time_limit_s)
    t0 = time.time()
    status = solver.solve(model)
    dt = time.time() - t0
    obj = solver.objective_value if status in (model_builder.SolveStatus.OPTIMAL, model_builder.SolveStatus.FEASIBLE) else None
    return obj, str(status).split('.')[-1], dt


def main():
    rows = []
    files = sorted(glob.glob("input/*.scm"))
    for f in files:
        I = LPInstance(f); I.solve()
        row = {"inst": f.split("/")[-1]}
        for v in ["A", "B", "C", "D"]:
            obj, st, dt = build_and_solve(I, v, time_limit_s=60.0)
            row[v] = (obj, st, dt)
        rows.append(row)
        print(f"{row['inst']:25s}", end=" ")
        for v in ["A", "B", "C", "D"]:
            obj, st, dt = row[v]
            tag = "" if st in ("OPTIMAL",) else f"({st[:3]})"
            print(f"{v}={math.ceil(obj) if obj is not None else '-':>7} [{dt:5.1f}s]{tag}", end="  ")
        print()

    print()
    print("Totals (ceil-sums, OPTIMAL or FEASIBLE-with-bound only):")
    for v in ["A", "B", "C", "D"]:
        total = sum(math.ceil(r[v][0]) for r in rows if r[v][0] is not None)
        print(f"  {v}: {total}")


if __name__ == "__main__":
    main()
