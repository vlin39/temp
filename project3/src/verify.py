"""
Standalone verifier for the supply-chain LP relaxation.

Runs LPInstance(input_file).solve() (same code path as src/main.py), then
independently re-checks every modeled constraint against the fractional
solution returned by SCIP and recomputes the objective from the fractional
y, x, v arrays.

This script is NOT invoked from run.sh / runAll.sh; it exists purely as a
sanity check, kept out of the grader-facing hot path so that ./run.sh's
output remains a single JSON line as the handout requires.

Usage:
    ./venv/bin/python3 src/verify.py <input_file> [--tol 1e-6]

Exit code 0 if the solution satisfies every constraint within tolerance and
the recomputed objective matches solver.objective_value; exit code 1
otherwise (with one violation per line printed to stdout).
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser

import numpy as np

from lpinstance import LPInstance


def verify(instance: LPInstance, tol: float = 1e-6) -> list[str]:
    violations: list[str] = []

    if instance.y_sol is None or instance.x_sol is None or instance.v_sol is None:
        violations.append("no fractional solution stored (solver did not return OPTIMAL)")
        return violations

    y = instance.y_sol
    x = instance.x_sol
    v = instance.v_sol

    C = instance.numCustomers
    F = instance.numFacilities
    V = float(instance.numMaxVehiclePerFacility)

    demand = instance.demandC.astype(float)
    capacity = instance.capacityF.astype(float)
    opening = instance.openingCostF.astype(float)
    alloc = instance.allocCostCF.astype(float)
    dist = instance.distanceCF.astype(float)
    T = float(instance.truckDistLimit)
    u = float(instance.truckUsageCost)

    # (0) Variable bounds.
    if np.any(y < -tol) or np.any(y > 1.0 + tol):
        bad = np.where((y < -tol) | (y > 1.0 + tol))[0]
        for f in bad:
            violations.append(f"y[{f}]={y[f]:.6g} out of [0,1]")
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        bad = np.argwhere((x < -tol) | (x > 1.0 + tol))
        for c, f in bad:
            violations.append(f"x[{c},{f}]={x[c, f]:.6g} out of [0,1]")
    if np.any(v < -tol) or np.any(v > V + tol):
        bad = np.where((v < -tol) | (v > V + tol))[0]
        for f in bad:
            violations.append(f"v[{f}]={v[f]:.6g} out of [0,{V}]")

    # (1) Full-service: sum_f x[c,f] == 1.
    row_sums = x.sum(axis=1)
    for c in range(C):
        if abs(row_sums[c] - 1.0) > tol:
            violations.append(
                f"customer {c}: sum_f x[c,f] = {row_sums[c]:.6g}, expected 1"
            )

    # (2) Open-before-serve: x[c,f] <= y[f].
    diff = x - y[np.newaxis, :]
    bad = np.argwhere(diff > tol)
    for c, f in bad:
        violations.append(
            f"open-before-serve: x[{c},{f}]={x[c, f]:.6g} > y[{f}]={y[f]:.6g}"
        )

    # (3) Facility capacity: sum_c demand[c] * x[c,f] <= capacity[f] * y[f].
    served_demand = demand @ x  # shape (F,)
    cap_avail = capacity * y
    for f in range(F):
        if served_demand[f] > cap_avail[f] + tol:
            violations.append(
                f"capacity f={f}: demand_served={served_demand[f]:.6g} "
                f"> capacity[{f}]*y[{f}]={cap_avail[f]:.6g}"
            )

    # (4) Vehicle workload: sum_c dist[c,f] * x[c,f] <= truckDistLimit * v[f].
    workload = (dist * x).sum(axis=0)  # shape (F,)
    workload_avail = T * v
    for f in range(F):
        if workload[f] > workload_avail[f] + tol:
            violations.append(
                f"workload f={f}: total_distance={workload[f]:.6g} "
                f"> truckDistLimit*v[{f}]={workload_avail[f]:.6g}"
            )

    # (5) Vehicle-requires-open: v[f] <= V * y[f].
    bad = np.where(v > V * y + tol)[0]
    for f in bad:
        violations.append(
            f"vehicle-requires-open f={f}: v[{f}]={v[f]:.6g} > V*y[{f}]={V * y[f]:.6g}"
        )

    # (7) Recomputed objective matches solver.objective_value.
    recomputed = float(opening @ y) + float((alloc * x).sum()) + u * float(v.sum())
    reported = float(instance.lp_objective_value)
    if abs(recomputed - reported) > max(tol, tol * abs(reported)):
        violations.append(
            f"objective mismatch: recomputed={recomputed:.6g}, "
            f"solver.objective_value={reported:.6g}"
        )

    return violations


def main() -> int:
    parser = ArgumentParser(description="Verify LP relaxation solution")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    instance = LPInstance(args.input_file)
    instance.solve()

    violations = verify(instance, tol=args.tol)
    if violations:
        for msg in violations:
            print(msg)
        return 1

    print(f"OK obj={instance.lp_objective_value:.6g} ceil={instance.objective_value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
