"""
Reference-only benchmark: solves the FULL INTEGER PROGRAM (not the LP
relaxation) so we can see how far our LP lower bound in results.log sits
below the true IP optimum.

This is NOT what gets submitted. The graded artifact is results.log, which
the handout requires to come from the LP relaxation. main.py / run.sh /
runAll.sh continue to call LPInstance.solve() unchanged. This script is
invoked directly (it does not go through run.sh) and writes benchmark.log
with the same JSON-per-line format as results.log.

Variables made integer:
  y[f] in {0,1}
  x[c,f] in {0,1}
  v[f] in {0, 1, ..., numCustomers}

On a per-instance time-out, the best primal solution SCIP found is an
*upper* bound on the optimum, NOT a valid lower bound. We therefore report
solver.best_objective_bound (the dual bound, always a valid LB), unioned
with the LP relaxation value as a floor.

Usage:
    ./venv/bin/python3 src/benchmark.py <input_file> [--tl 290]
    ./venv/bin/python3 src/benchmark.py --all input/ 290 benchmark.log
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from ortools.linear_solver.python import model_builder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpinstance import LPInstance


def solve_ip(inst: LPInstance, time_limit_s: float) -> dict:
    """Solve the full IP; return a dict with Result (ceil of best valid LB),
    Solution status, and timing/diagnostic info."""
    C, F = inst.numCustomers, inst.numFacilities
    V = inst.numMaxVehiclePerFacility

    model = model_builder.Model()
    solver = model_builder.Solver("SCIP")

    y = [model.new_int_var(0, 1, f"y_{f}") for f in range(F)]
    x = [[model.new_int_var(0, 1, f"x_{c}_{f}") for f in range(F)] for c in range(C)]
    v = [model.new_int_var(0, V, f"v_{f}") for f in range(F)]

    for c in range(C):
        model.add(sum(x[c][f] for f in range(F)) == 1)
    for c in range(C):
        for f in range(F):
            model.add(x[c][f] <= y[f])
    for f in range(F):
        model.add(
            sum(float(inst.demandC[c]) * x[c][f] for c in range(C))
            <= float(inst.capacityF[f]) * y[f]
        )
    for f in range(F):
        model.add(
            sum(float(inst.distanceCF[c, f]) * x[c][f] for c in range(C))
            <= float(inst.truckDistLimit) * v[f]
        )
    for f in range(F):
        model.add(v[f] <= float(V) * y[f])

    opening = sum(float(inst.openingCostF[f]) * y[f] for f in range(F))
    service = sum(
        float(inst.allocCostCF[c, f]) * x[c][f]
        for c in range(C)
        for f in range(F)
    )
    truck = float(inst.truckUsageCost) * sum(v[f] for f in range(F))
    model.minimize(opening + service + truck)

    solver.set_time_limit_in_seconds(time_limit_s)
    t0 = time.time()
    status = solver.solve(model)
    dt = time.time() - t0

    status_name = str(status).rsplit(".", 1)[-1]
    primal = solver.objective_value if status in (
        model_builder.SolveStatus.OPTIMAL,
        model_builder.SolveStatus.FEASIBLE,
    ) else None
    dual = None
    try:
        dual = solver.best_objective_bound
    except Exception:
        pass

    if status == model_builder.SolveStatus.OPTIMAL:
        # primal == dual == true optimum (within solver tolerance)
        lb = primal
        sol_tag = "OPT"
    elif primal is not None and dual is not None:
        # Time-out (or other non-optimal) with both bounds available.
        # The dual is the valid lower bound; the primal is an upper bound.
        lb = dual
        sol_tag = "DUAL"
    elif primal is not None:
        # We have a feasible solution but no dual bound. The primal is an
        # upper bound, not a lower bound, so we cannot safely report it.
        lb = None
        sol_tag = status_name
    else:
        lb = None
        sol_tag = status_name

    return {
        "status": status_name,
        "primal": primal,
        "dual": dual,
        "lb": lb,
        "sol_tag": sol_tag,
        "time": dt,
    }


def floor_with_lp(inst: LPInstance, ip_lb: float | None) -> tuple[int, str, float]:
    """Take max(LP LB, IP LB) as the final reported LB; both are valid."""
    inst.solve()
    lp_lb = float(inst.lp_objective_value) if inst.lp_objective_value is not None else None
    candidates = [b for b in (lp_lb, ip_lb) if b is not None]
    if not candidates:
        return -1, "NONE", lp_lb if lp_lb is not None else float("nan")
    best = max(candidates)
    return math.ceil(best), ("LP" if best == lp_lb else "IP"), best


def run_one(path: str, time_limit_s: float) -> dict:
    instance_name = os.path.basename(path)
    inst = LPInstance(path)
    t0 = time.time()
    ip = solve_ip(inst, time_limit_s=time_limit_s)
    result, source, raw = floor_with_lp(inst, ip["lb"])
    total_time = time.time() - t0
    return {
        "Instance": instance_name,
        "Time": f"{total_time:.2f}",
        "Result": result,
        "Solution": ip["sol_tag"],
        "_diag": {
            "status": ip["status"],
            "primal": ip["primal"],
            "dual": ip["dual"],
            "lb_source": source,
            "raw_lb": raw,
            "ip_time": ip["time"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="A .scm file or, with --all, a folder")
    parser.add_argument("--all", action="store_true",
                        help="Treat input_path as a folder of .scm files")
    parser.add_argument("--tl", type=float, default=290.0,
                        help="Per-instance time limit in seconds (default 290)")
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON-per-line log here (default benchmark.log when --all)")
    args = parser.parse_args()

    if args.all:
        files = sorted(glob.glob(os.path.join(args.input_path, "*.scm")))
        out_path = args.out or "benchmark.log"
        if os.path.exists(out_path):
            print(f"refusing to overwrite {out_path}", file=sys.stderr)
            return 1
        with open(out_path, "w") as fl:
            for f in files:
                print(f"# {f}", file=sys.stderr)
                row = run_one(f, time_limit_s=args.tl)
                diag = row.pop("_diag")
                line = json.dumps(row)
                fl.write(line + "\n")
                fl.flush()
                print(f"  {line}  ({diag})", file=sys.stderr)
        return 0
    else:
        row = run_one(args.input_path, time_limit_s=args.tl)
        diag = row.pop("_diag")
        print(json.dumps(row))
        print(f"# diag: {diag}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
