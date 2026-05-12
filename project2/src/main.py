import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from cpinstance import CPInstance
from model_timer import Timer

def main():
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit in seconds (default: none)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the solution against all constraints after solving.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "interleaved",
            "twophase",
            "impact",
            "twophase_ff_rand_max",
            "twophase_ff_min_max",
            "twophase_ff_center_max",
            "twophase_first_rand_max",
            "twophase_first_min_max",
            "twophase_ff_random_random",
            "twophase_size_max_max",
            "twophase_simple", 
            "twophase_ff_max", 
            ],
        default="impact",
        help="Search strategy: 'interleaved' (weighted-random, Python), 'twophase' (usingbuilt ins, two-phase), or 'impact' (basically tuning DefaultPhase).",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    filename = input_file.name

    instance = CPInstance(str(input_file))
    timer = Timer()
    timer.start()
    is_solution, n_fails, schedule = instance.solve(time_limit_seconds=args.time_limit, strategy=args.strategy)
    timer.stop()

    resultdict = {}
    resultdict["Instance"] = filename
    resultdict["Time"] = round(timer.getTime(), 2)
    resultdict["Result"] = str(n_fails)
    # Format: flat space-separated string of begin end pairs, employee by employee, day by day
    if is_solution and schedule:
        parts = []
        for e in range(len(schedule)):
            for d in range(len(schedule[e])):
                parts.append(str(schedule[e][d][0]))
                parts.append(str(schedule[e][d][1]))
        resultdict["Solution"] = " ".join(parts)

    # Pretty prints solution, uncomment to use
    # if is_solution:
    #     instance.prettyPrint(instance.numEmployees, instance.numDays, schedule)
    #     instance.generateVisualizerInput(instance.numEmployees, instance.numDays, schedule)
    print(json.dumps(resultdict))

    if args.check and is_solution:
        violations = instance.check_solution(schedule)
        if violations:
            print(f"CHECKER: {len(violations)} violation(s) found:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)

if __name__ == "__main__":
    main()
