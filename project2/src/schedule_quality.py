
import json
import statistics
import sys
from pathlib import Path


def parse_sched(path: str):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()

    num_shifts = int(data["Business_numShifts"])
    demand_flat = [int(x) for x in data["Business_minDemandDayShift"].split()]
    demand = [demand_flat[i:i + num_shifts] for i in range(0, len(demand_flat), num_shifts)]

    return {
        "numWeeks": int(data["Business_numWeeks"]),
        "numDays": int(data["Business_numDays"]),
        "numEmployees": int(data["Business_numEmployees"]),
        "numShifts": num_shifts,
        "minDemandDayShift": demand,
        "minDailyOperation": int(data["Business_minDailyOperation"]),
        "minConsecutiveWork": int(data["Employee_minConsecutiveWork"]),
        "maxDailyWork": int(data["Employee_maxDailyWork"]),
        "minWeeklyWork": int(data["Employee_minWeeklyWork"]),
        "maxWeeklyWork": int(data["Employee_maxWeeklyWork"]),
        "maxConsecutiveNightShift": int(data["Employee_maxConsecutiveNigthShift"]),
        "maxTotalNightShift": int(data["Employee_maxTotalNigthShift"]),
    }


def parse_solution(solution_str: str, num_employees: int, num_days: int):
    vals = [int(x) for x in solution_str.split()]
    expected = num_employees * num_days * 2
    if len(vals) != expected:
        raise ValueError(f"Expected {expected} integers, got {len(vals)}")
    out = []
    idx = 0
    for e in range(num_employees):
        row = []
        for d in range(num_days):
            row.append((vals[idx], vals[idx + 1]))
            idx += 2
        out.append(row)
    return out


def infer_shift_and_hours(begin: int, end: int):
    if begin == -1 and end == -1:
        return 0, 0
    hours = end - begin
    if begin == 0:
        return 1, hours
    if begin == 8:
        return 2, hours
    if begin == 16:
        return 3, hours
    return -1, hours


def stdev_or_zero(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def analyze_one(instance_path: Path, rec: dict):
    inst = parse_sched(str(instance_path))
    sol = rec.get("Solution")
    if not sol or sol == "--":
        return {"instance": instance_path.name, "solved": False, "note": "No solution present."}

    sched = parse_solution(sol, inst["numEmployees"], inst["numDays"])
    num_days = inst["numDays"]
    num_employees = inst["numEmployees"]

    shifts = [[0] * num_days for _ in range(num_employees)]
    hours = [[0] * num_days for _ in range(num_employees)]
    malformed_pairs = 0
    cross_shift_pairs = 0

    for e in range(num_employees):
        for d in range(num_days):
            begin, end = sched[e][d]
            s, h = infer_shift_and_hours(begin, end)
            shifts[e][d] = s
            hours[e][d] = h
            if s == -1:
                malformed_pairs += 1
                continue
            if s == 0 and h != 0:
                malformed_pairs += 1
            if s in (1, 2, 3):
                if h < inst["minConsecutiveWork"] or h > inst["maxDailyWork"]:
                    malformed_pairs += 1
                if s == 1 and not (0 <= begin < end <= 8):
                    cross_shift_pairs += 1
                if s == 2 and not (8 <= begin < end <= 16):
                    cross_shift_pairs += 1
                if s == 3 and not (16 <= begin < end <= 24):
                    cross_shift_pairs += 1

    shift_slacks = []
    zero_backup_shiftdays = 0
    daily_operation_slacks = []
    days_with_some_zero_backup = 0

    for d in range(num_days):
        counts = [0] * inst["numShifts"]
        total_hours = 0
        for e in range(num_employees):
            counts[shifts[e][d]] += 1
            total_hours += hours[e][d]

        any_zero = False
        for s in range(1, inst["numShifts"]):
            req = inst["minDemandDayShift"][d][s]
            slack = counts[s] - req
            shift_slacks.append(slack)
            if slack == 0:
                zero_backup_shiftdays += 1
                any_zero = True
        if any_zero:
            days_with_some_zero_backup += 1
        daily_operation_slacks.append(total_hours - inst["minDailyOperation"])

    weekly_hours = []
    employee_total_hours = []
    undesirable_counts = []
    night_counts = []
    min_length_shifts = 0
    total_worked_shifts = 0
    evening_to_night = 0
    rest_gaps_under_12h = 0

    for e in range(num_employees):
        employee_total_hours.append(sum(hours[e]))
        undesirable_counts.append(sum(1 for d in range(num_days) if shifts[e][d] in (1, 3)))
        night_counts.append(sum(1 for d in range(num_days) if shifts[e][d] == 1))

        for d in range(num_days):
            if hours[e][d] > 0:
                total_worked_shifts += 1
                if hours[e][d] == inst["minConsecutiveWork"]:
                    min_length_shifts += 1

        for w in range(inst["numWeeks"]):
            start = 7 * w
            end = min(start + 7, num_days)
            weekly_hours.append(sum(hours[e][start:end]))

        for d in range(num_days - 1):
            if shifts[e][d] == 3 and shifts[e][d + 1] == 1:
                evening_to_night += 1
            b1, e1 = sched[e][d]
            b2, e2 = sched[e][d + 1]
            if b1 != -1 and b2 != -1:
                rest_gap = (24 - e1) + b2
                if rest_gap < 12:
                    rest_gaps_under_12h += 1

    training_violations = 0
    first_days = min(inst["numShifts"], num_days)
    if first_days == inst["numShifts"]:
        for e in range(num_employees):
            if len(set(shifts[e][:first_days])) != first_days:
                training_violations += 1

    consecutive_night_violations = 0
    total_night_violations = 0
    for e in range(num_employees):
        total_nights = sum(1 for d in range(num_days) if shifts[e][d] == 1)
        if total_nights > inst["maxTotalNightShift"]:
            total_night_violations += 1
        window = inst["maxConsecutiveNightShift"] + 1
        for d in range(num_days - window + 1):
            nights = sum(1 for dd in range(d, d + window) if shifts[e][dd] == 1)
            if nights > inst["maxConsecutiveNightShift"]:
                consecutive_night_violations += 1
                break

    return {
        "instance": instance_path.name,
        "solved": True,
        "time": rec.get("Time"),
        "result": rec.get("Result"),
        "coverage": {
            "min_shift_slack": min(shift_slacks) if shift_slacks else None,
            "avg_shift_slack": round(sum(shift_slacks) / len(shift_slacks), 3) if shift_slacks else None,
            "zero_backup_shiftdays": zero_backup_shiftdays,
            "days_with_some_zero_backup": days_with_some_zero_backup,
            "min_daily_operation_slack": min(daily_operation_slacks) if daily_operation_slacks else None,
            "avg_daily_operation_slack": round(sum(daily_operation_slacks) / len(daily_operation_slacks), 3) if daily_operation_slacks else None,
        },
        "hours": {
            "employee_total_min": min(employee_total_hours) if employee_total_hours else None,
            "employee_total_max": max(employee_total_hours) if employee_total_hours else None,
            "employee_total_mean": round(sum(employee_total_hours) / len(employee_total_hours), 3) if employee_total_hours else None,
            "employee_total_stdev": round(stdev_or_zero(employee_total_hours), 3),
            "weekly_hours_below_40": sum(1 for h in weekly_hours if h < 40),
            "weekly_hours_at_or_above_40": sum(1 for h in weekly_hours if h >= 40),
            "short_shift_fraction": round(min_length_shifts / total_worked_shifts, 4) if total_worked_shifts else None,
        },
        "fairness": {
            "undesirable_shift_min": min(undesirable_counts) if undesirable_counts else None,
            "undesirable_shift_max": max(undesirable_counts) if undesirable_counts else None,
            "undesirable_shift_stdev": round(stdev_or_zero(undesirable_counts), 3),
            "night_shift_min": min(night_counts) if night_counts else None,
            "night_shift_max": max(night_counts) if night_counts else None,
            "night_shift_stdev": round(stdev_or_zero(night_counts), 3),
        },
        "pattern_quality": {
            "evening_to_night_transitions": evening_to_night,
            "rest_gaps_under_12h": rest_gaps_under_12h,
        },
        "validity_warnings": {
            "malformed_pairs": malformed_pairs,
            "cross_shift_pairs": cross_shift_pairs,
            "training_violations": training_violations,
            "consecutive_night_violations": consecutive_night_violations,
            "total_night_violations": total_night_violations,
        },
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python schedule_quality.py <results.log> <input_dir> [--json]")
        raise SystemExit(1)

    results_path = Path(sys.argv[1])
    input_dir = Path(sys.argv[2])
    emit_json = "--json" in sys.argv[3:]

    rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst_name = rec.get("Instance")
            if not inst_name:
                continue
            inst_path = input_dir / inst_name
            if not inst_path.exists():
                rows.append({"instance": inst_name, "solved": False, "note": f"Missing {inst_path}"})
            else:
                rows.append(analyze_one(inst_path, rec))

    if emit_json:
        print(json.dumps(rows, indent=2))
        return

    for r in rows:
        print("=" * 72)
        print(r["instance"])
        if not r.get("solved"):
            print("  not solved:", r.get("note", "no solution"))
            continue

        print(f"  time={r['time']}  result={r['result']}")
        c = r["coverage"]
        h = r["hours"]
        f = r["fairness"]
        p = r["pattern_quality"]
        w = r["validity_warnings"]

        print("  coverage:")
        print(f"    min shift slack            : {c['min_shift_slack']}")
        print(f"    avg shift slack            : {c['avg_shift_slack']}")
        print(f"    zero-backup shift-days     : {c['zero_backup_shiftdays']}")
        print(f"    days with some zero backup : {c['days_with_some_zero_backup']}")
        print(f"    min daily-op slack         : {c['min_daily_operation_slack']}")
        print(f"    avg daily-op slack         : {c['avg_daily_operation_slack']}")

        print("  hours:")
        print(f"    employee hours min/max     : {h['employee_total_min']} / {h['employee_total_max']}")
        print(f"    employee hours mean        : {h['employee_total_mean']}")
        print(f"    employee hours stdev       : {h['employee_total_stdev']}")
        print(f"    weekly hours below 40      : {h['weekly_hours_below_40']}")
        print(f"    weekly hours >= 40         : {h['weekly_hours_at_or_above_40']}")
        print(f"    short-shift fraction       : {h['short_shift_fraction']}")

        print("  fairness:")
        print(f"    undesirable shifts min/max : {f['undesirable_shift_min']} / {f['undesirable_shift_max']}")
        print(f"    undesirable shifts stdev   : {f['undesirable_shift_stdev']}")
        print(f"    night shifts min/max       : {f['night_shift_min']} / {f['night_shift_max']}")
        print(f"    night shifts stdev         : {f['night_shift_stdev']}")

        print("  pattern quality:")
        print(f"    evening->night transitions : {p['evening_to_night_transitions']}")
        print(f"    rest gaps under 12h        : {p['rest_gaps_under_12h']}")

        print("  validity warnings:")
        print(f"    malformed pairs            : {w['malformed_pairs']}")
        print(f"    cross-shift pairs          : {w['cross_shift_pairs']}")
        print(f"    training violations        : {w['training_violations']}")
        print(f"    consecutive-night viol.    : {w['consecutive_night_violations']}")
        print(f"    total-night violations     : {w['total_night_violations']}")


if __name__ == "__main__":
    main()
