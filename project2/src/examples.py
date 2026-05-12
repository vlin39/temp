from ortools.constraint_solver import pywrapcp


def run_australia():
    colors = ["red", "green", "blue"]
    solver = pywrapcp.Solver("AustraliaBinaryArray")
    wa = solver.IntVar(0, 2, "WA")
    nt = solver.IntVar(0, 2, "NT")
    sa = solver.IntVar(0, 2, "SA")
    q = solver.IntVar(0, 2, "Q")
    nsw = solver.IntVar(0, 2, "NSW")
    v = solver.IntVar(0, 2, "V")

    arr = [solver.IntVar(3, 5, f"arr_{i}") for i in range(3)]
    solver.Add(solver.AllDifferent(arr))

    solver.Add(wa != nt)
    solver.Add(wa != sa)
    solver.Add(nt != sa)
    solver.Add(nt != q)
    solver.Add(sa != q)
    solver.Add(sa != nsw)
    solver.Add(sa != v)
    solver.Add(q != nsw)
    solver.Add(nsw != v)

    all_vars = [wa, nt, sa, q, nsw, v] + arr


    db = solver.DefaultPhase(all_vars)
    solver.NewSearch(db)

    if solver.NextSolution():
        print()
        for name, var in [
            ("WesternAustralia", wa),
            ("NorthernTerritory", nt),
            ("SouthAustralia", sa),
            ("Queensland", q),
            ("NewSouthWales", nsw),
            ("Victoria", v),
        ]:
            print(f"{name}:    {colors[var.Value()]}")
        for x in arr:
            print(f"arr val is : {x.Value()}")
    else:
        print("No Solution found!")
    solver.EndSearch()


def run_send_more_money():
    solver = pywrapcp.Solver("SendMoreMoney")
    s = solver.IntVar(1, 9, "S")
    e = solver.IntVar(0, 9, "E")
    n = solver.IntVar(0, 9, "N")
    d = solver.IntVar(0, 9, "D")
    m = solver.IntVar(1, 9, "M")
    o = solver.IntVar(0, 9, "O")
    r = solver.IntVar(0, 9, "R")
    y = solver.IntVar(0, 9, "Y")

    vars_list = [s, e, n, d, m, o, r, y]
    solver.Add(solver.AllDifferent(vars_list))

    send = s * 1000 + e * 100 + n * 10 + d
    more = m * 1000 + o * 100 + r * 10 + e
    money = m * 10000 + o * 1000 + n * 100 + e * 10 + y
    solver.Add(money == send + more)

    db = solver.DefaultPhase(vars_list)
    solver.NewSearch(db)
    if solver.NextSolution():
        print(f"  {s.Value()} {e.Value()} {n.Value()} {d.Value()}")
        print(f"  {m.Value()} {o.Value()} {r.Value()} {e.Value()}")
        print(f"{m.Value()} {o.Value()} {n.Value()} {e.Value()} {y.Value()}")
    else:
        print("No Solution!")
    solver.EndSearch()


if __name__ == "__main__":
    import sys
    which = (sys.argv[1:] or ["australia"])[0].lower()
    if which == "australia":
        run_australia()
    elif which == "money":
        run_send_more_money()
    else:
        print("Usage: python cp_examples.py [australia|money]")
        print("  default: binary")
