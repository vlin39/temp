from __future__ import annotations

import json
import sys
from typing import Optional, List, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp

from search import build_search



class CPInstance:
    # BUSINESS parameters
    numWeeks: int ## SP: Is it days + weeks or either?
    numDays: int
    numEmployees: int
    numShifts: int  ## The off shift is denoted by 0 while work shifts, night, day, and evening are denoted by 1, 2, and 3 respectively. 
    numIntervalsInDay: int 
    minDemandDayShift: list[list[int]]  ## e.g. minDemandDayShift[d][s] = 2 means that on day d at least 2 employees should be working day shift s.
    minDailyOperation: int  ## a minimum demand needs to be met to ensure the daily operation for every day when considering all employees and shifts.
    
    # EMPLOYEE parameters
    minConsecutiveWork: int
    maxDailyWork: int
    minWeeklyWork: int
    maxWeeklyWork: int
    maxConsecutiveNightShift: int
    maxTotalNightShift: int

    # Solver
    solver: pywrapcp.Solver

    def __init__(self, filename: str):
        self.load_from_file(filename)
        self.solver = None

    def load_from_file(self, f: str):
        """
        Reads in a file and populates the instance parameters.
        """
        params = {} 
        if not f:
            print("No file provided")
            return
        with open(f, "r") as fl:
            lines = fl.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("Business_"):
                    key, value = line.split(":")
                    if key != "Business_minDemandDayShift":
                        params[key] = int(value)
                    else:
                        params[key] = [int(x) for x in value.split()]
                elif line.startswith("Employee_"):
                    key, value = line.split(":")
                    params[key] = int(value)
                
        self.numWeeks = params.get("Business_numWeeks")
        self.numDays = params.get("Business_numDays")
        self.numEmployees = params.get("Business_numEmployees")
        self.numShifts = params.get("Business_numShifts")
        self.numIntervalsInDay = params.get("Business_numIntervalsInDay")
        
        raw = params.get("Business_minDemandDayShift", [])
        self.minDemandDayShift = []
        if raw:
            for i in range(0, self.numDays * self.numShifts, self.numShifts):
                self.minDemandDayShift.append(raw[i : i + self.numShifts])
                
        self.minDailyOperation = params.get("Business_minDailyOperation")
        self.minConsecutiveWork = params.get("Employee_minConsecutiveWork")
        self.maxDailyWork = params.get("Employee_maxDailyWork")
        self.minWeeklyWork = params.get("Employee_minWeeklyWork")
        self.maxWeeklyWork = params.get("Employee_maxWeeklyWork")
        self.maxConsecutiveNightShift = params.get("Employee_maxConsecutiveNigthShift")
        self.maxTotalNightShift = params.get("Employee_maxTotalNigthShift")


    def solve(
        self,
        time_limit_seconds: Optional[float] = None,
        strategy: str = "interleaved",
    ):
        # From class: Two primary decision-variable matrices:
        #   shiftOfEmployeeDay[e][d]    -- shift label in {0 .. numShifts-1}
        #   durationOfEmployeeDay[e][d] -- hours worked in {0 .. maxDailyWork}
        #    Correspondence:   shiftOfEmployeeDay[e][d] == OFF_SHIFT  iff  durationOfEmployeeDay[e][d] == 0

        OFF_SHIFT   = 0
        NIGHT_SHIFT = 1
        DAYS_PER_WEEK = 7

        self.solver = pywrapcp.Solver("EmployeeScheduling")
        solver = self.solver

        shifts    = list(range(self.numShifts))
        days      = list(range(self.numDays))
        employees = list(range(self.numEmployees))

        # Decision variables 

        # shiftOfEmployeeDay[e][d]: which shift employee e works on day d.
        shiftOfEmployeeDay = [
            [solver.IntVar(0, self.numShifts - 1, f"shift_{e}_{d}") for d in days]
            for e in employees
        ]

        # durationOfEmployeeDay[e][d]: how many hours employee e works on day d.
        # Domain is [0, maxDailyWork]; the correspondence constraint below further
        # restricts it to 0 when off and [minConsecutiveWork, maxDailyWork] when working.
        durationOfEmployeeDay = [
            [solver.IntVar(0, self.maxDailyWork, f"dur_{e}_{d}") for d in days]
            for e in employees
        ]

        # Correspondence: shiftOfEmployeeDay[e][d] == off  ↔  duration == 0  #
        for e in employees:
            for d in days:
                is_off = solver.IsEqualCstVar(shiftOfEmployeeDay[e][d], OFF_SHIFT)
                solver.Add(durationOfEmployeeDay[e][d] <= self.maxDailyWork * (1 - is_off))
                solver.Add(durationOfEmployeeDay[e][d] >= self.minConsecutiveWork * (1 - is_off))

        # Business constraints

        # Min employees per shift per day via Global Cardinality (Distribute).
        # For each day d, the column of shift variables must satisfy:
        #   |{e : shiftOfEmployeeDay[e][d] == s}| >= minDemandDayShift[d][s]  for all s.
        for d in days:
            col       = [shiftOfEmployeeDay[e][d] for e in employees]
            card_mins = [self.minDemandDayShift[d][s] for s in shifts]
            card_maxs = [self.numEmployees] * self.numShifts
            solver.Add(solver.Distribute(col, card_mins, card_maxs))

        # Min total hours worked across all employees each day.
        for d in days:
            solver.Add(solver.Sum([durationOfEmployeeDay[e][d] for e in employees]) >= self.minDailyOperation)

        
        # Training phase: each employee sees every shift label exactly once across the first numShifts days (includes the off shift).
        for e in employees:
            training_vars = [shiftOfEmployeeDay[e][d] for d in range(self.numShifts)]
            solver.Add(solver.AllDifferent(training_vars))

        
        # Employee constraints 
        

        # Weekly work-hour bounds for each complete 7-day week.
        # TODO: Partial trailing weeks (numDays % 7 != 0) are ignored.
        num_full_weeks = self.numDays // DAYS_PER_WEEK
        for e in employees:
            for w in range(num_full_weeks):
                week_days    = list(range(w * DAYS_PER_WEEK, (w + 1) * DAYS_PER_WEEK))
                weekly_hours = solver.Sum([durationOfEmployeeDay[e][d] for d in week_days])
                solver.Add(weekly_hours >= self.minWeeklyWork)
                solver.Add(weekly_hours <= self.maxWeeklyWork)

        # Consecutive night-shift limit.
        if self.maxConsecutiveNightShift == 1:
            # Special case: night on day d implies NOT night on day d+1.
            for e in employees:
                for d in range(self.numDays - 1):
                    is_night = solver.IsEqualCstVar(shiftOfEmployeeDay[e][d], NIGHT_SHIFT)
                    is_next_night = solver.IsEqualCstVar(shiftOfEmployeeDay[e][d + 1], NIGHT_SHIFT)
                    solver.Add(is_night + is_next_night <= 1)
        else:
            # General case: in any window of (maxConsecutiveNightShift + 1) consecutive days,
            # at most maxConsecutiveNightShift can be night shifts.
            for e in employees:
                for d in range(self.numDays - self.maxConsecutiveNightShift):
                    window = [
                        solver.IsEqualCstVar(shiftOfEmployeeDay[e][d + k], NIGHT_SHIFT)
                        for k in range(self.maxConsecutiveNightShift + 1)
                    ]
                    solver.Add(solver.Sum(window) <= self.maxConsecutiveNightShift)

        # Total night-shift cap per employee.
        for e in employees:
            is_night = [solver.IsEqualCstVar(shiftOfEmployeeDay[e][d], NIGHT_SHIFT) for d in days]
            solver.Add(solver.Sum(is_night) <= self.maxTotalNightShift)


        shift_vars    = [shiftOfEmployeeDay[e][d]    for e in employees for d in days]
        duration_vars = [durationOfEmployeeDay[e][d] for e in employees for d in days]


        # Using custom heuristic from 'search' module.
        # db, _search_refs = build_search(solver, shift_vars, duration_vars, self.numShifts, self.maxDailyWork, strategy=strategy)
        db, _search_refs = build_search(
            solver,
            shift_vars,
            duration_vars,
            self.numShifts,
            self.maxDailyWork,
            self.numDays,
            self.minDemandDayShift,
            strategy=strategy,
        )

        # Luby restarts with a base unit of 100 failures.
        # Because the shift selector uses weighted-random value selection,
        # each restart explores a genuinely different region of the tree.
        # Earlier, we had a deterministic search strategy, which means 
        # that these restarts were sort of useless.
        restart = solver.LubyRestart(100)

        limits = [restart]
        if time_limit_seconds is not None:
            limits.append(solver.TimeLimit(int(time_limit_seconds * 1000)))
        solver.NewSearch(db, limits)

        if solver.NextSolution():
            hours_per_slot = self.numIntervalsInDay // (self.numShifts - 1)
            shift_start = {0: -1} 
            for s in range(1, self.numShifts):
                shift_start[s] = (s - 1) * hours_per_slot

            schedule = []
            for e in employees:
                row = []
                for d in days:
                    s = shiftOfEmployeeDay[e][d].Value()
                    dur = durationOfEmployeeDay[e][d].Value()
                    if s == OFF_SHIFT:
                        row.append((-1, -1))
                    else:
                        begin = shift_start[s]
                        row.append((begin, begin + dur))
                schedule.append(row)
            solver.EndSearch()
            return True, solver.Failures(), schedule
        else:
            solver.EndSearch()
            return False, solver.Failures(), []


    ## I'm putting this in here, but 
    ## maybe we should move it to a separate module?
    def check_solution(self, sched: list) -> list[str]:
        """
        Checks sched against every model constraint and returns a list of
        violation strings.  An empty list means the solution is valid.
        """
        violations = []

        ## THese are sort of ''immutable'' truths in the scehduling problem formualtion.
        OFF_SHIFT   = 0
        NIGHT_SHIFT = 1
        DAYS_PER_WEEK = 7
        hours_per_slot = self.numIntervalsInDay // (self.numShifts - 1)

        def shift_of(e, d):
            begin, _ = sched[e][d]
            return OFF_SHIFT if begin == -1 else begin // hours_per_slot + 1

        def duration_of(e, d):
            begin, end = sched[e][d]
            return 0 if begin == -1 else end - begin

        employees = range(self.numEmployees)
        days      = range(self.numDays)
        shifts    = range(self.numShifts)

        for e in employees:
            for d in days:
                dur = duration_of(e, d)
                s   = shift_of(e, d)
                if s == OFF_SHIFT:
                    if dur != 0:
                        violations.append(f"E{e+1} D{d}: off shift but duration={dur}")
                else:
                    if dur < self.minConsecutiveWork:
                        violations.append(f"E{e+1} D{d}: duration {dur} < minConsecutiveWork {self.minConsecutiveWork}")
                    if dur > self.maxDailyWork:
                        violations.append(f"E{e+1} D{d}: duration {dur} > maxDailyWork {self.maxDailyWork}")

        # Min employees per shift per day 
        for d in days:
            counts = {s: sum(1 for e in employees if shift_of(e, d) == s) for s in shifts}
            for s in shifts:
                demand = self.minDemandDayShift[d][s]
                if counts[s] < demand:
                    violations.append(
                        f"D{d} shift {s}: {counts[s]} employees < demand {demand}"
                    )

        # Min daily operation 
        for d in days:
            total = sum(duration_of(e, d) for e in employees)
            if total < self.minDailyOperation:
                violations.append(
                    f"D{d}: total hours {total} < minDailyOperation {self.minDailyOperation}"
                )

        # Training phase: AllDifferent across first numShifts days
        for e in employees:
            labels = [shift_of(e, d) for d in range(self.numShifts)]
            if len(set(labels)) != self.numShifts:
                violations.append(
                    f"E{e+1} training phase: shift labels not all-different: {labels}"
                )

        # Weekly work-hour bounds
        num_full_weeks = self.numDays // DAYS_PER_WEEK
        for e in employees:
            for w in range(num_full_weeks):
                week_days = range(w * DAYS_PER_WEEK, (w + 1) * DAYS_PER_WEEK)
                total = sum(duration_of(e, d) for d in week_days)
                if total < self.minWeeklyWork:
                    violations.append(
                        f"E{e+1} week {w}: {total}h < minWeeklyWork {self.minWeeklyWork}"
                    )
                if total > self.maxWeeklyWork:
                    violations.append(
                        f"E{e+1} week {w}: {total}h > maxWeeklyWork {self.maxWeeklyWork}"
                    )

        # Max consecutive night shifts
        for e in employees:
            for d in range(self.numDays - self.maxConsecutiveNightShift):
                window = sum(
                    1 for k in range(self.maxConsecutiveNightShift + 1)
                    if shift_of(e, d + k) == NIGHT_SHIFT
                )
                if window > self.maxConsecutiveNightShift:
                    violations.append(
                        f"E{e+1} D{d}-D{d+self.maxConsecutiveNightShift}: "
                        f"{window} consecutive nights > {self.maxConsecutiveNightShift}"
                    )

        # Max total night shifts
        for e in employees:
            total = sum(1 for d in days if shift_of(e, d) == NIGHT_SHIFT)
            if total > self.maxTotalNightShift:
                violations.append(
                    f"E{e+1}: {total} total night shifts > maxTotalNightShift {self.maxTotalNightShift}"
                )

        return violations


    def prettyPrint(self, numEmployees, numDays, sched):
        """
        Poor man's Gantt chart.
        Displays the employee schedules on the command line. 
        Each row corresponds to a single employee. 
        A "+" refers to a working hour and "." means no work
        The shifts are separated with a "|"
        The days are separated with "||"
        
        This might help you analyze your solutions. 
        
        @param numEmployees the number of employees
        @param numDays the number of days
        @param sched sched[e][d] = (begin, end) hours for employee e on day d
        """
        for e in range(numEmployees):
            print(f"E{e+1}: ", end="")
            if e < 9: print(" ", end="")
            for d in range(numDays):
                begin = sched[e][d][0]
                end = sched[e][d][1]
                for i in range(self.numIntervalsInDay):
                    if i % 8 == 0: print("|", end="")
                    if begin != end and i >= begin and i < end:
                         print("+", end="")
                    else:
                         print(".", end="")
                print("|", end="")
            print(" ")

    def generateVisualizerInput(self, numEmployees, numDays, sched):
        solString = f"{numDays} {numEmployees}\n"
        for d in range(numDays):
            for e in range(numEmployees):
                solString += f"{sched[e][d][0]} {sched[e][d][1]}\n"

        fileName = f"{numDays}_{numEmployees}_sol.txt"
        try:
            with open(fileName, "w") as fl:
                fl.write(solString)
            print(f"File created: {fileName}")
        except IOError as e:
            print(f"An error occured: {e}")
