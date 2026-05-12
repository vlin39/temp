from __future__ import annotations

import sys
from typing import Optional, List, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp


class CPInstance:
    # BUSINESS parameters
    numWeeks: int
    numDays: int
    numEmployees: int
    numShifts: int
    numIntervalsInDay: int
    minDemandDayShift: list[list[int]]
    minDailyOperation: int
    
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
    ):
        """
        Employee Scheduling Model 
        """
        self.solver = pywrapcp.Solver("employee_scheduling")

        # variables
        shift = [
            [self.solver.IntVar(0, self.numShifts - 1, f"shift_{e}_{d}") for d in range(self.numDays)]
            for e in range(self.numEmployees)
        ]
        hours = [
            [
                self.solver.IntVar(
                    [0] + list(range(self.minConsecutiveWork, self.maxDailyWork + 1)),
                    f"hours_{e}_{d}",
                )
                for d in range(self.numDays)
            ]
            for e in range(self.numEmployees)
        ]

        # constraints

        # off shift must have 0 hours; working shifts must have legal positive hours
        allowed_shift_hours = [(0, 0)]
        for s in range(1, self.numShifts):
            for h in range(self.minConsecutiveWork, self.maxDailyWork + 1):
                allowed_shift_hours.append((s, h))

        for e in range(self.numEmployees):
            for d in range(self.numDays):
                self.solver.Add(
                    self.solver.AllowedAssignments(
                        [shift[e][d], hours[e][d]], allowed_shift_hours
                    )
                )

        # first 4 days must contain each shift exactly once for every employee
        if self.numDays >= 4:
            for e in range(self.numEmployees):
                self.solver.Add(self.solver.AllDifferent([shift[e][d] for d in range(4)]))

        # minimum total hours per day and minimum demand per day/shift
        for d in range(self.numDays):
            self.solver.Add(
                self.solver.Sum([hours[e][d] for e in range(self.numEmployees)])
                >= self.minDailyOperation
            )
            for s in range(1, self.numShifts):
                assigned = [
                    self.solver.IsEqualCstVar(shift[e][d], s)
                    for e in range(self.numEmployees)
                ]
                self.solver.Add(self.solver.Sum(assigned) >= self.minDemandDayShift[d][s])

        # weekly min/max hours
        for e in range(self.numEmployees):
            for w in range(self.numWeeks):
                start_day = 7 * w
                end_day = min(start_day + 7, self.numDays)
                week_hours = [hours[e][d] for d in range(start_day, end_day)]
                self.solver.Add(self.solver.Sum(week_hours) >= self.minWeeklyWork)
                self.solver.Add(self.solver.Sum(week_hours) <= self.maxWeeklyWork)

        # night-shift restrictions
        for e in range(self.numEmployees):
            night_bools = [self.solver.IsEqualCstVar(shift[e][d], 1) for d in range(self.numDays)]
            self.solver.Add(self.solver.Sum(night_bools) <= self.maxTotalNightShift)
            window = self.maxConsecutiveNightShift + 1
            for d in range(self.numDays - window + 1):
                self.solver.Add(
                    self.solver.Sum(night_bools[d : d + window])
                    <= self.maxConsecutiveNightShift
                )

        # solve
        all_vars = []
        for e in range(self.numEmployees):
            for d in range(self.numDays):
                all_vars.append(shift[e][d])
                all_vars.append(hours[e][d])

        db = self.solver.DefaultPhase(all_vars)
        if time_limit_seconds is not None:
            self.solver.NewSearch(
                db, [self.solver.TimeLimit(int(time_limit_seconds * 1000))]
            )
        else:
            self.solver.NewSearch(db)

        if self.solver.NextSolution():
            flat_schedule = []
            for e in range(self.numEmployees):
                for d in range(self.numDays):
                    s = shift[e][d].Value()
                    h = hours[e][d].Value()
                    if s == 0:
                        begin, end = -1, -1
                    elif s == 1:
                        begin, end = 0, h
                    elif s == 2:
                        begin, end = 8, 8 + h
                    else:
                        begin, end = 16, 16 + h
                    flat_schedule.extend([str(begin), str(end)])

            schedule = " ".join(flat_schedule)
            failures = self.solver.Failures()
            self.solver.EndSearch()
            return True, failures, schedule
        else:
            failures = self.solver.Failures()
            self.solver.EndSearch()
            return False, failures, None
            

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
