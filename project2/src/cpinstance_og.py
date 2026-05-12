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
        pass
    
        # TODO: your model goes here

        # variables

        # constraints

        # solve
        db = self.solver.DefaultPhase(...)
        self.solver.NewSearch(db)

        if self.solver.NextSolution():
            schedule = ...
            return True, self.solver.Failures(), schedule
        else:
            return False, self.solver.Failures(), None
            

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
