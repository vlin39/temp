from __future__ import annotations

import math
from typing import Optional

import numpy as np
from ortools.linear_solver.python import model_builder


class LPInstance:
    # Problem parameters
    numCustomers: int  # the number of customers
    numFacilities: int  # the number of facilities
    allocCostCF: np.ndarray  # allocCostCF[c][f] is the service cost paid each time customer c is served by facility f
    demandC: np.ndarray  # demandC[c] is the demand of customer c
    openingCostF: np.ndarray  # openingCostF[f] is the opening cost of facility f
    capacityF: np.ndarray  # capacityF[f] is the capacity of facility f
    numMaxVehiclePerFacility: int  # maximum number of vehicles to use at an open facility
    truckDistLimit: float  # total driving distance limit for trucks
    truckUsageCost: float  # fixed usage cost paid if a truck is used
    distanceCF: np.ndarray  # distanceCF[c][f] is the roundtrip distance between customer c and facility f

    def __init__(self, filename: str):
        self.load_from_file(filename)
        self.solver: model_builder.Solver = model_builder.Solver('SCIP')
        self.model = model_builder.Model()
        self.solution = None
        self.objective_value = None

    def solve(self):
        """
            Supply Chain LP relaxation.

            Decision variables (all continuous):
              y[f]    in [0, 1]                     : fraction of facility f open
              x[c,f]  in [0, 1]                     : fraction of customer c served from facility f
              v[f]    in [0, numMaxVehiclePerFacility] : number of vehicles at facility f

            Minimize:
              sum_f openingCostF[f] * y[f]
              + sum_{c,f} allocCostCF[c,f] * x[c,f]
              + truckUsageCost * sum_f v[f]
        """

        # # from stencil, unused
        # # TODO: your model goes here

        model = self.model
        solver = self.solver

        C = self.numCustomers
        F = self.numFacilities
        V = self.numMaxVehiclePerFacility  # = numCustomers

        # Variables
        y = [model.new_num_var(0.0, 1.0, f"y_{f}") for f in range(F)]
        x = [[model.new_num_var(0.0, 1.0, f"x_{c}_{f}") for f in range(F)] for c in range(C)]
        v = [model.new_num_var(0.0, float(V), f"v_{f}") for f in range(F)]

        # Constraints

        # (1) Each customer is fully served across facilities.
        for c in range(C):
            model.add(sum(x[c][f] for f in range(F)) == 1)

        # (2) A customer can only be served from a (fractionally) open facility.
        for c in range(C):
            for f in range(F):
                model.add(x[c][f] <= y[f])

        # (3) Facility capacity, scaled by openness.
        for f in range(F):
            model.add(
                sum(float(self.demandC[c]) * x[c][f] for c in range(C))
                <= float(self.capacityF[f]) * y[f]
            )

        # (4) Vehicle workload: total round-trip distance served from facility f
        #     cannot exceed v[f] * truckDistLimit.
        for f in range(F):
            model.add(
                sum(float(self.distanceCF[c, f]) * x[c][f] for c in range(C))
                <= float(self.truckDistLimit) * v[f]
            )

        # (5) Vehicles only exist at (fractionally) open facilities.
        for f in range(F):
            model.add(v[f] <= float(V) * y[f])

        # Objective
        opening_cost = sum(float(self.openingCostF[f]) * y[f] for f in range(F))
        service_cost = sum(
            float(self.allocCostCF[c, f]) * x[c][f]
            for c in range(C)
            for f in range(F)
        )
        truck_cost = float(self.truckUsageCost) * sum(v[f] for f in range(F))
        model.minimize(opening_cost + service_cost + truck_cost)

        # Solve
        # # from stencil, unused
        # self.solution = ...
        # self.objective_value = ...
        # pass

        status = solver.solve(model)
        self.status = status

        if status == model_builder.SolveStatus.OPTIMAL:
            # Store fractional solution arrays for the standalone verifier.
            self.y_sol = np.array([solver.value(y[f]) for f in range(F)])
            self.x_sol = np.array(
                [[solver.value(x[c][f]) for f in range(F)] for c in range(C)]
            )
            self.v_sol = np.array([solver.value(v[f]) for f in range(F)])
            self.lp_objective_value = solver.objective_value
            self.solution = (self.y_sol, self.x_sol, self.v_sol)
            self.objective_value = math.ceil(solver.objective_value)
        else:
            # LP relaxation should always be feasible & bounded on these inputs;
            # if not, surface it instead of silently emitting a bogus number.
            self.y_sol = None
            self.x_sol = None
            self.v_sol = None
            self.lp_objective_value = None
            self.solution = None
            self.objective_value = None
            print(f"Solver returned non-optimal status: {status}")

    def load_from_file(self, filename: str):
        try:
            with open(filename, "r") as fl:
                numCustomers, numFacilities = [int(i) for i in fl.readline().split()]
                numMaxVehiclePerFacility = numCustomers
                print(
                    f"numCustomers: {numCustomers} numFacilities: {numFacilities} numVehicle: {numMaxVehiclePerFacility}")
                allocCostCF = np.zeros((numCustomers, numFacilities))

                allocCostraw = [float(i) for i in fl.readline().split()]
                index = 0
                for i in range(numCustomers):
                    for j in range(numFacilities):
                        allocCostCF[i, j] = allocCostraw[index]
                        index += 1

                demandC = np.array([float(i) for i in fl.readline().split()])
                openingCostF = np.array([float(i) for i in fl.readline().split()])
                capacityF = np.array([float(i) for i in fl.readline().split()])
                truckDistLimit, truckUsageCost = [float(i) for i in fl.readline().split()]

                distanceCF = np.zeros((numCustomers, numFacilities))
                distanceCFraw = [float(i) for i in fl.readline().split()]
                index = 0
                for i in range(numCustomers):
                    for j in range(numFacilities):
                        distanceCF[i, j] = distanceCFraw[index]
                        index += 1

                self.numCustomers = numCustomers
                self.numFacilities = numFacilities
                self.allocCostCF = allocCostCF
                self.demandC = demandC
                self.openingCostF = openingCostF
                self.capacityF = capacityF
                self.numMaxVehiclePerFacility = numMaxVehiclePerFacility
                self.truckDistLimit = truckDistLimit
                self.truckUsageCost = truckUsageCost
                self.distanceCF = distanceCF
        except Exception as e:
            print(f"Could not read problem instance file due to error: {e}")
            return None
