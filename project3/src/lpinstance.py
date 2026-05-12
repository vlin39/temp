from __future__ import annotations

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
            Supply Chain Model
        """

        # TODO: your model goes here

        # Variables

        # Constraints

        # Solve
        self.solution = ...
        self.objective_value = ...

        pass

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
