import math
import sys
import numpy as np

# from stencil, unused: model_builder ships SCIP, which is an IP solver. The
# project requires that OR-Tools be used only as an LP solver, so we use the
# pywraplp interface with GLOP (a pure-LP simplex solver) instead.
# from ortools.linear_solver.python import model_builder

from ortools.linear_solver import pywraplp

#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)


class IPInstance:
    numTests: int  # number of tests
    numDiseases: int  # number of diseases
    costOfTest: np.ndarray  # [numTests] the cost of each test
    A: np.ndarray  # [numTests][numDiseases] 0/1 matrix if test is positive for disease

    def __init__(self, filename: str) -> None:
        self.load_from_file(filename)
        # from stencil, unused: original IP-solver setup using SCIP via model_builder.
        # We perform Branch-and-Bound manually, calling pywraplp + GLOP only for the
        # LP relaxations at each BnB node (see solve()).
        # self.model = model_builder.Model()
        # self.solver = model_builder.Solver("SCIP")
        self.solution = None
        self.objective_value = None
        self._preprocess()

    # ---------------- Preprocessing ----------------
    def _preprocess(self):
        """Cast the problem as min-cost set cover over disease pairs.

        For every unordered pair of diseases (j, k), build the constraint
            sum_{i : A[i,j] != A[i,k]} x_i >= 1
        i.e. at least one selected test must differentiate j and k.
        """
        pairs = []
        for j in range(self.numDiseases):
            for k in range(j + 1, self.numDiseases):
                pairs.append((j, k))
        self.pairs = pairs
        self.numPairs = len(pairs)

        # B[p, i] = 1 iff test i differentiates pair p
        B = np.zeros((self.numPairs, self.numTests), dtype=np.int8)
        for p_idx, (j, k) in enumerate(pairs):
            B[p_idx] = (self.A[:, j] != self.A[:, k]).astype(np.int8)
        self.B = B

        # tests_for_pair[p] = list of test indices that differentiate pair p
        self.tests_for_pair = [np.where(B[p] == 1)[0].tolist() for p in range(self.numPairs)]

        # The IP is feasible iff every disease pair has >= 1 differentiating test
        self.feasible = all(len(t) > 0 for t in self.tests_for_pair)

        # If every cost is integer-valued we can use a ceiling lower bound
        self.integer_costs = bool(np.all(self.costOfTest == np.floor(self.costOfTest)))

    # ---------------- Branch and Bound ----------------
    def solve(self):
        """
        Healthcare Analytics Model

        Branch-and-Bound: at each node solve the LP relaxation with GLOP.
        Prune by bound (LP value >= incumbent) or by infeasibility,
        otherwise branch on the most fractional variable, trying x=1 first.
        """
        if not self.feasible:
            self.solution = False
            self.objective_value = None
            return self.solution, self.objective_value

        # Build LP solver once; bound changes between BnB nodes reuse it (warm-started simplex).
        lp = pywraplp.Solver.CreateSolver('GLOP')
        if lp is None:
            print("Could not create GLOP LP solver")
            sys.exit(1)
        self.lp = lp

        # Continuous decision variables x_i in [0, 1]
        self.x = [lp.NumVar(0.0, 1.0, f'x_{i}') for i in range(self.numTests)]

        # Cover constraints: one per disease pair
        for p in range(self.numPairs):
            ct = lp.Constraint(1.0, lp.infinity())
            for i in self.tests_for_pair[p]:
                ct.SetCoefficient(self.x[i], 1.0)

        # Objective: minimise total cost of selected tests
        obj = lp.Objective()
        for i in range(self.numTests):
            obj.SetCoefficient(self.x[i], float(self.costOfTest[i]))
        obj.SetMinimization()

        # Initial incumbent from a greedy set-cover heuristic
        ub, sol = self._greedy_set_cover()
        self.best_obj = ub
        self.best_sol = sol

        # Recurse
        self.nodes = 0
        sys.setrecursionlimit(max(10000, 50 * self.numTests + 1000))
        self._bnb()

        if self.best_sol is None:
            self.solution = False
            self.objective_value = None
        else:
            self.solution = True
            self.objective_value = int(round(self.best_obj)) if self.integer_costs else float(self.best_obj)

        return self.solution, self.objective_value

    def _greedy_set_cover(self):
        """Greedy min-cost set cover used as an initial upper bound."""
        uncovered = np.ones(self.numPairs, dtype=bool)
        used = np.zeros(self.numTests, dtype=bool)
        total_cost = 0.0

        while uncovered.any():
            best_i = -1
            best_ratio = float('inf')
            best_count = -1
            for i in range(self.numTests):
                if used[i]:
                    continue
                count = int(self.B[uncovered, i].sum())
                if count == 0:
                    continue
                cost_i = float(self.costOfTest[i])
                # Zero-cost tests strictly dominate, ordered by coverage size.
                ratio = cost_i / count if cost_i > 0 else -float(count)
                if (ratio < best_ratio) or (ratio == best_ratio and count > best_count):
                    best_ratio = ratio
                    best_count = count
                    best_i = i
            if best_i == -1:
                return float('inf'), None  # only reachable if infeasible
            used[best_i] = True
            total_cost += float(self.costOfTest[best_i])
            uncovered &= (self.B[:, best_i] == 0)

        return total_cost, [int(v) for v in used]

    def _bnb(self):
        """Depth-first BnB. Solves the LP at the current bound state,
        prunes, or branches on the most fractional variable."""
        EPS = 1e-7

        status = self.lp.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            return  # LP infeasible at this subtree -> prune

        self.nodes += 1
        lp_val = self.lp.Objective().Value()

        # Tighter LB for integer costs (any IP-feasible objective is an integer)
        lb = math.ceil(lp_val - EPS) if self.integer_costs else lp_val
        if lb >= self.best_obj - EPS:
            return

        # Pick branching variable: the one closest to 0.5
        frac_idx = -1
        max_frac = -1.0
        for i in range(self.numTests):
            v = self.x[i].solution_value()
            if EPS < v < 1 - EPS:
                f = v if v <= 0.5 else 1 - v
                if f > max_frac:
                    max_frac = f
                    frac_idx = i

        if frac_idx == -1:
            # LP solution is integral -> valid IP solution
            if lp_val < self.best_obj - EPS:
                self.best_obj = lp_val
                self.best_sol = [int(round(self.x[i].solution_value())) for i in range(self.numTests)]
            return

        old_lb = self.x[frac_idx].lb()
        old_ub = self.x[frac_idx].ub()

        # Up branch first: forcing a test in tends to satisfy more cover constraints quickly,
        # which yields tighter incumbents and more pruning on the down branch.
        self.x[frac_idx].SetBounds(1.0, 1.0)
        self._bnb()

        self.x[frac_idx].SetBounds(0.0, 0.0)
        self._bnb()

        self.x[frac_idx].SetBounds(old_lb, old_ub)

    # ---------------- I/O ----------------
    def load_from_file(self, filename: str):
        try:
            with open(filename, "r") as fl:
                self.numTests = int(fl.readline().strip())  # n
                self.numDiseases = int(fl.readline().strip())  # m

                self.costOfTest = np.array([float(i) for i in fl.readline().strip().split()])

                self.A = np.zeros((self.numTests, self.numDiseases))
                for i in range(0, self.numTests):
                    self.A[i, :] = np.array([int(i) for i in fl.readline().strip().split()])

        except Exception as e:
            print(f"Error reading instance file. File format may be incorrect.{e}")
            exit(1)

    def __str__(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0, self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out
