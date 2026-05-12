import math
import sys
import numpy as np

from ortools.linear_solver.python import model_builder

# from stencil, unused: an earlier draft of this file used the pywraplp + GLOP
# interface for the LP relaxations. The active code below uses the
# model_builder API with SCIP, matching the solver the stencil constructs in
# __init__. All BnB variables are *continuous*, so SCIP only solves LP
# relaxations -- never an IP, in compliance with the project requirement.
# from ortools.linear_solver import pywraplp

#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)


# SCIP parameters: ask SCIP to behave like a thin LP wrapper -- skip the
# presolving, cutting plane, propagating, conflict analysis, and primal
# heuristic machinery that is geared toward MIP, since every variable here
# is continuous and we only want the LP relaxation value.
_SCIP_LP_ONLY_PARAMS = "\n".join([
    "presolving/maxrounds = 0",
    "presolving/maxrestarts = 0",
    "separating/maxrounds = 0",
    "separating/maxroundsroot = 0",
    "separating/maxcuts = 0",
    "separating/maxcutsroot = 0",
    "propagating/maxrounds = 0",
    "propagating/maxroundsroot = 0",
    "conflict/enable = FALSE",
    "lp/initalgorithm = d",
    "lp/resolvealgorithm = d",
    "limits/totalnodes = 1",
])


class IPInstance:
    numTests: int  # number of tests
    numDiseases: int  # number of diseases
    costOfTest: np.ndarray  # [numTests] the cost of each test
    A: np.ndarray  # [numTests][numDiseases] 0/1 matrix if test is positive for disease

    def __init__(self, filename: str) -> None:
        self.load_from_file(filename)
        # Stencil-provided model + SCIP solver. We declare the decision variables
        # as continuous (new_num_var), so every call to self.solver.solve(self.model)
        # is a pure LP relaxation -- SCIP never branches itself.
        self.model = model_builder.Model()
        self.solver = model_builder.Solver("SCIP")
        # Configure SCIP to skip its MIP-oriented machinery (see comment on
        # _SCIP_LP_ONLY_PARAMS above).
        self.solver.set_solver_specific_parameters(_SCIP_LP_ONLY_PARAMS)
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
        # pairs_for_test[i] = list of pairs differentiated by test i
        self.pairs_for_test = [np.where(B[:, i] == 1)[0].tolist() for i in range(self.numTests)]

        # The IP is feasible iff every disease pair has >= 1 differentiating test
        self.feasible = all(len(t) > 0 for t in self.tests_for_pair)

        # If every cost is integer-valued we can use a ceiling lower bound
        self.integer_costs = bool(np.all(self.costOfTest == np.floor(self.costOfTest)))

    # ---------------- Branch and Bound ----------------
    def solve(self):
        """
        Healthcare Analytics Model

        Branch-and-Bound: at each node solve the LP relaxation with SCIP
        (continuous-only variables). Prune by bound (LP value >= incumbent)
        or by infeasibility, otherwise branch on the most fractional variable,
        trying x=1 first. Constraint propagation (unit-cover) eliminates many
        LP solves on subtrees that are obviously infeasible or have forced
        variables.
        """
        if not self.feasible:
            self.solution = False
            self.objective_value = None
            return self.solution, self.objective_value

        # Continuous decision variables x_i in [0, 1]
        self.x = [self.model.new_num_var(0.0, 1.0, f'x_{i}') for i in range(self.numTests)]

        # Cover constraints: one per disease pair
        for p in range(self.numPairs):
            tests = self.tests_for_pair[p]
            self.model.add(sum(self.x[i] for i in tests) >= 1)

        # Objective: minimise total cost of selected tests
        self.model.minimize(
            sum(float(self.costOfTest[i]) * self.x[i] for i in range(self.numTests))
        )

        # Per-test fixation state for propagation:
        #   fixed[i] = 0 (forced out), 1 (forced in), or -1 (free)
        # active_per_pair[p] = number of "free or forced-in" tests left for pair p
        # covered_per_pair[p] = True iff some forced-in test already covers p
        self._fixed = [-1] * self.numTests
        self._active_per_pair = np.array(
            [len(t) for t in self.tests_for_pair], dtype=np.int32
        )
        self._covered_per_pair = np.zeros(self.numPairs, dtype=bool)
        self._forced_cost = 0.0  # sum of c_i over fixed[i]==1

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
            self.objective_value = (
                int(round(self.best_obj)) if self.integer_costs else float(self.best_obj)
            )

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
                ratio = cost_i / count if cost_i > 0 else -float(count)
                if (ratio < best_ratio) or (ratio == best_ratio and count > best_count):
                    best_ratio = ratio
                    best_count = count
                    best_i = i
            if best_i == -1:
                return float('inf'), None
            used[best_i] = True
            total_cost += float(self.costOfTest[best_i])
            uncovered &= (self.B[:, best_i] == 0)

        return total_cost, [int(v) for v in used]

    # ---------- propagation helpers ----------
    def _set_var(self, i: int, value: int):
        """Fix x_i to 0 or 1 in both the LP and the propagation state.

        Returns (ok, undo_list).
            ok: False if propagation finds an infeasibility -> prune.
            undo_list: list of (action, payload) to replay in reverse on backtrack.
        """
        undo = []
        ok = True

        # Stack-based propagation queue of "force x_i = value" requests.
        # We start with the explicit branch decision and keep going as long as
        # unit-cover constraints turn up more forced variables.
        queue = [(i, value)]
        while queue and ok:
            j, v = queue.pop()
            cur = self._fixed[j]
            if cur == v:
                continue  # already fixed the same way
            if cur != -1 and cur != v:
                ok = False  # contradiction
                break

            # Apply: update fixed, the LP variable bounds, and per-pair counts.
            self._fixed[j] = v
            old_lb = self.x[j].lower_bound
            old_ub = self.x[j].upper_bound
            self.x[j].lower_bound = float(v)
            self.x[j].upper_bound = float(v)
            undo.append(('fix', j, old_lb, old_ub))

            if v == 1:
                self._forced_cost += float(self.costOfTest[j])
                undo.append(('cost', float(self.costOfTest[j])))
                # All pairs that j covers are now satisfied.
                for p in self.pairs_for_test[j]:
                    if not self._covered_per_pair[p]:
                        self._covered_per_pair[p] = True
                        undo.append(('cover', p))
            else:
                # v == 0: shrink the active set of every pair j touches.
                for p in self.pairs_for_test[j]:
                    if self._covered_per_pair[p]:
                        continue  # already covered by some forced-in test
                    self._active_per_pair[p] -= 1
                    undo.append(('active', p))
                    cnt = self._active_per_pair[p]
                    if cnt == 0:
                        ok = False
                        break
                    if cnt == 1:
                        # Unit cover: the single remaining differentiating test
                        # must be selected. Find it and queue it.
                        for k in self.tests_for_pair[p]:
                            if self._fixed[k] == -1:
                                queue.append((k, 1))
                                break

        return ok, undo

    def _undo(self, undo_list):
        """Reverse the side-effects recorded by _set_var, in LIFO order."""
        for entry in reversed(undo_list):
            tag = entry[0]
            if tag == 'fix':
                _, j, old_lb, old_ub = entry
                self._fixed[j] = -1
                self.x[j].lower_bound = old_lb
                self.x[j].upper_bound = old_ub
            elif tag == 'cost':
                self._forced_cost -= entry[1]
            elif tag == 'cover':
                self._covered_per_pair[entry[1]] = False
            elif tag == 'active':
                self._active_per_pair[entry[1]] += 1

    # ---------- BnB recursion ----------
    def _bnb(self):
        EPS = 1e-7

        # Cheap LB from forced-in tests alone: prune before even solving the LP.
        if self.integer_costs:
            quick_lb = math.ceil(self._forced_cost - EPS)
        else:
            quick_lb = self._forced_cost
        if quick_lb >= self.best_obj - EPS:
            return

        status = self.solver.solve(self.model)
        if status != model_builder.SolveStatus.OPTIMAL:
            return  # infeasible at this subtree -> prune

        self.nodes += 1
        lp_val = self.solver.objective_value

        lb = math.ceil(lp_val - EPS) if self.integer_costs else lp_val
        if lb >= self.best_obj - EPS:
            return

        # Pick the variable closest to 0.5 (most-fractional rule).
        frac_idx = -1
        max_frac = -1.0
        for i in range(self.numTests):
            if self._fixed[i] != -1:
                continue
            v = self.solver.value(self.x[i])
            if EPS < v < 1 - EPS:
                f = v if v <= 0.5 else 1 - v
                if f > max_frac:
                    max_frac = f
                    frac_idx = i

        if frac_idx == -1:
            # LP solution is integral -> valid IP solution.
            if lp_val < self.best_obj - EPS:
                self.best_obj = lp_val
                self.best_sol = [
                    int(round(self.solver.value(self.x[i]))) for i in range(self.numTests)
                ]
            return

        # Up branch first (forcing a test in covers many pairs and tightens the
        # incumbent earlier, which lets the down branch prune more aggressively).
        ok, undo = self._set_var(frac_idx, 1)
        if ok:
            self._bnb()
        self._undo(undo)

        ok, undo = self._set_var(frac_idx, 0)
        if ok:
            self._bnb()
        self._undo(undo)

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
