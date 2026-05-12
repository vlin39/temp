# Project IV - Healthcare Analytics (CSCI 2951-O)

**Team:** skadiogl

## 1. Constraint Model

We formulate the problem as a 0/1 minimum-cost set-cover Integer Program.

**Decision variables.** `x_i ∈ {0,1}` for every test `i = 1..n`, with `x_i = 1`
indicating that test `i` is selected.

**Objective.** Minimise total cost
`min  Σ_i c_i · x_i`.

**Constraints.** For every unordered pair of diseases `(j, k)`, at least one
selected test must give different results on the two diseases:

```
Σ_{i : A[i,j] ≠ A[i,k]} x_i ≥ 1     ∀ 1 ≤ j < k ≤ m
```

There are `m(m-1)/2` such cover constraints. Building the coefficient matrix
`B`, where `B[p,i] = 1` iff test `i` differentiates the pair `p = (j,k)`,
reduces the problem to standard min-cost set cover (elements = disease pairs,
sets = tests).

The LP relaxation replaces `x_i ∈ {0,1}` with `x_i ∈ [0,1]`.

## 2. Branch-and-Bound

Solver: **GLOP** (pywraplp) for the LP relaxations only, as required by the
project. SCIP / model_builder are not used to drive the search (only the
verification helper in `src/verify_with_ip.py` uses SCIP, which the handout's
"Hint" explicitly permits).

### Initial incumbent
A greedy set-cover heuristic (pick the un-used test with smallest
`cost / uncovered-coverage` ratio, ties broken by larger coverage) provides
a feasible upper bound before any LP is solved. This typically prunes a
large portion of the tree immediately.

### Bounding
At each BnB node we solve the LP relaxation by adjusting the variable bounds
in the *same* GLOP solver (so GLOP warm-starts simplex from the previous basis
across nodes — building the LP once is the only expensive setup). Since all
input costs are integer, any IP-feasible objective is itself an integer, so we
tighten the LP lower bound to `⌈lp_value⌉` and prune whenever
`⌈lp_value⌉ ≥ incumbent`.

### Branching
- **Variable choice:** the variable whose LP value is closest to 0.5 (most
  fractional). This is the classical SOS branching rule and produced the best
  results among the alternatives we tried (largest-coefficient, largest
  reduced-cost) on this benchmark.
- **Order:** up-branch first (`x_i = 1`). Forcing a test into the cover
  satisfies many constraints at once, which tightens the incumbent earlier
  in the DFS and increases pruning on the down branch.

### Feasibility check
If the LP at a node returns INFEASIBLE (i.e. some pair has no remaining
differentiating test once we have fixed several `x_i = 0`), the subtree is
pruned. We rely on GLOP to detect this, which is fast thanks to warm-starting.

## 3. Observations

- Costs are integers across all 14 input files, so the ceiling-based bound
  is a strict improvement and triggers many additional cuts compared to the
  continuous LP bound alone.
- "Up first" branching matters: switching to "down first" roughly doubles
  the running time on the 100x* instances because the incumbent stays loose
  until late in the search.
- The 0.25-density instances are harder than the 0.5-density ones at the same
  size — sparser disease columns produce sparser constraints, looser LP
  relaxations, and more fractional optima.
- The 100×200 instances build a 19,900-row LP; setting the constraint
  coefficients via the `MPConstraint.SetCoefficient` API (rather than `Sum`
  expressions) makes one-time setup notably faster.

## 4. Code layout

```
src/
  ipinstance.py       parser + Branch-and-Bound (all active code)
  main.py             entry point (unchanged from stencil)
  timer.py            timer (unchanged from stencil)
  verify_with_ip.py   optional helper: solves with SCIP IP to cross-check
```

Unused stencil snippets (the `model_builder` / SCIP imports in `ipinstance.py`)
are kept commented with `# from stencil, unused`.

## 5. Time spent
Approximately 4 hours.
