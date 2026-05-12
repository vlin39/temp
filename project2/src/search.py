"""Custom decision builders for the employee scheduling CP model."""
from __future__ import annotations

import random

from ortools.constraint_solver import pywrapcp

NIGHT_SHIFT = 1  # weak assumption: night shift is always shift 1

class InterleavedSelector(pywrapcp.PyDecisionBuilder):
    """
    Interleave shift and duration variables so constraint propagation between the two kicks in
    immediately — no more assigning *all* shifts before touching durations.

    Variable selection (both types):
      CHOOSE_MIN_SIZE_LOWEST_MIN — fail-first across the combined pool.
      Naturally handles training-phase days (AllDifferent), high-demand days
      (minDemandDayShift), and late-week budget pressure (minWeeklyWork /
      maxWeeklyWork) without any explicit priority rules.

    Value selection:
      • Shift vars  → weighted-random with night deprioritised (night is
        doubly capped by maxConsecutiveNightShift and maxTotalNightShift, so
        conserving it reduces backtracking). Randomness lets Luby restarts
        diversify the search tree.
      • Duration vars → ASSIGN_MAX_VALUE — maximise hours first to
        satisfy minDailyOperation / minWeeklyWork with fewer backtracks.
    """

    def __init__(self, shift_vars, duration_vars, num_shifts, max_daily_work, num_days, min_demand_day_shift,):
        super().__init__()
        self._shift_set = set(id(v) for v in shift_vars)
        self._all_vars      = list(shift_vars) + list(duration_vars)
        self._num_shifts    = num_shifts
        self._max_daily_work = max_daily_work

    def Next(self, solver_):
        # --- Variable selection: fail-first across combined shift+duration pool ---
        most_constrained_var = None
        fewest_choices    = self._num_shifts + 1
        lowest_domain_min = self._max_daily_work + 1
        for v in self._all_vars:
            if v.Bound():
                continue
            domain_size = v.Size()
            domain_min  = v.Min()
            if domain_size < fewest_choices or (domain_size == fewest_choices and domain_min < lowest_domain_min):
                most_constrained_var = v
                fewest_choices    = domain_size
                lowest_domain_min = domain_min

        if most_constrained_var is None:
            return None

        # --- Value selection ---
        if id(most_constrained_var) in self._shift_set:
            shift_weights = [1, 2, 3, 3]   # off, night, day, evening
            domain  = [s for s in range(self._num_shifts) if most_constrained_var.Contains(s)]
            weights = [shift_weights[s] for s in domain]
            chosen  = random.choices(domain, weights=weights, k=1)[0]
        else:
            chosen = most_constrained_var.Max()

        return solver_.AssignVariableValue(most_constrained_var, chosen)
    
class InterleavedSelector2(pywrapcp.PyDecisionBuilder):
    """
    Interleaved search with hopefully stronger tie-breaking and demand-aware value ordering?
    """

    def __init__(
        self,
        shift_vars,
        duration_vars,
        num_shifts,
        max_daily_work,
        num_days,
        min_demand_day_shift,
    ):
        super().__init__()

        self._shift_vars = list(shift_vars)
        self._duration_vars = list(duration_vars)
        self._all_vars = self._shift_vars + self._duration_vars

        self._shift_set = set(id(v) for v in self._shift_vars)
        self._num_shifts = num_shifts
        self._max_daily_work = max_daily_work
        self._num_days = num_days
        self._min_demand_day_shift = min_demand_day_shift

        self._meta = {}
        self._paired = {}

        for idx, v in enumerate(self._shift_vars):
            e = idx // num_days
            d = idx % num_days
            self._meta[id(v)] = ("shift", e, d)

        for idx, v in enumerate(self._duration_vars):
            e = idx // num_days
            d = idx % num_days
            self._meta[id(v)] = ("duration", e, d)

        for s_var, d_var in zip(self._shift_vars, self._duration_vars):
            self._paired[id(s_var)] = d_var
            self._paired[id(d_var)] = s_var

        # Static day difficulty score: higher means branch earlier
        self._day_score = []
        for d in range(num_days):
            demand = sum(min_demand_day_shift[d][1:])  # ignore off
            night = min_demand_day_shift[d][1]
            self._day_score.append(10 * demand + 3 * night)

    def _var_priority(self, v):
        kind, e, d = self._meta[id(v)]
        size = v.Size()
        vmin = v.Min()

        # Training days get priority
        training_bonus = 1000 if d < 4 and kind == "shift" else 0

        # Shift vars before duration vars on ties
        kind_bonus = 100 if kind == "shift" else 0

        # Duration vars become more useful if their paired shift is bound
        paired_bonus = 0
        if kind == "duration":
            paired_shift = self._paired[id(v)]
            if paired_shift.Bound():
                paired_bonus = 50

        # higher demand days first
        day_bonus = self._day_score[d]
        return (size, -(training_bonus + kind_bonus + paired_bonus + day_bonus), vmin)

    def _ordered_shift_values(self, d, domain):
        scores = []
        for s in domain:
            if s == 0:
                score = -1000
            else:
                demand = self._min_demand_day_shift[d][s]
                night_penalty = 2 if s == 1 else 0
                score = 10 * demand - night_penalty
            scores.append((score, s))

        scores.sort(reverse=True)
        return [s for _, s in scores]

    def Next(self, solver_):
        best = None
        best_key = None

        for v in self._all_vars:
            if v.Bound():
                continue
            key = self._var_priority(v)
            if best is None or key < best_key:
                best = v
                best_key = key

        if best is None:
            return None

        kind, e, d = self._meta[id(best)]

        if kind == "shift":
            domain = [s for s in range(self._num_shifts) if best.Contains(s)]
            ordered = self._ordered_shift_values(d, domain)
            chosen = ordered[0]

        else:
            paired_shift = self._paired[id(best)]
            if paired_shift.Bound():
                if paired_shift.Value() == 0:
                    chosen = 0
                else:
                    chosen = best.Max()
            else:
                chosen = best.Max()

        return solver_.AssignVariableValue(best, chosen)


def build_search(
    solver,
    shift_vars,
    duration_vars,
    num_shifts,
    max_daily_work,
    num_days,
    min_demand_day_shift,
    strategy="impact"):
    """
    Build a DecisionBuilder for the employee scheduling model.

    strategy:
      "interleaved" — InterleavedSelector: single interleaved pool of shift +
                   duration vars, fail-first variable selection, weighted-random
                   shift values, max-first duration values. All in Python —
                   smart but slow.

      "twophase"   — Pure C++ two-phase: solver.Phase with
                   CHOOSE_MIN_SIZE_LOWEST_MIN + ASSIGN_RANDOM_VALUE for shifts,
                   then ASSIGN_MAX_VALUE for durations. Fast but no domain-aware
                   value weighting.

      "impact"     — OR-Tools DefaultPhase with tuned parameters
                   (CHOOSE_MAX_AVERAGE_IMPACT + SELECT_MAX_IMPACT) and a
                   domain-aware fallback heuristic. Impact-based search learns
                   which variables/values cause the most propagation.
    """
    if strategy == "twophase":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_RANDOM_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MAX_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]
    
    elif strategy == "twophase_ff_max":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MAX_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MAX_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]
    
    elif strategy == "twophase_ff_min_max":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MIN_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MAX_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]

    elif strategy == "twophase_ff_center_max":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_CENTER_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MAX_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]
    
    elif strategy == "twophase_first_min_max":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MIN_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MAX_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]
    
    elif strategy == "twophase_first_rand_min":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_RANDOM_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MIN_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]
    
    elif strategy == "twophase_first_rand_max":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_RANDOM_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MAX_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]
    
    elif strategy == "twophase_ff_random_random":
        phase_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_RANDOM_VALUE,
        )
        phase_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_RANDOM_VALUE,
        )
        db = solver.Compose([phase_shifts, phase_durations])
        return db, [phase_shifts, phase_durations]

    elif strategy == "impact":
        all_vars = list(shift_vars) + list(duration_vars)
        params = pywrapcp.DefaultPhaseParameters()

        params.var_selection_schema = params.CHOOSE_MAX_AVERAGE_IMPACT
        params.value_selection_schema = params.SELECT_MAX_IMPACT

        # Fallback heuristic: when impact scores don't differentiate,
        # DefaultPhase delegates to this builder.
        fallback_shifts = solver.Phase(
            shift_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_RANDOM_VALUE,
        )
        fallback_durations = solver.Phase(
            duration_vars,
            solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
            solver.ASSIGN_MAX_VALUE,
        )
        fallback = solver.Compose([fallback_shifts, fallback_durations])
        params.decision_builder = fallback

        db = solver.DefaultPhase(all_vars, params)
        return db, [db, fallback, fallback_shifts, fallback_durations]

    else:  # "interleaved"
        db = InterleavedSelector(
            shift_vars,
            duration_vars,
            num_shifts,
            max_daily_work,
            num_days,
            min_demand_day_shift,
        )
        return db, [db]
