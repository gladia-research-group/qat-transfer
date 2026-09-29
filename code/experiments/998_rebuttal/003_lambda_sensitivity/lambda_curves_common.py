"""998 — Shared grid and curve math for the lambda sensitivity analysis.

Family-independent helpers imported by the three compute_lambda_curves_*.py
scripts.  Only the path logic differs between families, and that stays in the
per-family scripts, mirroring 001_zero_shot_reframing.

The reviewers asked how sensitive QV patching is to the scaling lambda.  The
answer is the shape of the accuracy curve around its optimum, summarised here by
four per-cell quantities:

    interval        the widest contiguous lambda range around the optimum that
                    stays above the reference level: how wrong one may be about
                    lambda before the patch stops paying off.
    plateau_width   the width of the region near the peak: how flat the optimum
                    is, i.e. how much precision in lambda actually buys.
    unit_retention  what the data-free default lambda = 1 retains of the best
                    achievable gain.
    unimodal        whether the curve rises then falls, as the local quadratic
                    picture behind Proposition 1 predicts.

Two baseline modes
------------------
`BASELINE_FP_PTQ` measures Delta(lambda) = acc(lambda) - acc(vanilla PTQ), which
is the quantity the paper reports and the one a "win" is defined against.  It
needs a vanilla-PTQ accuracy on the same evaluation split.

`BASELINE_UNIT` measures r(lambda) = acc(lambda) - acc(lambda = 1) instead: how
much accuracy is given up by moving off the data-free default.  It exists
because the dense lambda sweeps were only ever run on the validation split,
where no vanilla-PTQ baseline was ever evaluated, so Delta is not computable
there.  It answers "how sensitive is the method to lambda" directly, and pairs
with 001, which supplies the lambda = 1 versus PTQ comparison on test.

The mode changes the geometry of the curve in two ways that must not be glossed:

    The zero anchor.  Under `BASELINE_FP_PTQ`, Delta(0) = 0 exactly, since
    patching with a zero vector returns the fine-tuned checkpoint.  lambda = 0
    is therefore inserted as a known anchor.  Under `BASELINE_UNIT` no such
    anchor exists: r(0) is the vanilla-PTQ accuracy minus the default's, i.e.
    precisely the unmeasured quantity.  So intervals become censorable on the
    left as well as the right.

    The anchor is inserted at its position in the lambda ordering, not at the
    front.  While every grid was positive it was the same thing, and the anchor
    also bounded every interval from below.  Neither holds once negative lambdas
    are swept: the anchor then sits in the middle of the grid, and an interval
    may legitimately extend below zero -- which is exactly the case a negative
    sweep exists to detect, a donor whose QV is anti-aligned with the receiver's.
    Every walk in `_interval_above` and `_sign_changes` assumes the points are
    ordered by lambda, so prepending would silently corrupt both.

    The reference level.  Under `BASELINE_FP_PTQ` the interval is the region with
    Delta > 0, i.e. beating vanilla PTQ.  Under `BASELINE_UNIT` it is the region
    with r >= 0, i.e. doing at least as well as the default.  The comparison must
    be non-strict there, because r(1) = 0 holds by construction: a strict test
    would clip every interval at exactly 1.0 and report a width of zero.

Grid handling
-------------
Widths are always measured in lambda units with linearly interpolated crossings,
never by counting grid points.  Both grids in use are non-uniform -- the shared
one steps by 0.15 except between 1.0 and 1.05, which are 0.05 apart -- so point
counting would be a biased width estimator.

`SHARED_GRID` is the set of lambdas swept for every model, and restricting to it
makes curves directly comparable across backbones.  Some models were swept on a
finer 0.05..2.0 grid; `discover_grid` reads whatever is actually present on disk
so that resolution is not thrown away.  A grid that stops before the curve comes
back down leaves the interval right-censored, which is flagged rather than
silently truncated at the last grid point.
"""

import bisect
import os
import statistics
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The lambda values shared by every family and every model.
SHARED_GRID = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.05, 1.2, 1.35, 1.5]

# The negative lambdas swept for bert-base.  Deliberately NOT folded into
# SHARED_GRID: that constant defines cross-model comparability for results
# already computed on positive-only grids, and widening it would silently change
# what "shared" means for them.  The negative analysis runs with --grid full,
# which reads whatever is on disk.  Recorded here so the sweep is documented
# where the math that consumes it lives.
#   fine   -- step 0.05, matching the positive grid, so the merged curve is
#             uniformly spaced over [-0.5, 2.0] where the optimum plausibly lies.
#   coarse -- step 0.25, mirroring the positive extent for the symmetry check.
NEGATIVE_GRID_FINE   = [-0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5]
NEGATIVE_GRID_COARSE = [-0.75, -1.0, -1.25, -1.5, -1.75, -2.0]

# Baseline modes; see the module docstring.
BASELINE_FP_PTQ = "fp_ptq"
BASELINE_UNIT   = "unit"

# Delta(0) = 0 by construction, not by measurement.  Only meaningful under
# BASELINE_FP_PTQ, where lambda = 0 is the unpatched receiver.
ANCHOR_ALPHA = 0.0
ANCHOR_DELTA = 0.0

# The data-free default scaling.
UNIT_ALPHA = 1.0

# Under BASELINE_FP_PTQ a cell is "on its plateau" while it retains this
# fraction of its best gain.  Relative to the gain, so it needs the baseline.
PLATEAU_FRAC = 0.9

# Under BASELINE_UNIT there is no gain to take a fraction of, so the plateau is
# defined by an absolute accuracy tolerance instead: the region within this much
# accuracy of the peak.  Accuracies are stored as fractions, so this is 1 point.
PLATEAU_TOL = 0.01

# r(1) = 0 exactly under BASELINE_UNIT, so the reference-level comparison is
# loosened by this much to keep lambda = 1 inside its own interval.
TIE_EPS = 1e-12


# ---------------------------------------------------------------------------
# Grid discovery
# ---------------------------------------------------------------------------
def curve_key(alpha):
    """Stable string key for a grid lambda, for use in JSON objects."""
    return str(alpha)


def alpha_dir(alpha):
    """Directory name encoding a lambda in an evaluation path."""
    return f"qv=alpha={alpha}"


def discover_grid(cell_prefixes, split, strict=False):
    """Lambdas actually evaluated on `split`, as a sorted list of floats.

    `cell_prefixes` are the directories above the qv=alpha=... level, one per
    donor-receiver cell.  The union is taken across cells so that one model has
    one grid: a per-cell grid would make widths incomparable between cells.

    The qv=alpha=/split= layout is identical in every family, so this lives here
    rather than being duplicated in each per-family script.

    Unreadable prefixes and unparseable alpha directories used to be skipped
    silently, which made a path-grammar mismatch indistinguishable from "this
    cell was never swept": the grid simply came back empty and every downstream
    curve was flat.  Both are now counted and reported, and `strict` turns them
    into an error.
    """
    alphas = set()
    unreadable = []
    unparseable = []
    for prefix in cell_prefixes:
        try:
            entries = os.listdir(prefix)
        except OSError as exc:
            unreadable.append((prefix, str(exc)))
            continue
        for entry in entries:
            if not entry.startswith("qv=alpha="):
                continue
            if not os.path.isdir(os.path.join(prefix, entry, f"split={split}")):
                continue
            try:
                alphas.add(float(entry[len("qv=alpha="):]))
            except ValueError:
                unparseable.append(os.path.join(prefix, entry))
                continue

    if unreadable:
        print(f"[WARN] discover_grid: {len(unreadable)}/{len(cell_prefixes)} cell "
              f"prefixes unreadable", file=sys.stderr)
        for prefix, why in unreadable[:10]:
            print(f"  {why}", file=sys.stderr)
    if unparseable:
        print(f"[WARN] discover_grid: {len(unparseable)} alpha directories did not "
              f"parse as floats", file=sys.stderr)
        for path in unparseable[:10]:
            print(f"  {path}", file=sys.stderr)
    if not alphas:
        print(f"[ERROR] discover_grid: no lambda found on split={split} across "
              f"{len(cell_prefixes)} cell prefixes -- the constructed path grammar "
              f"does not match this tree.", file=sys.stderr)
        raise RuntimeError(f"empty lambda grid for split={split}")
    if strict and (unreadable or unparseable):
        raise RuntimeError(
            f"discover_grid: {len(unreadable)} unreadable prefixes, "
            f"{len(unparseable)} unparseable alpha dirs (strict mode)")

    return sorted(alphas)


def resolve_grid(mode, discovered):
    """The grid to use, given the --grid mode and what was found on disk.

    "shared" restricts to SHARED_GRID for cross-model comparability; "full"
    keeps every lambda present.  Either way the result is intersected with what
    exists, so a grid entry never silently becomes a missing-file report.
    """
    if mode == "shared":
        return [a for a in SHARED_GRID if a in set(discovered)]
    return list(discovered)


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------
def _anchored_points(curve, grid, baseline):
    """Grid points as a sorted (alpha, delta) list, anchored where meaningful.

    Returns `(points, anchor_idx)`, where `anchor_idx` is the index of the
    inserted lambda = 0 anchor, or None when no anchor applies.  The anchor goes
    at its sorted position: see the module docstring on why prepending is wrong
    as soon as the grid contains negative lambdas.
    """
    pts = [(a, curve[curve_key(a)]) for a in grid if curve_key(a) in curve]
    if baseline != BASELINE_FP_PTQ:
        return pts, None

    idx = bisect.bisect_left([a for a, _ in pts], ANCHOR_ALPHA)
    # A measured lambda = 0 would be the same point by construction; keep the
    # measurement rather than inserting a duplicate abscissa.
    if idx < len(pts) and pts[idx][0] == ANCHOR_ALPHA:
        return pts, idx
    return pts[:idx] + [(ANCHOR_ALPHA, ANCHOR_DELTA)] + pts[idx:], idx


def _cross(a_lo, d_lo, a_hi, d_hi, level):
    """Lambda at which the segment crosses `level`, by linear interpolation."""
    if d_hi == d_lo:
        return a_lo
    return a_lo + (level - d_lo) * (a_hi - a_lo) / (d_hi - d_lo)


def _interval_above(points, star_idx, level):
    """Widest contiguous interval around points[star_idx] with delta > level.

    Returns (lo, hi, left_censored, right_censored).  Crossings are linearly
    interpolated; an endpoint of the grid that is still above `level` is
    reported at the grid edge and flagged as censored.
    """
    # Walk left until the curve drops to or below the level, then interpolate.
    lo, left_censored = points[0][0], True
    for i in range(star_idx, 0, -1):
        a_prev, d_prev = points[i - 1]
        if d_prev <= level:
            lo = _cross(a_prev, d_prev, points[i][0], points[i][1], level)
            left_censored = False
            break

    # Same to the right.
    hi, right_censored = points[-1][0], True
    for i in range(star_idx, len(points) - 1):
        a_next, d_next = points[i + 1]
        if d_next <= level:
            hi = _cross(points[i][0], points[i][1], a_next, d_next, level)
            right_censored = False
            break

    return lo, hi, left_censored, right_censored


def _sign_changes(deltas):
    """Sign changes in the successive differences, ignoring exact ties."""
    signs = []
    for prev, cur in zip(deltas, deltas[1:]):
        diff = cur - prev
        if diff > 0:
            signs.append(1)
        elif diff < 0:
            signs.append(-1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _as_interval(lo, hi, left_censored, right_censored, extra=None):
    out = {
        "lo":             lo,
        "hi":             hi,
        "width":          hi - lo,
        "left_censored":  left_censored,
        "right_censored": right_censored,
    }
    if extra:
        out.update(extra)
    return out


def curve_stats(curve, grid, baseline):
    """Derived sensitivity statistics for one donor-receiver cell.

    `curve` maps curve_key(alpha) -> value over (a subset of) `grid`, where the
    value is Delta(lambda) under BASELINE_FP_PTQ and r(lambda) under
    BASELINE_UNIT.  Quantities that presuppose a positive gain are None when the
    cell never wins, since neither a plateau nor a retention fraction means
    anything there.
    """
    points, anchor_idx = _anchored_points(curve, grid, baseline)
    # The optimum is taken over the measured grid, never over the anchor: a cell
    # whose curve is negative everywhere should report the lambda that hurt it
    # least, not lambda = 0.  The anchor can now sit anywhere in `points`, so it
    # is dropped by index rather than by slicing off the front.
    if anchor_idx is None:
        measured = points
    else:
        measured = points[:anchor_idx] + points[anchor_idx + 1:]
    if not measured:
        return None

    measured_idx, (lambda_star, delta_max) = max(
        enumerate(measured), key=lambda kv: kv[1][1]
    )
    if anchor_idx is None or measured_idx < anchor_idx:
        star_idx = measured_idx
    else:
        star_idx = measured_idx + 1

    delta_unit = curve.get(curve_key(UNIT_ALPHA))
    deltas = [d for _, d in measured]

    stats = {
        "n_points":       len(measured),
        "lambda_star":    lambda_star,
        "delta_max":      delta_max,
        "delta_unit":     delta_unit,
        "n_sign_changes": _sign_changes(deltas),
        "unimodal":       _sign_changes(deltas) <= 1,
        "interval":       None,
        "plateau":        None,
        "unit_retention": None,
        "unit_regret":    None,
        "unit_is_optimal": None,
        "wins_at_unit":   None,
    }

    if baseline == BASELINE_FP_PTQ:
        stats["wins_at_unit"] = None if delta_unit is None else delta_unit > 0

        # A cell that never beats the baseline has no interval to report: the
        # whole grid sits at or below zero, and lambda_star is the anchor itself.
        if delta_max <= 0:
            return stats

        level = 0.0
        plateau_level = PLATEAU_FRAC * delta_max
        if delta_unit is not None:
            stats["unit_retention"] = delta_unit / delta_max
    else:
        # r(1) = 0 by construction, so delta_max >= 0 always and the peak is
        # never "a loss".  delta_max is exactly what the default gives up.
        stats["unit_regret"]     = delta_max
        stats["unit_is_optimal"] = lambda_star == UNIT_ALPHA

        # Non-strict, so that lambda = 1 is not excluded from its own interval.
        level = -TIE_EPS
        plateau_level = delta_max - PLATEAU_TOL

    lo, hi, left_c, right_c = _interval_above(points, star_idx, level)
    stats["interval"] = _as_interval(lo, hi, left_c, right_c, {"level": level})

    p_lo, p_hi, p_left, p_right = _interval_above(points, star_idx, plateau_level)
    stats["plateau"] = _as_interval(p_lo, p_hi, p_left, p_right,
                                    {"threshold": plateau_level})

    return stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def dist_stats(values):
    """Summary of a distribution, skipping None entries."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "mean": None, "median": None,
                "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n":      len(vals),
        "mean":   statistics.fmean(vals),
        "median": statistics.median(vals),
        "p25":    vals[len(vals) // 4],
        "p75":    vals[(3 * len(vals)) // 4],
        "min":    vals[0],
        "max":    vals[-1],
    }


def pooled_curve(cells, grid):
    """The curve pooled across cells, one summary per grid lambda.

    `frac_positive` at each lambda is the protocol-level answer to the
    sensitivity question: it says how much of the benefit survives a badly
    chosen scaling.  Its meaning follows the baseline mode -- beating vanilla
    PTQ under BASELINE_FP_PTQ, beating the lambda = 1 default under
    BASELINE_UNIT -- and is recorded in the config block of the output JSON.
    """
    out = {}
    for alpha in grid:
        key = curve_key(alpha)
        vals = sorted(c[key] for c in cells if key in c)
        if not vals:
            out[key] = {"n": 0, "mean": None, "median": None,
                        "p25": None, "p75": None, "frac_positive": None}
            continue
        out[key] = {
            "n":             len(vals),
            "mean":          statistics.fmean(vals),
            "median":        statistics.median(vals),
            "p25":           vals[len(vals) // 4],
            "p75":           vals[(3 * len(vals)) // 4],
            "frac_positive": sum(1 for v in vals if v > 0) / len(vals),
        }
    return out


def _frac(flags):
    """Fraction of True among the non-None flags."""
    vals = [f for f in flags if f is not None]
    if not vals:
        return None
    return sum(1 for f in vals if f) / len(vals)


def summarize_cells(pairs, grid, baseline):
    """Aggregate the per-cell statistics of one model over a set of pairs.

    The key set is the same in both baseline modes, so that the aggregate and
    the plotting script need no branching; the mode-specific quantities are
    present as None where they do not apply.
    """
    curves = [p["curve"] for p in pairs if p.get("curve")]
    stats  = [p["stats"] for p in pairs if p.get("stats")]
    if not stats:
        return {"n": 0}

    intervals = [s["interval"] for s in stats if s["interval"] is not None]
    plateaus  = [s["plateau"]  for s in stats if s["plateau"]  is not None]

    return {
        "n":              len(stats),
        "curve":          pooled_curve(curves, grid),
        "interval_width": dist_stats([i["width"] for i in intervals]),
        "interval_lo":    dist_stats([i["lo"]    for i in intervals]),
        "interval_hi":    dist_stats([i["hi"]    for i in intervals]),
        "plateau_width":  dist_stats([p["width"] for p in plateaus]),
        "lambda_star":    dist_stats([s["lambda_star"] for s in stats]),
        "unit_retention": dist_stats([s["unit_retention"] for s in stats]),
        "unit_regret":    dist_stats([s["unit_regret"] for s in stats]),
        "frac_unimodal":       _frac([s["unimodal"] for s in stats]),
        "frac_unit_optimal":   _frac([s["unit_is_optimal"] for s in stats]),
        "frac_wins_at_unit":   _frac([s["wins_at_unit"] for s in stats]),
        "frac_left_censored":  _frac([i["left_censored"]  for i in intervals]),
        "frac_right_censored": _frac([i["right_censored"] for i in intervals]),
        "n_no_interval":  sum(1 for s in stats if s["interval"] is None),
        "baseline":       baseline,
    }
