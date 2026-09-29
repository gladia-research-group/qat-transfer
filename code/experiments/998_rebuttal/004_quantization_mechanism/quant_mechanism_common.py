"""998 — Shared weight-space quantization math for the mechanism analysis.

Family-independent helpers imported by the compute_weight_quant_stats_*.py
scripts.  Only path logic differs between families, and that stays in the
per-family script, mirroring 001_zero_shot_reframing and 003_lambda_sensitivity.

The question
------------
The rebuttal asks *why* QV patching helps.  Two mechanisms are consistent with
every number in the paper today:

    H1  the patch makes the weights easier to quantize, so post-training
        quantization destroys less of them;
    H2  the patch leaves the quantization error the same size but moves it into
        directions the network's function is insensitive to.

This module measures H1.  It is deliberately blind to H2: nothing here touches
data, activations or the loss, so a flat result is evidence *for* H2 by
elimination rather than a failure of the measurement.

Why these metrics and not a distributional distance
---------------------------------------------------
The obvious analysis -- compare the patched weight histogram to the golden QAT
one with KL or JS -- cannot answer the question.  Proximity to QAT_tgt in weight
space is exactly what Proposition 1's cos^2 law already predicts, so it separates
neither hypothesis, and both KL and JS are additionally invariant to the *values*
of their bins: a row that pulled its outliers inward by 30%, which is precisely
the H1 signal, can score identically to one that did not.

What the quantizer's form implies is much sharper.  `_quantize_channelwise` is
symmetric absmax with one scale per output row,

    scale[r] = max|W[r]| / qmax,      qmax = 2**(bits-1) - 1  =  3 at 3 bits

so a single outlier sets the grid for its entire row, and the round-trip error
follows almost entirely from one shape statistic.  Treating the rounding error as
roughly uniform on [-scale/2, scale/2] gives

    q_rel[r] = ||E[r]|| / ||W[r]||  ~=  max|W[r]| / (sqrt(12) * qmax * rms(W[r]))

i.e. `q_rel` is the row's max-to-rms ratio divided by a constant (10.39 at 3
bits).  That identity is why `outlier` below is not an independent metric but the
*explanation* of the primary one, and it is why H1 is answerable from state dicts
alone, with no data, no forward pass and no GPU.

The approximation holds only while the rounding residual really is spread across
the row, and that has a regime boundary worth knowing.  On synthetic Gaussian
rows it is near-exact: predicted 0.325 against a measured 0.326.  But it assumes
a decent share of coordinates land on non-zero codes, and when a row's outlier
is extreme enough that almost everything rounds to zero, ||E|| stops being a
rounding residual and becomes the bulk's own norm.  Measured on rows with a 100:1
max-to-bulk ratio, an isotropic perturbation then drives `outlier` *down* 4% while
driving `q_rel` *up* from 0.267 to 0.385 -- the two move in opposite directions.
`dead_frac` is what says which regime a layer is in, which is one more reason it
is co-primary rather than decorative: at 3 bits even a plain Gaussian row already
sits at dead_frac ~ 0.42, so real backbones are not comfortably far from this
boundary.

The metrics, and why each earns its place
-----------------------------------------
    q_rel        fraction of a row destroyed by the quantization round trip.
                 The quantity actually paid for.
    dead_frac    fraction of a row's weights that round to code 0.  Co-primary,
                 because q_rel has a blind spot: a row of small weights dominated
                 by one outlier can have 7 of 8 weights annihilated to zero while
                 q_rel reads only 0.22, since the surviving outlier inflates the
                 denominator ||W||.  dead_frac reports that collapse directly
                 (87.5% in that example), and "vanilla PTQ zeroes X% of all
                 weights, patching brings it to Y%" is a legible sentence in a way
                 q_rel is not.
    outlier      max|W[r]| / rms(W[r]).  The ratio that sets the grid; explains
                 q_rel through the identity above.
    occupancy    normalized entropy of the code histogram over the achievable
                 levels.  Separates "spreads across the grid" from "piles onto
                 three levels", which q_rel alone cannot see.
    w_norm       per-row ||W||.  Not a quality metric: it is norm bookkeeping.
                 An orthogonal control direction strictly adds energy
                 (||W + lam*R||^2 = ||W||^2 + lam^2*||R||^2), so patched norms
                 must be visible in the output rather than silently driving
                 `outlier`.

Both a row-wise distribution and a global pooled value are reported for q_rel and
dead_frac.  The global value is the headline ("what fraction of the whole model
is destroyed"); the distribution is what shows whether an effect is uniform or
concentrated in a few layers, which is itself a likely finding.

Control directions
------------------
Three nulls, nesting by how much of the QV's structure they preserve.  All three
are near-orthogonal to the QV -- in ~86M dimensions a random direction has cos of
standard deviation 1/sqrt(d) ~ 1e-4, and a within-row permutation of a zero-mean
vector is likewise nearly orthogonal to itself -- so orthogonality is *not* what
distinguishes them.  It is what makes a direction structureless, which is the
point of a null.

    random    matches per-key norm only.  Whether it is inert depends on the
              row's tail shape, which is precisely why it must be measured rather
              than assumed.  For Gaussian-ish rows max/rms is scale-invariant, so
              an isotropic direction moves nothing: synthetic Gaussian rows shift
              `outlier` by -0.4% between lambda = 0 and lambda = 1 at a 30%
              relative displacement.  For rows carrying an isolated outlier the
              perturbation inflates the bulk faster than the max, and `outlier`,
              `q_rel` and `dead_frac` all improve by a few percent on their own.
              That offset is small but not zero, so the QV arm has to beat the
              random arm rather than merely move away from lambda = 0.
    shuffle   matches per-key norm *and* the exact per-row value multiset;
              destroys only the assignment of values to coordinates, hence the
              alignment with W.  If H1 metrics improve here too, "easier to
              quantize" is a property of the QV's marginal distribution rather
              than of its alignment.
    taskvec   FP_donor - PT, norm-matched to the QV.  A genuinely trained
              direction encoding task knowledge instead of quantization
              robustness.  The other two establish only "QV != noise"; this one
              tests the paper's actual claim, that the QV carries
              quantization-specific rather than merely trained structure.
"""

import math

import torch

from src.quantization import dequantize_tensor, quantize_tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Arms measured per donor-receiver cell, at every lambda.
ARM_QV       = "qv"
ARM_RANDOM   = "random"
ARM_SHUFFLE  = "shuffle"
ARM_TASKVEC  = "taskvec"
CELL_ARMS    = (ARM_QV, ARM_RANDOM, ARM_SHUFFLE, ARM_TASKVEC)

# Per-receiver anchors, independent of donor and lambda.  BASELINE_FP is what
# would be shipped today (vanilla PTQ of the receiver's FP checkpoint) and is the
# reference every arm is compared against; BASELINE_GOLDEN is the receiver's own
# QAT checkpoint, i.e. the ceiling.
BASELINE_FP     = "fp"
BASELINE_GOLDEN = "golden"
BASELINES       = (BASELINE_FP, BASELINE_GOLDEN)

# Metrics whose row-wise distribution is reported.
ROW_METRICS = ("q_rel", "dead_frac", "outlier", "occupancy", "w_norm")

# Metrics that also have a meaningful model-wide pooled value.
GLOBAL_METRICS = ("q_rel", "dead_frac")

# Guards a division by the norm of an all-zero row.  Matches the clamp already
# applied to abs_max inside _quantize_channelwise.
EPS = 1e-8

# A recovery ratio is reported only when QAT moved the metric by at least this
# fraction of the baseline.  See `recovery` for why a smaller gap makes the
# ratio meaningless rather than merely imprecise.
RECOVERY_MIN_REL = 0.01


# ---------------------------------------------------------------------------
# Key selection
# ---------------------------------------------------------------------------
def is_quantized_weight_key(key, tensor, skip_modules):
    """Whether `key` is a weight that `apply_ptq_` would fake-quantize.

    `apply_ptq_` walks modules and quantizes `nn.Linear.weight` unless some
    ancestor's attribute name appears in `skip_modules`.  A state dict has no
    module types, so this reconstructs the same set structurally: a Linear weight
    is the only 2D `.weight` tensor in these backbones.  LayerNorm weights are 1D
    and `patch_embed.proj.weight` is a 4D Conv2d kernel, so both drop out on rank
    alone rather than by name matching.

    This is a *reconstruction*, and reconstructions drift.  The per-family script
    exposes --verify-modules, which builds the real model, runs `apply_ptq_`, and
    asserts the returned module list matches this selection exactly.  Run it once
    per backbone before trusting any number downstream.

    Note the restriction to nn.Linear: open_clip's fused
    `MultiheadAttention.in_proj_weight` is also 2D but is not an nn.Linear
    weight, and `apply_ptq_open_clip_` handles it separately.  This predicate is
    therefore correct for timm and text backbones only.
    """
    if not key.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False
    parts = key.split(".")
    return not any(p in skip_modules for p in parts)


def select_quantized_keys(state_dict, skip_modules):
    """Sorted keys of every weight `apply_ptq_` would fake-quantize."""
    return sorted(
        k for k, v in state_dict.items()
        if torch.is_tensor(v) and is_quantized_weight_key(k, v, skip_modules)
    )


# ---------------------------------------------------------------------------
# Per-row quantization statistics
# ---------------------------------------------------------------------------
def _occupancy(codes, bits):
    """Normalized entropy of each row's code histogram.

    Bins span the full integer range the dtype allows, qmin..qmax, so a code
    landing outside the achievable band would still be counted rather than
    silently dropped.  Normalization is by log of the *achievable* level count,
    2*qmax+1: because scale = max|W|/qmax, the ratio W/scale never reaches qmin,
    so a perfectly spread row saturates at 2*qmax+1 levels, not 2**bits.  At 3
    bits that is 7 levels, not 8.
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    counts = torch.stack(
        [(codes == v).sum(dim=1) for v in range(qmin, qmax + 1)], dim=1
    ).to(torch.float64)

    p = counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))

    return (-plogp.sum(dim=1)) / math.log(2 * qmax + 1)


def row_metrics(weight, bits, granularity):
    """Quantization statistics for one weight matrix, per output row.

    Returns a dict of 1D float tensors (one entry per row) plus the pooled
    numerators and denominators needed to form model-wide global metrics without
    re-reading the tensor.

    The quantization itself is delegated to `quantize_tensor` /
    `dequantize_tensor` from src.quantization -- the very functions
    `fake_quantize_tensor` and hence `apply_ptq_` call.  Reimplementing the
    rounding here, however faithfully, would put the entire analysis one silent
    edit away from measuring a quantizer the evaluations never used.
    """
    W = weight.detach().to(torch.float32)
    if W.ndim != 2:
        raise ValueError(f"row_metrics expects a 2D weight, got shape {tuple(W.shape)}")

    codes, scale = quantize_tensor(W, bits, granularity)
    W_hat = dequantize_tensor(codes, scale)
    E = W - W_hat

    e_norm = E.norm(dim=1)
    w_norm = W.norm(dim=1)
    rms    = W.pow(2).mean(dim=1).sqrt()
    absmax = W.abs().amax(dim=1)
    dead   = (codes == 0).to(torch.float32).mean(dim=1)

    return {
        "rows": {
            "q_rel":     e_norm / w_norm.clamp(min=EPS),
            "dead_frac": dead,
            "outlier":   absmax / rms.clamp(min=EPS),
            "occupancy": _occupancy(codes, bits).to(torch.float32),
            "w_norm":    w_norm,
        },
        # Pooled sums, kept separate so the model-wide q_rel is the true
        # ||E||/||W|| over every quantized parameter rather than a mean of
        # per-row ratios, which would weight a 768-wide row and a 3072-wide row
        # equally and drift from the quantity a reader has in mind.
        "pooled": {
            "sq_err":   float(E.pow(2).sum()),
            "sq_w":     float(W.pow(2).sum()),
            "n_dead":   int((codes == 0).sum()),
            "n_params": int(W.numel()),
            "n_rows":   int(W.shape[0]),
        },
    }


# ---------------------------------------------------------------------------
# Pairwise comparison of two quantizations
# ---------------------------------------------------------------------------
def jaccard_independent(p):
    """Expected Jaccard of two *independent* random sets of density `p`.

    E|A n B| = p^2 and E|A u B| = 2p - p^2, so the ratio is p/(2-p).  At the
    observed dead-weight density of ~0.5016 this is ~0.335, and that is the
    number a measured Jaccard has to be read against.  1.0 alone is a useless
    reference: with the QV at 1.6% of weight norm most weights cannot cross any
    threshold whatever direction you push them in, so a Jaccard of 0.95 would be
    guaranteed and would demonstrate nothing.  The empirical control arms are the
    stronger reference still; this one is the analytic floor.
    """
    if p is None or p <= 0:
        return None
    return p / (2 - p)


def compare_codes(keys, get_ref, get_new, bits, granularity, per_layer=False):
    """How the quantized representation differs between two weight sets.

    Stage 1 established that PTQ zeroes the same *fraction* of FP and QAT weights
    (50.160% against 50.160%).  It cannot say whether they are the same *weights*,
    and that is the one weight-space mechanism its aggregates are blind to.  This
    answers it directly.

    `get_ref` is the reference (FP, i.e. what would be shipped) and `get_new` the
    thing being compared to it, so `dead_in` counts weights the change *killed*
    and `dead_out` weights it *revived*.

    Also emitted, free from the same quantization pass and treated as a bonus
    rather than a deliverable: how much of the movement in fq(W) comes from the
    integer codes changing versus from the per-row scale drifting.  Because
    scale[r] = max|W[r]|/qmax moves whenever the row's largest weight moves,
    fq(W) would shift even with zero code churn, and the two separate exactly:

        fq(W_new) - fq(W_ref)
            = (c_new - c_ref) * s_ref        <- code term
            + c_ref * (s_new - s_ref)        <- scale term
            + (c_new - c_ref) * (s_new - s_ref)   <- cross term

    That is an algebraic identity, not an approximation, so `residual_rel` in the
    output must come back at floating-point noise; anything larger is a bug in
    this function rather than a finding.
    """
    acc = {
        "inter": 0, "union": 0, "dead_in": 0, "dead_out": 0, "churn": 0,
        "n_params": 0, "dead_ref": 0, "dead_new": 0,
        "sq_total": 0.0, "sq_code": 0.0, "sq_scale": 0.0, "sq_cross": 0.0,
        "sq_residual": 0.0,
    }
    ratio_buffers = []
    layers = {}

    for key in sorted(keys):
        W_ref = get_ref(key).detach().to(torch.float32)
        W_new = get_new(key).detach().to(torch.float32)

        c_ref, s_ref = quantize_tensor(W_ref, bits, granularity)
        c_new, s_new = quantize_tensor(W_new, bits, granularity)

        dead_ref = c_ref == 0
        dead_new = c_new == 0

        inter = int((dead_ref & dead_new).sum())
        union = int((dead_ref | dead_new).sum())
        d_in  = int((~dead_ref & dead_new).sum())
        d_out = int((dead_ref & ~dead_new).sum())
        churn = int((c_ref != c_new).sum())

        cf_ref, cf_new = c_ref.to(torch.float32), c_new.to(torch.float32)
        d_code  = (cf_new - cf_ref) * s_ref
        d_scale = cf_ref * (s_new - s_ref)
        d_cross = (cf_new - cf_ref) * (s_new - s_ref)
        d_total = cf_new * s_new - cf_ref * s_ref

        stats = {
            "inter": inter, "union": union, "dead_in": d_in, "dead_out": d_out,
            "churn": churn, "n_params": int(W_ref.numel()),
            "dead_ref": int(dead_ref.sum()), "dead_new": int(dead_new.sum()),
            "sq_total":    float(d_total.pow(2).sum()),
            "sq_code":     float(d_code.pow(2).sum()),
            "sq_scale":    float(d_scale.pow(2).sum()),
            "sq_cross":    float(d_cross.pow(2).sum()),
            "sq_residual": float((d_code + d_scale + d_cross - d_total).pow(2).sum()),
        }
        for field, value in stats.items():
            acc[field] += value

        ratio_buffers.append((s_new / s_ref.clamp(min=EPS)).flatten())

        if per_layer:
            layers[key] = _overlap_summary(stats, None)

    return _overlap_summary(
        acc, torch.cat(ratio_buffers) if ratio_buffers else None, layers if per_layer else None
    )


def _overlap_summary(acc, scale_ratios, layers=None):
    """Format one accumulator of pairwise counts into the output dict."""
    n = acc["n_params"]
    total = math.sqrt(acc["sq_total"])

    def frac_of_total(field):
        return math.sqrt(acc[field]) / total if total > 0 else None

    p_ref = acc["dead_ref"] / n if n else None
    p_new = acc["dead_new"] / n if n else None
    p_mean = (p_ref + p_new) / 2 if p_ref is not None else None

    out = {
        "dead_jaccard":   acc["inter"] / acc["union"] if acc["union"] else None,
        "jaccard_independent_ref": jaccard_independent(p_mean),
        "dead_in":        acc["dead_in"] / n if n else None,
        "dead_out":       acc["dead_out"] / n if n else None,
        "code_churn":     acc["churn"] / n if n else None,
        "dead_frac_ref":  p_ref,
        "dead_frac_new":  p_new,
        "n_params":       n,
        "decomposition": {
            "code":     frac_of_total("sq_code"),
            "scale":    frac_of_total("sq_scale"),
            "cross":    frac_of_total("sq_cross"),
            # Must be ~0: the three terms sum to the total by algebra.
            "residual_rel": frac_of_total("sq_residual"),
        },
    }
    if scale_ratios is not None:
        out["scale_ratio"] = dist_stats(scale_ratios)
    if layers is not None:
        out["layers"] = layers
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def dist_stats(values):
    """Summary of a 1D tensor's distribution."""
    if values is None or values.numel() == 0:
        return {"n": 0, "mean": None, "median": None,
                "p25": None, "p75": None, "min": None, "max": None}

    v = values.detach().to(torch.float64).flatten()
    q = torch.quantile(v, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))

    return {
        "n":      int(v.numel()),
        "mean":   float(v.mean()),
        "median": float(q[1]),
        "p25":    float(q[0]),
        "p75":    float(q[2]),
        "min":    float(v.min()),
        "max":    float(v.max()),
    }


def measure_weights(keys, get_weight, bits, granularity, per_layer=False):
    """Quantization statistics over a set of weight tensors, pooled and per row.

    `get_weight(key) -> 2D tensor` is called once per key and its result released
    immediately, so a patched model is never materialized in full.  That matters
    at ViT-H scale, where holding theta(lambda) alongside the checkpoint it was
    built from would double a multi-GB footprint for no reason.

    Rows from every layer pool into one distribution per metric.  `per_layer`
    additionally keeps the same summary for each layer, which is what would
    reveal an effect concentrated in, say, late-block mlp.fc2.  It is off by
    default because it multiplies the output size by the layer count for every
    cell, lambda and arm.
    """
    row_buffers = {m: [] for m in ROW_METRICS}
    pooled = {"sq_err": 0.0, "sq_w": 0.0, "n_dead": 0, "n_params": 0, "n_rows": 0}
    layers = {}

    for key in sorted(keys):
        stats = row_metrics(get_weight(key), bits, granularity)

        for metric, values in stats["rows"].items():
            row_buffers[metric].append(values)

        for field, value in stats["pooled"].items():
            pooled[field] += value

        if per_layer:
            p = stats["pooled"]
            layers[key] = {
                "rows": {m: dist_stats(v) for m, v in stats["rows"].items()},
                "global": {
                    "q_rel":     math.sqrt(p["sq_err"] / p["sq_w"]) if p["sq_w"] > 0 else None,
                    "dead_frac": p["n_dead"] / p["n_params"] if p["n_params"] else None,
                    "n_params":  p["n_params"],
                    "n_rows":    p["n_rows"],
                },
            }

    out = {
        "rows": {
            m: dist_stats(torch.cat(bufs) if bufs else None)
            for m, bufs in row_buffers.items()
        },
        "global": {
            "q_rel":     math.sqrt(pooled["sq_err"] / pooled["sq_w"]) if pooled["sq_w"] > 0 else None,
            "dead_frac": pooled["n_dead"] / pooled["n_params"] if pooled["n_params"] else None,
            "n_params":  pooled["n_params"],
            "n_rows":    pooled["n_rows"],
            "n_layers":  len(keys),
        },
    }
    if per_layer:
        out["layers"] = layers
    return out


def measure_mapping(weights, bits, granularity, per_layer=False):
    """`measure_weights` over an already-materialized key -> tensor mapping."""
    return measure_weights(
        list(weights), weights.__getitem__, bits, granularity, per_layer=per_layer
    )


def measure_patched(base, direction, lam, keys, bits, granularity, per_layer=False):
    """Statistics of theta(lambda) = base + lambda * direction, one key at a time.

    The patched tensor for a key exists only for the duration of that key's
    measurement, which is why this takes the pieces rather than a patched state
    dict built by `patch`.
    """
    def get_weight(key):
        return base[key].detach().to(torch.float32) + lam * direction[key]

    return measure_weights(keys, get_weight, bits, granularity, per_layer=per_layer)


def recovery(value, baseline, ceiling):
    """Fraction of the way from the shippable baseline to the QAT ceiling.

    Mirrors the accuracy recovery ratio already used elsewhere in the project:
    0 means the patch changed nothing relative to vanilla PTQ of the receiver's
    FP checkpoint, 1 means it matched the receiver's own QAT run, negative means
    it moved the wrong way.

    Returns None unless QAT itself moved the metric by at least
    RECOVERY_MIN_REL of the baseline.  This is not defensive padding; it is the
    central failure mode of this analysis.  Measured on ViT-B/MNIST, QAT changes
    q_rel from 0.38578 to 0.38585 and dead_frac from 0.501597 to 0.501601 while
    changing accuracy by 84 points, so the denominator is a rounding artifact.
    Dividing by it manufactured recovery ratios of +3.41 and -5.58 out of
    differences in the sixth decimal place -- numbers that read as meaningful
    signal and are entirely noise.  "QAT did not move this metric, so the
    fraction of its movement that transfer recovers is undefined" is the honest
    report, and it is itself the H1 answer.
    """
    if value is None or baseline is None or ceiling is None:
        return None
    denom = baseline - ceiling
    if abs(denom) < max(EPS, RECOVERY_MIN_REL * abs(baseline)):
        return None
    return (baseline - value) / denom


# ---------------------------------------------------------------------------
# Direction construction
# ---------------------------------------------------------------------------
def difference(minuend, subtrahend, keys):
    """`minuend - subtrahend` over `keys`, as float32 tensors."""
    return {
        k: minuend[k].detach().to(torch.float32) - subtrahend[k].detach().to(torch.float32)
        for k in keys
    }


def norm_matched(direction, reference):
    """Rescale each key of `direction` to carry the norm of `reference`'s key.

    Per key rather than globally: the QV's energy is distributed very unevenly
    across layers, and a globally matched control would concentrate its
    displacement wherever the random draw happened to put it, confounding "wrong
    direction" with "wrong layer".
    """
    out = {}
    for k, v in direction.items():
        v_norm = float(v.norm())
        r_norm = float(reference[k].norm())
        out[k] = v * (r_norm / v_norm) if v_norm > EPS else torch.zeros_like(v)
    return out


def random_direction(reference, generator):
    """Isotropic Gaussian direction, per-key norm-matched to `reference`.

    Keys are drawn in sorted order so the draw is reproducible from the seed
    alone, independent of state-dict iteration order.
    """
    out = {}
    for k in sorted(reference):
        ref = reference[k]
        g = torch.randn(ref.shape, generator=generator, dtype=torch.float32)
        g_norm = float(g.norm())
        out[k] = g * (float(ref.norm()) / g_norm) if g_norm > EPS else torch.zeros_like(ref)
    return out


def shuffled_direction(reference, generator):
    """`reference` with its entries permuted within each output row.

    Preserves each row's exact multiset of values -- hence its norm, its maximum
    and its whole marginal distribution -- while destroying which weight each
    displacement lands on.  It is therefore the tightest of the structureless
    nulls: anything it reproduces cannot be an alignment effect.
    """
    out = {}
    for k in sorted(reference):
        v = reference[k]
        rows = v.shape[0]
        cols = v.numel() // rows
        flat = v.reshape(rows, cols)
        idx = torch.argsort(torch.rand(rows, cols, generator=generator), dim=1)
        out[k] = torch.gather(flat, 1, idx).reshape(v.shape)
    return out


def max_abs_deviation(base, direction, lam, target, keys):
    """max |base + lambda*direction - target| over `keys`.

    Used for the same-task identity check.  When donor == receiver and
    lambda = 1, FP_T + (QAT_T - FP_T) is algebraically QAT_T, so this must come
    out at float32 rounding noise.  Anything larger means the QV was built from
    checkpoints that are not the ones the accuracy numbers came from -- a
    mismatched epoch, seed or skip tag -- and every downstream statistic is then
    describing a model nobody evaluated.
    """
    worst = 0.0
    for k in keys:
        theta = base[k].detach().to(torch.float32) + lam * direction[k]
        worst = max(worst, float((theta - target[k].detach().to(torch.float32)).abs().max()))
    return worst
