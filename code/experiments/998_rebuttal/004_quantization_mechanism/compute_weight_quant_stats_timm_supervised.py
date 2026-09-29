"""998 — Weight-space quantization statistics for QV patching (timm supervised)

Reviewers accept *that* `FP_tgt + lambda * QV_donor` survives PTQ better than
`FP_tgt` does, and Proposition 1 says *how much* of the benefit transfers.
Neither says *why*.  The rebuttal page poses the mechanism question directly:

    is quantization performance improved by making the weights easier to
    quantize, or because the quantization error is confined to a subspace the
    model is invariant to?

Call those H1 and H2.  This script measures H1 and is deliberately blind to H2:
it never builds a model for inference, never touches a dataset and never runs a
forward pass, so it cannot see a function-space effect even in principle.  That
blindness is the point.  H1 and H2 both predict "closer to QAT is better", so
they are separable only by contrasting the *size* of the quantization error
against its *effect*, and the size is what lives here.  A flat result is
therefore evidence for H2 by elimination, not a failed measurement.

What is measured
----------------
For every weight `apply_ptq_` would fake-quantize, and every output row of it:

    q_rel      ||W - fq(W)|| / ||W||, the fraction of the row destroyed
    dead_frac  fraction of the row's weights that round to code 0
    outlier    max|W| / rms(W), the ratio that sets the row's scale
    occupancy  normalized entropy over the achievable code levels
    w_norm     ||W||, carried as norm bookkeeping for the control arms

See quant_mechanism_common.py for why these and not a KL or JS distance against
the golden QAT weight histogram, which cannot separate the hypotheses.

Three reference points, not two
-------------------------------
    fp      theta(0) = FP_tgt.  Vanilla PTQ of the receiver's own checkpoint:
            what would be shipped today, and therefore the baseline every arm is
            measured against.
    theta   FP_tgt + lambda * direction.  The measurement.
    golden  QAT_tgt.  The receiver's own QAT run: the ceiling.

`recovery` expresses an arm as the fraction of the way from `fp` to `golden`,
mirroring the accuracy recovery ratio used elsewhere in the project, so the
weight-space and accuracy stories are directly comparable per cell.

Arms
----
`qv` is the real thing.  `random`, `shuffle` and `taskvec` are nulls that nest by
how much of the QV's structure they keep -- norm only, norm plus per-row value
multiset, and a genuinely trained direction carrying task rather than
quantization content.  Without them a metric that moves proves nothing: an
orthogonal displacement adds energy and can lower `outlier` on its own.  The
module docstring of quant_mechanism_common.py argues each one.

Two panels
----------
Sweeping `--lambdas` for a fixed cell holds donor, receiver, model and data
constant so nothing but lambda varies; its accuracy counterpart r(lambda) is
already on disk from 003_lambda_sensitivity.  Running the full donor x receiver
grid at lambda = 1 gives the cross-section whose accuracy counterpart Delta comes
from 001_zero_shot_reframing.  Neither accuracy quantity is recomputed here --
this script emits only the explanatory variables, and the join happens in
aggregate_quant_mechanism.py.  Note the two panels answer to different
references: r(lambda) is measured on the val split against the lambda = 1
default, Delta on the test split against vanilla PTQ.  They must not be blended.

Deviations from the repo conventions, both deliberate
-----------------------------------------------------
This is an analysis script that reads CHECKPOINT_BASE_PATH, which the repo
conventions say analysis scripts never do.  It still builds no model for inference, needs no GPU
and needs no dataset, so it keeps the property the rule exists to protect.  It is
consequently *not* runnable in a code-only clone: the FP and QAT checkpoints
under `storage/` are required.  `--verify-modules` and the `taskvec` arm do
construct a timm model on CPU, the first to check the PTQ module set against the
real `apply_ptq_` and the second to obtain the pretrained backbone.

The output path uses a `weights/` leaf where the 003 grammar has `split={split}`.
A weight-space statistic does not depend on an evaluation split, and inventing
one would imply a provenance these numbers do not have.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import torch

from src.duration import mult_path_frag, mult_tag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import sanitize_timm_model_name

from quant_mechanism_common import (
    ARM_QV,
    ARM_RANDOM,
    ARM_SHUFFLE,
    ARM_TASKVEC,
    BASELINE_FP,
    BASELINE_GOLDEN,
    CELL_ARMS,
    GLOBAL_METRICS,
    compare_codes,
    difference,
    max_abs_deviation,
    measure_mapping,
    measure_patched,
    norm_matched,
    random_direction,
    recovery,
    select_quantized_keys,
    shuffled_direction,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_timm_supervised"
MODALITY = "vision"

EVAL_ROOT_OUT = "evaluations/998_rebuttal/004_quantization_mechanism"

# Keys of the classification head, excluded from the QV by 001's qv_transfer.py.
# Reproduced here so the direction differenced below is the same object the
# accuracy numbers were produced from.  The head is additionally excluded from
# the measured set by skip_modules, so this is belt and braces.
HEAD_PREFIX = "model.head."

# The same-task identity FP + 1.0 * (QAT - FP) == QAT holds algebraically; in
# float32 it holds to rounding.  Anything above this means the checkpoints being
# differenced are not the ones the accuracy numbers came from.
IDENTITY_TOL = 1e-4

# Key of the per-receiver QAT-versus-FP code comparison inside `baselines`.
OVERLAP_GOLDEN_VS_FP = "golden_vs_fp"

MODEL_DISPLAY_NAMES = {
    "vit_base_patch16_224.orig_in21k": "ViT-B/16",
    "vit_large_patch16_224.orig_in21k": "ViT-L/16",
    "vit_huge_patch14_224.orig_in21k": "ViT-H/14",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16 (AugReg2)",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k": "ViT-B/16 (CLIP)",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B/16",
    "deit3_base_patch16_224.fb_in22k_ft_in1k": "DeiT3-B/16 (IN22k)",
    "deit3_large_patch16_224.fb_in1k": "DeiT3-L/16",
    "deit3_large_patch16_224.fb_in22k_ft_in1k": "DeiT3-L/16 (IN22k)",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k": "Swin-L",
}


def _lam_key(lam):
    """Stable string key for a lambda, matching 003's curve_key."""
    return str(lam)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",   required=True, nargs="+",
                        help="timm model names, e.g. vit_base_patch16_224.orig_in21k")
    parser.add_argument("--seed",          required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")

    parser.add_argument("--optim",         required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",            required=True, type=float)
    parser.add_argument("--wd",            required=True, type=float)
    parser.add_argument("--ls",            required=True, type=float)
    parser.add_argument("--wl",            required=True, type=int)
    parser.add_argument("--max-grad-norm", required=True, type=float)
    parser.add_argument("--batch-sizes",   required=True, type=int, nargs="+",
                        help="Batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",      required=True, type=int,
                        help="Bit width of the QAT run the QV was produced by. "
                             "Selects the donor checkpoint; does not quantize anything here.")
    parser.add_argument("--ptq-bits",      required=True, type=int,
                        help="Bit width the statistics are computed at, i.e. the "
                             "PTQ configuration the patched model would be shipped under.")
    parser.add_argument("--granularity",   required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",  required=True, nargs="+",
                        help="Module names skipped during quantization "
                             "(no default: must be specified explicitly).")

    parser.add_argument("--lambdas",       required=True, type=float, nargs="+",
                        help="Scaling factors to measure. Panel 1 wants the 003 grid; "
                             "panel 2 wants just 1.0.")
    parser.add_argument("--arms",          required=True, nargs="+", choices=list(CELL_ARMS),
                        help="Which directions to patch with. 'qv' is the real one; "
                             "the rest are nulls and omitting them makes the result "
                             "uninterpretable.")

    parser.add_argument("--donors",        default=None, nargs="+",
                        help="Donor datasets. Default: every dataset with checkpoints present.")
    parser.add_argument("--receivers",     default=None, nargs="+",
                        help="Receiver datasets. Default: every dataset with checkpoints present.")

    parser.add_argument("--control-seed",  default=None, type=int,
                        help="Seed for the random and shuffle arms. Required when "
                             "either is requested, so a control is reproducible.")

    parser.add_argument("--per-layer",     action="store_true",
                        help="Also emit per-layer summaries. Multiplies output size by "
                             "the layer count for every cell, lambda and arm.")
    parser.add_argument("--verify-modules", action="store_true",
                        help="Build the model once and assert the measured weight set "
                             "matches what apply_ptq_ actually quantizes. Needs timm "
                             "weights; run it once per backbone before trusting output.")

    args = parser.parse_args()

    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")

    needs_control_seed = {ARM_RANDOM, ARM_SHUFFLE} & set(args.arms)
    if needs_control_seed and args.control_seed is None:
        parser.error(
            f"--control-seed is required for arms {sorted(needs_control_seed)}: "
            "an unreproducible null is not a control"
        )

    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
def _fp_ckpt_path(args, model_dir, optim_frag, dataset, epochs, epoch_mult):
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision", FAMILY, "fp", model_dir, dataset,
        optim_frag, mult_path_frag(epoch_mult), f"seed={args.seed}",
        f"classifier_epoch_{epochs}.pt",
    )


def _qat_ckpt_path(args, model_dir, optim_frag, dataset, epochs, epoch_mult):
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision", FAMILY, "qat", model_dir, dataset,
        optim_frag, mult_path_frag(epoch_mult), _qat_frag(args),
        f"seed={args.seed}",
        f"classifier_epoch_{epochs}.pt",
    )


def _load(path):
    """Load a checkpoint, or None if it is not on disk."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
def _out_dir(args):
    """Mirrors the 003 grammar, with a `weights` leaf in place of split=.

    A weight-space statistic has no evaluation split; emitting split=val or
    split=test would claim a provenance these numbers do not have.
    """
    return os.path.join(
        EVAL_ROOT_OUT,
        f"seed={args.seed}",
        # The aggregate spans a whole donor x receiver matrix at one pair of
        # budgets. Without these two components an aggregate computed at a
        # different multiplier would silently overwrite this one.
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        "weights",
    )


# ---------------------------------------------------------------------------
# Module-set verification
# ---------------------------------------------------------------------------
def verify_modules(args, model_name, reference_dataset, measured_keys):
    """Assert the measured weight set equals what apply_ptq_ actually quantizes.

    The measured set is reconstructed from a state dict by rank and name (see
    is_quantized_weight_key), because a state dict carries no module types.  This
    builds the real model and compares against the authoritative list apply_ptq_
    returns.  A mismatch means the analysis is describing a different set of
    tensors than the evaluations quantized, which invalidates every number.
    """
    from torch import nn

    from src.quantization import apply_ptq_
    from src.vision.ilharco_timm_supervised.modeling import ImageClassifier

    classifier = ImageClassifier(
        model_name=model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[reference_dataset],
    )

    all_linear = {
        name for name, module in classifier.named_modules() if isinstance(module, nn.Linear)
    }
    quantized = set(apply_ptq_(
        model=classifier,
        bits=args.ptq_bits,
        granularity=args.granularity,
        skip_modules=frozenset(args.skip_modules),
    ))

    expected = {f"{name}.weight" for name in quantized}
    measured = set(measured_keys)

    missing = sorted(expected - measured)
    extra   = sorted(measured - expected)

    result = {
        "model_name":          model_name,
        "reference_dataset":   reference_dataset,
        "n_linear_modules":    len(all_linear),
        "n_quantized_modules": len(quantized),
        "n_measured_keys":     len(measured),
        "missing_from_measured": missing,
        "extra_in_measured":     extra,
        "ok":                  not missing and not extra,
    }

    del classifier

    if result["ok"]:
        print(f"  [verify-modules] OK: {len(measured)} weights match apply_ptq_ exactly")
    else:
        print(f"  [verify-modules] MISMATCH for {model_name}", file=sys.stderr)
        if missing:
            print(f"    quantized by apply_ptq_ but not measured: {missing}", file=sys.stderr)
        if extra:
            print(f"    measured but not quantized by apply_ptq_: {extra}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Pretrained backbone (for the taskvec arm)
# ---------------------------------------------------------------------------
def load_pretrained_backbone(model_name, reference_dataset):
    """PT state dict, for TV = FP_donor - PT.

    `ImageClassifier.__init__` always loads timm pretrained weights, so no
    checkpoint is needed.  num_classes affects only the head, which is excluded
    from every measured and differenced key set.
    """
    from src.vision.ilharco_timm_supervised.modeling import ImageClassifier

    classifier = ImageClassifier(
        model_name=model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[reference_dataset],
    )
    sd = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
    del classifier
    return sd


# ---------------------------------------------------------------------------
# Measurement of one cell
# ---------------------------------------------------------------------------
def measure_cell(args, fp_tgt_sd, qat_tgt_sd, directions, keys, baseline, same_task):
    """Every requested arm at every requested lambda, for one donor-receiver cell."""
    arms = {}

    for arm, direction in directions.items():
        per_lambda = {}

        for lam in args.lambdas:
            stats = measure_patched(
                base=fp_tgt_sd,
                direction=direction,
                lam=lam,
                keys=keys,
                bits=args.ptq_bits,
                granularity=args.granularity,
                per_layer=args.per_layer,
            )
            stats["recovery"] = {
                metric: recovery(
                    stats["global"][metric],
                    baseline[BASELINE_FP]["global"][metric],
                    baseline[BASELINE_GOLDEN]["global"][metric],
                )
                for metric in GLOBAL_METRICS
            }

            # Which weights the patch kills or revives relative to vanilla PTQ of
            # the receiver's own FP checkpoint.  The reference is FP rather than
            # golden because FP is what would be shipped.
            stats["vs_fp"] = compare_codes(
                keys=keys,
                get_ref=lambda k: fp_tgt_sd[k],
                get_new=lambda k, _lam=lam, _d=direction: fp_tgt_sd[k] + _lam * _d[k],
                bits=args.ptq_bits,
                granularity=args.granularity,
                per_layer=args.per_layer,
            )

            # FP_T + 1.0 * (QAT_T - FP_T) is algebraically QAT_T.  Checking it
            # here rather than in a test keeps the assertion attached to the
            # exact checkpoints this run differenced.
            if same_task and arm == ARM_QV and lam == 1.0:
                stats["identity_max_abs_dev"] = max_abs_deviation(
                    fp_tgt_sd, direction, lam, qat_tgt_sd, keys
                )

            per_lambda[_lam_key(lam)] = stats

        arms[arm] = per_lambda

    return arms


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_golden_overlap(receiver, overlap):
    """The Stage 1b headline: does QAT kill the same weights FP does?

    `jaccard_independent_ref` is printed alongside because 1.0 is not the
    reference to read the Jaccard against -- see jaccard_independent().
    """
    print(
        f"  [{receiver}] QAT vs FP: "
        f"dead_jaccard={overlap['dead_jaccard']:.4f} "
        f"(indep ref {overlap['jaccard_independent_ref']:.4f})  "
        f"code_churn={overlap['code_churn'] * 100:.2f}%  "
        f"killed={overlap['dead_in'] * 100:.2f}% revived={overlap['dead_out'] * 100:.2f}%  "
        f"| moved by codes {overlap['decomposition']['code'] * 100:.1f}% "
        f"scale {overlap['decomposition']['scale'] * 100:.1f}%"
    )


def _print_cell(donor, receiver, arms, baseline):
    """One line per cell: the headline globals against their baseline."""
    fp_q    = baseline[BASELINE_FP]["global"]["q_rel"]
    gold_q  = baseline[BASELINE_GOLDEN]["global"]["q_rel"]

    qv = arms.get(ARM_QV, {})
    unit = qv.get(_lam_key(1.0))
    if unit is None:
        print(f"    {donor:>14s} -> {receiver:<14s}  (no lambda=1 measurement)")
        return

    bits = [
        f"q_rel {fp_q:.4f} -> {unit['global']['q_rel']:.4f} (QAT {gold_q:.4f})",
        f"J={unit['vs_fp']['dead_jaccard']:.4f}",
        f"churn={unit['vs_fp']['code_churn'] * 100:.2f}%",
    ]
    # The control arms are what the QV arm has to be read against: with the QV at
    # ~1.6% of weight norm, most weights cannot cross a threshold regardless of
    # direction, so a high Jaccard on its own means nothing.
    for arm in (ARM_RANDOM, ARM_SHUFFLE, ARM_TASKVEC):
        cell = arms.get(arm, {}).get(_lam_key(1.0))
        if cell is not None:
            bits.append(
                f"{arm}: J={cell['vs_fp']['dead_jaccard']:.4f} "
                f"churn={cell['vs_fp']['code_churn'] * 100:.2f}%"
            )

    print(f"    {donor:>14s} -> {receiver:<14s}  " + "  ".join(bits))


def _fmt(value):
    return "n/a" if value is None else f"{value:+.2f}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    datasets  = sorted(DATASET_NAME_TO_EPOCHS.keys())
    donors    = args.donors    if args.donors    else datasets
    receivers = args.receivers if args.receivers else datasets

    for name in list(donors) + list(receivers):
        if name not in DATASET_NAME_TO_EPOCHS:
            raise ValueError(f"unknown dataset {name!r}")

    skip_modules = frozenset(args.skip_modules)

    models  = {}
    missing = []

    for model_name, batch_size in zip(args.model_names, args.batch_sizes):
        model_dir  = sanitize_timm_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)

        print(f"\nLoading {model_name} ...")

        # Per-receiver anchors, computed once and reused across donors: they
        # depend on neither the donor nor lambda.  Only the summaries are cached,
        # never the state dicts.
        baselines = {}
        identity_checks = {}
        measured_keys = None
        pretrained_sd = None
        verification = None
        cells = []

        # Donor-outer, receiver-inner.  The QV costs two checkpoint loads and is
        # reused across every receiver, whereas receiver-outer would rebuild it
        # per cell: 22 donors x 22 receivers is ~550 loads this way against ~1000
        # the other.
        for donor in donors:
            donor_epochs = DATASET_NAME_TO_EPOCHS[donor]

            fp_src_path  = _fp_ckpt_path(args, model_dir, optim_frag, donor, donor_epochs, args.source_epoch_mult)
            qat_src_path = _qat_ckpt_path(args, model_dir, optim_frag, donor, donor_epochs, args.source_epoch_mult)

            fp_src_sd  = _load(fp_src_path)
            qat_src_sd = _load(qat_src_path)
            if fp_src_sd is None or qat_src_sd is None:
                for path, sd in ((fp_src_path, fp_src_sd), (qat_src_path, qat_src_sd)):
                    if sd is None:
                        missing.append(path)
                continue

            if measured_keys is None:
                measured_keys = [
                    k for k in select_quantized_keys(fp_src_sd, skip_modules)
                    if not k.startswith(HEAD_PREFIX)
                ]
                if not measured_keys:
                    raise ValueError(
                        f"no quantizable weights found for {model_name} with "
                        f"skip_modules={sorted(skip_modules)}"
                    )
                print(f"  measured weights: {len(measured_keys)}")
                if args.verify_modules:
                    verification = verify_modules(args, model_name, receivers[0], measured_keys)

            qv = difference(qat_src_sd, fp_src_sd, measured_keys)

            directions = {}
            if ARM_QV in args.arms:
                directions[ARM_QV] = qv
            if ARM_RANDOM in args.arms:
                gen = torch.Generator().manual_seed(args.control_seed)
                directions[ARM_RANDOM] = random_direction(qv, gen)
            if ARM_SHUFFLE in args.arms:
                gen = torch.Generator().manual_seed(args.control_seed)
                directions[ARM_SHUFFLE] = shuffled_direction(qv, gen)
            if ARM_TASKVEC in args.arms:
                if pretrained_sd is None:
                    pretrained_sd = load_pretrained_backbone(model_name, receivers[0])
                tv = difference(fp_src_sd, pretrained_sd, measured_keys)
                directions[ARM_TASKVEC] = norm_matched(tv, qv)

            del qat_src_sd, fp_src_sd

            for receiver in receivers:
                receiver_epochs = DATASET_NAME_TO_EPOCHS[receiver]

                fp_tgt_path  = _fp_ckpt_path(args, model_dir, optim_frag, receiver, receiver_epochs, args.target_epoch_mult)
                qat_tgt_path = _qat_ckpt_path(args, model_dir, optim_frag, receiver, receiver_epochs, args.target_epoch_mult)

                fp_tgt_sd = _load(fp_tgt_path)
                if fp_tgt_sd is None:
                    missing.append(fp_tgt_path)
                    continue

                qat_tgt_sd = _load(qat_tgt_path)
                if qat_tgt_sd is None:
                    missing.append(qat_tgt_path)
                    del fp_tgt_sd
                    continue

                if receiver not in baselines:
                    baselines[receiver] = {
                        BASELINE_FP: measure_mapping(
                            {k: fp_tgt_sd[k] for k in measured_keys},
                            args.ptq_bits, args.granularity, per_layer=args.per_layer,
                        ),
                        BASELINE_GOLDEN: measure_mapping(
                            {k: qat_tgt_sd[k] for k in measured_keys},
                            args.ptq_bits, args.granularity, per_layer=args.per_layer,
                        ),
                        # The headline Stage 1b comparison: does the receiver's
                        # own QAT checkpoint kill the same weights its FP
                        # checkpoint does, or merely the same number of them?
                        OVERLAP_GOLDEN_VS_FP: compare_codes(
                            keys=measured_keys,
                            get_ref=lambda k: fp_tgt_sd[k],
                            get_new=lambda k: qat_tgt_sd[k],
                            bits=args.ptq_bits,
                            granularity=args.granularity,
                            per_layer=args.per_layer,
                        ),
                    }
                    _print_golden_overlap(receiver, baselines[receiver][OVERLAP_GOLDEN_VS_FP])

                arms = measure_cell(
                    args=args,
                    fp_tgt_sd=fp_tgt_sd,
                    qat_tgt_sd=qat_tgt_sd,
                    directions=directions,
                    keys=measured_keys,
                    baseline=baselines[receiver],
                    same_task=(donor == receiver),
                )

                if donor == receiver:
                    dev = arms.get(ARM_QV, {}).get(_lam_key(1.0), {}).get("identity_max_abs_dev")
                    if dev is not None:
                        identity_checks[receiver] = dev
                        if dev > IDENTITY_TOL:
                            print(
                                f"  [IDENTITY FAIL] {receiver}: "
                                f"max|FP + QV - QAT| = {dev:.3e} > {IDENTITY_TOL:.0e}",
                                file=sys.stderr,
                            )

                cells.append({
                    "donor":     donor,
                    "receiver":  receiver,
                    "same_task": donor == receiver,
                    "arms":      arms,
                })

                _print_cell(donor, receiver, arms, baselines[receiver])

                del fp_tgt_sd, qat_tgt_sd

            del directions

        models[model_name] = {
            "display_name":    MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "model_dir":       model_dir,
            "batch_size":      batch_size,
            "measured_keys":   measured_keys or [],
            "n_measured_keys": len(measured_keys or []),
            "baselines":       baselines,
            "cells":           cells,
            "identity_checks": identity_checks,
            "verification":    verification,
        }

    results = {
        "family":   FAMILY,
        "modality": MODALITY,
        "config": {
            "seed":          args.seed,
            "optim":         args.optim,
            "lr":            args.lr,
            "wd":            args.wd,
            "ls":            args.ls,
            "wl":            args.wl,
            "max_grad_norm": args.max_grad_norm,
            "qat_bits":      args.qat_bits,
            "ptq_bits":      args.ptq_bits,
            "granularity":   args.granularity,
            "skip_modules":  list(args.skip_modules),
            "lambdas":       list(args.lambdas),
            "arms":          list(args.arms),
            "donors":        list(donors),
            "receivers":     list(receivers),
            "control_seed":  args.control_seed,
            "per_layer":     args.per_layer,
            "identity_tol":  IDENTITY_TOL,
            "hypothesis":    (
                "H1: patching makes weights easier to quantize. Predicts q_rel and "
                "dead_frac fall below their fp baseline and track the accuracy gain. "
                "Flat values are evidence for H2, which this script cannot see."
            ),
            "baseline_meaning": (
                "fp = vanilla PTQ of the receiver's FP checkpoint (what would be "
                "shipped); golden = the receiver's own QAT checkpoint (the ceiling); "
                "recovery = (fp - arm) / (fp - golden)."
            ),
        },
        "datasets": datasets,
        "models":   models,
        "missing":  missing,
    }

    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"weight_quant_stats_{FAMILY}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if missing:
        print(f"\n{len(missing)} missing checkpoints", file=sys.stderr)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
