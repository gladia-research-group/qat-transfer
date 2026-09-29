"""998 — Weight-space quantization statistics at the tuned lambda* (timm supervised)

compute_weight_quant_stats_timm_supervised.py measures whether QV patching
reduces the weight-space quantization error at a *global* list of lambdas —
in practice at lambda = 1, the zero-shot setting the paper's headline claim
rests on.  A reviewer can still object that the mechanism question was asked
at the wrong operating point: the paper also reports results at lambda*, the
per-pair scaling selected on the receiver's validation split, and if H1 (the
patch makes weights easier to quantize) were true only near the tuned optimum,
the lambda = 1 measurement would miss it.

This script closes that gap.  For every donor-receiver cell it measures the
qv arm at that cell's own lambda*, read from 001_zero_shot_reframing's
win_loss aggregate — the same object the accuracy-at-best-alpha numbers were
produced from, so weight-space statistic and accuracy statistic refer to the
identical operating point per cell.  Nothing else changes: metrics, baselines
(fp / golden), recovery and the vs_fp code comparison are all inherited from
the lambda = 1 script, which this one imports rather than re-implements.

Only the qv arm is measured.  The nulls exist to calibrate a *moving* metric,
and their lambda* is undefined — no accuracy sweep selected a scaling for a
random direction.  If the qv arm is flat here too, the lambda = 1 nulls
already say everything the nulls can say.

Output goes to the same `weights/` leaf as the lambda = 1 run, under a
distinct filename (`weight_quant_stats_best_alpha_{family}.json`), so the two
operating points sit side by side without overwriting each other.  Each cell
records the lambda* it was measured at.
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

from src.duration import mult_tag

os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name

from compute_weight_quant_stats_timm_supervised import (
    FAMILY,
    HEAD_PREFIX,
    IDENTITY_TOL,
    MODALITY,
    MODEL_DISPLAY_NAMES,
    OVERLAP_GOLDEN_VS_FP,
    _fp_ckpt_path,
    _load,
    _optim_frag,
    _out_dir,
    _print_cell,
    _print_golden_overlap,
    _qat_ckpt_path,
    measure_cell,
)
from quant_mechanism_common import (
    ARM_QV,
    BASELINE_FP,
    BASELINE_GOLDEN,
    compare_codes,
    difference,
    measure_mapping,
    select_quantized_keys,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIN_LOSS_ROOT = "evaluations/998_rebuttal/001_zero_shot_reframing"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model-names",   required=True, nargs="+",
                        help="timm model names, unsanitized")
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
                        help="one per model name, order-matched")
    parser.add_argument("--qat-bits",      required=True, type=int)
    parser.add_argument("--ptq-bits",      required=True, type=int)
    parser.add_argument("--granularity",   required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",  required=True, nargs="+")
    parser.add_argument("--donors",        default=None, nargs="+",
                        help="restrict donors; default is every dataset")
    parser.add_argument("--receivers",     default=None, nargs="+",
                        help="restrict receivers; default is every dataset")
    parser.add_argument("--per-layer",     action="store_true",
                        help="emit per-layer summaries alongside the pooled ones")

    args = parser.parse_args()

    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")

    # measure_cell reads these off args; the qv arm at one lambda per cell is
    # this script's whole point, and lambdas is rewritten per cell below.
    args.arms = [ARM_QV]
    args.lambdas = []
    args.control_seed = None

    return args


# ---------------------------------------------------------------------------
# lambda* lookup
# ---------------------------------------------------------------------------
def _win_loss_path(args):
    """001's aggregate for this family and config, split=test.

    alpha_best is *selected* on val; the test file merely records it next to
    the test-split accuracies, and is the file every best-alpha figure reads.
    """
    return os.path.join(
        WIN_LOSS_ROOT,
        f"seed={args.seed}",
        # The aggregate spans a whole donor x receiver matrix at one pair of
        # budgets. Without these two components an aggregate computed at a
        # different multiplier would silently overwrite this one.
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        "split=test",
        f"win_loss_{FAMILY}.json",
    )


def load_alpha_best(args, model_name):
    """{(donor, receiver): alpha_best} for one model, from 001's win_loss."""
    path = _win_loss_path(args)
    with open(path) as f:
        win_loss = json.load(f)

    model_entry = win_loss["models"].get(model_name)
    if model_entry is None:
        raise ValueError(f"model {model_name!r} not present in {path}")

    return {
        (pair["donor"], pair["receiver"]): pair["alpha_best"]
        for pair in model_entry["pairs"]
        if pair["alpha_best"] is not None
    }


# ---------------------------------------------------------------------------
# Main
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

        alpha_best = load_alpha_best(args, model_name)

        print(f"\nLoading {model_name} ...")

        baselines = {}
        identity_checks = {}
        measured_keys = None
        cells = []

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

            qv = difference(qat_src_sd, fp_src_sd, measured_keys)
            directions = {ARM_QV: qv}

            del qat_src_sd, fp_src_sd

            for receiver in receivers:
                lam_best = alpha_best.get((donor, receiver))
                if lam_best is None:
                    print(f"  [NO ALPHA_BEST] {donor} -> {receiver}: skipped",
                          file=sys.stderr)
                    continue

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

                # One lambda per cell: the operating point 001 selected on val.
                args.lambdas = [lam_best]

                arms = measure_cell(
                    args=args,
                    fp_tgt_sd=fp_tgt_sd,
                    qat_tgt_sd=qat_tgt_sd,
                    directions=directions,
                    keys=measured_keys,
                    baseline=baselines[receiver],
                    same_task=(donor == receiver),
                )

                if donor == receiver and lam_best == 1.0:
                    dev = arms[ARM_QV][str(1.0)].get("identity_max_abs_dev")
                    if dev is not None:
                        identity_checks[receiver] = dev
                        if dev > IDENTITY_TOL:
                            print(
                                f"  [IDENTITY FAIL] {receiver}: "
                                f"max|FP + QV - QAT| = {dev:.3e} > {IDENTITY_TOL:.0e}",
                                file=sys.stderr,
                            )

                cells.append({
                    "donor":      donor,
                    "receiver":   receiver,
                    "same_task":  donor == receiver,
                    "alpha_best": lam_best,
                    "arms":       arms,
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
            "lambda_source": _win_loss_path(args),
            "arms":          [ARM_QV],
            "donors":        list(donors),
            "receivers":     list(receivers),
            "per_layer":     args.per_layer,
            "identity_tol":  IDENTITY_TOL,
            "hypothesis": (
                "H1 at the tuned operating point: if patching made weights easier "
                "to quantize only near lambda*, the lambda = 1 measurement would "
                "miss it. Each cell is measured at its own alpha_best from 001."
            ),
        },
        "datasets": datasets,
        "models":   models,
        "missing":  missing,
    }

    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"weight_quant_stats_best_alpha_{FAMILY}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
    if missing:
        print(f"missing checkpoints: {len(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
