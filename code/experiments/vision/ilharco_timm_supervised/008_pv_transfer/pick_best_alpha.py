"""Pick the best QV scaling factor (alpha) per (donor, receiver) pair for 008.

Selection is on the receiver's **validation** split; the number worth reporting
is the **test** accuracy at the alpha selected there. Both are printed, because
reporting the val-selected val accuracy would be selecting and reporting on the
same data -- lambda* is not a zero-shot result and the split it was chosen on
has to stay visible.

This is the 008 counterpart of `001_qat_transfer/pick_best_alpha.py`, with one
structural difference. In 001 the alpha grid was swept by external hydra runs,
so that script also emitted re-run commands and wrote `best_alpha_*.json` files
that `qv_transfer.py` read back via `alpha="best"`. `qv_transfer_pv.py` sweeps
`qv.alphas` in-process for both splits, following 005/006/007, so no round-trip
through disk is needed and none is offered: this script is pure analysis over
cells that already exist. The `disk` mode is kept only so the selected alphas
are recorded next to the results they were selected from.

The alpha grid is deliberately the same `ALLOWED_ALPHAS` set 001 uses, so
lambda* distributions are comparable between the QAT-QV and PV-QV phases.

Usage
-----
uv run --active python code/experiments/vision/ilharco_timm_supervised/008_pv_transfer/pick_best_alpha.py \\
    --model-name vit_base_patch16_224.orig_in21k --seed 2038 \\
    --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 \\
    --pv-bits 3 --pv-granularity channel --pv-skip-modules head \\
    --pv-delta 0.9 --pv-tau 0.01 --pv-trust none --pv-pevery 1 --pv-temp 0.0 \\
    --ptq-bits 3 --ptq-granularity channel --ptq-skip-modules head \\
    --output table
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.duration import mult_path_frag, parse_role_frag, role_path_frag
from src.pv_tuning import pv_path_frag
from src.vision.utils import sanitize_timm_model_name


METRIC_KEYS = ("val_accuracy_fp_head_ptq", "val_accuracy_pv_head_ptq")

# Allowed QV scaling factors for the restricted sweep. Any qv=alpha=* directory
# on disk whose alpha is not in this set is silently ignored. Kept identical to
# 001_qat_transfer so lambda* is comparable across the two phases.
ALLOWED_ALPHAS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)
_ALPHA_TOL = 1e-9


def _is_allowed_alpha(alpha: float) -> bool:
    return any(abs(alpha - a) < _ALPHA_TOL for a in ALLOWED_ALPHAS)


EVAL_ROOT_QV = os.path.join(
    os.environ["EVALUATION_BASE_PATH"],
    "vision",
    "ilharco_timm_supervised",
    "008_pv_transfer",
    "vision",
    "qv_transfer_pv",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name",    required=True)
    p.add_argument("--seed",          required=True, type=int)
    p.add_argument("--source-epoch-mult", required=True, type=float,
                   help="Training-budget multiplier of the DONOR checkpoints. "
                        "Selection and the reported test number must use the "
                        "same budget, so this is explicit with no default.")
    p.add_argument("--target-epoch-mult", required=True, type=float,
                   help="Training-budget multiplier of the RECEIVER checkpoints.")

    p.add_argument("--lr",            required=True, type=float)
    p.add_argument("--wd",            required=True, type=float)
    p.add_argument("--ls",            required=True, type=float)
    p.add_argument("--wl",            required=True, type=int)
    p.add_argument("--max-grad-norm", required=True, type=float)
    p.add_argument("--batch-size",    required=True, type=int)

    p.add_argument("--pv-bits",         required=True, type=int)
    p.add_argument("--pv-granularity",  required=True, choices=["tensor", "channel"])
    p.add_argument("--pv-skip-modules", required=True, nargs="+")
    p.add_argument("--pv-delta",        required=True, type=float)
    p.add_argument("--pv-tau",          required=True, type=float)
    p.add_argument("--pv-trust",        required=True,
                   help='PV trust_ratio, or "none" if it was disabled')
    p.add_argument("--pv-pevery",       required=True, type=int)
    p.add_argument("--pv-temp",         required=True, type=float)

    p.add_argument("--ptq-bits",         required=True, type=int)
    p.add_argument("--ptq-granularity",  required=True, choices=["tensor", "channel"])
    p.add_argument("--ptq-skip-modules", required=True, nargs="+")

    p.add_argument("--output",        default="table",
                   choices=["table", "json", "disk"],
                   help="Output format (default: table)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers (mirror qv_transfer_pv.py eval_dir layout)
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if skip_modules else "none"


def _optim_frag(lr, wd, ls, wl, mgn, bs):
    return f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={mgn}_bs={bs}"


def _pv_frag(args):
    trust = None if args.pv_trust.lower() == "none" else float(args.pv_trust)
    return pv_path_frag(
        bits=args.pv_bits,
        granularity=args.pv_granularity,
        skip_modules=args.pv_skip_modules,
        delta_decay=args.pv_delta,
        max_code_change_per_step=args.pv_tau,
        trust_ratio=trust,
        p_every=args.pv_pevery,
        temperature=args.pv_temp,
    )


def _ptq_frag(bits, gran, skip_modules):
    return f"ptq=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _pair_dir(args, src, tgt):
    return os.path.join(
        EVAL_ROOT_QV,
        sanitize_timm_model_name(args.model_name),
        role_path_frag("src", src, args.seed, args.source_epoch_mult),
        role_path_frag("tgt", tgt, args.seed, args.target_epoch_mult),
        _optim_frag(args.lr, args.wd, args.ls, args.wl, args.max_grad_norm, args.batch_size),
        _pv_frag(args),
        _ptq_frag(args.ptq_bits, args.ptq_granularity, args.ptq_skip_modules),
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def _glob_val_results(args):
    pattern = os.path.join(
        _pair_dir(args, "*", "*"),
        "qv=alpha=*",
        "split=val",
        "eval_results.json",
    )

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No val results found for pattern:\n  {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} val result file(s).", file=sys.stderr)
    return files


def _read_test_accuracy(args, src, tgt, alpha, metric_key):
    """Read the test-split accuracy at the val-selected alpha.

    Returns None when the matching test cell has not been run; that is a
    reportable state, not an error, so it is surfaced as N/A rather than
    silently backfilled with the val number.
    """
    test_key = metric_key.replace("val_accuracy_", "test_accuracy_")
    path = os.path.join(
        _pair_dir(args, src, tgt),
        f"qv=alpha={alpha}",
        "split=test",
        "eval_results.json",
    )
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get(test_key)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None


def find_best_alphas_for_metric(args, files, metric_key):
    # best[src_dataset][tgt_dataset] = {"alpha", "val_acc", "test_acc"}
    best = {}
    alpha_re = re.compile(r"^qv=alpha=(.+)$")

    for fpath in files:
        parts = fpath.split(os.sep)

        src_dataset = tgt_dataset = alpha_val = None
        for part in parts:
            role = parse_role_frag(part)
            if role is not None:
                if role.role == "src":
                    src_dataset = role.dataset
                else:
                    tgt_dataset = role.dataset
                continue
            m = alpha_re.match(part)
            if m:
                try:
                    alpha_val = float(m.group(1))
                except ValueError:
                    pass
                else:
                    if not _is_allowed_alpha(alpha_val):
                        alpha_val = None

        if src_dataset is None or tgt_dataset is None or alpha_val is None:
            continue

        try:
            with open(fpath) as f:
                acc = json.load(f).get(metric_key)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [READ ERROR] {fpath}: {e}", file=sys.stderr)
            continue

        if acc is None:
            print(f"  [MISSING KEY] {metric_key} in {fpath}", file=sys.stderr)
            continue

        if src_dataset not in best:
            best[src_dataset] = {}

        prev = best[src_dataset].get(tgt_dataset)
        if prev is None or acc > prev["val_acc"]:
            best[src_dataset][tgt_dataset] = {"alpha": alpha_val, "val_acc": acc}

    # Selection is done; now read the test cell at each selected alpha.
    for src, inner in best.items():
        for tgt, entry in inner.items():
            entry["test_acc"] = _read_test_accuracy(
                args, src, tgt, entry["alpha"], metric_key
            )

    return best


def find_best_alphas(args):
    files = _glob_val_results(args)
    return {
        metric_key: find_best_alphas_for_metric(args, files, metric_key)
        for metric_key in METRIC_KEYS
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def _output_single_table(best, metric_key):
    print(f"### metric: {metric_key}  (alpha selected on val, accuracy reported on test)")

    src_datasets = sorted(best.keys(), key=str.lower)

    tgt_set = set()
    for inner in best.values():
        tgt_set.update(inner.keys())
    tgt_datasets = sorted(tgt_set, key=str.lower)

    src_w = max(len("donor"), max((len(s) for s in src_datasets), default=6))
    tgt_w = max(len("receiver"), max((len(t) for t in tgt_datasets), default=6))
    print(
        f"{'donor':<{src_w}}  {'receiver':<{tgt_w}}  "
        f"{'best_alpha':>10}  {'val_acc':>11}  {'test_acc':>11}"
    )
    print(f"{'-'*src_w}  {'-'*tgt_w}  {'-'*10}  {'-'*11}  {'-'*11}")

    for src in src_datasets:
        for tgt in tgt_datasets:
            entry = best[src].get(tgt)
            if entry is None:
                print(
                    f"{src:<{src_w}}  {tgt:<{tgt_w}}  {'N/A':>10}  "
                    f"{'N/A':>11}  {'N/A':>11}"
                )
                continue
            test_acc = (
                f"{entry['test_acc']:>11.4f}"
                if entry.get("test_acc") is not None
                else f"{'N/A':>11}"
            )
            print(
                f"{src:<{src_w}}  {tgt:<{tgt_w}}  {entry['alpha']:>10.4f}  "
                f"{entry['val_acc']:>11.4f}  {test_acc}"
            )


def output_table(all_best):
    for i, metric_key in enumerate(METRIC_KEYS):
        if i > 0:
            print()
        _output_single_table(all_best[metric_key], metric_key)


def output_json(all_best):
    print(json.dumps(all_best, indent=2, sort_keys=True))


def output_disk(all_best, args):
    written = 0
    for metric_key in METRIC_KEYS:
        best = all_best[metric_key]
        label = metric_key.removeprefix("val_accuracy_")
        for src in sorted(best.keys(), key=str.lower):
            for tgt in sorted(best[src].keys(), key=str.lower):
                entry = best[src][tgt]
                best_alpha_dir = _pair_dir(args, src, tgt)
                os.makedirs(best_alpha_dir, exist_ok=True)
                best_alpha_path = os.path.join(best_alpha_dir, f"best_alpha_{label}.json")
                payload = {
                    metric_key: {
                        "alpha": entry["alpha"],
                        "val_acc": entry["val_acc"],
                        "test_acc": entry.get("test_acc"),
                    }
                }
                with open(best_alpha_path, "w") as f:
                    json.dump(payload, f, indent=2)
                written += 1

    print(f"Wrote {written} best_alpha_*.json file(s).", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    all_best = find_best_alphas(args)

    if not any(all_best[m] for m in METRIC_KEYS):
        print("No best-alpha results found.", file=sys.stderr)
        sys.exit(1)

    if args.output == "table":
        output_table(all_best)
    elif args.output == "json":
        output_json(all_best)
    elif args.output == "disk":
        output_disk(all_best, args)


if __name__ == "__main__":
    main()
