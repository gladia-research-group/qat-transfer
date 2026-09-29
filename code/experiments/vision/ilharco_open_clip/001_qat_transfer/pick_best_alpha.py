"""Pick the best QV scaling factor (alpha) per (source, target) pair.

Reads val-split eval_results.json files produced by qv_transfer.py,
finds the alpha that maximises val_accuracy_patched_qat_ptq for each
(source_dataset, target_dataset) combination, and outputs the result
as a table, JSON, or ready-to-run hydra+submitit commands.

Usage
-----
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py \
    --model-name ViT-B-32 --pretrained openai --seed 1 \
    --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 \
    --bits 8 --granularity channel --skip-modules classification_head \
    --output table
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[4]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from dotenv import load_dotenv
load_dotenv()

from src.duration import mult_path_frag, parse_role_frag, role_path_frag
from src.vision.utils import sanitize_open_clip_model_name


METRIC_KEY = "val_accuracy_patched_qat_ptq"

# Allowed QV scaling factors for the restricted sweep. Any qv=alpha=* directory
# on disk whose alpha is not in this set is silently ignored.
ALLOWED_ALPHAS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)
_ALPHA_TOL = 1e-9


def _is_allowed_alpha(alpha: float) -> bool:
    return any(abs(alpha - a) < _ALPHA_TOL for a in ALLOWED_ALPHAS)

EVAL_ROOT_QV = os.path.join(
    os.environ["EVALUATION_BASE_PATH"],
    "vision",
    "ilharco_open_clip",
    "001_qat_transfer",
    "vision",
    "qv_transfer",
)

SCRIPT_PATH = "code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py"

# SLURM parameters for sbatch mode (must match config/hydra/launcher/submitit_slurm.yaml)
# Partition and account are cluster-specific and are not hardcoded: export
# SBATCH_PARTITION / SBATCH_ACCOUNT (see .env.example), which sbatch reads itself.
_SLURM_GRES = "gpu:1"
_SLURM_CPUS = 8
_SLURM_MEM = "128G"
_SLURM_PROJECT_ROOT = os.environ.get("SLURM_PROJECT_ROOT") or os.getcwd()
_SLURM_LOG_DIR = f"{_SLURM_PROJECT_ROOT}/logs/config/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer"
_SLURM_SETUP = (
    f"cd {_SLURM_PROJECT_ROOT}"
    f" && export PYTHONPATH='{_SLURM_PROJECT_ROOT}/code:$PYTHONPATH'"
    + (f" && export TORCHINDUCTOR_CACHE_DIR='{os.environ['TORCHINDUCTOR_CACHE_DIR']}'"
       if os.environ.get("TORCHINDUCTOR_CACHE_DIR") else "")
    + f" && mkdir -p {_SLURM_LOG_DIR}"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name",    required=True)
    p.add_argument("--pretrained",    required=True)
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

    p.add_argument("--bits",          required=True, type=int)
    p.add_argument("--granularity",   required=True, choices=["tensor", "channel"])
    p.add_argument("--skip-modules",  required=True, nargs="+")

    p.add_argument("--slurm-timeout",  required=True, type=int,
                   help="SLURM job timeout in minutes")
    p.add_argument("--slurm-job-name", required=True,
                   help="SLURM job name")

    p.add_argument("--output",        default="table",
                   choices=["table", "json", "commands", "commands-bg", "commands-sbatch", "disk"],
                   help="Output format (default: table)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers (mirror qv_transfer.py eval_dir layout)
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if skip_modules else "none"


def _optim_frag(lr, wd, ls, wl, mgn, bs):
    return f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={mgn}_bs={bs}"


def _qat_frag(bits, gran, skip_modules):
    return f"qat=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _ptq_frag(bits, gran, skip_modules):
    return f"ptq=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def find_best_alphas(args):
    model_dir  = sanitize_open_clip_model_name(args.model_name, args.pretrained)
    optim      = _optim_frag(args.lr, args.wd, args.ls, args.wl, args.max_grad_norm, args.batch_size)
    qat        = _qat_frag(args.bits, args.granularity, args.skip_modules)
    ptq        = _ptq_frag(args.bits, args.granularity, args.skip_modules)

    # Glob: src=*_seed=*/tgt=*_seed=*/<optim>/<qat>/<ptq>/qv=alpha=*/split=val/eval_results.json
    pattern = os.path.join(
        EVAL_ROOT_QV, model_dir,
        f"src=*_seed={args.seed}_{mult_path_frag(args.source_epoch_mult)}",
        f"tgt=*_seed={args.seed}_{mult_path_frag(args.target_epoch_mult)}",
        optim, qat, ptq,
        "qv=alpha=*",
        "split=val",
        "eval_results.json",
    )

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No val results found for pattern:\n  {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} val result file(s).", file=sys.stderr)

    # best[src_dataset][tgt_dataset] = {"alpha": float, "acc": float}
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
                        continue

        if src_dataset is None or tgt_dataset is None or alpha_val is None:
            print(f"  [SKIP] could not parse: {fpath}", file=sys.stderr)
            continue

        try:
            with open(fpath) as f:
                acc = json.load(f).get(METRIC_KEY)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [READ ERROR] {fpath}: {e}", file=sys.stderr)
            continue

        if acc is None:
            print(f"  [MISSING KEY] {METRIC_KEY} in {fpath}", file=sys.stderr)
            continue

        if src_dataset not in best:
            best[src_dataset] = {}

        prev = best[src_dataset].get(tgt_dataset)
        if prev is None or acc > prev["acc"]:
            best[src_dataset][tgt_dataset] = {"alpha": alpha_val, "acc": acc}

    return best


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def output_table(best):
    src_datasets = sorted(best.keys(), key=str.lower)

    tgt_set = set()
    for inner in best.values():
        tgt_set.update(inner.keys())
    tgt_datasets = sorted(tgt_set, key=str.lower)

    # Header
    src_w = max(len("source"), max((len(s) for s in src_datasets), default=6))
    tgt_w = max(len("target"), max((len(t) for t in tgt_datasets), default=6))
    print(f"{'source':<{src_w}}  {'target':<{tgt_w}}  {'best_alpha':>10}  {'val_acc_ptq':>11}")
    print(f"{'-'*src_w}  {'-'*tgt_w}  {'-'*10}  {'-'*11}")

    for src in src_datasets:
        for tgt in tgt_datasets:
            entry = best[src].get(tgt)
            if entry is None:
                print(f"{src:<{src_w}}  {tgt:<{tgt_w}}  {'N/A':>10}  {'N/A':>11}")
            else:
                print(f"{src:<{src_w}}  {tgt:<{tgt_w}}  {entry['alpha']:>10.4f}  {entry['acc']:>11.4f}")


def output_json(best):
    print(json.dumps(best, indent=2, sort_keys=True))


def _build_cmd(args, src, tgt, alpha, skip_list, *, submitit=True):
    parts = [f"uv run --active python {SCRIPT_PATH}"]
    if submitit:
        parts.extend([
            "-m hydra/launcher=submitit_slurm",
            f"hydra.launcher.timeout_min={args.slurm_timeout}",
            f"hydra.job.name={args.slurm_job_name}",
        ])
    parts.extend([
        f"model_name={args.model_name}",
        f"pretrained={args.pretrained}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"wd={args.wd}",
        f"ls={args.ls}",
        f"wl={args.wl}",
        f"max_grad_norm={args.max_grad_norm}",
        f"'source.dataset_names=[{src}]'",
        f"source.seed={args.seed}",
        f"source.epoch_mult={args.source_epoch_mult}",
        f"'target.dataset_names=[{tgt}]'",
        f"target.seed={args.seed}",
        f"target.epoch_mult={args.target_epoch_mult}",
        f"qat.bits={args.bits}",
        f"qat.granularity={args.granularity}",
        f"'qat.skip_modules=[{skip_list}]'",
        f"qv.alpha={alpha}",
        f"ptq.bits={args.bits}",
        f"ptq.granularity={args.granularity}",
        f"'ptq.skip_modules=[{skip_list}]'",
        "eval_split=test",
    ])
    return " ".join(parts)


def _sbatch_wrap(inner_cmd, args):
    return (
        f"sbatch"
        f" --gres={_SLURM_GRES}"
        f" --cpus-per-task={_SLURM_CPUS}"
        f" --mem={_SLURM_MEM}"
        f" --time={args.slurm_timeout}"
        f" --job-name={args.slurm_job_name}"
        f" --output={_SLURM_LOG_DIR}/%x_%j.out"
        f" --error={_SLURM_LOG_DIR}/%x_%j.err"
        f" --wrap=\"{_SLURM_SETUP} && {inner_cmd}\""
    )


def output_commands(best, args, *, bg=False):
    src_datasets = sorted(best.keys(), key=str.lower)
    skip_list = ",".join(sorted(args.skip_modules))
    total = sum(len(best[src]) for src in src_datasets)
    current = 0

    for src in src_datasets:
        for tgt in sorted(best[src].keys(), key=str.lower):
            current += 1
            entry = best[src][tgt]
            cmd = _build_cmd(args, src, tgt, entry["alpha"], skip_list, submitit=True)
            print(f"\n\necho '[progress] {current}/{total} src={src} tgt={tgt}'\n\n")
            print(f"{cmd} &" if bg else cmd)

    if bg:
        print("\nwait")


def output_commands_sbatch(best, args):
    src_datasets = sorted(best.keys(), key=str.lower)
    skip_list = ",".join(sorted(args.skip_modules))
    total = sum(len(best[src]) for src in src_datasets)
    current = 0

    for src in src_datasets:
        for tgt in sorted(best[src].keys(), key=str.lower):
            current += 1
            entry = best[src][tgt]
            inner = _build_cmd(args, src, tgt, entry["alpha"], skip_list, submitit=False)
            print(f"\n\necho '[progress] {current}/{total} src={src} tgt={tgt}'\n\n")
            print(_sbatch_wrap(inner, args))


def output_disk(best, args):
    model_dir = sanitize_open_clip_model_name(args.model_name, args.pretrained)
    optim = _optim_frag(args.lr, args.wd, args.ls, args.wl, args.max_grad_norm, args.batch_size)
    qat = _qat_frag(args.bits, args.granularity, args.skip_modules)
    ptq = _ptq_frag(args.bits, args.granularity, args.skip_modules)

    written = 0
    for src in sorted(best.keys(), key=str.lower):
        for tgt in sorted(best[src].keys(), key=str.lower):
            entry = best[src][tgt]
            best_alpha_dir = os.path.join(
                EVAL_ROOT_QV, model_dir,
                role_path_frag("src", src, args.seed, args.source_epoch_mult),
                role_path_frag("tgt", tgt, args.seed, args.target_epoch_mult),
                optim, qat, ptq,
            )
            os.makedirs(best_alpha_dir, exist_ok=True)
            best_alpha_path = os.path.join(best_alpha_dir, "best_alpha.json")
            payload = {METRIC_KEY: {"alpha": entry["alpha"], "acc": entry["acc"]}}
            with open(best_alpha_path, "w") as f:
                json.dump(payload, f, indent=2)
            written += 1

    print(f"Wrote {written} best_alpha.json file(s).", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    best = find_best_alphas(args)

    if not best:
        print("No best-alpha results found.", file=sys.stderr)
        sys.exit(1)

    if args.output == "table":
        output_table(best)
    elif args.output == "json":
        output_json(best)
    elif args.output == "commands":
        output_commands(best, args)
    elif args.output == "commands-bg":
        output_commands(best, args, bg=True)
    elif args.output == "commands-sbatch":
        output_commands_sbatch(best, args)
    elif args.output == "disk":
        output_disk(best, args)


if __name__ == "__main__":
    main()
