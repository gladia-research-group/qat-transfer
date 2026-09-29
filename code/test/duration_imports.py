"""Static check: nothing uses a duration helper it did not import.

Why this is its own test
------------------------
While threading the ``mult=`` axis through ~140 files, eight scripts ended up
calling ``mult_tag`` without importing it, and six finetuners called
``training_budget`` / ``mult_path_frag`` / ``run_meta`` without importing them.
Every one of those files passed ``py_compile`` -- Python resolves globals at call
time, not at compile time -- and several passed ``--help`` too, because argparse
exits before the path builders ever run.  The failure only surfaced deep inside
a job, after the expensive part.

That is precisely the shape of bug the duration axis exists to eliminate:
something that looks fine, runs for a while, and then either dies late or (worse)
silently writes to the wrong place.  So it gets a cheap static gate rather than
relying on someone exercising every code path.

Run:  uv run --active python code/test/duration_imports.py
"""

import ast
import glob
import pathlib
import sys

# Every public name in src/duration.py that a caller might use.
SYMBOLS = {
    "UNIT_MULT",
    "checkpoint_epochs",
    "clamped_warmup",
    "mult_path_frag",
    "mult_tag",
    "parse_role_frag",
    "resolve_duration",
    "role_path_frag",
    "run_meta",
    "training_budget",
    "unit_steps",
}

SELF = "src/duration.py"


def main():
    offenders = []
    scanned = 0
    for path in sorted(glob.glob("code/**/*.py", recursive=True)):
        if path.endswith(SELF):
            continue
        try:
            tree = ast.parse(pathlib.Path(path).read_text())
        except SyntaxError as exc:
            offenders.append((path, f"SYNTAX: {exc}"))
            continue
        scanned += 1

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.duration":
                imported |= {a.asname or a.name for a in node.names}

        used = {
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        missing = (used & SYMBOLS) - imported
        if missing:
            offenders.append((path, f"uses {sorted(missing)} without importing"))

    print(f"scanned {scanned} files under code/")
    if offenders:
        print(f"\n{len(offenders)} problem(s):")
        for path, why in offenders:
            print(f"  {path}: {why}")
        sys.exit(1)
    print("every duration helper used is imported.")


if __name__ == "__main__":
    main()
