"""Training duration as a first-class, explicitly-named axis.

Why this module exists
----------------------
Until now the length of a finetuning run was never *named* anywhere.  It was
re-derived, at every reader and every writer independently, as
``DATASET_NAME_TO_EPOCHS[dataset_name]``.  That table encodes the Ilharco-style
schedule, whose per-dataset epoch counts are chosen so that every dataset sees a
comparable number of optimizer steps -- roughly 2,000, with ImageNet the glaring
exception at 9,971:

    Flowers102  147 ep x    8 batches =  1,176 steps
    Cars         35 ep x   58 batches =  2,030 steps
    GTSRB        11 ep x  188 batches =  2,068 steps   <- median
    TinyImageNet  4 ep x  743 batches =  2,972 steps
    ImageNet      1 ep x 9971 batches =  9,971 steps   <- 4.82x the median

So ImageNet is confounded as a donor: it is simultaneously the most diverse task
*and*, by nearly 5x, the longest-trained one.  Separating those two explanations
requires running ImageNet on the common budget and running ordinary datasets on
ImageNet's budget -- neither of which the old scheme can express, because
duration is a table lookup keyed only by dataset name.

Why a multiplier rather than an absolute step count
---------------------------------------------------
The axis introduced here is ``epoch_mult``: a scalar multiplying each dataset's
existing schedule.  ``epoch_mult = 1.0`` is, by definition, exactly what the repo
has always run.  Experiment 1 is ImageNet at 0.25; experiment 2 is ordinary
datasets at 4.0.

Three reasons this beats naming absolute steps:

1. ``DATASET_NAME_TO_EPOCHS`` is never edited.  118 files read it, and in many of
   them it is not a duration table at all -- it is the canonical dataset registry
   (the three ``compute_win_loss_*.py``), the cost basis
   (``002_cost_amortization/compute_costs.py``), and a frozen invariant guarded by
   a RuntimeError tripwire (``005_qv_alignment/compute_euclidean_alignment.py``).
   A multiplier is orthogonal to all three; an absolute step count would have
   forced the table to change meaning and broken every one of them.

2. One scalar is meaningful applied to a *list*.  The transfer configs take
   ``source.dataset_names`` as a list but the duration knob as a scalar, and the
   dispatch runners hard-code all 22 donors as a single CSV.  "every donor at 4x
   its normal schedule" is one number; "every donor at 9,000 steps" is not, and
   would require a per-dataset mapping in config plus a new field in every
   dispatcher's queue-line grammar.

3. ``mult=1`` is true by construction.  Labelling Cars' real 2,030-step run
   ``steps=2500`` would put a number in the path describing a run that never
   happened -- precisely the class of error this axis exists to eliminate.

The multiplier is always written explicitly.  There is no elision, no
absence-means-default: a path that does not carry ``mult=`` predates this axis.

What lives here
---------------
``mult_path_frag``  the ONE place the ``mult=`` string is spelled, following the
                    convention already established by ``awq_path_frag`` in
                    ``src/awq.py`` and ``pv_path_frag`` in ``src/pv_tuning.py``.
``resolve_duration``  turns (dataset, multiplier, loader shape) into the step
                    budget, the epoch count the loop must iterate, and the base
                    epoch count that still names the checkpoint file.
``clamped_warmup``  guards ``cosine_lr`` against short budgets.
"""

import math
import re
from collections import namedtuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Multiplier meaning "the schedule this repo has always run".
UNIT_MULT = 1.0

#: Decimal places retained by ``mult_path_frag``.  A multiplier that is not
#: exactly representable at this precision is rejected rather than silently
#: truncated, because two distinct multipliers collapsing to one fragment would
#: merge two budgets into one directory.
_MULT_PRECISION = 6


Duration = namedtuple("Duration", ["max_steps", "loop_epochs", "base_epochs", "epoch_mult"])


# ---------------------------------------------------------------------------
# Path fragment
# ---------------------------------------------------------------------------
def mult_path_frag(epoch_mult):
    """The canonical ``mult=<m>`` path component.

    Never spell this fragment by hand.  Every writer and every reader of the
    duration axis must call this function, for the same reason ``awq_path_frag``
    exists: the sharpest failure mode of a new path axis is *formatting drift*.
    Hydra will hand this function ``4``, ``4.0`` and ``"4.0"`` depending on how
    an override was typed, and if those produced ``mult=4``, ``mult=4.0`` and
    ``mult=4.0`` respectively, a single budget would silently fork into two
    half-populated directory trees that no reader could reconcile.

    Canonical form is the shortest fixed-point decimal that round-trips:

        1, 1.0, "1.0", "1"  ->  "mult=1"
        0.25, "0.25"        ->  "mult=0.25"
        4, 4.0              ->  "mult=4"
        0.2                 ->  "mult=0.2"

    Raises ValueError on anything non-finite, non-positive, or not exactly
    representable at ``_MULT_PRECISION`` decimal places.
    """
    try:
        value = float(epoch_mult)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"epoch_mult is not a number: {epoch_mult!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"epoch_mult must be finite, got {value!r}")
    if value <= 0:
        raise ValueError(f"epoch_mult must be positive, got {value!r}")

    text = f"{value:.{_MULT_PRECISION}f}".rstrip("0").rstrip(".")
    if float(text) != value:
        raise ValueError(
            f"epoch_mult={value!r} is not exactly representable in "
            f"{_MULT_PRECISION} decimal places (would be written as {text!r}). "
            f"Choose a multiplier that round-trips, e.g. 0.25 rather than 1/3."
        )
    return f"mult={text}"


# ---------------------------------------------------------------------------
# Duration resolution
# ---------------------------------------------------------------------------
def unit_steps(base_epochs, num_batches, accum_steps):
    """Optimizer steps in one 1x run -- today's formula, unchanged.

    This is verbatim what every finetuner already computes as the ``cosine_lr`` /
    ``linear_lr`` length: ``epochs * num_batches // accum_steps``.  It is kept as
    its own function so the equivalence at ``mult=1`` is a property of one
    expression rather than of ten copies.
    """
    return base_epochs * num_batches // accum_steps


def resolve_duration(dataset_name, epoch_mult, num_batches, accum_steps, epochs_table):
    """Resolve a multiplier into a concrete step budget and loop length.

    Returns a ``Duration`` with:

    ``max_steps``    optimizer steps to run.  This is the scheduler length and
                     the point at which the training loop breaks.
    ``loop_epochs``  epochs the loop must iterate over to reach ``max_steps``.
                     For ``epoch_mult > 1`` this exceeds the table value; for
                     ``epoch_mult < 1`` the loop breaks mid-epoch.
    ``base_epochs``  the 1x table value.  This still names the checkpoint file
                     (``classifier_epoch_{base_epochs}.pt``) -- see below.
    ``epoch_mult``   the canonicalised multiplier.

    Why the filename keeps ``base_epochs`` rather than the realized count: the
    realized count depends on ``len(train_loader)``, i.e. on the dataset being
    present on disk.  Every ``evaluate_*.py`` and ``qv_transfer.py`` reconstructs
    checkpoint paths without ever building a dataset, so deriving the filename
    from anything loader-dependent would break the property that a code-only
    clone can reconstruct any load path.  The budget is therefore carried by the
    ``mult=`` directory component and the filename names the *reference*
    schedule; the realized epochs and steps are recorded in the results JSON.

    At ``epoch_mult == 1.0`` this is exactly today's behaviour: ``max_steps``
    equals ``unit_steps(...)`` and ``loop_epochs`` equals ``base_epochs``, so the
    loop runs every epoch to completion and the break never fires.
    """
    if dataset_name not in epochs_table:
        raise KeyError(
            f"{dataset_name!r} has no entry in the epoch table; the multiplier "
            f"axis scales an existing schedule and cannot invent one."
        )
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}")
    if accum_steps <= 0:
        raise ValueError(f"accum_steps must be positive, got {accum_steps}")

    # Canonicalise through the fragment so that anything nameable on disk is
    # exactly what gets trained -- and anything not nameable raises here, before
    # a GPU-hour is spent, rather than at save time.
    value = float(epoch_mult)
    mult_path_frag(value)

    base_epochs = epochs_table[dataset_name]
    steps_1x = unit_steps(base_epochs, num_batches, accum_steps)

    if value == UNIT_MULT:
        # Pinned rather than computed: the whole migration rests on 1x being
        # bit-for-bit today's behaviour, and that must not depend on rounding.
        return Duration(steps_1x, base_epochs, base_epochs, value)

    max_steps = max(1, round(value * steps_1x))
    loop_epochs = max(1, math.ceil(max_steps * accum_steps / num_batches))
    return Duration(max_steps, loop_epochs, base_epochs, value)


TrainingBudget = namedtuple(
    "TrainingBudget", ["loop_epochs", "max_steps", "ckpt_epochs", "duration"]
)


def training_budget(dataset_name, epoch_mult, num_batches, accum_steps,
                    epochs_table, limit_num_epochs=None):
    """The three numbers every finetuner needs, derived once rather than ten times.

    ``loop_epochs``  how many epochs to iterate.
    ``max_steps``    scheduler length, and the optimizer-step count at which the
                     loop breaks.
    ``ckpt_epochs``  the number that goes in the checkpoint filename.

    ``limit_num_epochs`` is the pre-existing dryrun knob and stays orthogonal to
    the multiplier: it *truncates* whatever budget the multiplier asked for and
    never extends it.  Clamping rather than overriding is deliberate -- it is
    what the finetuners already do, and the filename must name what was actually
    written, so the readers are aligned to the writers rather than the reverse.
    """
    duration = resolve_duration(
        dataset_name, epoch_mult, num_batches, accum_steps, epochs_table
    )
    loop_epochs = duration.loop_epochs
    max_steps = duration.max_steps
    ckpt_epochs = duration.base_epochs

    if limit_num_epochs is not None:
        loop_epochs = min(loop_epochs, limit_num_epochs)
        max_steps = min(max_steps, max(1, loop_epochs * num_batches // accum_steps))
        ckpt_epochs = min(ckpt_epochs, limit_num_epochs)

    return TrainingBudget(loop_epochs, max_steps, ckpt_epochs, duration)


def mult_tag(epoch_mult):
    """The canonical multiplier *number*, with no ``mult=`` prefix.

    For the run-id based phases (009, 010), whose identity dicts use a
    dash-separated style (``Cars-seed2038``) rather than the ``key=value``
    directory grammar.  Routed through ``mult_path_frag`` so those phases get the
    same canonicalisation -- otherwise 4 and 4.0 would drift apart there while
    remaining unified everywhere else.
    """
    return mult_path_frag(epoch_mult)[len("mult="):]


ROLE_FRAG_RX = re.compile(
    r"^(?P<role>src|tgt)=(?P<dataset>.+?)_seed=(?P<seed>\d+)_mult=(?P<mult>[0-9]+(?:\.[0-9]+)?)$"
)

RoleFrag = namedtuple("RoleFrag", ["role", "dataset", "seed", "epoch_mult"])


def role_path_frag(role, dataset_name, seed, epoch_mult):
    """The ``src=``/``tgt=`` component of a transfer path.

    Transfer paths name two datasets but carry a single shared ``optim=``
    fragment, so a per-role budget cannot live beside it as a sibling component
    the way it does in the baseline tree -- donor and receiver would have nowhere
    to disagree.  It therefore rides inside these fragments, exactly as ``seed``
    already does:

        src=CIFAR10_seed=2038_mult=1 / tgt=DTD_seed=2038_mult=4

    That asymmetry between the baseline and transfer grammars is not an
    inconsistency introduced here; it is the one the tree already had for seeds.
    """
    if role not in ("src", "tgt"):
        raise ValueError(f"role must be 'src' or 'tgt', got {role!r}")
    return f"{role}={dataset_name}_seed={seed}_{mult_path_frag(epoch_mult)}"


def parse_role_frag(fragment):
    """Inverse of ``role_path_frag`` -> ``RoleFrag(role, dataset, seed, epoch_mult)``.

    Roughly forty scripts -- every ``pick_best_alpha.py`` and most heatmap and
    table plotters -- recover the donor and receiver names by applying their own
    copy of ``^src=(.+?)_seed=\\d+$``.  Those copies all stop matching once the
    fragment carries a multiplier, and several of them swallow the failure and
    render an empty cell.  Parsing through one function instead means the
    grammar has exactly one definition and one place to update.

    Returns None if the fragment does not parse, so callers can count misses
    rather than silently skipping.
    """
    m = ROLE_FRAG_RX.match(fragment)
    if m is None:
        return None
    return RoleFrag(
        m.group("role"), m.group("dataset"), int(m.group("seed")), float(m.group("mult"))
    )


def checkpoint_epochs(dataset_name, epochs_table, limit_num_epochs=None):
    """The epoch count in a checkpoint filename, derived without a dataloader.

    This is the reader-side counterpart of ``training_budget``.  Every
    ``evaluate_*.py`` and every transfer script reconstructs checkpoint paths
    without ever building a dataset, so this deliberately depends only on the
    epoch table and the dryrun limit -- never on ``num_batches``.  That is the
    property that keeps a code-only clone able to name any checkpoint, and it is
    why the filename records the 1x reference schedule rather than the realized
    step count.

    Note the semantics: ``min(table, limit)``, matching what the finetuners
    write.  The readers previously used ``limit if limit is not None else
    table``, which disagrees whenever a limit exceeds the table value -- the
    writer would save one name and the reader look for another.  Readers follow
    writers, not the reverse.
    """
    if dataset_name not in epochs_table:
        raise KeyError(f"{dataset_name!r} has no entry in the epoch table")
    base_epochs = epochs_table[dataset_name]
    if limit_num_epochs is None:
        return base_epochs
    return min(base_epochs, limit_num_epochs)


def run_meta(budget, num_batches, accum_steps, warmup_length):
    """Provenance for the realized budget, which no path component records.

    The ``mult=`` fragment names the multiplier, and the filename names the 1x
    reference schedule, but neither says how many optimizer steps actually ran --
    that depends on the loader shape.  ``002_cost_amortization`` reconstructs
    exactly this quantity today by multiplying epochs by dataset size; recording
    it at source turns that derivation into a measurement.
    """
    return {
        "epoch_mult": budget.duration.epoch_mult,
        "base_epochs": budget.duration.base_epochs,
        "loop_epochs": budget.loop_epochs,
        "max_steps": budget.max_steps,
        "ckpt_epochs": budget.ckpt_epochs,
        "num_batches": num_batches,
        "accum_steps": accum_steps,
        "warmup_length": clamped_warmup(warmup_length, budget.max_steps)
        if warmup_length is not None else None,
    }


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def clamped_warmup(warmup_length, max_steps):
    """Warmup length that stays valid at short budgets.

    ``cosine_lr`` computes ``es = steps - warmup_length`` with no guard, so a run
    shorter than ``wl`` is warmup-only and then divides by zero or a negative
    number.  That was unreachable while duration came from a fixed table -- the
    shortest 1x schedule is Flowers102 at 1,176 steps against ``wl=500`` -- but a
    free multiplier puts it one override away: Flowers102 at ``mult=0.25`` is 294
    steps.

    ``wl`` is deliberately *not* rescaled by the multiplier.  It is fixed across
    every dataset in the existing protocol, so holding it fixed across budgets
    keeps short and long runs comparable to the runs already on disk.  The
    consequence -- warmup occupies a larger *fraction* of a short run -- is a
    real property of the experiment and belongs in the write-up, not hidden
    behind an automatic rescale.
    """
    if max_steps <= 1:
        return 0
    return min(warmup_length, max_steps - 1)
