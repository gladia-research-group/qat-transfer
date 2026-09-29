import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from common.run_id import hydra_override_arg, run_id_flat, run_id_path


def test_run_id_percent_encodes_unsafe_values():
    cfg = {"model": "a/b", "alpha": -0.05}
    assert str(run_id_path(cfg, ["model", "alpha"])) == "model=a%2Fb/alpha=-0.05"
    assert run_id_flat(cfg, ["model", "alpha"]) == "model=a%2Fb,alpha=-0.05"


def test_hydra_override_arg_quotes_string_delimiters():
    value = "evaluations/seed=2038/qat=bits=3/result.json"
    assert hydra_override_arg("outcome_path", value) == (
        'outcome_path="evaluations/seed=2038/qat=bits=3/result.json"'
    )
