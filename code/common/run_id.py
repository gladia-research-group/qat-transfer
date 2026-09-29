"""Stable run identities and full-config collision guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


def _plain(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return value


def _text(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _component(key: str, value: Any) -> str:
    safe = "-._~"
    return f"{quote(str(key), safe=safe)}={quote(_text(value), safe=safe)}"


def run_id_path(config: Mapping[str, Any], params: Sequence[str]) -> Path:
    return Path(*[_component(key, config[key]) for key in params])


def run_id_flat(config: Mapping[str, Any], params: Sequence[str]) -> str:
    return ",".join(_component(key, config[key]) for key in params)


def hydra_override_arg(key: str, value: Any) -> str:
    value = _plain(value)
    if isinstance(value, str):
        # Hydra parses the override after the shell has removed shell quoting.
        # Keep strings quoted in Hydra's grammar so embedded delimiters such as
        # ``=`` are treated as value data rather than override syntax.
        return f"{key}={json.dumps(value)}"
    return f"{key}={_text(value)}"


def guard_run_config(
    config: Mapping[str, Any], params: Sequence[str], eval_dir: str | Path
) -> Path:
    """Atomically snapshot a resolved config and reject run-id collisions."""
    resolved = {str(k): _plain(v) for k, v in dict(config).items()}
    for key in params:
        if key not in resolved:
            raise KeyError(f"run_id parameter missing from config: {key}")

    path = Path(eval_dir) / ".run_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != resolved:
            keys = sorted(set(previous) | set(resolved))
            differing = [k for k in keys if previous.get(k) != resolved.get(k)]
            raise RuntimeError(
                "run_id collision — re-elect run_id or migrate; differing params: "
                + ", ".join(differing)
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path
