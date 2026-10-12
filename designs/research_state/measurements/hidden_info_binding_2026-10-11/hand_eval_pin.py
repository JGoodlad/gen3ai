"""THE PINNED TWIN of the hand evaluator (design doc §6.5): the SAME source, the PIN's layout.

It loads ``src/main/hidden_info_study/hand_eval.py`` of the checkout that holds this folder BY FILE PATH
(``importlib.util.spec_from_file_location``), so there is no copy to drift; the module's own imports
(``agents.observation``, ``agents.gen3_data``) resolve to whatever is first on the path, which under
``run_at_pin.sh`` is the PIN's ``src/``. Every output row records the source's sha256 (``source_sha()``).

    run_at_pin.sh <end2|prod> 1 4 10m hand_eval_pin.py           # the pin check: prints it, writes
                                                                 # hand_eval_pin_check_<cp>.json here

The check: every ``get_layout()`` key path the evaluator reads (``hand_eval.LAYOUT_PATHS``) exists at the pin, the
layout resolves (the encoders' own read-backs find every cell), and the resolved offsets are printed for the
cross-pin comparison.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "src" / "main" / "hidden_info_study" / "hand_eval.py"


@functools.lru_cache(maxsize=1)
def load() -> ModuleType:
    """The evaluator module, from SOURCE (never a copy)."""
    spec = importlib.util.spec_from_file_location("hib_hand_eval", SOURCE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hib_hand_eval"] = mod          # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def source_sha() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def layout():
    return load().resolve_layout()


def _walk(d: Any, path: str) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            raise KeyError(path)
        d = d[k]
    return d


def check() -> Dict[str, Any]:
    """The pin check (no outcome is read): the key paths, the resolved layout, the source sha."""
    import agents
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    he = load()
    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    paths: Dict[str, Any] = {}
    missing = []
    for p in he.LAYOUT_PATHS:
        try:
            v = _walk(lay, p)
            paths[p] = v if isinstance(v, (int, float, str)) else json.loads(json.dumps(v))
        except KeyError:
            missing.append(p)
    resolved = None
    err = None
    if not missing:
        try:
            L = he.resolve_layout()
            resolved = {k: (sorted([list(x) if isinstance(x, tuple) else x, y] for x, y in v.items())
                            if isinstance(v, dict) else sorted(v) if isinstance(v, frozenset) else v)
                        for k, v in dataclasses.asdict(L).items()}
        except Exception as exc:  # noqa: BLE001 — reported, and the check fails
            err = f"{type(exc).__name__}: {exc}"
    return {"agents_from": str(Path(agents.__file__).resolve().parent), "pin": os.environ.get("VDP_PIN_SHA"),
            "source": str(SOURCE), "source_sha256": source_sha(), "paths_checked": len(he.LAYOUT_PATHS),
            "missing": missing, "resolve_error": err, "ok": not missing and err is None,
            "resolved": resolved, "paths": paths}


def main() -> None:
    cp = os.environ.get("VDP_CP")
    if cp not in ("end2", "prod"):
        raise SystemExit("run through run_at_pin.sh (VDP_CP unset)")
    pin = os.environ.get("VDP_PIN", "")
    import agents

    if not str(Path(agents.__file__).resolve()).startswith(pin):
        raise SystemExit(f"agents imported from {agents.__file__}, not the pin checkout {pin}")
    out = check()
    out["cp"] = cp
    print(json.dumps({k: out[k] for k in ("cp", "pin", "agents_from", "source_sha256", "paths_checked", "missing",
                                          "resolve_error", "ok")}, indent=1))
    print(json.dumps(out["resolved"]))
    (HERE / f"hand_eval_pin_check_{cp}.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
