"""THE OBS GOLDEN, owned by the Rust core (P6 of the poke-env retirement, 2026-10-08).

``golden_obs_fixture.json`` is the per-decision sha256 of the trainee's observation row over a fixed battle set — the
value-neutrality LINCHPIN: if a row any training run reads changes, this says so in the routine gate. Until P6 the
battles were PLAYED by the Python capture (``golden_obs_capture``: poke-env players over the bridge) and the rows
written by the Python encoder; the core was held to them (``rust_core_parity_test``, deleted in slice 6c). The battles are now BANKED as
input logs (``golden_obs_battles.json``, recorded once while both stacks agreed: core rows == Python rows ==
the committed hashes) and the golden is the CORE's: each log is replayed through ``core_events --obs`` (the step
chain + the parse-chain gate that ``sim_bridge``'s core observation mode ships) and the trainee's (p1's) rows are
hashed in order.

    python -m agents.training.golden_obs_core --check    # exit 0 = the core reproduces every committed hash
    python -m agents.training.golden_obs_core --write    # re-record the hashes (a DELIBERATE obs change only)

``--write`` prints the first changed decision and how many moved; review that diff before committing it (an obs
change that moves more decisions than you meant is the finding). The hashes file keeps its name and its JSON shape
(``main.policy_spectrum.bank.encoder_identity`` fingerprints its bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Both ship BESIDE this module (a module locating its own data files, not repo-root discovery).
HASHES_PATH = Path(__file__).with_name("golden_obs_fixture.json")
BATTLES_PATH = Path(__file__).with_name("golden_obs_battles.json")
#: The viewer whose rows the golden hashes (the capture's trainee: p1).
VIEWER = 0


def vector_hashes(vectors) -> List[str]:
    """Per-decision sha256 of the float32 bytes — the byte-exact value fingerprint."""
    return [hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest() for v in vectors]


def load_battles() -> list:
    from agents.battle.core_replay import RecordedBattle

    d = json.loads(BATTLES_PATH.read_text())
    return [RecordedBattle(label=b["label"], format_id=b["format_id"], seed=b["seed"], p1=b["p1"], p2=b["p2"],
                           commands=[list(c) for c in b["commands"]], init_seed=bool(b.get("init_seed")),
                           quick_claw=bool(b.get("quick_claw")))
            for b in d["logs"]]


def core_rows(battles=None) -> List[np.ndarray]:
    """The core's rows for the trainee at every decision of the golden battles, in order."""
    from agents.battle.core_obs import wrap_row
    from agents.battle.core_replay import run_core

    out: List[np.ndarray] = []
    for res in run_core(battles if battles is not None else load_battles(), trackers=True, obs=True):
        if not res["ok"]:
            raise RuntimeError(f"the core refused golden battle {res.get('label')}: {res.get('error')}")
        out.extend(wrap_row(c["obs"]) for c in res["trackers"][VIEWER] if "obs" in c)
    return out


def compare(got: List[str], want: List[str]) -> Tuple[int, int]:
    """``(first differing decision or -1, how many differ)`` — a length change counts every unmatched decision."""
    n = max(len(got), len(want))
    diff = [i for i in range(n) if i >= len(got) or i >= len(want) or got[i] != want[i]]
    return (diff[0] if diff else -1), len(diff)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    rows = core_rows()
    got = vector_hashes(rows)
    golden = json.loads(HASHES_PATH.read_text())
    first, n = compare(got, golden["hashes"])
    if a.check:
        if n:
            print(f"[golden] ✗ {n} of {len(got)} decisions differ from the committed golden; first at decision {first}")
            return 1
        print(f"[golden] ✓ the core reproduces all {len(got)} committed hashes")
        return 0
    from agents.battle.core_obs import obs_dim

    golden.update({"_comment": "Per-decision sha256 of the trainee's observation row over the banked golden battles "
                               "(golden_obs_battles.json), written by the RUST core. Regenerate ONLY for a deliberate "
                               "obs change: python -m agents.training.golden_obs_core --write",
                   "obs_dim": int(obs_dim()), "n_decisions": len(got), "hashes": got})
    HASHES_PATH.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"[golden] wrote {len(got)} hashes; {n} changed" + (f" (first at decision {first})" if n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
