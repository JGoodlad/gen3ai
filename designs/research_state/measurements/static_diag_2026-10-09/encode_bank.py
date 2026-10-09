"""Step 1: re-encode the Lane S bank ONCE at the pin (P_st 6c6d2e09) and cache rows + masks + the
decision metadata + the obs layout. Both arms read the same obs (token encoding is a MODEL choice), so
one encoding serves all checkpoints. Run with PYTHONPATH=<pin>/src (run_at_pin.sh). DESCRIPTIVE.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

PIN = Path("/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09")
BANK = PIN / "designs/research_state/measurements/m5_laneS/bank_v1"
OUT = Path("/home/goodlad/gen3ai_archive/static_diag_2026-10-09")


def pin_head() -> str:
    """The pin's detached HEAD, read from its git admin dir (no git process)."""
    gitdir = (PIN / ".git").read_text().split("gitdir:", 1)[1].strip()
    return (Path(gitdir) / "HEAD").read_text().strip()


def main():
    head = pin_head()
    assert head.startswith("6c6d2e09"), head
    import agents
    assert str(Path(agents.__file__).resolve()).startswith(str(PIN)), agents.__file__
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import reencode

    bank = load_bank(BANK)
    t = time.time()
    rows, masks, gate = reencode(bank, workers=2)
    print(f"re-encoded {len(rows)} rows in {time.time() - t:.0f}s; gate {gate['byte_equal']}/"
          f"{gate['recorded_rows_checked']} obs_as_recorded={gate['obs_as_recorded']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "bank_rows.npz", rows=rows, masks=masks)
    keep = ["id", "battle", "side", "n", "kind", "phase", "legal_bucket", "opp_class", "opp_name",
            "source", "outcome", "cats", "n_legal", "played", "turn"]
    meta = [{k: d.get(k) for k in keep} for d in bank.decisions]
    (OUT / "bank_meta.json").write_text(json.dumps(meta))
    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    (OUT / "obs_layout.json").write_text(json.dumps(lay, indent=1, default=str))
    (OUT / "encode_gate.json").write_text(json.dumps({k: v for k, v in gate.items()}, indent=1, default=str))
    print("obs dim", rows.shape[1], "layout top keys", list(lay)[:40])


if __name__ == "__main__":
    sys.exit(main())
