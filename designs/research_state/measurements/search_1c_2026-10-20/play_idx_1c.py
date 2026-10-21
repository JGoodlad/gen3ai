"""STAGE 1c, PHASE B's PLAY INDICES, computed at the DRIVER checkout (``c1c_common.play_indices``: a corpus arm from its
STRICT weights, AMENDMENT A1; F8 / C8 their first 8; any other weighted arm the driver's stored resample). The PLAY
side runs at X26's pin, whose ``main`` predates ``main.value_search.opp_likelihood``, so it cannot compute them itself:
this writes them once to ``<archive>/play/play_idx.json`` (with the sha256 of its content) and ``play_1c.py`` reads
them. ``read_1c.py b`` recomputes every played index with the same function and STOPS on any mismatch.

    python -I play_idx_1c.py      # from the DRIVER checkout, after result_1c_a.json exists; refuses to rewrite a different file
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1c_common as K  # noqa: E402

if str(K.REPO / "src") not in sys.path:
    sys.path.insert(1, str(K.REPO / "src"))


def main() -> None:
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(K.REPO / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not this checkout: run with -I from the DRIVER checkout")
    ra = json.loads((HERE / "result_1c_a.json").read_text())
    if not ra.get("bar", {}).get("met"):
        raise SystemExit("phase A's bar was not met: no playout is registered")
    arms = list(ra["bar"]["play"])
    rows = K.read_by_id(K.ARCH / "rows" / "rows.jsonl")
    idx = {arm: {did: K.play_indices(r, arm) for did, r in rows.items()} for arm in arms}
    body = json.dumps({"arms": arms, "idx": idx}, sort_keys=True)
    blob = {"sha256": hashlib.sha256(body.encode()).hexdigest(), "arms": arms, "idx": idx}
    out = K.d("play") / "play_idx.json"
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("sha256") != blob["sha256"]:
            raise SystemExit(f"{out} exists with other indices (sha256 {old.get('sha256')}): refusing to rewrite")
        print(f"[1c play_idx] {out} already holds these indices ({blob['sha256'][:12]})")
        return
    out.write_text(json.dumps(blob, sort_keys=True) + "\n")
    print(f"[1c play_idx] {len(arms)} arms x {len(rows)} states -> {out} ({blob['sha256'][:12]}); "
          f"no worlds: {sum(1 for a in arms for v in idx[a].values() if v is None)}")


if __name__ == "__main__":
    main()
