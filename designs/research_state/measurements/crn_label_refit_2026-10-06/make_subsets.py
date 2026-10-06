"""The split and the subsample (README §1, §5). Writes ``rows/subset_train.json`` and
``rows/subset_held.json`` in the ``truth run`` subset format ({"bank": sha, "ids": [...]}).

    python make_subsets.py [--max-battles N] [--pilot N]

Battles are taken in the order of ``sha256(SALT + ":size:" + battle)`` (lowest first) until
``--max-battles``; each taken battle goes wholly to TRAIN or HELD-OUT by ``sha256(SALT + ":" + battle)``.
``--pilot N`` instead writes ``rows/subset_pilot.json``: the first N held-out turns (throughput only).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANES = HERE.parent / "m5_laneS"
SALT = "crn_label_refit_2026-10-06"


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def is_held(battle: str) -> bool:
    return int(h(f"{SALT}:{battle}")[0], 16) % 2 == 0


def is_val(battle: str) -> bool:
    return int(h(f"{SALT}:val:{battle}")[0], 16) <= 2


def load():
    sub = json.loads((LANES / "truth_v2" / "gt_subset_v2.json").read_text())
    man = json.loads((LANES / "bank_v1" / "manifest.json").read_text())
    if sub["bank"] != man["content_sha256"]:
        raise SystemExit("subset / bank mismatch")
    dec = {}
    with gzip.open(LANES / "bank_v1" / "decisions.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            dec[d["id"]] = d
    return sub, dec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-battles", type=int, default=None)
    ap.add_argument("--pilot", type=int, default=None)
    a = ap.parse_args()
    sub, dec = load()
    ids = sub["ids"]
    battles = sorted({dec[i]["battle"] for i in ids}, key=lambda b: h(f"{SALT}:size:{b}"))
    out = HERE / "rows"
    out.mkdir(exist_ok=True)
    if a.pilot:
        held = [i for i in ids if is_held(dec[i]["battle"])][: a.pilot]
        (out / "subset_pilot.json").write_text(json.dumps({"bank": sub["bank"], "ids": held}, indent=1) + "\n")
        print(f"pilot: {len(held)} turns")
        return 0
    take = set(battles[: a.max_battles] if a.max_battles else battles)
    tr = [i for i in ids if dec[i]["battle"] in take and not is_held(dec[i]["battle"])]
    he = [i for i in ids if dec[i]["battle"] in take and is_held(dec[i]["battle"])]
    meta = {"salt": SALT, "max_battles": a.max_battles, "n_battles": len(take)}
    for name, lst in (("train", tr), ("held", he)):
        (out / f"subset_{name}.json").write_text(
            json.dumps(dict(meta, bank=sub["bank"], split=name, ids=lst), indent=1) + "\n")
    nb_tr = len({dec[i]['battle'] for i in tr})
    nb_he = len({dec[i]['battle'] for i in he})
    acts = lambda lst: sum(dec[i]["n_legal"] for i in lst)
    print(f"train: {len(tr)} turns / {nb_tr} battles / {acts(tr)} actions; "
          f"held: {len(he)} turns / {nb_he} battles / {acts(he)} actions; "
          f"playouts S8/S32: {8 * acts(tr) + 32 * acts(he)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
