"""Compare two fingerprints (a proof JSON, or a recorded golden entry) on EVERY field: init / post params,
every init / post group hash, every pinned loss (exact). `python compare.py A.json B.json [--golden-arm blob]`
where B may be the golden JSON (its entry for this torch build is read)."""
from __future__ import annotations

import json
import sys

FIELDS = ("init_params_sha256", "post_params_sha256", "init_group_sha256", "group_sha256", "losses")


def entry(path: str, arm: str | None) -> dict:
    d = json.load(open(path))
    if "entries" not in d:
        return d
    import torch
    blk = d if arm in (None, "blob") else d["arms"][arm]
    return blk["entries"][torch.__version__]


def main() -> int:
    a, b = sys.argv[1], sys.argv[2]
    arm = sys.argv[sys.argv.index("--golden-arm") + 1] if "--golden-arm" in sys.argv else None
    x, y = entry(a, arm), entry(b, arm)
    moved, same, absent = [], 0, []
    for f in FIELDS:
        if f not in x or f not in y:
            absent.append(f)
            continue
        if isinstance(x[f], dict):
            for k in sorted(set(x[f]) | set(y[f])):
                if x[f].get(k) != y[f].get(k):
                    moved.append(f"{f}.{k}")
                else:
                    same += 1
        elif x[f] != y[f]:
            moved.append(f)
        else:
            same += 1
    print(json.dumps({"a": a, "b": b, "identical_fields": same, "moved": moved, "not_compared": absent}, indent=1))
    return 0 if not moved else 1


if __name__ == "__main__":
    raise SystemExit(main())
