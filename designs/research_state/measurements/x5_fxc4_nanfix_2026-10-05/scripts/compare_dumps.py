"""Compare two `codehash.py` dumps (HEAD vs the fix), region by region. Two normalisations, both of
COMMENTS that name process state, not code: (1) the per-process compile-cache temp dir in Inductor's
`# kernel path:` comments (`.../gen3ai_compile_<pid>_<random>/`; the kernel FILE NAMES after it are
content hashes and stay compared); (2) the host ADDRESS at the end of a `_tensor_constantN = None  #
<device> <dtype> <shape> <stride> <address>` comment (the constant's device / dtype / shape / stride
stay compared). Everything else is compared byte for byte.
Usage: compare_dumps.py <dump_a> <dump_b> <out.json>"""
import hashlib
import json
import re
import sys
from pathlib import Path

TMP = re.compile(r"/[^\s'\"]*/gen3ai_compile_\d+_[A-Za-z0-9_]+/")
ADDR = re.compile(r"^(_tensor_constant\d+ = None  # .*) [0-9a-f]{6,}$", re.M)


def norm(text: str) -> str:
    return ADDR.sub(r"\1 <ADDR>", TMP.sub("<COMPILE_TMP>/", text))


def digests(d: Path):
    return sorted(hashlib.sha256(norm(p.read_text()).encode()).hexdigest() for p in sorted(d.glob("*.py")))


a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
row = {}
for region in ("R1", "T2"):
    da, db = digests(a / region), digests(b / region)
    raw_equal = sorted(p.read_bytes() for p in (a / region).glob("*.py")) == \
        sorted(p.read_bytes() for p in (b / region).glob("*.py"))
    row[region] = {"files": [len(da), len(db)], "identical_after_comment_normalisation": da == db,
                   "identical_raw": raw_equal,
                   "combined_sha_a": hashlib.sha256("".join(da).encode()).hexdigest(),
                   "combined_sha_b": hashlib.sha256("".join(db).encode()).hexdigest(),
                   "only_in_a": len(set(da) - set(db)), "only_in_b": len(set(db) - set(da))}
json.dump(row, open(out, "w"), indent=1)
print(json.dumps(row, indent=1))
