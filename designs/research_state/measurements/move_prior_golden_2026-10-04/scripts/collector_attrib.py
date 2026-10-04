"""Attribute collector_integration_test's `len(rows) > 200`: run its `_run("ffi")` with the move prior
optionally HELD at a file's values and print the distinct trainee row-slot count."""
import json
import sys
from pathlib import Path

from agents.gen3_data import priors as P
if len(sys.argv) > 1:
    t = json.loads(Path(sys.argv[1]).read_text())
    P.move_raw = lambda: t
from agents.training.rust_rollout import collector_integration_test as C  # noqa: E402
from agents.training.rust_rollout import testkit as TK  # noqa: E402
TK.build_selfcheck()
r = C._run("ffi", TK.production_spaces())
print(json.dumps({"prior": sys.argv[1] if len(sys.argv) > 1 else "tree", "rows": len(r["rec"].rows)}))
