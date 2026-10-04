"""Revert evidence for F-X5-47's tests: temporarily revert the fix in `src/agents/gen3_data/priors.py`, run
`priors_species_usage_test.py`, restore the file byte-for-byte (finally), print the pytest summary.

  --mode code         the whole facade module at the parent commit (`git show <parent>:...`)
  --mode denominator  only `species_usage`'s weight line back to `Raw count` (the guard kept)

    python revert_check.py --mode code --parent 7bed4347      # run from the tree's root
"""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("code", "denominator"), required=True)
ap.add_argument("--parent", default="7bed4347")
a = ap.parse_args()

target = Path("src/agents/gen3_data/priors.py")
fixed = target.read_bytes()
try:
    if a.mode == "code":
        target.write_bytes(subprocess.run(["git", "show", f"{a.parent}:{target}"], check=True,
                                          capture_output=True).stdout)
    else:
        old = "        w = _weighted_count(sp_data)\n        if sid and w > 0.0:\n            out[sid] = w\n"
        new = "        w = float(sp_data.get(\"Raw count\") or 0.0)\n        if sid and w > 0.0:\n            out[sid] = w\n"
        t = fixed.decode()
        assert t.count(old) == 1, "the fixed weight line is not where this script expects it"
        target.write_text(t.replace(old, new))
    r = subprocess.run([sys.executable, "-m", "pytest", "src/agents/gen3_data/priors_species_usage_test.py", "-q",
                        "-p", "no:cacheprovider"], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED") or " passed" in ln or " failed" in ln]
    print(f"mode={a.mode}\n" + "\n".join(lines))
finally:
    target.write_bytes(fixed)
assert target.read_bytes() == fixed
