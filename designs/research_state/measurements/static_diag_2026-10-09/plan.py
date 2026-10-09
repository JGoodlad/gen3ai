"""Which checkpoints the diagnostic reads: every FINISHED screen seed (final_model.zip present), at
~2M, ~5M, ~10M (the checkpoint nearest each) and the final. Prints `<zip> <label>` lines."""
import re
from pathlib import Path

MODELS = Path("/home/goodlad/dev/gen3ai/models")
TARGETS = {"2M": 2_000_000, "5M": 5_000_000, "10M": 10_000_000}

for arm, tag in (("legacy", "L"), ("static", "S")):
    for d in sorted(MODELS.glob(f"rb_st_{arm}_s1*")):
        if not (d / "final_model.zip").exists():
            continue
        seed = d.name[-1]
        ck = {int(re.search(r"_(\d+)_steps", p.name).group(1)): p for p in (d / "checkpoints").glob("checkpoint_*_steps.zip")}
        for lab, t in TARGETS.items():
            s = min(ck, key=lambda x: abs(x - t))
            print(ck[s], f"{tag}{seed}_{lab}")
        print(d / "final_model.zip", f"{tag}{seed}_final")
