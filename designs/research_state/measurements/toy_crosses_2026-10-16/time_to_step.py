"""The time-to-step curves behind R-T5's EQUAL-COMPUTE checkpoint (toy_crosses_2026-10-16/PREREG.md).

Reads every SB3 `time/` table (time_elapsed, total_timesteps) of a run's launcher_child.full.log. A restart starts a new
SEGMENT (time_elapsed falls); SB3's time_elapsed counts from that process's learn() start, so it covers rollouts,
updates and the in-learn callbacks (eval included) but NOT process startup / compile before learn().

Two curves per run:
  EFFECTIVE: time on the surviving step path. A segment that resumes at step s0 starts at the earlier curve's time at
             s0 (linear interpolation), so work redone after a restart is not counted twice.
  GROSS    : every logged second, summed over segments (redone work counted; startup still excluded).
Usage: time_to_step.py <run_dir> [<run_dir> ...]  -> JSON on stdout. Read-only.
"""
import json, re, sys
from pathlib import Path

PAT_T = re.compile(r"\|\s+time_elapsed\s+\|\s+([0-9.e+]+)")
PAT_S = re.compile(r"\|\s+total_timesteps\s+\|\s+([0-9.e+]+)")


def tables(log: Path):
    t = None
    for line in log.open(errors="replace"):
        m = PAT_T.search(line)
        if m:
            t = float(m.group(1)); continue
        m = PAT_S.search(line)
        if m and t is not None:
            yield t, int(float(m.group(1))); t = None


def segments(log: Path):
    segs, cur, last_t = [], [], None
    for t, s in tables(log):
        if last_t is not None and t < last_t:
            segs.append(cur); cur = []
        cur.append((t, s)); last_t = t
    if cur: segs.append(cur)
    return segs


def interp(curve, s):
    """time at step s on a curve of (step, time) sorted by step; None outside it."""
    for (s0, t0), (s1, t1) in zip(curve, curve[1:]):
        if s0 <= s <= s1:
            return t0 + (t1 - t0) * (s - s0) / (s1 - s0) if s1 > s0 else t0
    return None


def curves(run: Path, base_curve=None, base_step=None):
    segs = segments(run / "launcher_child.full.log")
    eff, gross_total = [], 0.0
    if base_curve:
        eff = [p for p in base_curve if p[0] <= base_step]
    for seg in segs:
        # the segment's first table follows one rollout+update after the resume step; anchor its zero at the step
        # it resumed from (first step minus one iteration's steps, read from the first two tables when present)
        s_first = seg[0][1]
        per_iter = (seg[1][1] - seg[0][1]) if len(seg) > 1 else 0
        s_resume = s_first - per_iter
        if eff and eff[-1][0] >= s_resume:
            t_base = interp(eff, s_resume)
            if t_base is None: t_base = eff[-1][1]
            eff = [p for p in eff if p[0] <= s_resume] + [(s_resume, t_base)]
        else:
            t_base = eff[-1][1] if eff else 0.0
            eff.append((s_resume if eff or s_resume > 0 else 0, t_base))
        for t, s in seg:
            eff.append((s, t_base + t))
        gross_total += seg[-1][0]
    eff = sorted(set(eff))
    return eff, gross_total, len(segs)


def main():
    out = {}
    for a in sys.argv[1:]:
        run = Path(a)
        md = json.loads((run / "metadata.json").read_text())
        fp = (md.get("lineage") or {}).get("fork_parent")
        base = None; bstep = None
        if fp:
            parent = Path(fp["run_dir"]); bstep = int(re.search(r"checkpoint_(\d+)_steps", fp["path"]).group(1))
            base, _, _ = curves(parent)
        eff, gross, nseg = curves(run, base, bstep)
        rec = {"segments": nseg, "gross_logged_s": gross, "last": eff[-1]}
        for m in (5, 15, 30, 40, 45, 50, 55, 60, 70, 100):
            t = interp(eff, m * 1_000_000)
            if t is not None: rec[f"t_eff_{m}M_s"] = round(t, 1)
        rec["curve"] = eff[:: max(1, len(eff) // 400)]
        out[run.name] = rec
    print(json.dumps(out))


if __name__ == "__main__":
    main()
