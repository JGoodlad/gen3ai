"""Build the LEARNER BATTERY's four fork argvs from the new lineage's base argv, and ASSERT the design.

    # the registered argvs (D_g = N0's final lr, 2 s.f., by the lineage's rule, §3.1): run AFTER N0 ends
    python build_argvs.py --final --dg 2.8e-05
    # the committed placeholder form (D_g unknown until N0 ends): argv_<ARM>.txt carry __DG__ / __2DG__
    python build_argvs.py --final
    # the STAND-IN used to validate by executing (N0's latest periodic checkpoint, its current lr)
    python build_argvs.py --standin --parent models/ai_v14_01_base/checkpoints/checkpoint_4800000_steps.zip \
        --parent-step 4800000 --dg 4.3e-04

Registration: designs/research_state/learner_battery_2026-09-26.md (§2 the arms, §3 the dose).
The base is `../../new_lineage_2026-09-26/argv_base.txt` (N0 = ai_v14_01_base, token-exact). Moves, and
NOTHING else (asserted):

  common to all four arms (the lineage's continuation-block construction, §3.1 of the lineage, plus the
  flag deletions and the two battery levers stated explicitly so each arm differs from C in ONE value):
    - the six flags deleted at e3ef16db, with their values:
      --bias-additivity --mat-alive-weight --no-progress-penalty --self-ko-hp-penalty
      --switch-bias-weight --no-hand-shaping
    - --arch production: a FRESH-run flag, REFUSED on a resume (`combination_checks`); a fork inherits
      its parent's architecture surface from the recorded model_config.json (checkargs, 2026-09-26)
    --run-name <arm run>        --pin-commit <PIN>           --steps <parent step + 8,000,000>
    + --model <parent .zip>     + --fork-lr <lr> --fork-lr-freeze   + --checkpoint-every-steps 500000
    + --policy-gae-lambda 0.8   + --matmul-precision highest
    + --max-lr <2 D_g>          ONLY when 2 D_g > 2 x --lr (6e-4): the freeze clamps its pin into
                                [--min-lr, --max-lr], and E5's pin is 2 D_g. Inert otherwise (the KL
                                controller is frozen on every arm), so it is added to ALL four or none.
  per arm (exactly one value differs from C):
    C   : --fork-lr D_g
    E5  : --n-epochs 5  AND  --fork-lr 2 D_g   (ONE lever: the passes per rollout at a MATCHED DOSE,
                                                lr x n_epochs held at D_g x 10; §3)
    T32 : --matmul-precision high
    L95 : --policy-gae-lambda 0.95
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
BASE = KIT.parent / "new_lineage_2026-09-26" / "argv_base.txt"

PIN = "2cc830804a72824dbb1870d7e1f278f9e1370ff5"   # 2cc83080: --policy-gae-lambda, --matmul-precision, per-epoch diag (v123)
FINAL_PARENT = "models/ai_v14_01_base/final_model.zip"
FINAL_PARENT_STEP = 75_005_952
BLOCK = 8_000_000          # requested; lands +8,060,928 (82 rollouts of 98,304)
ROLLOUT = 48 * 2048

RUNS = {"C": "ai_v14_02_lbat_ctrl", "E5": "ai_v14_03_lbat_e5",
        "T32": "ai_v14_04_lbat_t32", "L95": "ai_v14_05_lbat_l95"}
DELETED_WITH_VALUE = ["--bias-additivity", "--mat-alive-weight", "--no-progress-penalty",
                      "--self-ko-hp-penalty", "--switch-bias-weight"]
DELETED_BARE = ["--no-hand-shaping"]
BASE_LR = 3e-4             # the recipe's --lr; the default --max-lr is 2x it


def flags(tokens):
    out, key = {}, None
    for t in tokens:
        if t.startswith("--"):
            key = t
            out.setdefault(key, [])
        else:
            out[key].append(t)
    return out


def drop(tokens, flag, with_value):
    i = tokens.index(flag)
    del tokens[i:i + (2 if with_value else 1)]
    assert flag not in tokens, f"{flag} appears twice in the base argv"


def setv(tokens, flag, value):
    if flag in tokens:
        i = tokens.index(flag)
        tokens[i + 1] = value
    else:
        tokens += [flag, value]


def fmt(x: float) -> str:
    return f"{x:.1e}"      # 2 significant figures, e.g. 2.8e-05


def build(arm: str, parent: str, parent_step: int, dg: str | None, pin: str) -> list:
    toks = shlex.split(BASE.read_text())
    for f in DELETED_WITH_VALUE:
        drop(toks, f, True)
    for f in DELETED_BARE:
        drop(toks, f, False)
    drop(toks, "--arch", True)      # FRESH-run only: resolve_config refuses it on a resume/fork
    setv(toks, "--run-name", RUNS[arm])
    setv(toks, "--pin-commit", pin)
    setv(toks, "--steps", str(parent_step + BLOCK))
    toks += ["--model", parent]
    dgv = None if dg is None else float(dg)
    lr_c = "__DG__" if dgv is None else fmt(dgv)
    lr_e5 = "__2DG__" if dgv is None else fmt(2 * dgv)
    toks += ["--fork-lr", lr_e5 if arm == "E5" else lr_c, "--fork-lr-freeze",
             "--checkpoint-every-steps", "500000",
             "--policy-gae-lambda", "0.95" if arm == "L95" else "0.8",
             "--matmul-precision", "high" if arm == "T32" else "highest"]
    if arm == "E5":
        setv(toks, "--n-epochs", "5")
    if dgv is not None and 2 * dgv > 2 * BASE_LR:
        toks += ["--max-lr", fmt(2 * dgv)]
    return toks


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--final", action="store_true")
    g.add_argument("--standin", action="store_true")
    ap.add_argument("--dg", default=None, help="D_g, N0's final lr at 2 s.f. (the lineage's rule)")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--parent-step", type=int, default=None)
    ap.add_argument("--pin", default=PIN)
    a = ap.parse_args()
    if a.standin:
        assert a.parent and a.parent_step and a.dg, "--standin needs --parent, --parent-step and --dg"
        parent, step, outdir, suffix = a.parent, a.parent_step, HERE, "_STANDIN"
    else:
        parent, step, outdir, suffix = FINAL_PARENT, FINAL_PARENT_STEP, KIT, ""
        assert a.parent is None and a.parent_step is None, "the FINAL parent is registered, not chosen"
    if a.dg is not None:
        v = float(a.dg)
        assert float(fmt(v)) == v, f"D_g must be at 2 significant figures: {a.dg}"
    base = shlex.split(BASE.read_text())
    arms = {arm: build(arm, parent, step, a.dg, a.pin) for arm in RUNS}
    fc = flags(arms["C"])
    for arm, toks in arms.items():
        f = flags(toks)
        moved = sorted(k for k in set(fc) | set(f) if fc.get(k) != f.get(k))
        want = {"C": [], "E5": ["--fork-lr", "--n-epochs"], "T32": ["--matmul-precision"],
                "L95": ["--policy-gae-lambda"]}[arm]
        assert moved == sorted(want + (["--run-name"] if arm != "C" else [])), (arm, moved)
        for d in DELETED_WITH_VALUE + DELETED_BARE + ["--arch"]:
            assert d not in f, (arm, d)
        assert f["--model"] == [parent] and f["--steps"] == [str(step + BLOCK)]
        assert "--fork-lr-freeze" in f and f["--seed"] == ["1001"]
        if a.dg is not None:   # THE DOSE: lr x n_epochs identical on every arm (E5 = 2 D_g x 5)
            d = float(f["--fork-lr"][0]) * int(f["--n-epochs"][0])
            assert abs(d - float(a.dg) * 10) < 1e-12, (arm, d)
        (outdir / f"argv_{arm}{suffix}.txt").write_text(" ".join(toks) + "\n")
    fb = flags(base)
    print(f"== base ai_v14_01_base: {len(base)} tokens; C: {len(arms['C'])} tokens")
    for k in sorted(set(fb) | set(fc)):
        if fb.get(k) != fc.get(k):
            print(f"   C vs base  {k}: {fb.get(k)} -> {fc.get(k)}")
    for arm in ("E5", "T32", "L95"):
        f = flags(arms[arm])
        print(f"   {arm:3s} vs C   " + "; ".join(f"{k}: {fc.get(k)} -> {f.get(k)}"
                                          for k in sorted(set(fc) | set(f)) if fc.get(k) != f.get(k)))
    if a.dg is not None:
        eb = 2048 * 32
        print(f"   dose (lr x n_epochs / {eb}) = {float(a.dg) * 10 / eb:.3e} on EVERY arm "
              f"({float(a.dg) * 10 / eb / 2.145e-08:.2f}x v8)")
    print(f"wrote {', '.join(str(outdir / f'argv_{x}{suffix}.txt') for x in RUNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
