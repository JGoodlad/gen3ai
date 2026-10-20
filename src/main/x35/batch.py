"""X35 BATCH 1's GATES: the orchestrator's rulings on the 2026-10-19 adversarial review (``X35_BATCH1_REVIEW.md``),
made BEFORE any X35 launch. Pure functions over files; nothing here launches, trains or plays. ``main.x35.kit``
exposes each as a subcommand, and ``launch_x35.sh`` calls them as pre-checks (a refusal is a HOLD, exit 7).

* **AM-1, ONE PIN.** :data:`PIN_FILE` (``x35_pin_batch1.txt`` in the stage directory) is written ONCE, by the first
  launch's GO: the pin's full sha, the date and the T62 verdict that chose it. Every later launch's ``--pin`` must
  EQUAL it, and every existing batch-1 run's RECORDED pin (``metadata.json``'s ``git_hash`` and every
  ``pin_history`` span, the record ``main.sidecar_audit`` reads) must equal it. A batch-1 run whose pin cannot be
  read is a HOLD too (it is not evidence of agreement).
* **POOL, ONE SEEDED SET.** Every fork of a side seeds its pool from the start node's run (T54: the parent's
  ``snapshot_*.zip`` at or below the fork step). The first launch of a side records that SET (names + sha256 and its
  digest) in :data:`POOL_FILE`; every later launch recomputes it and HOLDs on any difference, and the post-spawn
  check compares the fork's ``pool_seed.json`` names with the record. The pool's METADATA (``summary.json``, pruned
  by T54 but carrying the parent's LATEST scalars, F31) is recorded descriptively, never compared: X26 has ended, so
  production's cannot move; the toy's parent X34 is live, so the toy record is written at the first TOY launch.
* **AM-2, THE VARIANCE GATE G1** (after row 6: A1 and C0 at s3501-s3503). σ̂ = the POOLED WITHIN-ARM SD of the six
  finals' population scores S (pp; 4 df). **PASS iff σ̂ ≤ 1.2 pp**; σ̂ > 1.2, or within :data:`RULE8_EPS` of 1.2
  (rule 8: counted as above), is **HOLD**: D1, A2 and the toy calibration wait for the orchestrator's re-plan; C0
  s3504-s3505 run next and σ̂ is re-estimated on 5 C0 + 3 A1 (6 df, reported). It reads VARIANCE only, never an
  arm's Δ̂. Written once to :data:`G1_FILE`.
* **AM-7, THE ORDER, and the conditional stages** (:data:`STAGES`): the first six forks → G1 → C0 s3504-s3505 →
  A1 s3504-s3505 (only if A1's look 1 is CONTINUE) → D1, A2 (only if G1 PASSes) → toy C0 + X47 → X47 look 2
  (AM-4: X47 and toy C0 s3604-s3605, only if X47's look 1 is CONTINUE) → toy A1 / toy A2 (AM-3: only after
  production A1 / A2 has a FINAL class GAIN, NULL or LOSS, and G1 PASSed). :func:`stage_status` says, from the
  saved read records, whether a fork is ALLOWED, PENDING (its record does not exist yet) or REFUSED.
* **AM-6, K9(b) AT ARM END.** :func:`k9b_read` reads ``behaviour/max_abs_dlogp_judged`` and
  ``behaviour/p99_abs_dlogp_current`` over the fork's OWN rows (steps > its fork step: the inherited rows are
  X26's, ``gen3_tb_inherit_v1``). VOID iff any judged max ≥ 1e-3, any p99 ≥ 1e-4, any non-finite value, or the child
  log's NON-FINITE K9(b) line; a value within 1e-9 RELATIVE of a ceiling is recorded and does not fire (rule 8).
  No post-fork row, or an unknown fork step, is INCONCLUSIVE (never a PASS). A VOID arm enters no read and is re-run
  at the same pin and seed after its cause is fixed.

THE READ RECORDS the stage rules read are ``<stage>/reads/<arm>_look<k>.json``: the saved JSON of
``python -m main.x35.readout decide`` (A1 / A2 and the toy pair), ``decide-d1`` (D1) or ``python -m main.x35.x47
decide`` (X47). The record names its arm and look; a record for another arm or look is a refusal.

DECLARED HERE (choices the rulings leave open; each a FINDING of this build):
* AM-1's "every existing ``rb_x35*`` run" is read as every BATCH-1 run (:func:`batch1_run_names`, a moved-aside
  ``<run>.<SUFFIX>`` included): a later batch's runs carry their own pin and are listed as not judged here.
* The pin file is ``key: value`` text; the T62 verdict is one of :data:`T62_VERDICTS` (``PENDING`` = C1 (a):
  production launches at main HEAD with T62 before the battery read lands).
* G1's input is the six finals' S; a missing, duplicated or non-finite S is a refusal, never a verdict.
* D1 / A2 look 2 (3504-3505) and the toy pair's look 2 are not in AM-7's order; the kit allows them only on their own
  look-1 record reading CONTINUE (and AM-3 / G1 for the toy pair), and :data:`BATCH1` does not schedule them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

RULE8_EPS = 1e-9
PIN_FILE = "x35_pin_batch1.txt"
POOL_FILE = "x35_pool_batch1_{side}.json"
POOL_SCHEMA = "x35_pool_batch1_v1"
G1_FILE = "x35_gate_g1.json"
G1_SCHEMA = "x35_gate_g1_v1"
READS_DIR = "reads"
K9B_DIR = "k9b"
K9B_SCHEMA = "x35_k9b_arm_end_v1"
#: The battery's T62 verdicts (``gpu_window_battery_2026-10-19/README.md``) and PENDING (C1 (a), 2026-10-19).
T62_VERDICTS = ("ADOPT", "FP32_ALT", "INCONCLUSIVE", "INCOMPLETE", "PENDING")
#: AM-2's bar on σ̂, pp.
G1_SIGMA_BAR_PP = 1.2
G1_SEEDS = ("3501", "3502", "3503")
#: AM-6's ceilings (X26's Amendment 3; X34's for the toy are the same numbers, ``kit.K9B_TOY_CEILINGS``).
K9B_TAGS = {"judged_max": "behaviour/max_abs_dlogp_judged", "p99": "behaviour/p99_abs_dlogp_current"}
K9B_CEILINGS = {"judged_max": 1e-3, "p99": 1e-4}
K9B_NONFINITE_LINE = "FATAL: a NON-FINITE |d log pi|"
ALLOWED, PENDING, REFUSED = "ALLOWED", "PENDING", "REFUSED"
PASS, HOLD, VOID, INCONCLUSIVE = "PASS", "HOLD", "VOID", "INCONCLUSIVE"
FINAL_CLASSES = ("GAIN", "NULL", "LOSS")

#: AM-7's order (orchestrator, 2026-10-19), stage by stage. ``BATCH1`` (kit) is this list flattened.
STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("P1", (("a1", 3501), ("c0", 3501), ("c0", 3502), ("a1", 3502), ("a1", 3503), ("c0", 3503))),
    ("P2", (("c0", 3504), ("c0", 3505))),
    ("P3", (("a1", 3504), ("a1", 3505))),
    ("P4", (("d1", 3501), ("d1", 3502), ("d1", 3503), ("a2", 3501), ("a2", 3502), ("a2", 3503))),
    ("T1", (("toy_c0", 3601), ("toy_x47", 3601), ("toy_x47", 3602), ("toy_c0", 3602), ("toy_c0", 3603),
            ("toy_x47", 3603))),
    ("T1b", (("toy_x47", 3604), ("toy_c0", 3604), ("toy_c0", 3605), ("toy_x47", 3605))),
    ("T2", (("toy_a1", 3601), ("toy_a1", 3602), ("toy_a1", 3603), ("toy_a2", 3601), ("toy_a2", 3602),
            ("toy_a2", 3603))),
)
#: The arms of batch 1 (the AM-1 pin and the POOL record cover their runs).
BATCH1_ARMS = ("c0", "a1", "a2", "d1", "toy_c0", "toy_a1", "toy_a2", "toy_x47")
#: The arm label each read record carries (``readout`` A1 / A2, ``d1`` X38, ``x47`` X47).
READ_LABEL = {"a1": "A1", "a2": "A2", "d1": "X38", "toy_a1": "A1", "toy_a2": "A2", "toy_x47": "X47"}
CONTROLS = ("c0", "toy_c0")
LOOK2_SEEDS = {"production": (3504, 3505), "toy": (3604, 3605)}


class BatchError(ValueError):
    """A gate's input is unreadable or invalid (named; the launch script HOLDs on it)."""


def _near(x: float, b: float) -> bool:
    return abs(x - b) <= RULE8_EPS


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def batch1_run_names() -> Dict[str, Tuple[str, int]]:
    """``{run name: (arm, seed)}`` of every batch-1 arm at every registered seed."""
    from main.x35.kit import ARMS, run_name

    return {run_name(a, s): (a, s) for a in BATCH1_ARMS for s in ARMS[a].seeds}


def _batch1_dirs(models_dir: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``([(dir name, run name)], [other rb_x35* dirs])``; ``<run>.<SUFFIX>`` (moved aside) counts as its run."""
    names = batch1_run_names()
    ours: List[Tuple[str, str]] = []
    other: List[str] = []
    if not os.path.isdir(models_dir):
        return ours, other
    for d in sorted(os.listdir(models_dir)):
        if not d.startswith("rb_x35") or not os.path.isdir(os.path.join(models_dir, d)):
            continue
        base = d.split(".", 1)[0]
        if base in names:
            ours.append((d, base))
        else:
            other.append(d)
    return ours, other


# ------------------------------------------------------------------------------------------------- AM-1: the pin
def read_pin_file(path: str) -> Optional[Dict[str, str]]:
    """The pin file's ``key: value`` lines, or None when it does not exist. A file without a 40-hex ``sha`` is a
    refusal (a corrupt record never reads as "no pin yet")."""
    if not os.path.exists(path):
        return None
    out: Dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", out.get("sha", "")):
        raise BatchError(f"{path}: no 40-hex 'sha:' line ({out.get('sha')!r})")
    return out


def write_pin_file(path: str, sha: str, t62: str, now: Optional[str] = None) -> Dict[str, str]:
    """Write the batch pin ONCE (refuses an existing file and a verdict outside :data:`T62_VERDICTS`)."""
    if os.path.exists(path):
        raise BatchError(f"{path} exists: the batch pin is written ONCE")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise BatchError(f"the pin {sha!r} is not a full 40-hex sha")
    if t62 not in T62_VERDICTS:
        raise BatchError(f"T62 verdict {t62!r} is not one of {T62_VERDICTS}")
    doc = {"sha": sha, "written_at": now or _now(), "t62_verdict": t62,
           "rule": "AM-1 (orchestrator 2026-10-19): every batch-1 arm, look 2 included, launches at this pin"}
    Path(path).write_text("# X35 batch 1's ONE pin (X35_PIN_1), written once by the first launch's GO.\n"
                          + "".join(f"{k}: {v}\n" for k, v in doc.items()))
    return doc


def recorded_pins(run_dir: str) -> List[str]:
    """Every git hash a run RECORDED as its code: ``metadata.json``'s ``git_hash`` and each ``pin_history`` span
    (``main.sidecar_audit.audit_run``'s fields). Empty when the run has written no metadata yet."""
    p = os.path.join(run_dir, "metadata.json")
    if not os.path.isfile(p):
        return []
    try:
        meta = json.loads(Path(p).read_text())
    except (OSError, ValueError) as exc:
        raise BatchError(f"{p}: unreadable ({exc})") from exc
    hs = [str(meta["git_hash"])] if meta.get("git_hash") else []
    hs += [str(e.get("git_hash")) for e in meta.get("pin_history") or [] if isinstance(e, dict) and e.get("git_hash")]
    return hs


def _same_sha(a: str, b: str) -> bool:
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def pin_check(pin: str, stage: str, models_dir: str) -> Tuple[bool, List[str]]:
    """AM-1's pre-check: ``(ok, lines)``. ``pin`` is the full sha the launch would use."""
    lines: List[str] = []
    ok = True
    rec = read_pin_file(os.path.join(stage, PIN_FILE))
    if rec is None:
        lines.append(f"no {PIN_FILE} yet: the first GO writes {pin[:12]} as X35_PIN_1")
    elif rec["sha"] != pin:
        ok = False
        lines.append(f"the pin {pin[:12]} != X35_PIN_1 {rec['sha'][:12]} ({PIN_FILE}, written {rec.get('written_at')})")
    else:
        lines.append(f"the pin equals X35_PIN_1 {pin[:12]} ({PIN_FILE}, T62 {rec.get('t62_verdict')})")
    want = rec["sha"] if rec is not None else pin
    ours, other = _batch1_dirs(models_dir)
    for d, _ in ours:
        hs = recorded_pins(os.path.join(models_dir, d))
        if not hs:
            ok = False
            lines.append(f"{d}: no recorded pin (metadata.json absent or without git_hash): cannot show it is X35_PIN_1")
            continue
        bad = sorted({h for h in hs if not _same_sha(h, want)})
        if bad:
            ok = False
            lines.append(f"{d}: recorded pin(s) {[h[:12] for h in bad]} != {want[:12]}")
    if ours:
        lines.append(f"{len(ours)} batch-1 run(s) checked against {want[:12]}")
    if other:
        lines.append(f"not batch 1, not judged here: {other}")
    return ok, lines


def pin_write_if_absent(pin: str, stage: str, t62: Optional[str]) -> str:
    """At GO, after every pre-check: write the pin file if it does not exist (needs ``t62``), else confirm it."""
    path = os.path.join(stage, PIN_FILE)
    rec = read_pin_file(path)
    if rec is not None:
        if rec["sha"] != pin:
            raise BatchError(f"the pin {pin[:12]} != X35_PIN_1 {rec['sha'][:12]}")
        return f"X35_PIN_1 {pin[:12]} already recorded"
    if t62 is None:
        raise BatchError(f"the first GO writes {PIN_FILE} and needs --t62 <{'|'.join(T62_VERDICTS)}>")
    write_pin_file(path, pin, t62)
    return f"X35_PIN_1 {pin[:12]} WRITTEN to {PIN_FILE} (T62 {t62})"


# --------------------------------------------------------------------------------------- POOL: one seeded set
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_pool(parent_run_dir: str, fork_step: int) -> Dict[str, Any]:
    """The set T54's ``seed_pool`` copies from the parent (its own filter: every ``snapshot_*.zip`` whose step is
    ``<= fork_step`` or unparseable), with each file's sha256 and the set's digest."""
    from agents.training.pool_seed import POOL_METADATA_FILES, snapshot_step, snapshot_zips

    src = os.path.join(parent_run_dir, "snapshots")
    snaps = []
    for z in snapshot_zips(src):
        st = snapshot_step(z)
        if st is not None and st > fork_step:
            continue
        snaps.append({"name": os.path.basename(z), "sha256": sha256_of(z)})
    if not snaps:
        raise BatchError(f"{src}: no snapshot at or below {fork_step} (T54 would refuse the fork)")
    digest = hashlib.sha256("".join(f"{s['name']} {s['sha256']}\n" for s in snaps).encode()).hexdigest()
    meta = {n: sha256_of(os.path.join(src, n)) for n in POOL_METADATA_FILES if os.path.isfile(os.path.join(src, n))}
    return {"parent_run_dir": os.path.abspath(parent_run_dir), "fork_step": int(fork_step), "snapshots": snaps,
            "n_snapshots": len(snaps), "digest": digest, "metadata_sha256_descriptive": meta}


def pool_check(stage: str, side: str, call: Mapping[str, Any], parent_run_dir: str, *,
               write: bool = False) -> Tuple[bool, str]:
    """The POOL pre-check of one launch: recompute the seeded set and compare it with the side's record (or, on the
    first GO of a side, write the record). ``(ok, line)``."""
    path = os.path.join(stage, POOL_FILE.format(side=side))
    p = call["pstar"]
    now = expected_pool(parent_run_dir, int(p["step"]))
    if not os.path.exists(path):
        if not write:
            return True, (f"no {os.path.basename(path)} yet: the first {side} GO records {now['n_snapshots']} "
                          f"snapshot(s), digest {now['digest'][:12]}")
        doc = dict(now, schema=POOL_SCHEMA, side=side, pstar_sha256=p["sha256"], written_at=_now())
        Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return True, f"pool RECORDED: {now['n_snapshots']} snapshot(s), digest {now['digest'][:12]}"
    rec = json.loads(Path(path).read_text())
    probs = []
    if rec.get("schema") != POOL_SCHEMA:
        probs.append(f"schema {rec.get('schema')!r}")
    if rec.get("pstar_sha256") != p["sha256"]:
        probs.append(f"the record is for start node {str(rec.get('pstar_sha256'))[:12]}, not {str(p['sha256'])[:12]}")
    if rec.get("digest") != now["digest"]:
        was = {s["name"]: s["sha256"] for s in rec.get("snapshots") or []}
        is_ = {s["name"]: s["sha256"] for s in now["snapshots"]}
        probs.append(f"the seeded set moved: digest {str(rec.get('digest'))[:12]} -> {now['digest'][:12]}; "
                     f"gone {sorted(set(was) - set(is_))}, new {sorted(set(is_) - set(was))}, "
                     f"changed {sorted(n for n in set(was) & set(is_) if was[n] != is_[n])}")
    if probs:
        return False, "; ".join(probs)
    return True, f"pool set == the {side} record ({now['n_snapshots']} snapshot(s), digest {now['digest'][:12]})"


def recorded_pool_names(stage: str, side: str) -> Optional[List[str]]:
    """The recorded seeded set's names (the post-spawn comparison), or None when no record exists."""
    path = os.path.join(stage, POOL_FILE.format(side=side))
    if not os.path.exists(path):
        return None
    return [s["name"] for s in json.loads(Path(path).read_text())["snapshots"]]


# ------------------------------------------------------------------------------------- AM-2: the variance gate
@dataclass(frozen=True)
class G1:
    verdict: str
    sigma_pp: float
    df: int
    near: bool
    reestimate: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": G1_SCHEMA, "verdict": self.verdict, "sigma_hat_pp": self.sigma_pp, "df": self.df,
                "bar_pp": G1_SIGMA_BAR_PP, "near_bar": self.near, "reestimate": self.reestimate,
                "rule": "AM-2: PASS iff the pooled within-arm SD of S over A1 / C0 s3501-s3503 <= 1.2 pp "
                        "(within 1e-9 of 1.2 counts as above); HOLD holds D1, A2 and the toy calibration"}


def pooled_sd(groups: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """The pooled within-group SD and its df (Σ (n_g − 1))."""
    ss = math.fsum(math.fsum((x - math.fsum(g) / len(g)) ** 2 for x in g) for g in groups)
    df = sum(len(g) - 1 for g in groups)
    if df <= 0:
        raise BatchError("no within-group degrees of freedom")
    return math.sqrt(ss / df), df


def _scores(d: Mapping[str, Any], arm: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise BatchError(f"{arm} seed {k}: S {v!r} is not a finite number")
        out[str(k)] = float(v)
    return out


def gate_g1(doc: Mapping[str, Any]) -> G1:
    """AM-2 from ``{"a1": {seed: S_pp}, "c0": {seed: S_pp}}`` (C0 may carry 3504 / 3505: then the 6-df re-estimate
    on 5 C0 + 3 A1 is reported; the verdict reads s3501-s3503 only)."""
    a1, c0 = _scores(doc.get("a1") or {}, "A1"), _scores(doc.get("c0") or {}, "C0")
    for arm, d in (("A1", a1), ("C0", c0)):
        miss = [s for s in G1_SEEDS if s not in d]
        if miss:
            raise BatchError(f"{arm}: no S for seed(s) {miss} (G1 reads the six finals of rows 1-6)")
    extra = sorted(set(a1) - set(G1_SEEDS))
    if extra:
        raise BatchError(f"A1 seeds {extra}: G1 reads A1 s3501-s3503 only (A1's look 2 is not an input)")
    sd, df = pooled_sd([[a1[s] for s in G1_SEEDS], [c0[s] for s in G1_SEEDS]])
    near = _near(sd, G1_SIGMA_BAR_PP)
    verdict = PASS if (sd < G1_SIGMA_BAR_PP and not near) else HOLD
    re = None
    if set(c0) - set(G1_SEEDS):
        sd2, df2 = pooled_sd([[a1[s] for s in G1_SEEDS], [c0[s] for s in sorted(c0)]])
        re = {"sigma_hat_pp": sd2, "df": df2, "c0_seeds": sorted(c0), "a1_seeds": list(G1_SEEDS),
              "reported_only": True}
    return G1(verdict, sd, df, near, re)


def write_g1(stage: str, g: G1, doc: Mapping[str, Any]) -> str:
    path = os.path.join(stage, G1_FILE)
    if os.path.exists(path):
        raise BatchError(f"{path} exists: G1 is written ONCE")
    Path(path).write_text(json.dumps(dict(g.as_dict(), input=doc, written_at=_now()), indent=1, sort_keys=True)
                          + "\n")
    return path


def read_g1(stage: str) -> Optional[str]:
    """G1's recorded verdict, or None when it is not read yet."""
    path = os.path.join(stage, G1_FILE)
    if not os.path.exists(path):
        return None
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != G1_SCHEMA or doc.get("verdict") not in (PASS, HOLD):
        raise BatchError(f"{path}: not a G1 record")
    return str(doc["verdict"])


# ---------------------------------------------------------------- AM-7 / AM-3 / AM-4: which fork may launch now
def read_record(stage: str, arm: str, look: int) -> Optional[Dict[str, Any]]:
    """The saved look-``look`` read of ``arm`` (``reads/<arm>_look<k>.json``), checked to be that arm's and look's."""
    path = os.path.join(stage, READS_DIR, f"{arm}_look{look}.json")
    if not os.path.exists(path):
        return None
    doc = json.loads(Path(path).read_text())
    want = READ_LABEL.get(arm)
    if str(doc.get("arm")) != want or int(doc.get("look", -1)) != look:
        raise BatchError(f"{path}: the record is arm {doc.get('arm')!r} look {doc.get('look')!r}, not {want} look "
                         f"{look}")
    return dict(doc)


def final_class(stage: str, arm: str) -> Tuple[Optional[str], str]:
    """``(class, why)``: the production arm's FINAL R1 class (AM-3), from its latest saved look. ``None`` = none yet."""
    for look in (2, 1):
        rec = read_record(stage, arm, look)
        if rec is None:
            continue
        r1 = rec.get("r1") or {}
        if not r1.get("final"):
            return None, f"{arm} look {look}: R1 is not final (outcome {rec.get('verdict')})"
        return str(r1.get("class")), f"{arm} look {look}: R1 FINAL class {r1.get('class')}"
    return None, f"{arm}: no saved read"


def _look2_licence(stage: str, arm: str) -> Tuple[str, str]:
    rec = read_record(stage, arm, 1)
    if rec is None:
        return PENDING, f"{arm} look 1 is not read yet (look 2 runs only on CONTINUE)"
    if rec.get("verdict") == "CONTINUE":
        return ALLOWED, f"{arm} look 1 is CONTINUE"
    return REFUSED, f"{arm} look 1 is {rec.get('verdict')}, not CONTINUE: no look 2"


def stage_of(arm: str, seed: int) -> Optional[str]:
    for name, forks in STAGES:
        if (arm, seed) in forks:
            return name
    return None


def stage_status(stage: str, arm: str, seed: int) -> Tuple[str, str]:
    """ALLOWED / PENDING / REFUSED for ``arm`` × ``seed`` now (module docs), with the reason."""
    from main.x35.kit import ARMS

    if arm not in BATCH1_ARMS:
        return ALLOWED, f"{arm} is not a batch-1 arm (no AM-7 condition)"
    side = ARMS[arm].side
    look2 = seed in LOOK2_SEEDS[side]
    if arm == "c0" and look2:
        g = read_g1(stage)
        return (ALLOWED, f"G1 is read ({g})") if g else (PENDING, "C0's look-2 seeds run after G1 (AM-7)")
    if arm == "toy_c0" and look2:
        why = []
        for b in ("toy_x47", "toy_a1", "toy_a2"):
            st, w = _look2_licence(stage, b)
            if st == ALLOWED:
                return ALLOWED, w
            why.append(w)
        return PENDING if any("not read yet" in w for w in why) else REFUSED, "; ".join(why)
    if arm in ("d1", "a2", "toy_a1", "toy_a2"):
        g = read_g1(stage)
        if g is None:
            return PENDING, f"{arm} waits for G1 (AM-2)"
        if g != PASS:
            return REFUSED, f"G1 is {g}: {arm} is HELD for the orchestrator's re-plan (AM-2)"
    if arm in ("toy_a1", "toy_a2"):
        prod = arm[len("toy_"):]
        cls, why = final_class(stage, prod)
        if cls is None:           # not read yet, or look 1 said CONTINUE: look 2 may still give a final class
            return PENDING, f"AM-3: {why}"
        if cls not in FINAL_CLASSES:
            return REFUSED, f"AM-3: production {prod}'s final class is {cls}: the calibration is NOT SHOWN, no toy fork"
    if look2 and arm not in CONTROLS:
        return _look2_licence(stage, arm)
    return ALLOWED, f"{arm} s{seed}: no unmet condition (stage {stage_of(arm, seed)})"


# --------------------------------------------------------------------------------- AM-6: K9(b) at each arm's end
@dataclass(frozen=True)
class K9b:
    run: str
    verdict: str
    fork_step: Optional[int]
    n_rows: Dict[str, int]
    max_value: Dict[str, Optional[float]]
    fires: Tuple[str, ...]
    near: Tuple[str, ...]
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": K9B_SCHEMA, "run": self.run, "verdict": self.verdict, "fork_step": self.fork_step,
                "rows_after_fork": self.n_rows, "max_after_fork": self.max_value, "ceilings": K9B_CEILINGS,
                "fires": list(self.fires), "near_ceiling": list(self.near), "reasons": list(self.reasons)}


def judge_k9b(run: str, fork: Optional[int], series: Mapping[str, Sequence[Tuple[int, float]]],
              log_text: str = "") -> K9b:
    """The deterministic AM-6 verdict on the fork's own rows (``series`` = ``{key: [(step, value)]}`` for
    :data:`K9B_TAGS`' keys, already restricted to steps > ``fork``)."""
    fires: List[str] = []
    near: List[str] = []
    reasons: List[str] = []
    n_rows: Dict[str, int] = {}
    mx: Dict[str, Optional[float]] = {}
    if fork is None:
        reasons.append("the fork step is unknown: the inherited rows cannot be excluded")
    for key, bar in K9B_CEILINGS.items():
        rows = [(s, v) for s, v in series.get(key) or () if fork is None or s > fork]
        n_rows[key] = len(rows)
        if not rows:
            reasons.append(f"{K9B_TAGS[key]}: no row after the fork step (absence is not a pass)")
            mx[key] = None
            continue
        finite = [v for _, v in rows if math.isfinite(v)]
        mx[key] = max(finite) if finite else None
        for s, v in rows:
            if not math.isfinite(v):
                fires.append(f"{key} non-finite {v!r} at step {s}")
            elif abs(v - bar) <= RULE8_EPS * bar:
                near.append(f"{key} {v!r} at step {s} within 1e-9 relative of {bar:g} (recorded, no fire)")
            elif v >= bar:
                fires.append(f"{key} {v!r} >= {bar:g} at step {s}")
    if K9B_NONFINITE_LINE in log_text:
        fires.append(f"the child log carries '{K9B_NONFINITE_LINE}'")
    verdict = VOID if fires else INCONCLUSIVE if reasons else PASS
    return K9b(run, verdict, fork, n_rows, mx, tuple(fires), tuple(near), tuple(reasons))


def k9b_read(run_dir: str) -> K9b:
    """AM-6 on a real run directory (read-only): its own TensorBoard rows after the fork step, and its child log."""
    from main.x35.tbrows import fork_step, load_segments

    fork = fork_step(run_dir)
    segs = load_segments(run_dir, list(K9B_TAGS.values()), after_step=fork) if fork is not None else []
    series: Dict[str, List[Tuple[int, float]]] = {k: [] for k in K9B_TAGS}
    for seg in segs:
        for key, tag in K9B_TAGS.items():
            series[key] += seg.rows.get(tag, [])
    log = os.path.join(run_dir, "launcher_child.full.log")
    text = Path(log).read_text(errors="replace") if os.path.isfile(log) else ""
    return judge_k9b(os.path.basename(os.path.normpath(run_dir)), fork, series, text)


def k9b_record(stage: str, k: K9b) -> str:
    """Write (or overwrite: a re-read of a finished arm is the same read) ``k9b/<run>.json``."""
    d = os.path.join(stage, K9B_DIR)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{k.run}.json")
    Path(path).write_text(json.dumps(dict(k.as_dict(), read_at=_now()), indent=1, sort_keys=True) + "\n")
    return path


def k9b_batch(stage: str, models_dir: str, *, reader: Any = None) -> Tuple[bool, List[str]]:
    """The launch pre-check: every FINISHED batch-1 run (``final_model.zip``) has a K9(b) arm-end record that PASSes;
    a missing record is read now (``reader``, default :func:`k9b_read`) and written. VOID / INCONCLUSIVE ⇒ HOLD."""
    read = reader or k9b_read
    ok, lines = True, []
    ours, _ = _batch1_dirs(models_dir)
    for d, _base in ours:
        rd = os.path.join(models_dir, d)
        if d != _base or not os.path.isfile(os.path.join(rd, "final_model.zip")):
            continue
        path = os.path.join(stage, K9B_DIR, f"{d}.json")
        if os.path.exists(path):
            v = json.loads(Path(path).read_text()).get("verdict")
        else:
            k = read(rd)
            k9b_record(stage, k)
            v = k.verdict
        if v != PASS:
            ok = False
        lines.append(f"{d}: K9(b) arm end {v}")
    return ok, lines or ["no finished batch-1 arm yet"]
