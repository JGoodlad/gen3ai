r"""X59-S's ≥ 1500 FOLLOW-UP (``x59s_matched_students_2026-10-25/PREREG.md`` A2.7, Amendment 7): three arms at
N = ``hr_spec_v1.r2``'s ≥ 1500-actor TRAIN rows, seeds 5901-5905 (the primary arms' inits), §3's recipe, read
in-house only.

    python -m main.human_data.followup derive   --arm h1500|hmix|sscut --r2 <r2>/rows --ss <ss_rows_v2> --root <R>
    python -m main.human_data.followup verify   --arm h1500|hmix|sscut --r2 <r2>/rows --ss <ss_rows_v2> --root <R>
    python -m main.human_data.followup register --root <R> --r2 <r2>/rows --ss <ss_rows_v2> \
        --parent-registration <results/registration_a2.json> --out <results/registration_a7.json>
    python -m main.human_data.followup emit-chain --base <chain> --name <name> --worktree <WT> --require-commit <sha> \
        --root <R> [--write <chain file>]
    python -m main.human_data.followup export  --root <R> --r2 <r2>/rows --ss <ss_rows_v2>      # CPU, after the window
    python -m main.human_data.followup inhouse --root <R> --tree <play tree> [--play] --out <results/fu1500_inhouse.json>

THE ARMS (:data:`ARMS`; the student arm in ``student.FOLLOWUP_ARMS``, each with its own registration key):

* ``h1500`` (S-H≥1500): r2's TRAIN rows whose actor's pre-battle rating is ≥ :data:`RATING_HI`, in r2's order;
* ``hmix`` (S-H-mix): r2's TRAIN accounts (``np.unique`` of ``acct``, ascending) permuted by
  ``np.random.default_rng(MIX_SEED)``, taken WHOLE in that order until N rows; the last one taken is trimmed to its
  first rows in r2's order (:func:`mix_mask`); the kept rows stay in r2's order;
* ``sscut`` (S-S cut): S-S's first N TRAIN rows in shard order.

Each is re-cut into ``rowset.SHARD``-row shards (``rowset``'s shard format; the last shard holds the remainder), so
every arm's step budget is the same ``3 x Σ ceil(buffer rows / 512)`` for every seed. Validation: r2's validation
shard for the two human arms, S-S's for the cut, each a byte copy. ``npz`` files carry their write time, so a shard's
sha256 differs between two runs of the same selection; the manifest therefore also records ``content_sha256`` (the
rows' bytes, field by field) and ``verify`` re-selects from the parents and checks it, writing nothing.

THE GPU WINDOW (``emit-chain``): ONE arm-chain window item (``designs/ops/arm_chain.md`` §2), the pattern of
``main.ops.gpu_fill_window``: one process per step, the runner wraps every GPU step in ``gpu_lock.sh mem_cap.sh <GB>
timeout <s>`` (nothing here wraps itself), its OWN preempt marker :data:`MARKER`, ``on_fail: hold``:

1. ``preflight`` (CPU): the registration passes and names the rows on disk, every shard verifies, the lease token
   validates; a failure WRITES the marker (no later step runs);
2. ``cuda_preflight`` (GPU): ``main.human_data.cuda_preflight --arm human_1500`` (A2.9's 8 checks on S-H≥1500);
3. ``preflight_gate`` (CPU): the pre-flight's JSON must read PASS, else it writes the marker (A2.9: HOLD);
4. the 15 students (:func:`jobs`: seed-major, S-H≥1500, S-H-mix, S-S cut), ONE per step, serially: ``train-step``
   runs ``student train`` chunks of :data:`CHUNK_MIN` minutes in-process, continuing ONLY on exit 3 (at most
   :data:`MAX_CHUNKS`); any other outcome writes the marker;
5. ``check`` (CPU): every student finished at its registered budget, on its registered rows, one init per seed.

The budget is the brief's rule, computed (:func:`window_budget`): :data:`MS_PER_STEP` x the 15 students' steps x 1.5
+ 5 min (``gpu_fill_window.window_budget_min``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.human_data import rowset as RS

N_EXPECTED = 270_260            # PREREG Amendment 2b: r2's ≥ 1500-actor TRAIN rows (A2.7's N)
RATING_HI = 1500
MIX_SEED = 5930
SEEDS = (5901, 5902, 5903, 5904, 5905)
EPOCHS = 3
#: dir label -> (student arm, student dir prefix, registration key)
ARMS: Dict[str, Tuple[str, str, str]] = {
    "h1500": ("human_1500", "s_h1500", "rows_1500_manifest_sha256"),
    "hmix": ("human_mix", "s_hmix", "rows_mix_manifest_sha256"),
    "sscut": ("selfplay_cut", "s_scut", "selfplay_cut_manifest_sha256"),
}
MS_PER_STEP = 225.6             # the PASSing pre-flight's solo ms a step (~/gen3ai_archive/x59s/results/cuda_preflight.json: 225.56)
MARKER = "NO_GO_X59FU"
ON_FAIL = "hold"
MEM_GB_STUDENT = 10             # window.MEM_GB (PREREG Amendment 3: peak host RSS 7.27-7.7 GiB)
MEM_GB_PREFLIGHT = 16           # cuda_preflight's documented cap
CHUNK_MIN = 15.0                # ~2.5x a student's expected 6 min solo (1,584 steps x 225.6 ms)
MAX_CHUNKS = 3
STUDENT_TIMEOUT_S = 3600        # > MAX_CHUNKS x CHUNK_MIN + loads and evaluations
PREFLIGHT_TIMEOUT_S = 3600      # the pre-flight measured 11-12 min (status.txt X59W / X59W2)
INHOUSE_LABEL = "x59s_fu1500_inhouse"
#: the two registered pairings (A2.7): the player is S-H≥1500_k; each with §5's class names (+ NO-DIFFERENCE-AT-4PP)
PAIRINGS: Dict[str, Tuple[str, str, Tuple[str, str, str]]] = {
    "rating": ("h1500", "hmix", ("HIGHER-RATED-BETTER", "MIX-BETTER", "NOT-RESOLVED")),
    "data": ("h1500", "sscut", ("HUMAN-DATA-BETTER", "SELF-PLAY-DATA-BETTER", "NOT-RESOLVED")),
}
PANELS = ("heldout72_hpfix", "forum72", "p72")
PAIRS = 200


class FollowupError(RuntimeError):
    """A follow-up input or precondition that does not hold (named)."""


# ---------------------------------------------------------------------------------------------------- layout
def rows_dir(root: Path, arm: str) -> Path:
    return Path(root) / arm / "rows"


def student_dir(root: Path, arm: str, seed: int) -> Path:
    return Path(root) / "students" / f"{ARMS[arm][1]}_{seed}"


def jobs() -> List[Tuple[str, int]]:
    """The 15 students in the registered order: seed-major, then h1500, hmix, sscut."""
    return [(a, s) for s in SEEDS for a in ARMS]


# ---------------------------------------------------------------------------------------------------- selection
def mix_mask(acct: np.ndarray, n: int, seed: int = MIX_SEED) -> Tuple[np.ndarray, Dict[str, Any]]:
    """S-H-mix's rows (module docstring) over the CONCATENATED TRAIN ``acct`` column: a boolean mask with exactly
    ``n`` True, and what was taken."""
    accts, counts = np.unique(acct, return_counts=True)
    if int(counts.sum()) < n:
        raise FollowupError(f"{int(counts.sum())} TRAIN rows, fewer than N = {n}")
    order = np.random.default_rng(seed).permutation(len(accts))
    cum = np.cumsum(counts[order])
    j = int(np.searchsorted(cum, n))                   # the first position whose cumulative count reaches n
    whole = accts[order[:j]]
    last = accts[order[j]]
    need = n - (int(cum[j - 1]) if j else 0)
    m = np.isin(acct, whole)
    li = np.nonzero(acct == last)[0][:need]
    m[li] = True
    info = {"seed": seed, "accounts_total": int(len(accts)), "accounts_taken": j + 1, "accounts_whole": j,
            "last_account": int(last), "last_account_rows": int(counts[order[j]]), "last_account_rows_kept": need}
    if int(m.sum()) != n:
        raise FollowupError(f"the mix selected {int(m.sum())} rows, not {n}")
    return m, info


def _train_recs(rows: Path) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    man, sha = RS.load_manifest(rows)
    return man, sha, list(man["train_shards"])


def _masks(arm: str, r2: Path, ss: Path, n: int) -> Tuple[Path, List[Dict[str, Any]], List[np.ndarray], Dict[str, Any]]:
    """The source rows dir, its TRAIN shard records, one keep-mask per shard, and the selection's record."""
    if arm == "sscut":
        _, _, recs = _train_recs(ss)
        masks, left = [], n
        for rec in recs:
            k = min(left, int(rec["rows"]))
            m = np.zeros(int(rec["rows"]), bool)
            m[:k] = True
            masks.append(m)
            left -= k
        if left:
            raise FollowupError(f"S-S holds fewer than N = {n} TRAIN rows")
        return ss, recs, masks, {"rule": f"S-S's first {n} TRAIN rows in shard order", "n": n}
    _, _, recs = _train_recs(r2)
    if arm == "h1500":
        masks = [RS.load_shard(r2, rec, ("rating",))["rating"] >= RATING_HI for rec in recs]
        got = int(sum(int(m.sum()) for m in masks))
        if got != n:
            raise FollowupError(f"r2 holds {got} TRAIN rows with actor rating >= {RATING_HI}, not the registered "
                                f"N = {n} (Amendment 2b): STOP")
        return r2, recs, masks, {"rule": f"r2's TRAIN rows whose actor's pre-battle rating >= {RATING_HI}, r2's order",
                                 "n": n}
    if arm == "hmix":
        accts = [RS.load_shard(r2, rec, ("acct",))["acct"] for rec in recs]
        m, info = mix_mask(np.concatenate(accts), n)
        cuts = np.cumsum([0] + [len(a) for a in accts])
        masks = [m[cuts[i]:cuts[i + 1]] for i in range(len(recs))]
        return r2, recs, masks, {"rule": f"r2's TRAIN accounts (np.unique, ascending) permuted by "
                                         f"np.random.default_rng({MIX_SEED}), taken whole in that order to {n} rows, "
                                         f"the last trimmed to its first rows in r2's order; r2's row order kept",
                                 "n": n, **info}
    raise FollowupError(f"arm {arm!r} is not one of {list(ARMS)}")


def iter_shards(src: Path, recs: Sequence[Dict[str, Any]], masks: Sequence[np.ndarray]
                ) -> Iterator[Dict[str, np.ndarray]]:
    """The kept rows, in source order, re-cut into ``rowset.SHARD``-row shards (the last holds the remainder)."""
    carry: List[Dict[str, np.ndarray]] = []
    n_carry = 0
    for rec, m in zip(recs, masks):
        if not m.any():
            continue
        z = RS.load_shard(src, rec)
        if len(m) != len(z["act"]):
            raise FollowupError(f"{rec['name']}: a mask of {len(m)} rows for {len(z['act'])}")
        carry.append({f: z[f][m] for f in RS.FIELDS})
        n_carry += int(m.sum())
        while n_carry >= RS.SHARD:
            cat = RS._concat(carry)
            yield {f: cat[f][:RS.SHARD] for f in RS.FIELDS}
            carry = [{f: cat[f][RS.SHARD:] for f in RS.FIELDS}]
            n_carry -= RS.SHARD
    if n_carry:
        yield RS._concat(carry)


def content_update(h: "hashlib._Hash", arr: Mapping[str, np.ndarray]) -> None:
    """The rows' bytes, field by field (name, dtype, shape, data): independent of the npz's write time."""
    for f in RS.FIELDS:
        a = np.ascontiguousarray(arr[f])
        h.update(f"{f}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())


def step_budget(sizes: Sequence[int], seeds: Sequence[int] = SEEDS, epochs: int = EPOCHS) -> Dict[int, int]:
    """Each seed's step budget, exactly as the trainer computes it (``student.plan``)."""
    from main.human_data import student as ST

    return {int(s): int(sum(nb for *_, nb in ST.plan(list(sizes), int(s), epochs))) for s in seeds}


# ---------------------------------------------------------------------------------------------------- derive
def _parent_val(arm: str, r2: Path, ss: Path) -> Tuple[Path, Dict[str, Any]]:
    src = ss if arm == "sscut" else r2
    man, _ = RS.load_manifest(src)
    return src, dict(man["val"])


def derive(arm: str, r2: Path, ss: Path, root: Path, n: int = N_EXPECTED) -> Dict[str, Any]:
    """Write ``<root>/<arm>/rows`` (module docstring) and its ``rows_manifest.json``."""
    from utils.git import get_git_hash

    rdir = rows_dir(root, arm)
    if (rdir / "rows_manifest.json").exists():
        raise FollowupError(f"{rdir} already holds a rowset: a registered input is never rewritten")
    src, recs, masks, sel = _masks(arm, r2, ss, n)
    rdir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    shards: List[Dict[str, Any]] = []
    for arr in iter_shards(src, recs, masks):
        content_update(h, arr)
        shards.append(RS._write_shard(rdir, f"train_{len(shards):04d}", arr))
    vsrc, vrec = _parent_val(arm, r2, ss)
    RS.load_shard(vsrc, vrec, ("act",))                # sha-checks the parent's validation shard
    shutil.copyfile(vsrc / f"{vrec['name']}.npz", rdir / "val.tmp.npz")
    os.replace(rdir / "val.tmp.npz", rdir / "val.npz")
    vrec = {**vrec, "name": "val"}
    if RS.S_sha(rdir / "val.npz") != vrec["sha256"]:
        raise FollowupError("the copied validation shard differs from its parent's record")
    pman, psha = RS.load_manifest(src)
    drop = ("train_shards", "val", "train_rows", "val_rows", "train_hidden", "row_profile", "rowset", "derived",
            "derive_commit")
    man = {k: v for k, v in pman.items() if k not in drop}
    sizes = [int(s["rows"]) for s in shards]
    man.update({"rowset": f"x59s_fu1500_{arm}", "parent_rowset": pman.get("rowset"),
                "parent_rows_manifest_sha256": psha,
                "followup": {"arm": arm, "student_arm": ARMS[arm][0], **sel, "content_sha256": h.hexdigest(),
                             "validation": f"the parent's validation shard, a byte copy ({vrec['sha256']})",
                             "step_budget": step_budget(sizes)},
                "train_shards": shards, "val": vrec, "train_rows": int(sum(sizes)), "val_rows": int(vrec["rows"]),
                "train_hidden": int(sum(s["hidden"] for s in shards)), "followup_commit": get_git_hash()})
    if man["train_rows"] != n:
        raise FollowupError(f"{arm}: {man['train_rows']} TRAIN rows written, not N = {n}")
    man["row_profile"] = RS._row_profile(rdir, shards, vrec)
    (rdir / "rows_manifest.json").write_text(json.dumps(man, indent=1))
    return {"arm": arm, "rows_manifest_sha256": RS.S_sha(rdir / "rows_manifest.json"), "train_rows": man["train_rows"],
            "val_rows": man["val_rows"], "shards": len(shards), "followup": man["followup"]}


def verify(arm: str, r2: Path, ss: Path, root: Path, n: int = N_EXPECTED) -> Dict[str, Any]:
    """Re-select from the parents (writing nothing) and compare with the written rowset: the content digest, every
    shard's rows, and every shard file's sha256 against the manifest."""
    man, sha = RS.load_manifest(rows_dir(root, arm))
    src, recs, masks, _ = _masks(arm, r2, ss, n)
    h = hashlib.sha256()
    sizes = []
    for arr in iter_shards(src, recs, masks):
        content_update(h, arr)
        sizes.append(len(arr["act"]))
    hw = hashlib.sha256()
    for rec in man["train_shards"]:
        content_update(hw, RS.load_shard(rows_dir(root, arm), rec))   # sha-checks each shard file too
    _, vrec = _parent_val(arm, r2, ss)
    ok = (h.hexdigest() == man["followup"]["content_sha256"] == hw.hexdigest()
          and sizes == [int(s["rows"]) for s in man["train_shards"]] and man["val"]["sha256"] == vrec["sha256"]
          and RS.S_sha(rows_dir(root, arm) / "val.npz") == vrec["sha256"])
    return {"arm": arm, "rows_manifest_sha256": sha, "content_sha256_reselected": h.hexdigest(),
            "content_sha256_written": hw.hexdigest(), "content_sha256_manifest": man["followup"]["content_sha256"],
            "pass": bool(ok)}


# ---------------------------------------------------------------------------------------------------- register
def register(root: Path, r2: Path, ss: Path, parent_reg: Mapping[str, Any], n: int = N_EXPECTED) -> Dict[str, Any]:
    """The follow-up's registration (``student.check_registration`` reads its keys): the three rows manifests and
    the cross-checks the read relies on."""
    _, r2sha = RS.load_manifest(r2)
    ssman, sssha = RS.load_manifest(ss)
    r2man, _ = RS.load_manifest(r2)
    problems: List[str] = []
    if not parent_reg.get("pass") or parent_reg.get("rows_manifest_sha256") != r2sha:
        problems.append("r2's rows are not the ones Amendment 2b registered")
    if parent_reg.get("selfplay_manifest_sha256") != sssha:
        problems.append("S-S's rows are not the ones Amendment 2b registered")
    if int(r2man["row_profile"]["rating_ge1500_rows"]) != n:
        problems.append(f"r2's profile records {r2man['row_profile']['rating_ge1500_rows']} >= 1500 rows, not N = {n}")
    out: Dict[str, Any] = {"schema": "gen3_x59s_fu1500_registration_v1", "amendment": "PREREG Amendment 7 (A2.7)",
                           "n": n, "seeds": list(SEEDS), "epochs": EPOCHS, "mix_seed": MIX_SEED,
                           "parent_rows_manifest_sha256": r2sha, "parent_selfplay_manifest_sha256": sssha,
                           "pstar_sha256": ssman.get("pstar_sha256"), "arms": {}}
    budgets = {}
    for arm, (st_arm, _, key) in ARMS.items():
        man, sha = RS.load_manifest(rows_dir(root, arm))
        parent = sssha if arm == "sscut" else r2sha
        if man.get("parent_rows_manifest_sha256") != parent:
            problems.append(f"{arm}: made from other parent rows")
        if man["train_rows"] != n:
            problems.append(f"{arm}: {man['train_rows']} TRAIN rows, not N = {n}")
        sizes = [int(s["rows"]) for s in man["train_shards"]]
        b = step_budget(sizes)
        if b != {int(k): int(v) for k, v in man["followup"]["step_budget"].items()}:
            problems.append(f"{arm}: the step budget differs from its manifest's")
        budgets[arm] = b
        vparent = (ssman if arm == "sscut" else r2man)["val"]["sha256"]
        if man["val"]["sha256"] != vparent:
            problems.append(f"{arm}: the validation shard is not its parent's")
        out[key] = sha
        out["arms"][arm] = {"student_arm": st_arm, "rows_manifest_sha256": sha, "train_rows": man["train_rows"],
                            "val_rows": man["val_rows"], "shards": len(sizes), "shard_sizes_tail": sizes[-2:],
                            "content_sha256": man["followup"]["content_sha256"], "step_budget": b,
                            "labels": ("P*'s sampled action (S-S's act column)" if arm == "sscut"
                                       else "the human's own action, or a hidden move's set (r2's act / moveset)")}
    if len({json.dumps(b, sort_keys=True) for b in budgets.values()}) != 1:
        problems.append(f"the step budgets differ across the arms: {budgets}")
    out["step_budget"] = next(iter(budgets.values()))
    out["steps_all_students"] = int(sum(sum(b.values()) for b in budgets.values()))
    out["problems"], out["pass"] = problems, not problems
    return out


# ---------------------------------------------------------------------------------------------------- window steps
def _stop(marker: str, why: str) -> int:
    Path(marker).write_text(f"written by main.human_data.followup: {why}\n")
    print(f"[fu1500] STOP: {why} -> wrote the preempt marker {marker}: no later step runs", flush=True)
    return 3


def preflight(root: Path, registration: Path, token_file: str, marker: str,
              token_valid: Optional[Callable[[str], Tuple[bool, str]]] = None) -> int:
    """The window's CPU precondition step (module docstring, step 1)."""
    probs: List[str] = []
    try:
        reg = json.loads(Path(registration).read_text())
    except (OSError, ValueError) as exc:
        return _stop(marker, f"the registration {registration} is unreadable ({type(exc).__name__})")
    if not reg.get("pass"):
        probs.append(f"the registration {registration} did not pass")
    for arm, (_, _, key) in ARMS.items():
        rd = rows_dir(root, arm)
        try:
            man, sha = RS.load_manifest(rd)
            if reg.get(key) != sha:
                probs.append(f"{arm}: {rd} is not the registered rowset")
            for rec in [*man["train_shards"], man["val"]]:
                if RS.S_sha(rd / f"{rec['name']}.npz") != rec["sha256"]:
                    probs.append(f"{arm}: {rec['name']} differs from its manifest")
        except (OSError, ValueError, KeyError) as exc:
            probs.append(f"{arm}: {rd} unreadable ({type(exc).__name__})")
    if token_valid is None:
        from main.ops.arm_chain.window import lease_token_valid as token_valid
    ok, why = token_valid(token_file)
    if not ok:
        probs.append(f"the GPU lease token: {why}")
    if probs:
        return _stop(marker, "; ".join(probs[:6]))
    (Path(root) / "results").mkdir(parents=True, exist_ok=True)
    (Path(root) / "students").mkdir(parents=True, exist_ok=True)
    print(f"[fu1500] preflight PASS: 3 registered rowsets verified, the lease token valid; students -> "
          f"{Path(root) / 'students'}", flush=True)
    return 0


def gate(preflight_json: Path, marker: str) -> int:
    """A2.9: the CUDA pre-flight's FAIL (or a missing result) HOLDs the window: it writes the marker."""
    try:
        res = json.loads(Path(preflight_json).read_text())
    except (OSError, ValueError) as exc:
        return _stop(marker, f"no readable CUDA pre-flight result {preflight_json} ({type(exc).__name__})")
    if not res.get("pass"):
        failed = [k for k, v in (res.get("checks") or {}).items() if not v.get("pass")]
        return _stop(marker, f"the CUDA pre-flight FAILED: {failed or res.get('error')}")
    print(f"[fu1500] the CUDA pre-flight PASSED ({res.get('checks', {}).get('cost')})", flush=True)
    return 0


def chunk_action(rc: int, chunks: int, max_chunks: int = MAX_CHUNKS) -> str:
    """A2.9's rule: exit 0 is done, exit 3 (the chunk budget) continues up to ``max_chunks``, anything else stops."""
    if rc == 0:
        return "done"
    if rc == 3 and chunks < max_chunks:
        return "continue"
    return "stop"


def train_argv(arm: str, seed: int, root: Path, registration: Path, device: str = "cuda",
               chunk_min: float = CHUNK_MIN) -> List[str]:
    return ["train", "--arm", ARMS[arm][0], "--rows", str(rows_dir(root, arm)), "--seed", str(seed),
            "--epochs", str(EPOCHS), "--device", device, "--budget-min", f"{chunk_min:g}",
            "--registration", str(registration), "--out", str(student_dir(root, arm, seed))]


def train_step(arm: str, seed: int, root: Path, registration: Path, marker: str, device: str = "cuda",
               chunk_min: float = CHUNK_MIN, max_chunks: int = MAX_CHUNKS,
               run: Optional[Callable[[List[str]], int]] = None) -> int:
    """One student to completion (module docstring, step 4)."""
    if run is None:
        from main.human_data.student import main as run
    argv = train_argv(arm, seed, root, registration, device, chunk_min)
    chunks = 0
    while True:
        chunks += 1
        try:
            rc = run(argv)
        except Exception as exc:  # noqa: BLE001 -- recorded in the marker, then re-raised
            _stop(marker, f"{ARMS[arm][1]}_{seed} chunk {chunks} raised {type(exc).__name__}: {str(exc)[:200]}")
            raise
        act = chunk_action(rc, chunks, max_chunks)
        print(f"[fu1500] {ARMS[arm][1]}_{seed} chunk {chunks}: exit {rc} -> {act}", flush=True)
        if act == "done":
            return 0
        if act == "stop":
            return _stop(marker, f"{ARMS[arm][1]}_{seed} chunk {chunks} exited {rc}")


def check_students(root: Path, reg: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 5: every student finished at its registered budget, on its registered rows; one init per seed."""
    import torch as th

    bad: List[str] = []
    inits: Dict[int, Dict[str, str]] = {}
    for arm, seed in jobs():
        p = student_dir(root, arm, seed) / "state.pt"
        if not p.exists():
            bad.append(f"{p}: missing")
            continue
        st = th.load(p, weights_only=False, map_location="cpu")
        want = int({int(k): v for k, v in reg["arms"][arm]["step_budget"].items()}[seed])
        if not st.get("done") or int(st["step"]) != want or int(st["total"]) != want:
            bad.append(f"{p}: step {st.get('step')} of {st.get('total')}, done {st.get('done')}; registered {want}")
        if st["arm"] != ARMS[arm][0] or st["rows_manifest_sha256"] != reg[ARMS[arm][2]]:
            bad.append(f"{p}: arm {st['arm']} / rows {str(st['rows_manifest_sha256'])[:12]} are not the registered")
        inits.setdefault(seed, {})[arm] = str(st["init_sha256"])
    for seed, d in sorted(inits.items()):
        if len(set(d.values())) != 1:
            bad.append(f"seed {seed}: init_sha256 differs across the arms {d}")
    return {"schema": "gen3_x59s_fu1500_window_check_v1", "students": len(jobs()), "problems": bad, "pass": not bad}


# ---------------------------------------------------------------------------------------------------- the chain
def window_budget(reg: Mapping[str, Any]) -> float:
    """The brief's budget (module docstring): whole minutes, rounded up."""
    from main.ops.gpu_fill_window import window_budget_min

    return window_budget_min([MS_PER_STEP * int(reg["steps_all_students"]) / 60000.0])


def window_item(*, worktree: str, require_commit: str, python: str, root: str, job_dir: str, token_file: str,
                reg: Mapping[str, Any]) -> Dict[str, Any]:
    from main.ops.gpu_fill_window import _cpu, _gpu

    reg_path = os.path.join(worktree, "designs", "research_state", "measurements",
                            "x59s_matched_students_2026-10-25", "results", "registration_a7.json")
    marker = os.path.join(job_dir, MARKER)
    me = [python, "-m", "main.human_data.followup"]
    pf_json = os.path.join(root, "results", "cuda_preflight.json")
    steps = [_cpu("preflight", [*me, "preflight", "--root", root, "--registration", reg_path,
                                "--token-file", token_file, "--marker", marker]),
             _gpu("cuda_preflight", [python, "-m", "main.human_data.cuda_preflight", "--rows",
                                     str(rows_dir(Path(root), "h1500")), "--registration", reg_path, "--scratch",
                                     os.path.join(root, "preflight_scratch"), "--out", pf_json, "--device", "cuda",
                                     "--arm", ARMS["h1500"][0], "--students", str(len(jobs()))],
                  MEM_GB_PREFLIGHT, PREFLIGHT_TIMEOUT_S),
             _cpu("preflight_gate", [*me, "gate", "--preflight-json", pf_json, "--marker", marker])]
    for arm, seed in jobs():
        steps.append(_gpu(f"student_{ARMS[arm][1]}_{seed}",
                          [*me, "train-step", "--arm", arm, "--seed", str(seed), "--root", root,
                           "--registration", reg_path, "--marker", marker], MEM_GB_STUDENT, STUDENT_TIMEOUT_S))
    steps.append(_cpu("check", [*me, "check", "--root", root, "--registration", reg_path,
                                "--out", os.path.join(root, "students", "window_check.json")]))
    return {"kind": "window", "name": "x59s-fu1500", "worktree": worktree, "require_commit": require_commit,
            "on_fail": ON_FAIL, "preempt_marker": marker, "budget_min": window_budget(reg),
            "x_registration": (f"X59-S PREREG Amendment 7 (A2.7): 15 students, {reg['steps_all_students']} steps at "
                               f"{MS_PER_STEP} ms x 1.5 + 5 min; any FAIL writes {marker} and HOLDs; "
                               f"students -> {root}/students"),
            "steps": steps}


# ---------------------------------------------------------------------------------------------------- the read
def export_all(root: Path, r2: Path, ss: Path) -> Dict[str, Any]:
    """Each trained student exported into ``<root>/read/students/<label>`` (``state.pt`` symlinked, the training dir
    untouched) with both yardsticks; VOID 3 and the learned-nothing guard at the registered budget."""
    from main.human_data import student as ST
    from main.human_data.x59s_ops import last_epoch_rose, void3

    reg_budget = None
    sides: Dict[str, Dict[str, Any]] = {}
    out = Path(root) / "read" / "students"
    for arm, seed in jobs():
        lab = f"{ARMS[arm][1]}_{seed}"
        src, dst = student_dir(root, arm, seed), out / lab
        side_p = dst / "model" / "student_sidecar.json"
        if not side_p.exists():
            dst.mkdir(parents=True, exist_ok=True)
            link = dst / "state.pt"
            if not link.is_symlink():
                link.symlink_to((src / "state.pt").resolve())
            (dst / "rows_path.txt").write_text((src / "rows_path.txt").read_text())
            ST.export(dst, {"human_val_human_labels": (r2, "human"), "selfplay_val_pstar_labels": (ss, "selfplay")})
        s = json.loads(side_p.read_text())
        if s["arm"] != ARMS[arm][0] or int(s["seed"]) != seed:
            raise FollowupError(f"{side_p}: arm {s['arm']} seed {s['seed']} is not {arm} {seed}")
        s["zip_path"] = str(dst / "model" / Path(s["zip"]).name)
        s["val_rose_last_epoch"] = last_epoch_rose(s["val_hist"])
        sides[lab] = s
        reg_budget = reg_budget or int(s["steps"])
    doc = {"schema": "gen3_x59s_fu1500_students_v1", "students": sides,
           "void3": void3(sides, steps=int(reg_budget or 0))}
    (out / "students.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def inhouse_cells(students: Mapping[str, Any]) -> List[Dict[str, Any]]:
    side = students["students"]
    return [{"contrast": c, "seed": k, "panel": p, "player": side[f"{ARMS[a][1]}_{k}"]["zip_path"],
             "opponent": side[f"{ARMS[b][1]}_{k}"]["zip_path"], "name": f"{c}_{k}_{p}"}
            for c, (a, b, _) in PAIRINGS.items() for k in SEEDS for p in PANELS]


def inhouse_play(state: Path, students: Mapping[str, Any], tree: Path, py: str, attempts: int = 2) -> None:
    """Every cell from the play tree ``tree`` (its ``src`` on PYTHONPATH, its checkout as cwd), CPU, sequentially."""
    from main.human_data.x59s_ops import _summary_of, inhouse_argv

    state.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(Path(tree) / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    for cell in inhouse_cells(students):
        log = state / f"{cell['name']}.log"
        k = 0
        while (_summary_of(log) or {}).get("pairs") != PAIRS and k < attempts:
            with open(log, "a") as f:
                subprocess.run(inhouse_argv(cell, state / "ledger", py, label=INHOUSE_LABEL), cwd=str(tree), env=env,
                               stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
            k += 1


def inhouse_read(state: Path, students: Mapping[str, Any]) -> Dict[str, Any]:
    """Each pairing's 5 rates (the equal-weight panel mean of S-H≥1500's mirrored-pair score), its class
    (``x59s_read.inhouse``: §5's classes + NO-DIFFERENCE-AT-4PP), the engines the ledger ran on (exactly one)."""
    from main.human_data import x59s_read as XR
    from main.human_data.x59s_ops import _summary_of, ledger_engines

    out: Dict[str, Any] = {"schema": "gen3_x59s_fu1500_inhouse_v1", "pairs": PAIRS, "cells": {}, "rates": {},
                           "classes": {}, "short": []}
    for cell in inhouse_cells(students):
        s = _summary_of(state / f"{cell['name']}.log")
        ok = s is not None and s.get("pairs") == PAIRS and s.get("score") is not None
        out["cells"][cell["name"]] = {"player": cell["player"], "opponent": cell["opponent"],
                                      "score": s.get("score") if ok else None, "pairs": (s or {}).get("pairs"),
                                      "regime_id": (s or {}).get("regime_id")}
        if not ok:
            out["short"].append(cell["name"])
    out["core_stamps"] = ledger_engines(state / "ledger")
    for c, (_, _, names) in PAIRINGS.items():
        rates = []
        for k in SEEDS:
            sc = [out["cells"][f"{c}_{k}_{p}"]["score"] for p in PANELS]
            rates.append(None if any(x is None for x in sc) else sum(sc) / len(sc))
        out["rates"][c] = rates
        if all(r is not None for r in rates) and len(out["core_stamps"]) == 1:
            out["classes"][c] = XR.inhouse(rates, names)
        else:
            out["classes"][c] = {"class": None, "why": "a short cell or not exactly one engine"}
    return out


# ---------------------------------------------------------------------------------------------------- cli
def _env_python() -> str:
    return str(Path.home() / "miniconda3" / "envs" / "gen3ai_torch28" / "bin" / "python3")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.followup", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    v = sub.add_parser("verify")
    for p in (d, v):
        p.add_argument("--arm", required=True, choices=tuple(ARMS))
    r = sub.add_parser("register")
    r.add_argument("--parent-registration", required=True)
    r.add_argument("--out", required=True)
    x = sub.add_parser("export")
    for p in (d, v, r, x):
        p.add_argument("--r2", required=True, help="hr_spec_v1.r2's rows dir")
        p.add_argument("--ss", required=True, help="S-S's rows dir (ss_rows_v2)")
        p.add_argument("--root", required=True)
    e = sub.add_parser("emit-chain", help="the window's chain file (JSON), CPU only")
    e.add_argument("--base", required=True, help="the queue's chain file: job_dir, backend, lease and models_dir")
    e.add_argument("--name", required=True)
    e.add_argument("--worktree", required=True, help="a clean, bootstrapped SCRATCH worktree (never main)")
    e.add_argument("--require-commit", required=True)
    e.add_argument("--root", required=True)
    e.add_argument("--python", default=_env_python())
    e.add_argument("--write", default="", help="write the chain here (refuses an existing file); else stdout")
    pf = sub.add_parser("preflight")
    pf.add_argument("--root", required=True)
    pf.add_argument("--registration", required=True)
    pf.add_argument("--token-file", required=True)
    pf.add_argument("--marker", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--preflight-json", required=True)
    g.add_argument("--marker", required=True)
    t = sub.add_parser("train-step")
    t.add_argument("--arm", required=True, choices=tuple(ARMS))
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--root", required=True)
    t.add_argument("--registration", required=True)
    t.add_argument("--marker", required=True)
    t.add_argument("--device", default="cuda")
    c = sub.add_parser("check")
    c.add_argument("--root", required=True)
    c.add_argument("--registration", required=True)
    c.add_argument("--out", required=True)
    h = sub.add_parser("inhouse")
    h.add_argument("--root", required=True)
    h.add_argument("--tree", required=True, help="the play tree (a clean checkout at the read's pin)")
    h.add_argument("--python", default=_env_python())
    h.add_argument("--play", action="store_true")
    h.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd in ("derive", "verify", "register", "export"):
            RS._guard_out(Path(a.root))
            r2, ss, root = Path(a.r2), Path(a.ss), Path(a.root)
            if a.cmd == "derive":
                res = derive(a.arm, r2, ss, root)
            elif a.cmd == "verify":
                res = verify(a.arm, r2, ss, root)
            elif a.cmd == "register":
                res = register(root, r2, ss, json.loads(Path(a.parent_registration).read_text()))
                Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
            else:
                res = export_all(root, r2, ss)
                res = {"students": len(res["students"]), "void3": res["void3"], "pass": not res["void3"]}
            print(json.dumps(res, indent=1), flush=True)
            return 0 if res.get("pass", True) else 1
        if a.cmd == "emit-chain":
            from main.ops.gpu_fill_window import FillError, _outside, chain_doc
            from utils.git import get_main_repo_root

            base = json.loads(Path(a.base).read_text())
            wt = os.path.realpath(a.worktree)
            if wt == os.path.realpath(get_main_repo_root(cwd=wt)):
                raise FollowupError(f"--worktree {a.worktree} is the MAIN checkout: the window runs in a scratch worktree")
            models_dir = str(base.get("models_dir") or os.path.join(base["repo"], "models"))
            try:
                _outside(a.root, (wt, models_dir, base["repo"]))
            except FillError as exc:
                raise FollowupError(str(exc)) from exc
            reg_p = Path(wt) / "designs/research_state/measurements/x59s_matched_students_2026-10-25/results/registration_a7.json"
            reg = json.loads(reg_p.read_text())
            if not reg.get("pass"):
                raise FollowupError(f"{reg_p} did not pass")
            item = window_item(worktree=wt, require_commit=a.require_commit, python=a.python, root=a.root,
                               job_dir=base["job_dir"], token_file=base["backend"]["lease_token_file"], reg=reg)
            text = json.dumps(chain_doc(base, a.name, [item]), indent=1) + "\n"
            if a.write:
                if os.path.exists(a.write):
                    raise FollowupError(f"--write {a.write} exists: a chain file is never overwritten")
                Path(a.write).write_text(text)
                print(f"[fu1500] wrote {a.write}: {item['name']} ({len(item['steps'])} steps, budget "
                      f"{item['budget_min']:g} min, on_fail {item['on_fail']}, marker {item['preempt_marker']})")
            else:
                sys.stdout.write(text)
            return 0
        if a.cmd == "preflight":
            return preflight(Path(a.root), Path(a.registration), a.token_file, a.marker)
        if a.cmd == "gate":
            return gate(Path(a.preflight_json), a.marker)
        if a.cmd == "train-step":
            return train_step(a.arm, a.seed, Path(a.root), Path(a.registration), a.marker, a.device)
        if a.cmd == "check":
            res = check_students(Path(a.root), json.loads(Path(a.registration).read_text()))
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
            print(json.dumps(res, indent=1), flush=True)
            return 0 if res["pass"] else 1
        st = json.loads((Path(a.root) / "read" / "students" / "students.json").read_text())
        state = Path(a.root) / "read" / "inhouse"
        if a.play:
            inhouse_play(state, st, Path(a.tree), a.python)
        res = inhouse_read(state, st)
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"out": a.out, "short": res["short"], "core_stamps": res["core_stamps"],
                          "classes": {k: v.get("class") for k, v in res["classes"].items()}}), flush=True)
        return 0
    except (FollowupError, RS.RowsetError) as exc:
        print(f"[fu1500] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
