"""The probe battery's DECISION BANK: games between checkpoints on the Rust eval core, replayed through the core with
both sides' views, then a deterministic stratified selection (`gen3_probe_battery_v1`, ``designs/prober/probe_battery.md``).

The bank stores INPUTS and TRUTH, never a model's reading: each game's input log (``games.jsonl.gz``), and per selected
decision the obs row the checkout's encoder wrote (``rows.npy``), its mask, the viewer's ``present()`` view (``V``) and
the opponent's own view at the same board (``W`` — the opponent's TRUE state), the opponent's choice there and the
game's winner (``decisions.jsonl.gz``). Facts are derived from these by :mod:`main.probe_battery.facts`.

Every step is deterministic: the games are seeded (mirrored pairs, one cycle seed per cell, ONE engine on CPU), the
replay is a pure function of the input log, and the selection ranks decisions by ``sha256(bank_seed:decision id)``
inside each (archetype, phase) stratum — never by chance.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "gen3_probe_battery_v1"
WORKER = Path(__file__).with_name("pin_worker.py")
#: the stratification axes' declared values (archetype from ``agents.training.team_archetypes``)
ARCHETYPES = ("hyper_offense", "offense", "balance", "semi_stall", "stall")
PHASES = ("opening", "midgame", "endgame")


class BatteryError(RuntimeError):
    """A typed refusal of the probe battery."""


# ------------------------------------------------------------------------------------------ refusals / paths
def refuse_models_output(path: Path) -> None:
    """``models/`` is read-only (root standing rule 6): every output of the battery lands outside it."""
    from utils.paths import main_models_dir

    p = Path(path).expanduser().resolve()
    roots = [Path(os.environ["GEN3AI_MODELS_DIR"]).resolve()] if os.environ.get("GEN3AI_MODELS_DIR") else []
    md = main_models_dir()
    if md is not None:
        roots.append(md.resolve())
    for r in roots:
        if p == r or r in p.parents:
            raise BatteryError(f"REFUSED: {path} is under models/ ({r}), which is read-only — write outside it")
    # a CHECKOUT's own models/ (a worktree's twin of the archive) is refused the same way
    for q in [p] + list(p.parents):
        if q.name == "models" and ((q.parent / ".git").exists() or (q.parent / "src").is_dir()):
            raise BatteryError(f"REFUSED: {path} is under a checkout's models/ ({q}) — write outside it")


class gz_writer:
    """A gzip text writer with a FIXED header (no name, mtime 0), so the same content is the same bytes."""

    def __init__(self, path: Path):
        import io

        self._raw = open(path, "wb")
        self._text = io.TextIOWrapper(gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0),
                                      encoding="utf-8")

    def write(self, s: str) -> int:
        return self._text.write(s)

    def __enter__(self) -> "gz_writer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._text.close()
        self._raw.close()


def worker_argv(checkout: Path, expect_commit: Optional[str], args: Sequence[str], *, mem_gb: int = 24,
                timeout_s: int = 3 * 3600, name: str = "probebat") -> Tuple[List[str], Dict[str, str]]:
    """The command that runs :mod:`pin_worker` under ``checkout``'s code: CPU only (CUDA hidden), ``nice 19``, in its
    own memory-capped scope (``scripts/ops/mem_cap.sh``), with a timeout INSIDE the cap."""
    from utils.paths import repo_path

    checkout = Path(checkout).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(checkout / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    ce = checkout / "src" / "rust_sim" / "target" / "release" / "core_events"
    if ce.exists():
        env["POKESIM_CORE_EVENTS_BIN"] = str(ce)
    py = os.environ.get("GEN3AI_PYTHON") or sys.executable
    head = ["nice", "-n", "19", str(repo_path("scripts", "ops", "mem_cap.sh")), "--name", name, str(mem_gb),
            "timeout", str(timeout_s), py, str(WORKER), "--checkout", str(checkout)]
    if expect_commit:
        head += ["--expect-commit", expect_commit]
    return head + list(args), env


def run_worker(checkout: Path, expect_commit: Optional[str], args: Sequence[str], **kw: Any) -> Dict[str, Any]:
    argv, env = worker_argv(checkout, expect_commit, args, **kw)
    print("[probe_battery] $ " + " ".join(shlex.quote(a) for a in argv), file=sys.stderr, flush=True)
    p = subprocess.run(argv, env=env, stdout=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        raise BatteryError(f"the worker failed (exit {p.returncode}): {' '.join(args[:1])}")
    last = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    return json.loads(last[-1]) if last else {}


# -------------------------------------------------------------------------------------------------- the plan
def plan_cells(legacy: Sequence[Tuple[str, str]], static: Sequence[Tuple[str, str]], pairs: int, seed: int
               ) -> List[Dict[str, Any]]:
    """The games' cells: each legacy seed vs the static seed of the same index (the cross's diagonal), each legacy
    seed vs the next legacy seed, each static seed vs the next static seed (a ring) — so both arms play, against
    both arms. ``legacy`` / ``static`` = ``[(label, zip)]``. One cycle seed per cell: ``seed + 1000 * k``."""
    cells: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    for i in range(min(len(legacy), len(static))):
        cells.append((legacy[i], static[i]))
    for arm in (legacy, static):
        if len(arm) >= 2:
            for i in range(len(arm)):
                cells.append((arm[i], arm[(i + 1) % len(arm)]))
    out = []
    for k, ((pl, pz), (ol, oz)) in enumerate(cells):
        out.append({"cell": f"{pl}x{ol}", "player": pz, "opponent": oz, "player_label": pl, "opponent_label": ol,
                    "pairs": int(pairs), "seed": int(seed) + 1000 * k})
    return out


# ---------------------------------------------------------------------------------------------- the strata
def archetype_table(checkout: Path, expect_commit: Optional[str], out: Path) -> Dict[str, str]:
    """``{packed team: archetype}`` for every team of ``checkout``'s pool, computed BY THE WORKER at that checkout
    (its teambuilder packs the teams exactly as its eval core did; ``classify_team`` reads an export)."""
    dst = Path(out) / "archetypes.json"
    if not dst.exists():
        run_worker(checkout, expect_commit, ["archetypes", "--out", str(dst)], name="probebat-arch")
    return json.loads(dst.read_text())


def phase_of(d: Dict[str, Any]) -> str:
    V = d["V"]
    if int(V.get("turn") or 0) <= 3:
        return "opening"
    alive = [sum(1 for m in V[s]["mons"] if not m.get("fainted")) + max(0, int(V[s].get("team_size") or 6)
             - len(V[s]["mons"])) for s in ("ours", "opp")]
    return "endgame" if min(alive) <= 2 else "midgame"


def rank_key(bank_seed: int, did: str) -> str:
    return hashlib.sha256(f"{bank_seed}:{did}".encode()).hexdigest()


def decision_id(d: Dict[str, Any]) -> str:
    return f"{d['game']}/p{int(d['side']) + 1}/{int(d['n'])}"


def select(decisions: Sequence[Dict[str, Any]], *, target: int, bank_seed: int, min_legal: int = 2
           ) -> List[int]:
    """Indices of the selected decisions: those with >= ``min_legal`` legal actions, ``target // 5`` per viewer-team
    ARCHETYPE (a short archetype gives all it has; the slack goes to the others, in declared order), each archetype's
    quota ranked by ``sha256(bank_seed:id)`` inside its phase strata in proportion to the phase's share. Deterministic:
    the same inputs give the same indices."""
    by: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    for i, d in enumerate(decisions):
        if sum(int(x) for x in d["mask"]) < min_legal:
            continue
        by[d["archetype"]][d["phase"]].append((rank_key(bank_seed, decision_id(d)), i))
    quota = {a: 0 for a in ARCHETYPES}
    avail = {a: sum(len(v) for v in by[a].values()) for a in ARCHETYPES}
    left = int(target)
    open_ = [a for a in ARCHETYPES if avail[a] > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for a in open_:
            take = min(share, avail[a] - quota[a], left)
            quota[a] += take
            left -= take
            if quota[a] < avail[a]:
                nxt.append(a)
            if left <= 0:
                break
        open_ = nxt
    picked: List[int] = []
    for a in ARCHETYPES:
        q = quota[a]
        if q <= 0:
            continue
        strata = {p: sorted(by[a][p]) for p in PHASES}
        tot = sum(len(v) for v in strata.values())
        # proportional per phase, rounding by largest remainder (deterministic ties: declared phase order)
        raw = {p: q * len(strata[p]) / tot for p in PHASES}
        base = {p: int(np.floor(raw[p])) for p in PHASES}
        rem = q - sum(base.values())
        for p in sorted(PHASES, key=lambda p: (-(raw[p] - base[p]), PHASES.index(p)))[:rem]:
            base[p] += 1
        for p in PHASES:
            picked += [i for _, i in strata[p][:base[p]]]
    return sorted(picked)


# --------------------------------------------------------------------------------------------- reading
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_decisions(bank: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(Path(bank) / "decisions.jsonl.gz"))


def content_sha256(path: Path) -> str:
    """sha256 of a gzip file's DECOMPRESSED bytes (a gzip header carries an mtime; the content is the record)."""
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(*, checkout: Path, expect_commit: Optional[str], legacy: Sequence[Tuple[str, str]],
          static: Sequence[Tuple[str, str]], pairs: int, seed: int, target: int, out: Path,
          n_envs: int = 32, threads: int = 2, torch_threads: int = 3) -> Dict[str, Any]:
    """Play → replay → select → write the bank under ``out`` (resumable: a finished step is not redone)."""
    out = Path(out).expanduser()
    refuse_models_output(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = plan_cells(legacy, static, pairs, seed)
    (out / "cells.json").write_text(json.dumps(cells, indent=1))
    games = out / "games.jsonl.gz"
    play = run_worker(checkout, expect_commit, ["play", "--cells", str(out / "cells.json"), "--out", str(games),
                                                 "--n-envs", str(n_envs), "--threads", str(threads),
                                                 "--torch-threads", str(torch_threads)], name="probebat-play")
    full = out / "replay_all"
    if not (full / "rows.npy").exists():
        run_worker(checkout, expect_commit, ["replay", "--games", str(games), "--out", str(full)],
                   name="probebat-replay")
    teams = {}
    for g in iter_jsonl(games):
        start = json.loads(g["script"].split("\n", 1)[0][len("START "):])
        gid = f"{g['cell']}#{g['game']}"
        teams[gid] = (start["p1"]["team"], start["p2"]["team"])
    table = archetype_table(checkout, expect_commit, out)
    missing = {p for pair in teams.values() for p in pair if p not in table}
    if missing:
        raise BatteryError(f"{len(missing)} played teams are not in {checkout}'s pool table — cannot stratify")
    decs = []
    for d in iter_jsonl(full / "decisions.jsonl.gz"):
        d["archetype"] = table[teams[d["game"]][int(d["side"])]]
        d["phase"] = phase_of(d)
        decs.append(d)
    idx = select(decs, target=target, bank_seed=seed)
    rows_all = np.load(full / "rows.npy", mmap_mode="r")
    rows = np.ascontiguousarray(rows_all[idx], dtype=np.float32)
    masks = np.array([decs[i]["mask"] for i in idx], dtype=np.int8)
    np.save(out / "rows.npy", rows)
    shutil.copyfile(full / "obs_layout.json", out / "obs_layout.json")
    np.save(out / "masks.npy", masks)
    with gz_writer(out / "decisions.jsonl.gz") as f:
        for k, i in enumerate(idx):
            d = dict(decs[i])
            d["bank_idx"] = k
            d["id"] = decision_id(d)
            f.write(json.dumps(d, sort_keys=True) + "\n")
    sel = [decs[i] for i in idx]
    man = {
        "schema": SCHEMA, "checkout": str(Path(checkout).resolve()), "commit": expect_commit, "seed": seed,
        "pairs_per_cell": pairs, "cells": len(cells), "games": sum(1 for _ in iter_jsonl(games)),
        "decisions_replayed": len(decs), "decisions": len(idx), "obs_dim": int(rows.shape[1]),
        "by_archetype": dict(Counter(d["archetype"] for d in sel)),
        "by_phase": dict(Counter(d["phase"] for d in sel)),
        "by_seat_arm": dict(Counter(str(d["seat_label"])[:1] for d in sel)),
        "battles": len({d["game"] for d in sel}),
        "sha256": {"rows.npy": file_sha256(out / "rows.npy"), "masks.npy": file_sha256(out / "masks.npy"),
                   "decisions.jsonl.gz (content)": content_sha256(out / "decisions.jsonl.gz"),
                   "games.jsonl.gz (content)": content_sha256(games)},
        "play": play,
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man
