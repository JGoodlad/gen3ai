"""One unit of the cutover stress — each kind returns ONE row (a JSON-able dict).

A unit never raises past :func:`run_unit`: an exception is a row with ``status: "error"`` and its
traceback (a RESULT to be triaged, not a retry). A unit killed mid-run leaves no row and is re-run
by the next driver (the row is written atomically, so its presence means it is complete).
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from main.rust_core_cutover import plan as PL

#: Numeric census fields summed into a unit's totals, per slice.
_NUM = {
    "E": ("battles", "viewers", "events", "lines"),
    "V": ("battles", "viewers", "decisions", "fields"),
    "T": ("battles", "viewers", "decisions", "rewards", "terminal_rewards", "window_rows"),
    "O": ("battles", "viewers", "decisions", "rows_equal", "value_cells", "byte_only_cells", "nan_cells", "tokens"),
}
#: Counter fields merged into a unit's totals (printed; the divergences are the verdict).
_CTR = {
    "E": ("kinds", "selfcheck"),
    "V": ("board_checks", "rules_fired", "known", "known_decisions", "out_of_scope"),
    "T": ("labels", "out_of_scope"),
    "O": ("nonzero_blocks",),
}


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=repr))


def _census_row(slices: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for name, c in slices.items():
        row = {k: getattr(c, k) for k in _NUM[name] if hasattr(c, k)}
        for k in _CTR[name]:
            if hasattr(c, k):
                row[k] = dict(getattr(c, k))
        row["divergences"] = dict(c.divergences)
        row["refused"] = list(c.refused)
        out[name] = row
    return out


def merge_totals(total: Dict[str, Dict[str, Any]], add: Dict[str, Dict[str, Any]]) -> None:
    """Fold one battle's census row into a unit's totals (sums, counter merges, refusal lists)."""
    for name, row in add.items():
        t = total.setdefault(name, {})
        for k, v in row.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                t[k] = t.get(k, 0) + v
            elif isinstance(v, dict):
                c = collections.Counter(t.get(k, {}))
                c.update(v)
                t[k] = dict(c)
            elif isinstance(v, list):
                t[k] = (t.get(k, []) + v)[:50]


def _divergent(row: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    return {s: {**r["divergences"], **({f"REFUSED: {x[:200]}": 1 for x in r["refused"]})}
            for s, r in row.items() if r["divergences"] or r["refused"]}


def known_divergence(stream: str, key: int, classes: Dict[str, Dict[str, int]]) -> Optional[str]:
    """The NAMED known divergence this battle reproduces EXACTLY, or ``None``.

    Only ``ladder_full_a`` replays the MILESTONE tier's ladder battles byte for byte (same key,
    same teams, same seeds), so only there can a ``LADDER_KNOWN_DIVERGENCES`` entry fire, and only
    in exactly its named slice-V census keys with every other slice clean."""
    if stream != "ladder_full_a":
        return None
    from agents.battle.rust_core_parity import LADDER_KNOWN_DIVERGENCES

    entry = LADDER_KNOWN_DIVERGENCES.get(key)
    if entry is None:
        return None
    name, want, _where = entry
    if set(classes) == {"V"} and set(classes["V"]) == set(want):
        return name
    return None


# ---------------------------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------------------------

def _teams_for(stream: PL.Stream, i: int) -> List[str]:
    from utils import ladder_corpus, team_sources

    src = stream.params["source"]
    if src == "pool":
        return team_sources.team_list("pool")
    if src == "ladder":
        return ladder_corpus.teams(stream.params["tier"])
    if src == "procedural":
        return team_sources.procedural_teams(2 * len(stream.unit_range(i)), PL.procedural_seed(stream, i))
    raise ValueError(src)


def run_parity(stream: PL.Stream, i: int, out: Path) -> Dict[str, Any]:
    import logging

    from agents.battle import rust_core_parity as P
    from agents.battle.rust_core_parity_obs import ObsCensus
    from agents.battle.rust_core_parity_trackers import TrackerCensus
    from agents.battle.rust_core_parity_views import ViewCensus

    logging.getLogger("poke-env").setLevel(logging.CRITICAL)
    teams = _teams_for(stream, i)
    policy = None
    if stream.params["policy"]:
        import torch

        torch.set_num_threads(1)
        policy = P.load_production_policy()
    totals: Dict[str, Dict[str, Any]] = {}
    divergent, errors, known = [], [], []
    covered = []
    for pb in PL.parity_battles(stream, i, len(teams)):
        slices = {"E": P.Census(), "V": ViewCensus(), "T": TrackerCensus(), "O": ObsCensus()}
        t0 = time.monotonic()
        try:
            live = P.play(pb.key, policy=policy, teams=(teams[pb.t1], teams[pb.t2]))
            live.recorded.label = pb.label
            P.compare_live(live, slices["E"])
            P.check_battles([live.recorded], slices["E"], views=slices["V"], trackers=slices["T"],
                            obs=slices["O"])
        except Exception as e:  # a battle that cannot be played or checked is a finding
            errors.append({"label": pb.label, "key": pb.key, "t1": pb.t1, "t2": pb.t2,
                           "error": f"{type(e).__name__}: {e}"[:2000],
                           "traceback": traceback.format_exc()[-4000:]})
            continue
        row = _census_row(slices)
        merge_totals(totals, row)
        covered.append([pb.t1, pb.t2])
        classes = _divergent(row)
        if classes:
            name = known_divergence(stream.name, pb.key, classes)
            rec = {"label": pb.label, "key": pb.key, "t1": pb.t1, "t2": pb.t2, "classes": classes,
                   "wall_s": round(time.monotonic() - t0, 2)}
            path = out / "divergences" / f"{PL.unit_id(stream.name, i)}__{pb.label}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "recorded": dataclasses.asdict(live.recorded), "classes": classes,
                "examples": {s: _jsonable(c.examples) for s, c in slices.items() if c.examples},
                "known": name}, indent=1, default=repr))
            rec["repro"] = str(path)
            (known if name else divergent).append({**rec, "known": name})
    return {"battles": len(covered), "errors": errors, "divergent": divergent, "known": known,
            "totals": totals, "covered": covered, "n_teams": len(teams)}


def run_corpora(stream: PL.Stream, i: int, out: Path) -> Dict[str, Any]:
    from agents.battle import rust_core_parity as P

    census = P.check_battles(P.protocol_battles(2) + P.byte_fuzz_battles(), P.Census())
    row = _census_row({"E": census})
    return {"battles": census.battles, "errors": [], "totals": row, "known": [],
            "divergent": ([{"label": "corpora", "classes": _divergent(row)}] if _divergent(row) else [])}


# ---------------------------------------------------------------------------------------------
# the A/B fuzzers
# ---------------------------------------------------------------------------------------------

def _summary_json(text: str) -> Optional[dict]:
    """The JSON object each fuzzer prints after its ``… SUMMARY`` line (the last one wins)."""
    idx = text.rfind("SUMMARY\n")
    if idx < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[idx + len("SUMMARY\n"):].lstrip())
        return obj
    except ValueError:
        return None


def fuzz_env(pin_target: Path) -> Dict[str, str]:
    """Every fuzzer pointed at the PIN's self-check build: no fuzzer builds into /tmp or a
    worktree (the three bridge fuzzers run `cargo build` into their target dir; here that is the
    pin's own, already built, so the build is a no-op fingerprint check)."""
    env = dict(os.environ)
    env.update(POKESIM_AB_REPLAY_BIN=str(pin_target / "selfcheck" / "ab_replay"),
               POKESIM_BRIDGE_TARGET=str(pin_target), POKESIM_SIMBRIDGE_TARGET=str(pin_target),
               POKESIM_EMISSION_SELFCHECK="1")
    return env


def run_fuzz(stream: PL.Stream, i: int, out: Path, timeout_s: float = 4 * 3600) -> Dict[str, Any]:
    from utils.paths import repo_path

    rng = stream.unit_range(i)
    udir = out / "fuzz" / PL.unit_id(stream.name, i)
    udir.mkdir(parents=True, exist_ok=True)
    script = repo_path("src", "rust_sim", "harness", stream.params["script"])
    seed = stream.params["seed_base"] + i
    argv = ["node", str(script), *stream.params["argv"], "--battles", str(len(rng)),
            "--master-seed", str(seed), "--out", str(udir)]
    log = udir / "log.txt"
    with open(log, "w") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           env=fuzz_env(repo_path("src", "rust_sim", "target")), timeout=timeout_s,
                           check=False)
    text = log.read_text(errors="replace")
    summary = _summary_json(text)
    repros = sorted(str(d) for d in (udir / "divergences").glob("*")) if (udir / "divergences").exists() else []
    allowed = len(list((udir / "allowlisted").glob("*"))) if (udir / "allowlisted").exists() else 0
    hard = None
    if summary is not None:
        hard = sum(int(summary.get(k) or 0) for k in ("diverged", "panic", "parse_error", "errored"))
    errors = []
    if summary is None:
        errors.append({"error": f"no SUMMARY in the fuzzer output (exit {p.returncode})",
                       "tail": text[-3000:]})
    elif stream.params["argv"][:1] == ["--protocol"] or stream.params["script"] != "ab_fuzz.js":
        # a green-gated fuzzer: its exit code IS its verdict and must agree with the counts
        gated = sum(int(summary.get(k) or 0) for k in ("diverged", "panic", "parse_error"))
        if (p.returncode != 0) != bool(gated):
            errors.append({"error": f"exit {p.returncode} disagrees with the summary's gated count {gated}"})
    battles = int(summary.get("battles") or 0) if summary else 0
    return {"battles": battles, "argv": argv[1:], "master_seed": seed, "exit": p.returncode,
            "hard_failures": hard, "allowlisted": (summary or {}).get("allowlisted", allowed),
            "summary": summary, "repros": repros, "errors": errors,
            "divergent": ([{"label": f"seed {seed}", "hard": hard, "repros": repros}] if hard else []),
            "known": []}


# ---------------------------------------------------------------------------------------------
# the SOAK
# ---------------------------------------------------------------------------------------------

def rss_kib(pid: int) -> Optional[int]:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def run_soak(stream: PL.Stream, i: int, out: Path, sample_every: int = 250) -> Dict[str, Any]:
    """ONE persistent bridge child driven by a real ``Gen3Env`` (the training transport, the
    self-check build) for ``per_unit`` episodes: random legal trainee actions (seeded), a
    ``RandomPlayer`` opponent, pool teams. The child's and this process's RSS are sampled every
    ``sample_every`` episodes; a child that is replaced (a crash) is counted."""
    import numpy as np

    from main.rust_core_cutover.envs import build_soak_env

    rng = np.random.default_rng(900_000 + i)
    env, session = build_soak_env(f"Soak{stream.name[-3:]}{i}", obs_source=stream.params["obs_source"])
    samples, pids, episodes, steps, errors = [], [], 0, 0, []
    t0 = time.monotonic()
    try:
        for ep in range(len(stream.unit_range(i))):
            obs, _ = env.reset()
            for _ in range(2000):
                legal = np.flatnonzero(np.asarray(obs["action_mask"]).astype(bool))
                obs, _r, term, trunc, _info = env.step(int(rng.choice(legal)) if legal.size else 0)
                steps += 1
                if term or trunc:
                    break
            else:
                raise AssertionError(f"episode {ep} did not end in 2000 steps")
            episodes += 1
            pid = session._proc.pid if session._proc is not None else None
            if pid is not None and (not pids or pids[-1] != pid):
                pids.append(pid)
            if ep % sample_every == 0 or ep == len(stream.unit_range(i)) - 1:
                samples.append([ep, round(time.monotonic() - t0, 1), rss_kib(pid) if pid else None,
                                rss_kib(os.getpid())])
    except Exception as e:
        errors.append({"error": f"{type(e).__name__}: {e}"[:2000], "at_episode": episodes,
                       "traceback": traceback.format_exc()[-4000:]})
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {"battles": episodes, "steps": steps, "rss_samples": samples,
            "child_pids": pids, "child_replacements": max(0, len(pids) - 1),
            "recycles": getattr(session, "_n_recycles", None),
            "errors": errors, "divergent": [], "known": []}


# ---------------------------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------------------------

def run_unit(stream: PL.Stream, i: int, out: Path) -> Dict[str, Any]:
    fn = {"parity": run_parity, "corpora": run_corpora, "fuzz": run_fuzz, "soak": run_soak}
    if stream.kind == "envn":
        from main.rust_core_cutover.envs import run_envn

        fn["envn"] = run_envn
    started = time.time()
    load0 = os.getloadavg()[0]
    try:
        body = fn[stream.kind](stream, i, out)
        status = "ok"
    except Exception as e:
        body = {"battles": 0, "divergent": [], "known": [],
                "errors": [{"error": f"{type(e).__name__}: {e}"[:2000],
                            "traceback": traceback.format_exc()[-6000:]}]}
        status = "error"
    if body.get("errors"):
        status = "error"
    return {"schema": PL.SCHEMA, "unit": PL.unit_id(stream.name, i), "stream": stream.name,
            "kind": stream.kind, "i": i, "status": status, "started": started,
            "finished": time.time(), "wall_s": round(time.time() - started, 1),
            "load1": [round(load0, 1), round(os.getloadavg()[0], 1)], "pid": os.getpid(), **body}
