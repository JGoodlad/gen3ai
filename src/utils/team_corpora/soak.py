"""A CRASH SOAK of a team corpus on the Rust ENV core — the core training plays, LABEL PATH INCLUDED (§4.1 step 5 of
``designs/endstate/design_diversity_teams_opponents.md``).

    python -m utils.team_corpora.soak w_forum_v1 [--tier ABC] [--profile selfcheck] [--out FILE] [--only ID,ID]
    python -m utils.team_corpora.soak n1_proc_v1 --rounds 5 --first-round 5 --seed 3803 --out CHUNK2   # one chunk
    python -m utils.team_corpora.soak merge CHUNK1 CHUNK2 --out FILE                                   # the record

A large corpus soaks in CHUNKS of rounds (``--first-round``), each its own process inside rule 5b's <= 45-min chunk;
``merge`` refuses chunks of different corpora / settings or with overlapping or missing rounds, and its record reads
like one soak over every round (``chunks`` lists the parts).

Every team of the tier plays TWO battles per round (``--rounds``) on the env core training uses: once in seat p1 against the next team (p2,
piloted by the Rust ``heuristic`` bot), once in seat p2 (piloted by the bot) against the previous team. p1 draws
uniformly over its legal mask from a seeded generator, so the battles reach moves a strong policy would not choose.
The teams are packed by the TRAINING teambuilder (``Gen3Teambuilder``: validated, the gen-3 Hidden Power IV fix), and
the ``selfcheck`` profile is the emission self-check build. The battle seeds are a function of ``--seed`` alone.

🚨 **The core declares EVERY core label family** (``utils.rust_env.columns.LABEL_FAMILIES``: a superset of what any
training run declares), so every p1 decision runs the label writers with the p2 team as the TRUTH side, and each team
sits in p2 once per round. Schema ``gen3_team_corpus_soak_v1`` declared NO label family and PASSED a corpus holding a
level-5 team (``sfv2_5051``) that crash-looped D1 s3501 with ``labels p2: krabby: the declared spread ... at L100``
(2026-10-22, launch sheet AM-D1-FIX): a soak that skips the label path cannot pass for the core training runs.

What it proves: 0 core errors (an encoder raise, a belief-tracker contradiction, a LABEL-writer fault, a rejected
choice the core cannot recover from: each kills a training run, not one battle) over every team, in both seats. What it
does NOT prove: sim vs port agreement (the Node differential gates), or a crash a stronger policy reaches. A core error
stops the soak and names the in-flight battles' teams.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: v2: the core declares every core label family (v1 declared none and could not see a label fault).
SOAK_SCHEMA = "gen3_team_corpus_soak_v2"
BOT = {"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}


def all_label_families() -> List[str]:
    """Every family the core writes (``labels::FAMILIES``): the soak declares them all."""
    from utils.rust_env import columns as C

    return list(C.LABEL_FAMILIES)


def soak(name: str, tier: str = "", *, profile: str = "selfcheck", seed: int = 3802, rounds: int = 1,
         n_envs: int = 8, threads: int = 3, turn_limit: int = 250, only: Sequence[str] = (),
         first_round: int = 0) -> Dict[str, Any]:
    """Soak the verified ``tier`` of the registered corpus ``name`` (``only``: just those team ids, a diagnosis)."""
    from utils.team_corpora import load_corpus

    corpus = load_corpus(name, tier)
    ids, texts = list(corpus.ids), list(corpus.texts)
    if only:
        missing = sorted(set(only) - set(ids))
        if missing:
            raise ValueError(f"--only names ids not in {name}@{corpus.tier}: {missing}")
        keep = [k for k, i in enumerate(ids) if i in set(only)]
        ids, texts = [ids[k] for k in keep], [texts[k] for k in keep]
    res = soak_texts(ids, texts, corpus.ident(), profile=profile, seed=seed, rounds=rounds, n_envs=n_envs,
                     threads=threads, turn_limit=turn_limit, first_round=first_round)
    if only:
        res["only"] = list(only)
        res["verdict"] = "PARTIAL_" + res["verdict"]   # a subset never stands for the corpus (corpus-check reads PASS)
    return res


def soak_texts(ids: Sequence[str], texts: Sequence[str], ident: Dict[str, Any], *, profile: str = "selfcheck",
               seed: int = 3802, rounds: int = 1, n_envs: int = 8, threads: int = 3,
               turn_limit: int = 250, first_round: int = 0) -> Dict[str, Any]:
    """The soak over explicit team texts (``ident`` is recorded as the corpus block)."""
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    from utils.rust_env.build import ensure_built
    from utils.teambuilder import Gen3Teambuilder

    labels = all_label_families()
    packed = Gen3Teambuilder(list(texts)).packed_teams
    if len(packed) != len(texts):
        raise RuntimeError(f"{len(texts) - len(packed)} corpus teams failed the training teambuilder's validation")
    n_teams = len(packed)
    # round r pairs team i (p1) with team i + 1 + r (p2): every team plays 2 x rounds games, each against new teams
    rounds_run = list(range(int(first_round), int(first_round) + int(rounds)))
    pairs = [(i, (i + 1 + r) % n_teams) for r in rounds_run for i in range(n_teams)]
    n = len(pairs)
    ensure_built(profile)
    spec = P.spec_json(n=n_envs, threads=threads, teams=list(packed), names=("soakone", "soaktwo"),
                       turn_limit=turn_limit, refusal_budget=4, bank_dir=None, labels=tuple(labels),
                       opponents=[dict(BOT)])
    core = ffi.FfiCore(spec, lib=ffi.load(ffi.default_path(profile)))
    c = core.cols
    rng = np.random.default_rng(seed)
    nxt = 0
    in_flight: Dict[int, Optional[int]] = {e: None for e in range(n_envs)}
    staged: Dict[int, Optional[int]] = {e: None for e in range(n_envs)}
    finished = 0
    decisions = 0
    stall_forfeits = 0
    end_turns: List[int] = []
    errors: List[Dict[str, Any]] = []
    t0 = time.time()

    def stage(envs) -> None:
        nonlocal nxt
        for e in envs:
            k = nxt if nxt < n else n + int(rng.integers(0, n))   # past the end: filler games, not counted
            i, j = pairs[k % n]
            c["ep_team"][e] = (i, j)
            c["ep_seed"][e] = rng.integers(0, 65536, 4)
            c["ep_opp"][e] = 0
            staged[e] = k
            nxt += 1

    def fault(exc: BaseException) -> None:  # a core error: name the in-flight battles (staged ones at a reset)
        live = {e: k for e, k in in_flight.items() if k is not None} or {e: k for e, k in staged.items() if k is not None}
        errors.append({"error": f"{type(exc).__name__}: {exc}"[:2000],
                       "in_flight": [{"env": e, "pair": pairs[k % n], "ids": [ids[pairs[k % n][0]], ids[pairs[k % n][1]]]}
                                     for e, k in live.items()]})

    stage(range(n_envs))
    try:
        try:
            core.reset()   # the first decision of every env runs here, label writers included
        except Exception as exc:
            fault(exc)
        else:
            for e in range(n_envs):
                in_flight[e] = staged[e]
            stage(range(n_envs))
        while not errors and finished < n:
            pend = [e for e in range(n_envs) if c["need"][e, 0] and int(c["turn"][e]) < turn_limit]
            for e in pend:
                m = np.flatnonzero(c["mask"][e, 0].astype(bool))
                c["action"][e, 0] = int(rng.choice(m))
            decisions += len(pend)
            try:
                core.step()
            except Exception as exc:
                fault(exc)
                break
            for f in core.finished():
                e = int(f["env"])
                k = in_flight[e]
                if k is not None and k < n:
                    finished += 1
                    if int(f.get("forfeit", 0) or 0):
                        stall_forfeits += 1
                    end_turns.append(int(f["end_turn"]))
                in_flight[e] = staged[e]
                stage([e])
    finally:
        close = getattr(core, "close", None)
        if close is not None:
            close()
    return {"schema": SOAK_SCHEMA, "corpus": dict(ident), "labels": labels, "profile": profile, "seed": seed, "n_envs": n_envs,
            "threads": threads, "turn_limit": turn_limit, "opponent": BOT, "rounds": int(rounds),
            **({"first_round": int(first_round)} if first_round else {}),
            "battles_planned": n, "battles_finished": finished, "decisions_p1": decisions,
            "seats": "per round, every team once as p1 (uniform over its mask) and once as p2 (the heuristic bot); "
                     "round r pairs team i with team i + 1 + r",
            "forfeits": stall_forfeits,
            "end_turn": ({"median": int(np.median(end_turns)), "max": int(max(end_turns))} if end_turns else None), "core_errors": errors, "wall_s": round(time.time() - t0, 1),
            "verdict": "PASS" if not errors and finished == n else "FAIL"}


#: Fields every chunk of one soak must share (``merge``).
_CHUNK_SAME = ("schema", "corpus", "labels", "profile", "n_envs", "threads", "turn_limit", "opponent")


def merge(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One record from soak CHUNKS over consecutive rounds 0..R-1 (each its own ``--first-round`` / ``--seed``).
    Raises ``ValueError`` on chunks of different corpora or settings, a repeated seed, or rounds that overlap or leave a
    gap. ``end_turn.median`` is the battle-weighted mean of the chunk medians (descriptive)."""
    if not chunks:
        raise ValueError("no chunks")
    for k in _CHUNK_SAME:
        if any(c.get(k) != chunks[0].get(k) for c in chunks):
            raise ValueError(f"soak chunks differ in {k!r}")
    if len({c["seed"] for c in chunks}) != len(chunks):
        raise ValueError("two soak chunks share a seed (the same dice twice)")
    spans = sorted((int(c.get("first_round", 0)), int(c["rounds"])) for c in chunks)
    at = 0
    for r0, k in spans:
        if r0 != at:
            raise ValueError(f"soak chunks cover rounds with a gap or an overlap at round {at} (next starts {r0})")
        at = r0 + k
    out = {k: chunks[0][k] for k in _CHUNK_SAME}
    fin = sum(int(c["battles_finished"]) for c in chunks)
    ends = [c["end_turn"] for c in chunks if c.get("end_turn")]
    out.update({
        "rounds": at, "seed": [c["seed"] for c in sorted(chunks, key=lambda c: int(c.get("first_round", 0)))],
        "battles_planned": sum(int(c["battles_planned"]) for c in chunks), "battles_finished": fin,
        "decisions_p1": sum(int(c["decisions_p1"]) for c in chunks), "seats": chunks[0]["seats"],
        "forfeits": sum(int(c["forfeits"]) for c in chunks),
        "end_turn": ({"median": round(sum(e["median"] * int(c["battles_finished"]) for c, e in
                                          zip([c for c in chunks if c.get("end_turn")], ends)) / max(fin, 1)),
                      "max": max(e["max"] for e in ends)} if ends else None),
        "core_errors": [e for c in chunks for e in c["core_errors"]],
        "wall_s": round(sum(float(c["wall_s"]) for c in chunks), 1),
        "chunks": [{"first_round": int(c.get("first_round", 0)), "rounds": int(c["rounds"]), "seed": c["seed"],
                    "battles_finished": c["battles_finished"], "wall_s": c["wall_s"], "verdict": c["verdict"]}
                   for c in sorted(chunks, key=lambda c: int(c.get("first_round", 0)))],
    })
    out["verdict"] = "PASS" if all(c["verdict"] == "PASS" for c in chunks) and not out["core_errors"] \
        and fin == out["battles_planned"] else "FAIL"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["merge"]:
        mp = argparse.ArgumentParser(description="merge soak chunks into one record")
        mp.add_argument("chunks", nargs="+")
        mp.add_argument("--out", required=True)
        m = mp.parse_args(argv[1:])
        res = merge([json.load(open(p)) for p in m.chunks])
        with open(m.out, "w") as f:
            f.write(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(res["verdict"], res["battles_finished"], "battles", len(res["core_errors"]), "core errors")
        return 0 if res["verdict"] == "PASS" else 1
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--tier", default="")
    ap.add_argument("--profile", default="selfcheck", choices=("selfcheck", "release"))
    ap.add_argument("--seed", type=int, default=3802)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--first-round", type=int, default=0, help="the first round of this CHUNK (round r pairs i, i+1+r)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="", help="comma-separated team ids: a diagnosis on a subset (verdict PARTIAL_*)")
    a = ap.parse_args(argv)
    only = tuple(x for x in a.only.split(",") if x)
    res = soak(a.name, a.tier, profile=a.profile, seed=a.seed, rounds=a.rounds, only=only, first_round=a.first_round)
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text if len(text) < 4000 else text[:4000])
    return 0 if res["verdict"] in ("PASS", "PARTIAL_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
