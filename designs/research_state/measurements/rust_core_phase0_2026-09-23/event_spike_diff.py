"""SPIKE harness (`rust_core_phase0_2026-09-23`): source-emitted events vs `Gen3Battle`'s reading.

    export PYTHONPATH=$PYTHONPATH:src
    cargo build --release --bin sim_bridge --manifest-path src/rust_sim/Cargo.toml
    cargo build --release --features event_spike --bin event_spike \
        --manifest-path src/rust_sim/Cargo.toml --target-dir src/rust_sim/target/spike
    POKESIM_SIM_BRIDGE_BIN=$PWD/src/rust_sim/target/release/sim_bridge \
    python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/event_spike_diff.py \
        --battles 40 --key0 0 --out <dir>

Per battle (fixed teams by pool index, per-player RNG, fixed sim seed, concurrency 1 — the
project's reproducible-fuzz recipe):

1. PLAY it for real over the production rust `sim_bridge` with two seeded random players whose
   battle class is `Gen3Battle` — so each side's event log is exactly what a live `Gen3Env`
   player folds. The per-side chunks and the `__RECON__` record are captured.
2. REPLAY the recorded commands through the `event_spike` binary (the SAME `BridgeSession`,
   built with `--features event_spike`), which returns the per-side chunks again plus one
   SOURCE record per omniscient line, typed at the emit site, with the engine's action SCOPE.
   The replayed chunks must be BYTE-IDENTICAL to the live ones, or the battle is refused.
3. For each viewer (p1, p2) build the source events in `BattleEvent` shape — side relative to
   the viewer, HP in the viewer's rendering — and diff them against that viewer's live log,
   field by field, per event kind:
     * `raw`   — source facts vs poke-env's reading. Every row is a place the READING is not the
                 sim's truth.
     * `ruled` — the same source facts with the named poke-env READING RULES applied
                 (`READING_RULES` below). Whatever survives is unexplained.
4. PARSE-BACK: re-derive every typed source record from its own omniscient LINE TEXT plus the
   scope inferred from line order, and count what the text alone cannot reproduce.

Nothing here is imported by production code. Divergences are FINDINGS: nothing is allowlisted.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import difflib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from poke_env import AccountConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.battle_event import OPP, OURS, EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.training.obs_roundtrip_fuzz_test import SeededRandomPlayer
from utils.bridge import reconstruction
from utils.bridge.local_battle_runner import run_local_battles
from utils.paths import repo_path
from utils.team_loader import TeamLoader

K = EventKind
SUBSET = (K.MOVE, K.SWITCH, K.DRAG, K.FAINT, K.DAMAGE, K.HEAL, K.STATUS, K.CURESTATUS, K.CANT,
          K.CRIT, K.MISS, K.FAIL, K.IMMUNE, K.RESISTED, K.SUPEREFFECTIVE)
KW_KIND = {
    "move": K.MOVE, "switch": K.SWITCH, "drag": K.DRAG, "faint": K.FAINT, "-damage": K.DAMAGE,
    "-heal": K.HEAL, "-status": K.STATUS, "-curestatus": K.CURESTATUS, "-cureteam": K.CURESTATUS,
    "cant": K.CANT, "-crit": K.CRIT, "-miss": K.MISS, "-fail": K.FAIL, "-notarget": K.FAIL,
    "-nothing": K.FAIL, "-immune": K.IMMUNE, "-resisted": K.RESISTED,
    "-supereffective": K.SUPEREFFECTIVE,
}
OUTCOME = (K.CRIT, K.MISS, K.FAIL, K.IMMUNE, K.RESISTED, K.SUPEREFFECTIVE)
MULT = {K.IMMUNE: 0.0, K.RESISTED: 0.5, K.SUPEREFFECTIVE: 2.0}

#: The poke-env READING RULES the `ruled` column applies to source facts. Each is pinned to the
#: line it mirrors. Filled in from what the `raw` column found — see the README's table.
READING_RULES = {
    "R1_outcome_owner_is_last_move_line": (
        "an outcome line (-crit/-miss/-fail/-immune/-resisted/-supereffective) belongs to the "
        "side of the most recent |move| line, reset only at |turn| — NOT to the action the sim "
        "is running", "poke_env/battle/abstract_battle.py:711 (set) / :1634 (end_turn reset); "
        "agents/battle/gen3_battle.py:716,733"),
    "R2_immune_owner_fallback": (
        "an -immune/-resisted/-supereffective with no open move is owned by the side OPPOSITE the "
        "named defender", "agents/battle/gen3_battle.py:735-738"),
    "R3_miss_target_is_user": (
        "a MISS/FAIL event's target is the mon NAMED at index 2 — for -miss that is the USER",
        "agents/battle/gen3_battle.py:717,727"),
    "R4_still_move_target_is_foe_active": (
        "a [still] move (empty target field) is given the other side's ACTIVE as its target",
        "agents/battle/gen3_battle.py:532-539"),
    "R5_miss_suffix_synthetic": (
        "|move|…|[miss] emits a SECOND MISS event (from='move-suffix') beside the standalone "
        "|-miss| line — one miss is two MISS events in the log",
        "agents/battle/gen3_battle.py:272-274,488-502"),
    "R7_effectiveness_drops_cause": (
        "an -immune/-resisted/-supereffective event carries only its multiplier: the line's "
        "[from] ability: <X> (Levitate / Immunity / Volt Absorb …) is NOT on the event",
        "agents/battle/gen3_battle.py:744-749 (value={'multiplier': …} only)"),
    "R8_cureteam_status_is_the_from_clause": (
        "a -cureteam event's `status` is sm[3] verbatim — for gen3 Aromatherapy that is the "
        "string '[from] move: Aromatherapy', not a status", "agents/battle/gen3_battle.py:687-691"),
    "R6_hp_presentation": (
        "HP is the viewer's rendering: own side x/maxhp exact, opponent ceil(100x/maxhp)/100 "
        "(99 if <max), 0 when fainted", "src/rust_sim/src/bridge.rs::hp_percent + fold_hp_line; "
        "poke_env Pokemon.current_hp_fraction"),
}


# ---------------------------------------------------------------------------
# 1. play one battle for real
# ---------------------------------------------------------------------------


def play(key: int, tag: str):
    pool = TeamLoader().get_all_teams()
    t1, t2 = pool[key % len(pool)], pool[(key + 1) % len(pool)]
    common = dict(battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                  start_listening=False, max_concurrent_battles=1, battle_class=Gen3Battle)
    p1 = SeededRandomPlayer(rng_seed=1000 + key, team=t1,
                            account_configuration=AccountConfiguration(f"{tag}a{key}", "pw"),
                            **common)
    p2 = SeededRandomPlayer(rng_seed=2000 + key, team=t2,
                            account_configuration=AccountConfiguration(f"{tag}b{key}", "pw"),
                            **common)
    sink: list = []
    asyncio.run(run_local_battles(p1, p2, 1, seed=[11 + key, 22 + key, 33 + key, 44 + key],
                                  impl="rust", chunk_sink=sink))
    (b1,) = list(p1.battles.values())
    (b2,) = list(p2.battles.values())
    rec = reconstruction.pop_record(b1.battle_tag)
    if rec is None:
        raise RuntimeError(f"no __RECON__ for {b1.battle_tag}")
    return b1, b2, rec, sink


# ---------------------------------------------------------------------------
# 2. replay through the spike binary
# ---------------------------------------------------------------------------


def replay(rec, spike_bin: str) -> dict:
    players = rec.players()
    start = {"formatid": rec.format_id, "seed": rec.prng_seed,
             "p1": players["p1"], "p2": players["p2"]}
    lines = ["START " + json.dumps(start)]
    for side, choice in rec.commands:
        lines.append(f"FORCELOSE {choice}" if side == "forcelose" else f"CHOOSE {side} {choice}")
    lines.append("END")
    p = subprocess.run([spike_bin], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"event_spike failed: {p.stderr.strip()}")
    return json.loads(p.stdout)


def chunks_identical(spike: dict, live: list) -> Tuple[bool, str]:
    rep = [("p1" if c["side"] == 0 else "p2", "\n".join(c["lines"])) for c in spike["chunks"]]
    if rep == list(live):
        return True, ""
    for i, (a, b) in enumerate(zip(rep, live)):
        if a != b:
            return False, f"chunk {i}: replay {a[0]} {a[1][:120]!r} vs live {b[0]} {b[1][:120]!r}"
    return False, f"chunk count replay {len(rep)} vs live {len(live)}"


# ---------------------------------------------------------------------------
# 3. source events in BattleEvent shape, per viewer
# ---------------------------------------------------------------------------


def _parse_ident(ident: str) -> Tuple[Optional[int], Optional[str]]:
    ident = ident.strip()
    if len(ident) >= 2 and ident[0] == "p" and ident[1] in "12":
        side = int(ident[1]) - 1
        name = ident.split(": ", 1)[1] if ": " in ident else None
        return side, name
    return None, None


def _pct(hp: int, mx: int) -> int:
    if mx == 0:
        return 0
    p = -(-100 * hp // mx)
    if p == 100 and hp < mx:
        p = 99
    return p


def _frac(hp: Tuple[int, int], side: int, viewer: int) -> float:
    cur, mx = hp
    if cur == 0:
        return 0.0
    if side == viewer:
        return cur / mx
    return _pct(cur, mx) / 100.0


def source_events(spike: dict, viewer: int, apply_rules: bool) -> Tuple[list, dict]:
    """Walk the source records in order and emit events in BattleEvent shape for ``viewer``."""
    species: Dict[Tuple[int, str], str] = {}
    sheet = {0: {n: to_id_str(s) for n, s in spike["names"]["p1"].items()},
             1: {n: to_id_str(s) for n, s in spike["names"]["p2"].items()}}
    active: Dict[int, Optional[str]] = {0: None, 1: None}
    hpfrac: Dict[Tuple[int, str], float] = {}
    last_mover: Optional[int] = None          # R1's state: poke-env's _current_move_user_side
    stats = collections.Counter()
    out: List[dict] = []

    def rel(side: Optional[int]) -> Optional[str]:
        if side is None:
            return None
        return OURS if side == viewer else OPP

    def sp(side: Optional[int], name: Optional[str]) -> Optional[str]:
        if side is None or name is None:
            return None
        return species.get((side, name)) or sheet[side].get(name) or to_id_str(name)

    for r in spike["recs"]:
        line = r["line"]
        parts = line.split("|")
        kw = parts[1] if len(parts) > 1 else ""
        if kw == "turn":
            last_mover = None
            continue
        kind = KW_KIND.get(kw)
        t = r["typed"]
        if t is not None and t["k"] in ("switch", "drag"):
            m = t["mon"]
            key = (m["side"], m["name"])
            species[key] = to_id_str(t["details"].split(",")[0])
            active[m["side"]] = m["name"]
            hpfrac[key] = _frac(tuple(t["hp"]), m["side"], viewer)
        if t is not None and t["k"] == "move":
            last_mover = t["user"]["side"]
        if kind is None:
            continue
        ev: Dict[str, Any] = {"kind": kind, "turn": r["turn"], "line": line, "scope": r["scope"],
                              "typed": t is not None, "value": {}}
        if t is None:
            stats[f"untyped:{kw}"] += 1
            ev.update(side=None, actor=None, target=None)
            out.append(ev)
            continue
        k = t["k"]
        v = ev["value"]
        if k == "move":
            u = t["user"]
            tgt = t["target"]
            ev.update(side=rel(u["side"]), actor=sp(u["side"], u["name"]),
                      target=sp(tgt["side"], tgt["name"]) if tgt else None)
            if (tgt is None or r["still_suffix"]) and apply_rules:             # R4
                foe = 1 - u["side"]
                ev["target"] = sp(foe, active[foe])
            mid = to_id_str(t["move"])
            v["move_id"] = mid
            frm = to_id_str(t["from"]) if t["from"] else None
            if frm and frm != mid and frm != "lockedmove":
                v["from_move"] = frm
            out.append(ev)
            if (r["miss_suffix"] or t.get("miss")) and apply_rules:            # R5
                out.append({"kind": K.MISS, "turn": r["turn"], "line": line, "scope": r["scope"],
                            "typed": True, "synthetic": True, "side": rel(u["side"]),
                            "actor": sp(u["side"], u["name"]), "target": None,
                            "value": {"from": "move-suffix"}})
            continue
        if k in ("switch", "drag", "faint"):
            m = t["mon"]
            ev.update(side=rel(m["side"]), actor=sp(m["side"], m["name"]), target=None)
            if k == "faint":
                hpfrac[(m["side"], m["name"])] = 0.0
            out.append(ev)
            continue
        if k in ("damage", "heal"):
            m = t["mon"]
            key = (m["side"], m["name"])
            after = _frac(tuple(t["hp"]), m["side"], viewer)
            before = hpfrac.get(key, after)
            hpfrac[key] = after
            ev.update(side=rel(m["side"]), actor=sp(*key), target=None)
            v["amount"] = after - before
            v["hp_after"] = after
            if t["cause"]:
                v["reason"] = t["cause"]
            if t["of"]:
                ev["source_of"] = sp(t["of"]["side"], t["of"]["name"])
            out.append(ev)
            continue
        if k == "status":
            m = t["mon"]
            ev.update(side=rel(m["side"]), actor=sp(m["side"], m["name"]), target=None)
            v["status"] = t["status"]
            if t["cause"]:
                v["reason"] = t["cause"]
            out.append(ev)
            continue
        if k == "curestatus":
            if t["mon"]:
                s, n = t["mon"]["side"], t["mon"]["name"]
            else:
                s, n = _parse_ident(t["ident_raw"])
            ev.update(side=rel(s), actor=sp(s, n), target=None)
            v["status"] = t["status"] or None
            if apply_rules and not t["status"] and t["cause"]:                  # R8
                v["status"] = "[from] " + t["cause"]
            out.append(ev)
            continue
        if k == "cant":
            m = t["mon"]
            ev.update(side=rel(m["side"]), actor=sp(m["side"], m["name"]), target=None)
            v["reason"] = t["reason"]
            v["move"] = to_id_str(t["move"]) if t["move"] else None
            of = t["of"]
            v["of_side"] = rel(of["side"]) if of else None
            v["of_actor"] = sp(of["side"], of["name"]) if of else None
            out.append(ev)
            continue
        # ---- outcome kinds: the SOURCE owner is the engine's action scope ----
        sc = r["scope"]
        mover = int(sc.split(":p")[1]) - 1 if sc.startswith("move:") else None
        if apply_rules:
            mover = last_mover                                                 # R1
        if k == "miss":
            named_side, named = (t["user"]["side"], t["user"]["name"]) if t["user"] else \
                _parse_ident(t["user_raw"])
            tgt = t["target"]
            target = sp(named_side, named) if apply_rules else (             # R3
                sp(tgt["side"], tgt["name"]) if tgt else None)
        else:
            m = t["mon"]
            target = sp(m["side"], m["name"])
            if mover is None and apply_rules and kind in MULT:                 # R2
                mover = 1 - m["side"]
        ev.update(side=rel(mover), actor=sp(mover, active[mover]) if mover is not None else None,
                  target=target)
        if kind in MULT:
            v["multiplier"] = MULT[kind]
        if t.get("cause") and not (apply_rules and kind in MULT):             # R7
            v["from"] = t["cause"]
        out.append(ev)
    return out, stats


def python_events(battle: Gen3Battle) -> Tuple[list, int]:
    out, synth = [], 0
    for e in battle.events:
        if e.kind not in SUBSET:
            continue
        if e.value.get("from") == "move-suffix":
            synth += 1
        out.append({"kind": e.kind, "turn": e.turn, "side": e.side, "actor": e.actor_species,
                    "target": e.target_species, "value": dict(e.value), "line": "|".join(e.raw),
                    "synthetic": e.value.get("from") == "move-suffix"})
    return out, synth


# ---------------------------------------------------------------------------
# the field diff
# ---------------------------------------------------------------------------

COMPARED = {
    K.MOVE: ("side", "actor", "target", "turn", "move_id", "from_move"),
    K.SWITCH: ("side", "actor", "turn"), K.DRAG: ("side", "actor", "turn"),
    K.FAINT: ("side", "actor", "turn"),
    K.DAMAGE: ("side", "actor", "turn", "amount", "hp_after", "reason"),
    K.HEAL: ("side", "actor", "turn", "amount", "hp_after", "reason"),
    K.STATUS: ("side", "actor", "turn", "status", "reason"),
    K.CURESTATUS: ("side", "actor", "turn", "status"),
    K.CANT: ("side", "actor", "turn", "reason", "move", "of_side", "of_actor"),
}
for _k in OUTCOME:
    COMPARED[_k] = ("side", "actor", "target", "turn", "from", "multiplier")


def _get(ev: dict, f: str):
    return ev[f] if f in ("side", "actor", "target", "turn") else ev["value"].get(f)


def _eq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def diff(src: list, py: list, tally: collections.Counter, examples: dict, battle: str) -> None:
    sk = [e["kind"] for e in src]
    pk = [e["kind"] for e in py]
    sm = difflib.SequenceMatcher(None, sk, pk, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for a, b in zip(src[i1:i2], py[j1:j2]):
                tally[(a["kind"].name, "__matched__")] += 1
                if not a.get("typed", True):
                    tally[(a["kind"].name, "__untyped_source__")] += 1
                    continue
                for f in COMPARED[a["kind"]]:
                    x, y = _get(a, f), _get(b, f)
                    if not _eq(x, y):
                        key = (a["kind"].name, f)
                        tally[key] += 1
                        ex = examples.setdefault(key, [])
                        if len(ex) < 4:
                            ex.append({"battle": battle, "source": x, "python": y,
                                       "scope": a.get("scope"), "src_line": a["line"],
                                       "py_line": b["line"]})
        else:
            for a in src[i1:i2]:
                key = (a["kind"].name, "__source_only__")
                tally[key] += 1
                ex = examples.setdefault(key, [])
                if len(ex) < 4:
                    ex.append({"battle": battle, "src_line": a["line"], "scope": a.get("scope")})
            for b in py[j1:j2]:
                key = (b["kind"].name, "__python_only__" + ("(synthetic)" if b.get("synthetic") else ""))
                tally[key] += 1
                ex = examples.setdefault(key, [])
                if len(ex) < 4:
                    ex.append({"battle": battle, "py_line": b["line"]})


# ---------------------------------------------------------------------------
# 4. parse-back: typed facts + scope from the omniscient LINE TEXT alone
# ---------------------------------------------------------------------------


def _from_of(parts: List[str]) -> Tuple[Optional[str], Optional[str]]:
    frm = of = None
    for tok in parts[3:]:
        tok = tok.strip()
        if tok.startswith("[from]"):
            frm = tok[len("[from]"):].strip()
        elif tok.startswith("[of]"):
            of = tok[len("[of]"):].strip()
    return frm, of


def parse_back(spike: dict, tally: collections.Counter, examples: dict, battle: str) -> None:
    """Re-derive each typed record's facts from its line, and the OWNER of each outcome line
    (the side whose MOVE the engine was running, or None) from line order alone.

    The text rule, stated: `|move|X` opens X's move; `|switch|` (any, incl. a Baton Pass entry),
    `|turn|`, `|upkeep` and the bare `|` batch/phase separator close it; `|drag|` does not (a
    Roar's drag is inside the Roar). poke-env's rule (R1) is the coarser one — it closes only at
    `|turn|` — and the `raw` column counts where the two disagree."""
    owner: Optional[int] = None
    for r in spike["recs"]:
        parts = r["line"].split("|")
        kw = parts[1] if len(parts) > 1 else ""
        # --- the owner state machine (the parser's only memory) ---
        if kw == "move":
            owner, _ = _parse_ident(parts[2])
        elif kw in ("switch", "upkeep", "turn") or (kw == "" and len(parts) == 2):
            owner = None
        scope = f"move:p{owner + 1}" if owner is not None else "none"
        t = r["typed"]
        if t is None:
            continue
        # --- typed facts from text ---
        k = t["k"]
        frm, of = _from_of(parts)
        facts_ok = True
        if k in ("damage", "heal"):
            want = t["cause"]
            if (frm or None) != want:
                facts_ok = False
            if (of is not None) != (t["of"] is not None):
                facts_ok = False
        elif k == "move":
            if parts[3] != t["move"]:
                facts_ok = False
        tally[("parse_back", k, "facts_ok" if facts_ok else "facts_differ")] += 1
        if k in ("crit", "miss", "fail", "immune", "resisted", "supereffective"):
            inferred = scope
            src = r["scope"] if r["scope"].startswith("move:") else "none"
            ok = inferred == src
            tally[("parse_back", "outcome_owner", "ok" if ok else f"{src}->{inferred}")] += 1
            if not ok:
                ex = examples.setdefault(("parse_back_scope", src, inferred), [])
                if len(ex) < 3:
                    ex.append({"battle": battle, "line": r["line"]})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=20)
    ap.add_argument("--key0", type=int, default=0)
    ap.add_argument("--tag", default="Es")
    ap.add_argument("--spike-bin", default=str(repo_path(
        "src", "rust_sim", "target", "spike", "release", "event_spike")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.environ.get("POKESIM_SIM_BRIDGE_BIN"):
        print("set POKESIM_SIM_BRIDGE_BIN to THIS worktree's sim_bridge", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    raw_t, ruled_t, pb_t = collections.Counter(), collections.Counter(), collections.Counter()
    raw_ex: dict = {}
    ruled_ex: dict = {}
    pb_ex: dict = {}
    untyped = collections.Counter()
    n_ok = n_refused = 0
    synth_py = synth_src = 0
    t0 = time.time()
    for key in range(a.key0, a.key0 + a.battles):
        name = f"key{key}"
        try:
            b1, b2, rec, live = play(key, a.tag)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: PLAY FAILED {type(e).__name__}: {e}")
            n_refused += 1
            continue
        spike = replay(rec, a.spike_bin)
        ok, why = chunks_identical(spike, live)
        if not ok:
            print(f"  {name}: REFUSED — replay is not the live battle: {why}")
            n_refused += 1
            continue
        n_ok += 1
        for viewer, battle in ((0, b1), (1, b2)):
            py, s = python_events(battle)
            synth_py += s
            src_raw, st = source_events(spike, viewer, apply_rules=False)
            src_ruled, _ = source_events(spike, viewer, apply_rules=True)
            synth_src += sum(1 for e in src_ruled if e.get("synthetic"))
            untyped.update(st)
            py_nosynth = [e for e in py if not e.get("synthetic")]
            diff(src_raw, py_nosynth, raw_t, raw_ex, f"{name}/p{viewer + 1}")
            diff(src_ruled, py, ruled_t, ruled_ex, f"{name}/p{viewer + 1}")
        parse_back(spike, pb_t, pb_ex, name)
        with open(os.path.join(a.out, f"{name}_spike.json"), "w") as fh:
            json.dump(spike, fh)
        print(f"  {name}: ok  turns={b1.turn}  recs={len(spike['recs'])}  "
              f"py_events={len(b1.events)}/{len(b2.events)}", flush=True)

    def table(t: collections.Counter) -> list:
        rows = collections.defaultdict(dict)
        for (kind, field), n in t.items():
            rows[kind][field] = n
        return [{"kind": k, **v} for k, v in sorted(rows.items())]

    report = {
        "battles_ok": n_ok, "battles_refused": n_refused, "wall_s": round(time.time() - t0, 1),
        "python_move_suffix_synthetics": synth_py, "ruled_source_synthetics": synth_src,
        "untyped_source_lines": dict(untyped),
        "raw": table(raw_t), "ruled": table(ruled_t),
        "parse_back": {" ".join(k[1:]): v for k, v in sorted(pb_t.items())},
        "raw_examples": {f"{k[0]}.{k[1]}": v for k, v in raw_ex.items()},
        "ruled_examples": {f"{k[0]}.{k[1]}": v for k, v in ruled_ex.items()},
        "parse_back_examples": {" ".join(map(str, k)): v for k, v in pb_ex.items()},
        "reading_rules": READING_RULES,
    }
    with open(os.path.join(a.out, "event_spike_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("battles_ok", "battles_refused", "wall_s",
                                               "python_move_suffix_synthetics",
                                               "ruled_source_synthetics",
                                               "untyped_source_lines")}, indent=1))
    for name, t in (("RAW (source vs reading)", raw_t), ("RULED (source + reading rules)", ruled_t)):
        print(f"\n{name}")
        for row in table(t):
            print("  ", row)
    print("\nPARSE-BACK")
    for k, v in sorted(pb_t.items()):
        print("  ", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
