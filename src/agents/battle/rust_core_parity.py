"""The Rust Core parity harness — slice E (events) — its corpus, its replay, its comparison.

``gen3_core_parity_events_v1`` (``designs/endstate/program_rust_core.md`` §3). ONE harness, one
slice per milestone; M1 lands slice **E**: per viewer, per event, ``seq · turn · kind · side ·
actor · target · value · raw`` of the Rust core's READING projection against the event log of a
``Gen3Battle`` fed the same per-side text. **No allowlist**: a divergence fails and prints its
census; the fix is in the core or a NAMED reading rule (``src/rust_sim/src/core_events/reading.rs``).

Both paths re-derive from a recorded INPUT log — no players, no player RNG:

* the CORE replays the recorded commands through the production bridge session with source
  recording on (``src/rust_sim/src/bin/core_events.rs``), which also refuses unless ``parse`` of
  each side's text reproduces its step-path events, and returns the per-side chunks;
* the REFERENCE feeds those chunks into a fresh ``Gen3Battle`` per viewer through
  :mod:`agents.battle.offline_feed` (a dispatch mirror of ``Player._handle_battle_message``).

The per-side chunks are checked against the bytes the battle's players received when it was
RECORDED (a digest in the fixture) — the gate refuses if the core no longer reproduces them.

Corpora (``python -m agents.battle.rust_core_parity --help``):

* **COMMIT** (``rust_core_parity_fixtures/commit_tier.json.gz``): 6 seeded-random battles over 12
  distinct pool teams + 2 production-policy battles + 2 seeded-random Baton Pass battles, recorded
  once; plus the byte-fuzz fixtures that carry the four ambiguity-prone shapes. Seconds; routine
  gate.

Slice **V** (views + legality — the TRUTH AUDIT of the training observation path,
:mod:`agents.battle.rust_core_parity_views`) rides the SAME call: ``check_battles(…,
views=ViewCensus())`` asks the core for its decision boards too (``core_events --views``).
* **MILESTONE** (``slow``): 2 seeds × 200 seeded-random battles and 2 × 50 production-policy
  battles PLAYED live (the live ``Gen3Battle`` logs must also equal the offline feed's), plus the
  22-scenario protocol capture corpus × 2 seeds and every byte-fuzz fixture. The pool content
  hash and the checkpoint sha256 are pinned in ``rust_core_parity_fixtures/manifest.json`` and the
  tier REFUSES on a mismatch (a pool change regenerates the manifest in the same commit).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.offline_feed import feed_chunk, new_battle, player_names
from utils.paths import repo_path

FIXTURES = repo_path("src", "agents", "battle", "rust_core_parity_fixtures")
COMMIT_FIXTURE = FIXTURES / "commit_tier.json.gz"
MANIFEST = FIXTURES / "manifest.json"
PROTOCOL_GOLDEN = repo_path("src", "rust_sim", "tests", "vectors", "protocol_capture_golden.txt")
BYTE_FUZZ_DIR = repo_path("src", "rust_sim", "tests", "vectors", "byte_fuzz_corpus")
#: The golden corpus of persisted records (`gen3_core_event_v1`): the COMMIT tier's recorded
#: battles + one battle per protocol capture scenario, both viewers, gzip'd (the per-side text
#: carries every `|request|` frame, so the plain corpus is ~8 MB). Gated by
#: `rust_core_parity_test.py::test_the_golden_records_round_trip_and_reparse`.
RECORD_GOLDEN_DIR = FIXTURES / "records"

#: The byte-fuzz fixtures that carry the four shapes the seeded-random corpus never reached
#: (Phase 0, F8): Damp's `[of]` cant, a slot-less future-move `-miss`, a Heal Bell-class bench
#: `-curestatus`, `[from] lockedmove`. The COMMIT tier runs them; the MILESTONE tier runs all.
SHAPE_FIXTURES = (
    "42_damp_blocks_explosion_move_still",
    "29_future_move_miss_benched_caster_source_ref",
    "30_future_move_miss_no_retro_tag_prior_turn",
    "31_future_move_miss_after_forced_replacement_flush",
    "70_attract_end_skips_a_fainted_holder",
    "52_charge_selfko_no_end",
)

#: The COMMIT tier's recorded battles: 6 seeded-random (12 distinct pool teams) + 2 policy.
COMMIT_RANDOM_KEYS = (0, 2, 4, 6, 8, 10)
COMMIT_POLICY_KEYS = (20, 22)
#: + 2 seeded-random BATON PASS battles, so the class that motivated slice V is pinned in the
#: routine gate forever: key 177 passes Calm Mind stages (the entrant reads spa+2/spd+2 in the
#: SIM), key 34 passes a Substitute (the §4b "missing `substitute`" finding's own board).
COMMIT_BATON_PASS_KEYS = (34, 177)
#: The MILESTONE tier's key ranges and policy keys, two seeds each. The random seeds STRIDE the
#: whole pool: teams are `key`, `key+1` (mod the pool), so the even keys pair (0,1), (2,3), … and
#: the odd keys (1,2), (3,4), … — every one of the 719 teams plays twice per seed, against two
#: different partners. (The Phase-0 ranges 0–199 and 5000–5199 wrapped onto overlapping teams and
#: covered ~234 of 719 — which is how a Pressure-Aerodactyl team, 3 in the pool, never reached the
#: truth audit.)
MILESTONE_RANDOM_KEYS = (range(0, 720, 2), range(1, 720, 2))
MILESTONE_POLICY_KEYS = (range(100, 150), range(6000, 6050))

#: The compared fields of one event (``BattleEvent`` attribute, core JSON key).
FIELDS = (("seq", "seq"), ("turn", "turn"), ("kind", "kind"), ("side", "side"),
          ("actor_species", "actor"), ("target_species", "target"), ("value", "value"),
          ("raw", "raw"))


@dataclass
class RecordedBattle:
    """One battle's INPUT log (what both paths re-derive from) + the bytes its players got."""

    label: str
    format_id: str
    seed: str
    p1: Dict[str, str]
    p2: Dict[str, str]
    commands: List[List[str]]
    init_seed: bool = False
    quick_claw: bool = False
    #: sha256 of the per-side chunks as RECORDED (``None`` for a corpus with no live capture).
    chunks_sha: Optional[str] = None

    def script(self) -> List[str]:
        start = {"label": self.label, "formatid": self.format_id, "seed": self.seed,
                 "p1": self.p1, "p2": self.p2, "init_seed": self.init_seed,
                 "quick_claw": self.quick_claw}
        out = ["START " + json.dumps(start)]
        for cmd in self.commands:
            side, choice = cmd[0], cmd[1]
            if side == "forcelose":
                out.append(f"FORCELOSE {choice}")
            else:
                # A capture golden's per-decision choice is fed only while the side's choice is
                # open (`CHOOSEIF`); a live recording's command is fed as the child received it.
                verb = "CHOOSEIF" if len(cmd) > 2 and cmd[2] == "if_open" else "CHOOSE"
                out.append(f"{verb} {side} {choice}")
        out.append("END")
        return out


def chunks_sha(chunks: Sequence[Tuple[str, str]]) -> str:
    """The digest of a battle's per-side chunk stream, ``[(side, text), …]`` in flush order."""
    h = hashlib.sha256()
    for side, text in chunks:
        h.update(f"{side}\x00{text}\x01".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the core
# ---------------------------------------------------------------------------

#: The EMISSION SELF-CHECK's stderr summary (``emission_check::summary``), printed by a
#: ``core_events`` built with the check on — a production build prints none.
_SELFCHECK_RE = re.compile(r"^emission_selfcheck omniscient=(\d+) per_viewer=(\d+) split=(\d+) frame=(\d+)$")
SELFCHECK_KINDS = ("omniscient", "per_viewer", "split", "frame")


def selfcheck_counts(stderr: str) -> Optional[Dict[str, int]]:
    """The self-check counts a ``core_events`` run reported on stderr, or ``None`` when the binary
    ran without the check (a production build)."""
    for line in stderr.splitlines():
        m = _SELFCHECK_RE.match(line.strip())
        if m:
            return dict(zip(SELFCHECK_KINDS, map(int, m.groups())))
    return None


def run_core(battles: Sequence[RecordedBattle], record_dir: Optional[str] = None,
             commit: str = "unknown", views: bool = False,
             selfcheck: Optional[collections.Counter] = None) -> List[dict]:
    """Replay ``battles`` through the core in ONE process; one result dict per battle.
    ``views`` also captures slice V's decision boards (``core_events --views``). ``selfcheck``
    accumulates the EMISSION SELF-CHECK's counts (``selfcheck["runs_without"]`` counts a process
    that ran without the check)."""
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    argv = [resolve_core_events_bin()]
    if views:
        argv.append("--views")
    if record_dir:
        argv += ["--record-dir", record_dir, "--commit", commit]
    stdin = "\n".join(line for b in battles for line in b.script()) + "\n"
    p = subprocess.run(argv, input=stdin, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"core_events failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")
    if selfcheck is not None:
        counts = selfcheck_counts(p.stderr)
        if counts is None:
            selfcheck["runs_without"] += 1
        else:
            selfcheck.update(counts)
    out = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    if len(out) != len(battles):
        raise RuntimeError(f"core_events answered {len(out)} battles for {len(battles)}")
    return out


def core_chunks(res: dict) -> List[Tuple[str, str]]:
    return [("p1" if side == 0 else "p2", "\n".join(lines)) for side, lines in res["chunks"]]


# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------

def reference(chunks: Sequence[Tuple[str, str]], viewer: str) -> Gen3Battle:
    """``viewer``'s ``Gen3Battle`` fed its per-side chunks offline."""
    lines = [ln for side, c in chunks if side == viewer for ln in c.split("\n")]
    b = new_battle(viewer, player_names(lines))
    for side, c in chunks:
        if side == viewer:
            feed_chunk(b, c)
    return b


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def _same(a: Any, b: Any) -> bool:
    """Type-strict equality: an int is not a float, whatever their values (the obs reads both)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@dataclass
class Census:
    battles: int = 0
    viewers: int = 0
    events: int = 0
    lines: int = 0
    kinds: collections.Counter = field(default_factory=collections.Counter)
    divergences: collections.Counter = field(default_factory=collections.Counter)
    examples: Dict[str, Any] = field(default_factory=dict)
    refused: List[str] = field(default_factory=list)
    #: The EMISSION SELF-CHECK's per-kind counts over every core process this census ran.
    selfcheck: collections.Counter = field(default_factory=collections.Counter)

    def diverge(self, key: str, example: Any) -> None:
        self.divergences[key] += 1
        self.examples.setdefault(key, example)

    def render(self) -> str:
        head = (f"{self.battles} battles, {self.viewers} viewers, {self.lines} per-side lines, "
                f"{self.events} events compared")
        if self.selfcheck:
            head += ", emission self-check " + " ".join(
                f"{k}={self.selfcheck[k]}" for k in (*SELFCHECK_KINDS, "runs_without") if k in self.selfcheck)
        if not self.divergences and not self.refused:
            return f"✅ 0 divergences — {head}"
        out = [f"❌ {sum(self.divergences.values())} divergences in {len(self.divergences)} "
               f"classes, {len(self.refused)} refused — {head}"]
        for k, n in self.divergences.most_common():
            out.append(f"   {n:7d}  {k}\n            e.g. {self.examples[k]}")
        for r in self.refused[:10]:
            out.append(f"   REFUSED {r}")
        return "\n".join(out)


def compare_viewer(core_events: List[dict], ref: Gen3Battle, label: str, census: Census) -> None:
    """Every core reading against the reference log, in order, field by field."""
    core = [r for ev in core_events for r in ev["readings"]]
    ref_events = list(ref.events)
    census.viewers += 1
    census.lines += len(core_events)
    if len(core) != len(ref_events):
        ck = collections.Counter(r["kind"] for r in core)
        rk = collections.Counter(e.kind.name for e in ref_events)
        census.diverge("[event COUNT]", (label, len(core), len(ref_events), dict(ck - rk), dict(rk - ck)))
        return
    for c, e in zip(core, ref_events):
        census.events += 1
        census.kinds[c["kind"]] += 1
        for attr, key in FIELDS:
            want = getattr(e, attr)
            got = c[key]
            if attr == "kind":
                want = want.name
            elif attr == "raw":
                want, got = list(want), list(got)
            if not _same(got, want):
                census.diverge(f"{e.kind.name}.{attr}", (label, "core", got, "gen3battle", want,
                                                        "|".join(e.raw)))


#: Battles per ``core_events`` process when slice V is on: a view capture is ~40 KB per decision
#: board, so a whole milestone corpus in one stdout would be gigabytes held at once.
VIEW_BATCH = 16


def check_battles(battles: Sequence[RecordedBattle], census: Census,
                  record_dir: Optional[str] = None, commit: str = "unknown",
                  views: "Optional[Any]" = None) -> Census:
    """Run the core on ``battles``, check each against its recorded bytes and its references.

    ``views`` (a :class:`agents.battle.rust_core_parity_views.ViewCensus`) also runs slice V —
    the TRUTH AUDIT of every decision's ``LiveView`` — on the SAME core replay."""
    if views is not None:
        from agents.battle.rust_core_parity_views import check_views

    step = VIEW_BATCH if views is not None else max(len(battles), 1)
    for lo in range(0, len(battles), step):
        batch = battles[lo:lo + step]
        results = run_core(batch, record_dir=record_dir, commit=commit, views=views is not None,
                           selfcheck=census.selfcheck)
        for b, res in zip(batch, results):
            census.battles += 1
            if not res["ok"]:
                census.refused.append(f"{b.label}: {res['error']}")
                continue
            chunks = core_chunks(res)
            if b.chunks_sha is not None and chunks_sha(chunks) != b.chunks_sha:
                census.refused.append(f"{b.label}: the core no longer reproduces the recorded chunks")
                continue
            for i, viewer in enumerate(("p1", "p2")):
                compare_viewer(res["viewers"][i], reference(chunks, viewer), f"{b.label}/{viewer}",
                               census)
            if views is not None:
                check_views(b.label, chunks, res.get("views") or [], views,
                            teams={"p1": b.p1["team"], "p2": b.p2["team"]},
                            format_id=b.format_id)
    return census


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------

def commit_corpus() -> List[RecordedBattle]:
    """The COMMIT tier: the 8 recorded battles, the six shape fixtures, and the first battle of
    each of the 22 protocol capture scenarios."""
    return (load_commit_fixture() + byte_fuzz_battles(SHAPE_FIXTURES) + protocol_battles(1)
            + constructed_battles())


def load_commit_fixture() -> List[RecordedBattle]:
    with gzip.open(COMMIT_FIXTURE, "rt") as fh:
        return [RecordedBattle(**d) for d in json.load(fh)["battles"]]


def golden_battles(text: str, tag: str, per_scenario: Optional[int] = None) -> List[RecordedBattle]:
    """The ``SCEN/TEAM/FMT/INIT/DEC`` grammar (the protocol capture golden and every byte-fuzz
    fixture) as recorded battles from the post-construction seed. ``per_scenario`` keeps the first
    N battles of each scenario."""
    teams: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    fmts: Dict[str, str] = {}
    out: List[RecordedBattle] = []
    seen: collections.Counter = collections.Counter()
    by_key: Dict[Tuple[str, str], RecordedBattle] = {}
    last: Dict[str, RecordedBattle] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "TEAM":
            teams[f[1]][f[2]] = line.split("\t", 3)[3]
        elif f[0] == "FMT":
            fmts[f[1]] = f[2]
        elif f[0] == "INIT":
            fuzz = "," not in f[3]
            key, seed = ("0", f[2]) if fuzz else (f[2], f[3])
            qc = fuzz and len(f) == 5 and f[4] == "1"
            b = RecordedBattle(
                label=f"{tag}{f[1]}_{key}", format_id=fmts.get(f[1], "gen3customgame"), seed=seed,
                p1={"name": "P1", "team": teams[f[1]]["p1"]},
                p2={"name": "P2", "team": teams[f[1]]["p2"]},
                commands=[], init_seed=True, quick_claw=qc)
            b._fuzz = fuzz  # type: ignore[attr-defined]
            seen[f[1]] += 1
            if per_scenario is None or seen[f[1]] <= per_scenario:
                out.append(b)
            by_key[(f[1], key)] = b
            last[f[1]] = b
        elif f[0] == "DEC":
            b = by_key.get((f[1], f[2]))
            c = 7
            if b is None or getattr(b, "_fuzz", False):
                b, c = last[f[1]], 6
            for side, tok in (("p1", f[c]), ("p2", f[c + 1])):
                if tok == "-":
                    continue
                verb = "move" if tok[0] == "m" else "switch"
                b.commands.append([side, f"{verb} {int(tok[1:]) + 1}", "if_open"])
    return out


FORECAST_GOLDEN = repo_path("src", "rust_sim", "tests", "vectors", "forecast_golden.txt")


def sweep_battles(text: str, tag: str, per_scenario: int = 1) -> List[RecordedBattle]:
    """A class-sweep golden (`INIT scen initSeed …`, `DEC scen key req f1 f2 c1 c2 …`, no
    protocol lines) as recorded battles — the port's own emission is what both paths read."""
    teams: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    out: List[RecordedBattle] = []
    seen: collections.Counter = collections.Counter()
    cur: Optional[RecordedBattle] = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "TEAM":
            teams[f[1]][f[2]] = line.split("\t", 3)[3]
        elif f[0] == "INIT":
            seen[f[1]] += 1
            cur = RecordedBattle(label=f"{tag}{f[1]}_{seen[f[1]] - 1}", format_id="gen3customgame",
                                 seed=f[2], p1={"name": "P1", "team": teams[f[1]]["p1"]},
                                 p2={"name": "P2", "team": teams[f[1]]["p2"]}, commands=[],
                                 init_seed=True)
            if seen[f[1]] <= per_scenario:
                out.append(cur)
        elif f[0] == "DEC" and cur is not None:
            for side, tok in (("p1", f[6]), ("p2", f[7])):
                if tok != "-":
                    verb = "move" if tok[0] == "m" else "switch"
                    cur.commands.append([side, f"{verb} {int(tok[1:]) + 1}", "if_open"])
    return out


def constructed_battles() -> List[RecordedBattle]:
    """Shapes no committed battle golden plays but the port models: Forecast's `-formechange`
    (every scenario of the forecast class sweep) and a Ditto `-transform` (constructed here)."""
    ditto = RecordedBattle(
        label="constructed.ditto_transform", format_id="gen3customgame", seed="1,2,3,4",
        p1={"name": "P1", "team": "Ditto|||Limber|transform|Relaxed|252,,252,,4,|N||||"},
        p2={"name": "P2", "team": "Snorlax|||Immunity|splash,bodyslam|Careful|252,,252,,,|N||||"},
        commands=[["p1", "move 1"], ["p2", "move 1"], ["p1", "move 2"], ["p2", "move 2"],
                  ["p1", "move 1"], ["p2", "move 2"]],
        init_seed=True)
    return sweep_battles(FORECAST_GOLDEN.read_text(), "forecast.") + [ditto]


def protocol_battles(per_scenario: Optional[int] = None) -> List[RecordedBattle]:
    return golden_battles(PROTOCOL_GOLDEN.read_text(), "proto.", per_scenario)


def byte_fuzz_battles(only: Optional[Iterable[str]] = None) -> List[RecordedBattle]:
    names = sorted(p.stem for p in BYTE_FUZZ_DIR.glob("*.txt"))
    if only is not None:
        keep = set(only)
        missing = keep - set(names)
        if missing:
            raise FileNotFoundError(f"byte-fuzz fixtures missing: {sorted(missing)}")
        names = [n for n in names if n in keep]
    out: List[RecordedBattle] = []
    for n in names:
        out += golden_battles((BYTE_FUZZ_DIR / f"{n}.txt").read_text(), f"fuzz.{n}.")
    return out


# ---------------------------------------------------------------------------
# live play (recording the commit fixture; the milestone tier's random + policy halves)
# ---------------------------------------------------------------------------

def pool_hash() -> str:
    from utils.team_loader import TeamLoader

    h = hashlib.sha256()
    for t in TeamLoader().get_all_teams():
        h.update(t.encode() if isinstance(t, str) else repr(t).encode())
        h.update(b"\x00")
    return h.hexdigest()


OU_RANDOM_TEAMS_JS = repo_path("src", "rust_sim", "harness", "ou_random_teams.js")


def procedural_teams(n: int, seed: int) -> List[str]:
    """``n`` packed gen3ou teams from the PROCEDURAL generator (``ou_random_teams.js --emit``):
    Smogon-derived, validated by Showdown's own ``TeamValidator('gen3ou')``, reproducible from
    ``seed`` — the fuzz surface beyond the pool's fixed teams (``src/rust_sim/CLAUDE.md``,
    ``--mode ourandom``). The teams are not the pool's, so no pool hash pins them; the seed does."""
    p = subprocess.run(["node", str(OU_RANDOM_TEAMS_JS), "--emit", str(n), "--seed", str(seed)],
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ou_random_teams --emit failed: {p.stderr.strip()[-2000:]}")
    teams = [t for t in p.stdout.splitlines() if t.strip()]
    if len(teams) != n:
        raise RuntimeError(f"ou_random_teams --emit returned {len(teams)} teams for {n}")
    return teams


def _players(key: int, tag: str, policy=None, teams: Optional[Tuple[str, str]] = None):
    from poke_env import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.training.obs_roundtrip_fuzz_test import SeededRandomPlayer
    from utils.team_loader import TeamLoader

    if teams is not None:
        t1, t2 = teams
    else:
        pool = TeamLoader().get_all_teams()
        t1, t2 = pool[key % len(pool)], pool[(key + 1) % len(pool)]
    common = dict(battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                  start_listening=False, max_concurrent_battles=1, battle_class=Gen3Battle)
    if policy is None:
        return (SeededRandomPlayer(rng_seed=1000 + key, team=t1, account_configuration=AccountConfiguration(f"{tag}a{key}", "pw"), **common),
                SeededRandomPlayer(rng_seed=2000 + key, team=t2, account_configuration=AccountConfiguration(f"{tag}b{key}", "pw"), **common))
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings

    mk = lambda team, seed, name: RLPlayer(  # noqa: E731
        model=policy, team=team, mappings=load_mappings(), stochastic=True, policy_seed=seed,
        account_configuration=AccountConfiguration(name, "pw"), **common)
    return mk(t1, 3000 + key, f"{tag}a{key}"), mk(t2, 4000 + key, f"{tag}b{key}")


@dataclass
class LiveBattle:
    """A battle PLAYED for real: its input log, the two players' live battles, and every chunk."""

    recorded: RecordedBattle
    p1: Gen3Battle
    p2: Gen3Battle
    chunks: List[Tuple[str, str]]


def play(key: int, tag: str = "Rc", policy=None,
         teams: Optional[Tuple[str, str]] = None) -> LiveBattle:
    """Play ONE battle for real over the production rust ``sim_bridge`` (the key recipe: teams
    ``key``/``key+1``, per-player RNG, sim seed ``[11+key, 22+key, 33+key, 44+key]``, concurrency 1)
    and return its input log, the two LIVE battles and the per-side chunks they received.
    ``teams`` overrides the pool pair (the procedural generator's teams)."""
    from utils.bridge import reconstruction
    from utils.bridge.local_battle_runner import run_local_battles

    p1, p2 = _players(key, tag, policy, teams)
    sink: list = []
    asyncio.run(run_local_battles(p1, p2, 1, seed=[11 + key, 22 + key, 33 + key, 44 + key],
                                  impl="rust", chunk_sink=sink))
    (b1,) = list(p1.battles.values())
    (b2,) = list(p2.battles.values())
    rec = reconstruction.pop_record(b1.battle_tag)
    if rec is None:
        raise RuntimeError(f"no __RECON__ for {b1.battle_tag}")
    players = rec.players()
    battle = RecordedBattle(
        label=f"{'policy' if policy is not None else 'procedural' if teams else 'random'}_{key}",
        format_id=rec.format_id,
        seed=rec.prng_seed, p1=players["p1"], p2=players["p2"],
        commands=[list(c) for c in rec.commands], chunks_sha=chunks_sha(sink))
    return LiveBattle(battle, b1, b2, list(sink))


def compare_live(live: LiveBattle, census: Census) -> None:
    """Each LIVE player's log == the offline feed's of the same chunks — the reference's own
    faithfulness, checked wherever a battle is actually played."""
    for viewer, battle in (("p1", live.p1), ("p2", live.p2)):
        _compare_live_viewer(battle, reference(live.chunks, viewer), f"{live.recorded.label}/{viewer}", census)


def _compare_live_viewer(live: Gen3Battle, off: Gen3Battle, label: str, census: Census) -> None:
    a, b = list(live.events), list(off.events)
    if len(a) != len(b):
        census.diverge("[live != offline COUNT]", (label, len(a), len(b)))
        return
    for x, y in zip(a, b):
        for attr, _ in FIELDS:
            if getattr(x, attr) != getattr(y, attr):
                census.diverge(f"[live != offline] {x.kind.name}.{attr}", (label, getattr(x, attr), getattr(y, attr)))


def production_checkpoint() -> Tuple[str, str]:
    """``(zip_path, sha256)`` of the ``production`` baseline, verified against its registry sha256."""
    from agents.training import baselines

    r = baselines.resolve("production")
    want = baselines.load_registry()["baselines"]["production"]["sha256"]
    with open(r.zip_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != want:
        raise RuntimeError(f"production checkpoint {r.zip_path}: sha256 {digest} != registry {want}")
    return r.zip_path, digest


def load_production_policy():
    """The ``production`` baseline checkpoint (``baselines.load`` — the by-name frozen load)."""
    from agents.training import baselines

    production_checkpoint()
    model = baselines.load("production", device="cpu")
    for m in model.policy.modules():
        if hasattr(m, "_debugger"):
            m._debugger = None
    return model


def record_commit_fixture(append: bool = False) -> None:
    """(Re)record the COMMIT tier's 10 battles: 6 seeded-random over 12 distinct pool teams, 2
    production-policy, 2 seeded-random Baton Pass battles. Every battle's live logs must equal the
    offline feed's, the core's replay must reproduce its chunks, and BOTH slices (E, V) must be
    clean, or nothing is written. ``append`` keeps the recorded battles and plays only the keys the
    fixture does not hold yet (a new key never re-records — and so never perturbs — the rest)."""
    from agents.battle.rust_core_parity_views import ViewCensus

    census = Census()
    kept = load_commit_fixture() if append else []
    have = {b.label for b in kept}
    want = [(k, None) for k in COMMIT_RANDOM_KEYS] + [(k, "policy") for k in COMMIT_POLICY_KEYS]
    want += [(k, None) for k in COMMIT_BATON_PASS_KEYS]
    model = load_production_policy() if any(
        p and f"policy_{k}" not in have for k, p in want) else None
    lives = [play(k, policy=model if p else None) for k, p in want
             if f"{'policy' if p else 'random'}_{k}" not in have]
    for lv in lives:
        compare_live(lv, census)
    battles = kept + [lv.recorded for lv in lives]
    views = ViewCensus()
    check_battles(battles, census, views=views)
    print(census.render())
    print(views.render())
    if census.divergences or census.refused or views.divergences or views.refused:
        raise SystemExit("not written: the recorded battles do not pass the gate")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "gen3_core_parity_commit_fixture_v1", "pool_sha256": pool_hash(),
               "battles": [b.__dict__ for b in battles]}
    with gzip.open(COMMIT_FIXTURE, "wt") as fh:
        json.dump(payload, fh, sort_keys=True)
    print(f"wrote {COMMIT_FIXTURE} ({COMMIT_FIXTURE.stat().st_size} bytes)")


def write_golden_records() -> int:
    """(Re)write the golden record corpus from the core's step path (both viewers)."""
    import shutil
    import tempfile

    from utils.git import get_git_hash

    with tempfile.TemporaryDirectory() as td:
        census = check_battles(load_commit_fixture() + protocol_battles(1), Census(),
                               record_dir=td, commit=get_git_hash() or "unknown")
        print(census.render())
        if census.divergences or census.refused:
            return 1
        if RECORD_GOLDEN_DIR.exists():
            shutil.rmtree(RECORD_GOLDEN_DIR)
        RECORD_GOLDEN_DIR.mkdir(parents=True)
        from pathlib import Path

        for f in sorted(Path(td).glob("*.jsonl")):
            # mtime=0: the gzip bytes are a pure function of the record (a regeneration diffs clean).
            with open(f, "rb") as src, gzip.GzipFile(RECORD_GOLDEN_DIR / (f.name + ".gz"), "wb",
                                                      mtime=0) as dst:
                dst.write(src.read())
    print(f"wrote {len(list(RECORD_GOLDEN_DIR.glob('*.gz')))} records under {RECORD_GOLDEN_DIR}")
    return 0


def check_golden_records() -> List[str]:
    """Every golden record: ``write(read(bytes)) == bytes`` and its stored text re-parses to its
    stored typed stream (the core does both; ``core_events --check-records``). Returns the
    failures (empty = clean)."""
    import tempfile
    from pathlib import Path

    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    files = sorted(RECORD_GOLDEN_DIR.glob("*.jsonl.gz"))
    if not files:
        return [f"no golden records under {RECORD_GOLDEN_DIR}"]
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for f in files:
            out = Path(td) / f.name[:-3]
            out.write_bytes(gzip.decompress(f.read_bytes()))
            paths.append(str(out))
        p = subprocess.run([resolve_core_events_bin(), "--check-records", *paths],
                           capture_output=True, text=True, check=False)
    bad = [ln for ln in p.stdout.splitlines() if not ln.startswith("ok ")]
    if p.returncode != 0 and not bad:
        bad = [p.stderr.strip()[-2000:]]
    if len([ln for ln in p.stdout.splitlines() if ln.startswith("ok ")]) != len(files) and not bad:
        bad = [f"checked {p.stdout.count('ok ')} of {len(files)} records"]
    return bad


def manifest_now() -> dict:
    """What the MILESTONE tier's corpus resolves to TODAY on this box."""
    _, digest = production_checkpoint()
    return {
        "schema": "gen3_core_parity_manifest_v1",
        "pool_sha256": pool_hash(),
        "production_sha256": digest,
        "random_keys": [[r.start, r.stop, r.step] for r in MILESTONE_RANDOM_KEYS],
        "policy_keys": [[r.start, r.stop] for r in MILESTONE_POLICY_KEYS],
        "protocol_per_scenario": 2,
    }


def check_manifest() -> dict:
    """REFUSE unless the committed manifest is what the corpus resolves to now: a pool change,
    a re-pointed baseline or an edited key range regenerates the manifest in the same commit
    (``python -m agents.battle.rust_core_parity write-manifest``)."""
    want = json.loads(MANIFEST.read_text())
    have = manifest_now()
    diff = {k: (want.get(k), have[k]) for k in have if want.get(k) != have[k]}
    if diff:
        raise RuntimeError(f"rust core parity MANIFEST mismatch {diff} — regenerate it in the same "
                           "commit: python -m agents.battle.rust_core_parity write-manifest")
    return want


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("record-commit", help="(re)record the COMMIT tier fixture")
    rc.add_argument("--append", action="store_true",
                    help="keep the recorded battles; play only keys the fixture lacks")
    sub.add_parser("write-manifest", help="(re)write the MILESTONE tier manifest")
    sub.add_parser("write-records", help="(re)write the golden record corpus")
    c = sub.add_parser("check", help="run the COMMIT tier corpus and print the census")
    c.add_argument("--record-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "record-commit":
        record_commit_fixture(append=a.append)
        return 0
    if a.cmd == "write-records":
        return write_golden_records()
    if a.cmd == "write-manifest":
        FIXTURES.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest_now(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST}")
        return 0
    census = Census()
    check_battles(commit_corpus(), census, record_dir=a.record_dir)
    print(census.render())
    return 1 if census.divergences or census.refused else 0


if __name__ == "__main__":
    sys.exit(main())
