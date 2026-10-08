"""The Rust core's recorded COMMIT corpus, held to the core alone (P6 slice 6c of the poke-env retirement, 2026-10-08).

Until slice 6c the same recorded battles were also compared with the Python ``Gen3Battle`` / ``LiveView`` /
``EpisodeTracker`` / Python encoder (``rust_core_parity*``, slices E / V / T / O). That second implementation was never
a truth (``designs/rust_sim/present.md`` §3) and is deleted; what holds the core now is the four checks of
``designs/rust_sim/encoder.md`` §6a (the ENGINE-TRUTH audit, the ROUND-TRIP chain, TWO ROADS ONE ROW, FROZEN GOLDENS).
This file is the part of the old harness that read ONLY the core, kept as a routine-gate test:

* the corpus (``rust_core_parity_fixtures/commit_tier.json.gz``: 6 seeded-random + 2 policy + 2 Baton Pass + 2 LADDER
  battles, recorded once as INPUT logs) replays ``ok`` through ``core_events --views --trackers --obs``. That one
  replay IS the parse == step gate (the core refuses unless ``parse`` of each side's text reproduces its step-path
  events), the parse-chain encode gate (``gen3_core_parse_obs_gate_v1``) and the engine-truth BOARD audit
  (``present::audit::check_view`` at every decision, both viewers); the per-side chunks must also be the bytes the
  battles' players received when they were recorded (a digest in the fixture);
* every row is fully WRITTEN (the self-check build NaN-prefills a row, so an unwritten cell reads NaN) and carries no
  signed zero, and every decision's choice tokens are exactly its legal actions;
* the persisted RECORD corpus (``rust_core_parity_fixtures/records``) reads, re-writes byte-identically and re-parses
  to its stored typed stream (the migrate-by-reparse path a schema change must pass);
* ACTION DENIAL's information boundary holds on the core's native record.

The obs rows' VALUES are held by ``agents/training/golden_obs_core_test.py`` (the committed per-decision hashes).
"""

from __future__ import annotations

import base64
import collections
import gzip
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
import pytest

from agents.battle.core_obs import wrap_row
from agents.battle.core_replay import RecordedBattle, chunks_sha, core_chunks, run_core
from utils.paths import repo_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FIXTURES = repo_path("src", "agents", "battle", "rust_core_parity_fixtures")
COMMIT_FIXTURE = FIXTURES / "commit_tier.json.gz"
RECORD_GOLDEN_DIR = FIXTURES / "records"

#: What the recorded fixture replays to (measured 2026-10-08 on the 12 recorded battles; the per-side chunk digest pins
#: the bytes, so these are exact): battles, tracker-bearing decisions over both viewers, and the floors below.
N_BATTLES = 12
N_DECISIONS = 1807
MIN_BOARD_CHECKS = 300_000
MIN_WINDOW_ROWS = 40_000
#: The obs blocks the corpus must see NONZERO somewhere (a replay cannot be green on zeros).
OBS_BLOCKS = ("our_team", "opp_team", "context", "global", "board", "pair_history", "event_window", "obs_facts")


def load_commit_fixture() -> List[RecordedBattle]:
    with gzip.open(COMMIT_FIXTURE, "rt") as fh:
        return [RecordedBattle(**d) for d in json.load(fh)["battles"]]


def _decision_caps(res: Mapping[str, Any]):
    """``(viewer index, capture)`` for every DECISION of one battle (the final ``terminal`` entry is not one)."""
    for vi, viewer in enumerate(res["trackers"]):
        for cap in viewer:
            if "trackers" in cap:
                yield vi, cap


def _obs_spans() -> Tuple[Tuple[str, int, int], ...]:
    from agents.observation import constants as C

    return (("our_team", C.OFFSET_OUR_TEAM, C.OFFSET_OPP_TEAM),
            ("opp_team", C.OFFSET_OPP_TEAM, C.OFFSET_CONTEXT),
            ("context", C.OFFSET_CONTEXT, C.OFFSET_GLOBAL),
            ("global", C.OFFSET_GLOBAL, C.OFFSET_REACTIVE),
            ("board", C.OFFSET_REACTIVE, C.OFFSET_PAIR_HISTORY),
            ("pair_history", C.OFFSET_PAIR_HISTORY, C.OFFSET_EVENT_WINDOW),
            ("event_window", C.OFFSET_EVENT_WINDOW, C.OFFSET_OBS_FACTS),
            ("obs_facts", C.OFFSET_OBS_FACTS, C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM))


def unwritten_or_signed_zero(row: np.ndarray) -> Tuple[List[int], List[int]]:
    """``(NaN cells, -0.0 cells)`` of one row. The self-check build NaN-prefills a row, so a NaN is a cell the encoder
    never wrote; ``-0.0`` equals ``0.0`` as a number but not as the BYTES the goldens hash."""
    nan = np.flatnonzero(np.isnan(row)).tolist()
    neg0 = np.flatnonzero(row.view(np.uint32) == np.uint32(0x80000000)).tolist()
    return nan, neg0


def boundary_violations(window: Sequence[Mapping]) -> List[Tuple[int, Any]]:
    """The INFORMATION BOUNDARY on one viewer's native record: a denied or refused opponent action carries
    ``"choice": "opp"`` and nothing else (the viewer never saw what the opponent chose); one of our own carries
    ``{"own": …}``. Returns ``(action index, action)`` for every violation."""
    out = []
    for i, a in enumerate(window):
        if a.get("kind") not in ("denied", "cant"):
            continue
        side = (a.get("actor") if a["kind"] == "denied" else a.get("mon"))[0]
        ch = a.get("choice")
        ok = ch == "opp" if side == "opp" else (isinstance(ch, dict) and set(ch) == {"own"})
        if not ok:
            out.append((i, a))
    return out


@pytest.fixture(scope="module")
def replay():
    """The commit corpus replayed ONCE through the core with views, trackers and obs on: ``(battles, results,
    self-check counts)``."""
    battles = load_commit_fixture()
    selfcheck: collections.Counter = collections.Counter()
    results = run_core(battles, views=True, trackers=True, obs=True, selfcheck=selfcheck)
    return battles, results, selfcheck


# ---------------------------------------------------------------------------
# the corpus replays clean
# ---------------------------------------------------------------------------

def test_the_commit_corpus_replays_clean_through_the_core(replay):
    """Every recorded battle replays ``ok`` (parse == step, the parse-chain encode, the board audit), the core
    reproduces the recorded per-side bytes, and the replay is not vacuous."""
    battles, results, selfcheck = replay
    # the EMISSION SELF-CHECK ran (`gen3_core_emission_selfcheck_v1`): a production `core_events` prints no counts, and
    # this gate then REFUSES rather than pass without the check (and without the NaN-poisoned prefill below)
    assert not selfcheck.get("runs_without"), (
        f"{selfcheck['runs_without']} core_events run(s) had NO emission self-check — the binary is a production "
        "build; unset POKESIM_CORE_EVENTS_BIN or point it at target/selfcheck/")
    assert selfcheck.get("omniscient", 0) > 0 and selfcheck.get("per_viewer", 0) >= 2 * selfcheck["omniscient"] * 0.9, \
        dict(selfcheck)

    assert len(battles) == N_BATTLES, f"the fixture holds {len(battles)} battles, the test pins {N_BATTLES}"
    assert len(results) == len(battles)
    decisions = board_checks = window_rows = terminal_rewards = 0
    labels: collections.Counter = collections.Counter()
    for b, res in zip(battles, results):
        assert res["ok"], f"{b.label}: the core refused the replay: {res['error']}"
        assert res["ended"] and not res["truncated"], f"{b.label}: the battle did not run to its end"
        # the recorded-bytes check has teeth: the digest in the fixture is of the bytes the players received
        assert b.chunks_sha is not None and chunks_sha(core_chunks(res)) == b.chunks_sha, \
            f"{b.label}: the core no longer reproduces the recorded per-side chunks"
        for vi, cap in _decision_caps(res):
            decisions += 1
            window_rows += len(cap["trackers"]["window"]["rows"])
            labels[cap["trackers"]["label"]["kind"]] += 1
            assert "obs" in cap, f"{b.label}: viewer {vi} decision after chunk {cap['after']} carries no obs row"
        for viewer in res["trackers"]:
            terminal_rewards += sum(1 for c in viewer if c.get("terminal"))
        for cap in res["views"]:
            for audit in cap["audit"]:
                board_checks += int(audit["checks"])
                assert not audit["divergences"], (b.label, cap["after"], audit["divergences"])
    assert decisions == N_DECISIONS, f"{decisions} decisions replayed, the recorded corpus has {N_DECISIONS}"
    assert board_checks >= MIN_BOARD_CHECKS, f"only {board_checks} board checks ran — the audit is vacuous"
    assert window_rows >= MIN_WINDOW_ROWS, f"only {window_rows} event-window rows — the replay is vacuous"
    assert all(labels[k] > 0 for k in (0, 1, 2)), f"an intent-label kind never fired: {dict(labels)}"
    assert terminal_rewards > 0, "no terminal reward recorded"


def test_the_corpus_rows_are_fully_written_and_each_decisions_tokens_are_its_legal_actions(replay):
    """Every core row is the observation (wrapped by ``wrap_row``), wholly written (no NaN) with no signed zero,
    every block nonzero somewhere, and the choice tokens are exactly the legal actions of the 11-dim mask."""
    battles, results, _ = replay
    spans = _obs_spans()
    nonzero: collections.Counter = collections.Counter()
    rows = tokens = 0
    for b, res in zip(battles, results):
        for vi, cap in _decision_caps(res):
            where = f"{b.label}/p{vi + 1}@chunk{cap['after']}"
            row = wrap_row(cap["obs"])
            nan, neg0 = unwritten_or_signed_zero(row)
            assert not nan, f"{where}: {len(nan)} cell(s) the encoder never wrote (NaN), first {nan[:6]}"
            assert not neg0, f"{where}: {len(neg0)} cell(s) are -0.0, first {neg0[:6]}"
            for name, lo, hi in spans:
                if np.any(row[lo:hi] != 0):
                    nonzero[name] += 1
            rows += 1
            mask = cap["mask"]
            assert len(mask) == 11 and set(mask) <= {0, 1}, (where, mask)
            legal = {i for i, bit in enumerate(mask) if bit}
            assert {int(k) for k in cap["tokens"]} == legal, (where, sorted(cap["tokens"]), sorted(legal))
            assert cap["choice"] in cap["tokens"].values(), (where, cap["choice"])
            tokens += len(cap["tokens"])
    assert rows == N_DECISIONS and tokens > 0
    missing = [name for name in OBS_BLOCKS if nonzero[name] == 0]
    assert not missing, f"obs blocks never nonzero: {missing} ({dict(nonzero)})"


def test_the_row_check_has_teeth(replay):
    """The NaN poison: a core row with ONE NaN cell (a cell the encoder never wrote) is caught, as is a ``-0.0``
    where ``+0.0`` belongs — and an untouched core row is clean."""
    _, results, _ = replay
    cap = next(c for _, c in _decision_caps(results[0]))
    row = wrap_row(cap["obs"]).copy()
    assert unwritten_or_signed_zero(row) == ([], [])
    zero_at = int(np.flatnonzero(row == 0)[0])
    poisoned = row.copy()
    poisoned[zero_at] = np.nan
    assert unwritten_or_signed_zero(poisoned) == ([zero_at], [])
    signed = row.copy()
    signed[zero_at] = -0.0
    assert unwritten_or_signed_zero(signed) == ([], [zero_at])
    # and the wire frame round-trips the bytes: a poisoned frame wraps to the same poisoned row
    frame = dict(cap["obs"], b64=base64.b64encode(poisoned.astype("<f4").tobytes()).decode())
    assert unwritten_or_signed_zero(wrap_row(frame)) == ([zero_at], [])


def test_the_recorded_bytes_check_has_teeth():
    """The digest in the fixture is sensitive to what the core ships: replay a battle with its last commands cut off and
    its per-side chunks are no longer the recorded ones."""
    b = load_commit_fixture()[0]
    full = run_core([b])[0]
    assert chunks_sha(core_chunks(full)) == b.chunks_sha
    cut = RecordedBattle(**dict(b.__dict__, commands=b.commands[:-4]))
    short = run_core([cut])[0]
    assert short["ok"] and chunks_sha(core_chunks(short)) != b.chunks_sha


# ---------------------------------------------------------------------------
# the information boundary
# ---------------------------------------------------------------------------

def test_the_information_boundary_holds_on_the_core_record_and_the_check_has_teeth(replay):
    """ACTION DENIAL's information boundary: every denied / refused action in the core's native record over the
    COMMIT corpus carries ``"opp"`` for an opponent (the viewer never saw the opponent's choice) and ``{"own": …}``
    for our own — and the check FAILS on a real opponent denial whose choice is made to leak."""
    _, results, _ = replay
    seen = {"opp": [], "ours": []}
    for res in results:
        for _vi, cap in _decision_caps(res):
            assert boundary_violations(cap.get("window") or ()) == [], cap["after"]
            for a in cap.get("window") or ():
                if a["kind"] in ("denied", "cant"):
                    seen[(a["actor"] if a["kind"] == "denied" else a["mon"])[0]].append(a)
    assert len(seen["opp"]) >= 20 and len(seen["ours"]) >= 20, {k: len(v) for k, v in seen.items()}
    assert boundary_violations(seen["opp"] + seen["ours"]) == []
    leak = dict(seen["opp"][0], choice={"own": "move 1"})
    assert boundary_violations([leak]) == [(0, leak)]
    hidden = dict(seen["ours"][0], choice="opp")
    assert boundary_violations([hidden]) == [(0, hidden)]


# ---------------------------------------------------------------------------
# the persisted record corpus
# ---------------------------------------------------------------------------

def check_golden_records() -> List[str]:
    """Every golden record: ``write(read(bytes)) == bytes`` and its stored text re-parses to its stored typed stream
    (the core does both: ``core_events --check-records``). Returns the failures (empty = clean)."""
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


def test_the_golden_records_round_trip_and_reparse():
    """The persisted RECORD (`gen3_core_event_v1`, `core_events::record`): every golden record — the COMMIT tier's
    battles + one battle per protocol scenario, both viewers — reads, re-writes BYTE-IDENTICALLY, and re-parses from
    its stored text to its stored typed stream (the migrate-by-reparse path a schema change must pass)."""
    records = sorted(RECORD_GOLDEN_DIR.glob("*.jsonl.gz"))
    assert len(records) >= 60, f"only {len(records)} golden records"
    assert check_golden_records() == []


def test_the_record_check_has_teeth(tmp_path):
    """A record that is not byte-for-byte what the core would write back is reported (a corrupted line is not ``ok``)."""
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    src = sorted(RECORD_GOLDEN_DIR.glob("*.jsonl.gz"))[0]
    text = gzip.decompress(src.read_bytes()).decode()
    bad = tmp_path / "tampered.jsonl"
    bad.write_text(text.replace('"kind"', '"kinD"', 1))
    p = subprocess.run([resolve_core_events_bin(), "--check-records", str(bad)],
                       capture_output=True, text=True, check=False)
    assert not all(ln.startswith("ok ") for ln in p.stdout.splitlines() if ln.strip()) or p.returncode != 0, \
        (p.stdout, p.stderr)
