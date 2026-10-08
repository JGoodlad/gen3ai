"""The PER-SIDE oracle reveal on the REAL engine (CPU, tiny): ``off`` is the engine of before, bit for bit; a reveal
reaches exactly the sides its mode names; every row is stamped. Seeded PERTURBED-fresh checkpoints (the h2h conftest's
pair A, B), T2 eager, the in-process core.

An ORACLE checkpoint here is A's own zip beside a ``model_config.json`` that records ``oracle_reveal`` (the weights are
A's: the reveal is an observation mode with no weight). So the same game under ``off`` and under a reveal differs ONLY
in what the revealed side's observation carries, and the per-side claim is read on the turn-1 decisions, which no
earlier action can have moved: a revealed side's first decision MOVES (its observation states the other team), an
unrevealed side's first decision is BIT-IDENTICAL to ``off``'s (its observation is ``off``'s, the forward is the same
slot on the same batch). The ROWS are the Rust core's: ``rust_env/tests/oracle_reveal_test.rs::
a_per_side_reveal_writes_each_side_exactly_as_the_symmetric_core_of_its_level`` proves a split core's side writes
exactly the training core's row of its level, and this file proves the engine declares that core.

The heavy plays are MODULE fixtures (the tier budget is per test call); the tests only read them."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

from agents.training import eval_ledger as L
from main.h2h import play as PL
from main.h2h import reveal as RV

pytestmark = [pytest.mark.sim, pytest.mark.integration]

COMPUTE = dict(device="cpu", backend="eager", threads=2, torch_threads=2, front="ffi", profile="selfcheck")
SCHEDULE_KEY = "h2h:test"
PAIRS, N_ENVS = 12, 8

#: The OFF/OFF play of (A, B) — 12 pairs, 8 envs, the cycle seed of (0, "h2h:test", 0) — RECORDED at `c7b4d03e`, the
#: commit before the per-side reveal, with this file's own digest: every game's result, end turn, teams and actions,
#: every p1 margin and log-prob bit, every p2 decision (dec_n, action, argmax, margin bits). Reproduced at the build.
#: RE-RECORDED 2026-10-06 for gen3_move_legality_by_id_v1 (the per-move-slot legality now lands on its own move;
#: with the old positional write restored the `c7b4d03e` digests 56135c83… / 36a0376a… reproduce exactly).
#: RE-RECORDED 2026-10-07 for gen3_op_ability_status_gigo_v1 (the damage op's opponent-ability / status-landing fix;
#: was 32972493… / 4baf87e0…).
#: RE-RECORDED 2026-10-07 for gen3_format_spec_priors_v1 (banned entities get prior 0: the Smogon ability / item priors
#: and the move / item / species tables; was 395e7d7b… / 82982a8f…).
#: RE-RECORDED 2026-10-07 for the X5 version break (gen3_x5_version_break_v1, config v144): the conftest's production
#: checkpoints are X5 now (blob deleted), a different architecture and so different games; was 9bffc7d6… / 54eca92a….
#: COMPLETED at the break's ONE golden re-record (parts 2 / 4 / 5 moved the checkpoints' init and forward, part 3 appended
#: the OBS-FACTS block, obs 2761 -> 2845); part 1's record was c06a6c51… / ce837ceb….
#: RE-RECORDED 2026-10-07 for gen3_mon_tied_gain_v1 (config v145): the op's out_gain shrank 99 -> 29, and the
#: conftest's ORDER-keyed `perturb_` draws every later parameter's noise from one stream, so the checkpoints moved;
#: was 4dd7ba19… / e4c9f4e8….
OFF_BITS = "fa189a9413cde29573bc6ce70004f2af2b124e0e79e703d5ae218313326df35f"
OFF_OUTCOME_ALL = "ee8512eeda767c34f9c9b186c31692943524389eab7131ce1dd6bff2fbfef9fc"


def bits_digest(games):
    h = hashlib.sha256()
    for g in sorted(games, key=lambda g: g["game"]):
        h.update(json.dumps([g["game"], g["result"], g["end_turn"], list(g["teams"]), list(g["actions"])]).encode())
        for x in list(g["margins"]) + list(g["logp"]):
            h.update(struct.pack("<d", float(x)))
        for o in g["opp"]:
            h.update(json.dumps([int(o[0]), int(o[1]), int(o[2])]).encode())
            h.update(struct.pack("<d", float(o[3])))
    return h.hexdigest()


def oracle_copy(zip_path: str, root: Path, level: str) -> str:
    """``zip_path``'s bytes in a run dir of its own whose ``model_config.json`` records ``oracle_reveal = level``."""
    src = Path(zip_path)
    dst = root / f"run_h2h_oracle_{level}"
    dst.mkdir()
    shutil.copy(src, dst / src.name)
    cfg = json.loads((src.parent / "model_config.json").read_text())
    cfg["oracle_reveal"] = level
    (dst / "model_config.json").write_text(json.dumps(cfg))
    return str(dst / src.name)


def run_engine(cells, mode, specs, ledger=None):
    """Play every cell (``PAIRS`` pairs at the fixed cycle seed) on ONE engine declared for ``cells`` under ``mode``;
    ``specs`` collects each eval core's startup declaration. With ``ledger`` (a root outside ``models/``), each cell
    also plays a 4-pair request through ``plan_edge`` / ``play_planned`` — the rows ``main.h2h`` writes."""
    import agents.training.rust_rollout.build as RB

    real = RB.open_core

    def spying(decl, spec_json):
        specs.append(json.loads(spec_json)["oracle_reveal"])
        return real(decl, spec_json)

    refs = [(PL.resolve_player(p), PL.resolve_player(o)) for p, o in cells]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RB, "open_core", spying)
        decl = PL.declare_engine(refs, PL._load_host, mode)
        eng = PL.H2HEngine(refs[0][0], refs[0][1], PL.Compute(n_envs=N_ENVS, **COMPUTE), emit=lambda _m: None,
                           decl=decl)
    out = {}
    try:
        for i, (p, o) in enumerate(refs):
            if i:
                eng.set_cell(p, o)
            games, _st = eng.play_batch(PAIRS, PL.cycle_seed(0, SCHEDULE_KEY, 0), sink=[])
            out[(p.zip_path, o.zip_path)] = ({g["game"]: g for g in games}, eng.levels)
            if ledger is not None:
                root, writer = PL.open_writer(str(ledger), "audit")
                ep = PL.plan_edge(writer, root, eng.regime, p, o, pairs=4, batch_pairs=4, schedule_seed=0,
                                  schedule_key=None, purpose="audit", request_id=None, family=None, request_kind=None)
                PL.play_planned(eng, writer, ep, schedule_seed=0, purpose="audit", run_label="h2h_reveal_test",
                                commit="test", emit=lambda _m: None)
                writer.close()
    finally:
        eng.close()
    return out, decl


@pytest.fixture(scope="module")
def plays(built, checkpoints, tmp_path_factory):
    a, b = checkpoints
    root = tmp_path_factory.mktemp("h2h_reveal")
    a_sp = oracle_copy(a, root, "species")
    ledger = root / "ledger"
    ledger.mkdir()
    specs = {"off": [], "one_sided": [], "both_sided": []}
    off, _ = run_engine([(a, b), (b, a)], "off", specs["off"], ledger=ledger)
    one, one_decl = run_engine([(a_sp, b), (b, a_sp)], "one_sided", specs["one_sided"], ledger=ledger)
    both, _ = run_engine([(a_sp, b)], "both_sided", specs["both_sided"], ledger=ledger)
    return dict(a=a, b=b, a_sp=a_sp, off=off, one=one, both=both, specs=specs, ledger=ledger, one_decl=one_decl)


def first_wave(games):
    """The games that START at the cycle's first step: each shard's first game (an env plays its shard's games in
    sequence, and every shard has an env: asserted). Their turn-1 decisions are batched exactly alike under any reveal
    — a LATER game starts beside games the reveal may already have steered elsewhere, so the batch its turn-1 row is
    forwarded in can differ, and an eager CPU forward's low bits move with the batch (``play_mirror_integration_test``
    shows the same for the env count). The bit-exact per-side claim is therefore read on the first wave ONLY: excluded,
    never tolerated (standing rule 8)."""
    shards = {}
    for k, g in games.items():
        shards.setdefault(g["shard"], []).append(k)
    assert len(shards) <= N_ENVS, f"{len(shards)} shards on {N_ENVS} envs: a shard's first game would wait for an env"
    wave = sorted(min(ks) for ks in shards.values())
    assert len(wave) >= 6, f"only {len(wave)} first-wave games: the per-side claim would be vacuous"
    return wave


def first(g):
    """A game's two turn-1 decisions: p1's (margin, log-prob) and p2's (action, argmax, margin)."""
    return (g["margins"][0], g["logp"][0]), (g["opp"][0][1], g["opp"][0][2], g["opp"][0][3])


def test_off_off_plays_the_engine_of_before_bit_for_bit(plays):
    games, levels = plays["off"][(plays["a"], plays["b"])]
    assert levels == RV.OFF_OFF
    bits = bits_digest(list(games.values()))
    outcome = L.outcome_digest(PL.outcome_vector(list(games.values())), [])
    assert bits == OFF_BITS, f"the off/off games moved against the pre-reveal record (now {bits} / {outcome})"
    assert outcome == OFF_OUTCOME_ALL, f"the off/off outcomes moved (now {outcome})"
    assert plays["specs"]["off"] == ["off"], "an off engine declares ONE core with the spec's string form, as before"


def test_one_sided_reaches_only_the_oracle_side_in_either_seat(plays):
    a, b, a_sp = plays["a"], plays["b"], plays["a_sp"]
    # the engine declared one core per seat the oracle sits in, each with the per-side form
    assert plays["specs"]["one_sided"] == [["species", "off"], ["off", "species"]]
    assert [lv for _gp, _go, lv in plays["one_decl"].cores] == [("species", "off"), ("off", "species")]
    for (cell, off_cell, oracle_seat) in (((a_sp, b), (a, b), 0), ((b, a_sp), (b, a), 1)):
        games, levels = plays["one"][cell]
        ref, _ = plays["off"][off_cell]
        assert levels == (("species", "off") if oracle_seat == 0 else ("off", "species"))
        assert sorted(games) == sorted(ref) == list(range(2 * PAIRS))
        for k in games:
            assert games[k]["teams"] == ref[k]["teams"] and games[k]["seed"] == ref[k]["seed"], "not the same game"
        assert first_wave(games) == first_wave(ref)
        # the ORACLE side's turn-1 decision moves in EVERY game (its observation states the other team) ...
        assert all(first(games[k])[oracle_seat] != first(ref[k])[oracle_seat] for k in games)
        # ... and on the first wave the other side's is bit-identical to off's
        for k in first_wave(games):
            moved = [x != y for x, y in zip(first(games[k]), first(ref[k]))]
            want = [True, False] if oracle_seat == 0 else [False, True]
            assert moved == want, (f"game {k}, oracle in seat p{oracle_seat + 1}: turn-1 decisions moved {moved} "
                                   f"(p1, p2), want {want} — the reveal must reach the oracle side and ONLY it")


def test_both_sided_reaches_both_sides_and_the_oracle_side_reads_what_it_reads_one_sided(plays):
    a, b, a_sp = plays["a"], plays["b"], plays["a_sp"]
    assert plays["specs"]["both_sided"] == ["species"], "both seats at the oracle's level: the symmetric string"
    games, levels = plays["both"][(a_sp, b)]
    ref, _ = plays["off"][(a, b)]
    one, _ = plays["one"][(a_sp, b)]
    assert levels == ("species", "species")
    assert first_wave(games) == first_wave(ref) == first_wave(one)
    for k in first_wave(games):
        p1, p2 = first(games[k])
        assert p1 != first(ref[k])[0] and p2 != first(ref[k])[1], f"game {k}: a side the reveal did not reach"
        assert p1 == first(one[k])[0], f"game {k}: p1's turn-1 observation is the same in both modes"


def test_every_row_is_stamped_with_its_protocol_and_per_side_levels(plays):
    from main.h2h import cli as CLI

    a, b = plays["a"], plays["b"]
    sha = {p: PL.resolve_player(p).sha256 for p in (a, b)}       # a_sp's bytes are a's
    want = {
        ("off", sha[a], sha[b]): None, ("off", sha[b], sha[a]): None,
        ("one_sided", sha[a], sha[b]): {"mode": "one_sided", "player": "species", "opponent": "off"},
        ("one_sided", sha[b], sha[a]): {"mode": "one_sided", "player": "off", "opponent": "species"},
        ("both_sided", sha[a], sha[b]): {"mode": "both_sided", "player": "species", "opponent": "species"},
    }
    reads = {}
    for decl in (CLI.H2H_READ, CLI.H2H_READ_ONE_SIDED, CLI.H2H_READ_BOTH_SIDED):
        reads.update(L.read_by_regime(decl, root=plays["ledger"]))
    rows = [r for got in reads.values() for r in got.rows]
    assert len(reads) == 3 and len(rows) == 5, f"{len(reads)} regimes, {len(rows)} rows"
    mode_of = {pr: m for m, pr in RV.PROTOCOL_OF_MODE.items()}
    got = {}
    for r in rows:
        L.validate_row(r)
        key = (mode_of[r["regime"]["protocol"]], r["player"]["sha256"], r["opponent"]["sha256"])
        got[key] = r["compute"].get("oracle_reveal")
        if key[0] == "off":
            assert "oracle_reveal" not in r["compute"], "an off row is the row of before: no stamp key at all"
    assert got == want
