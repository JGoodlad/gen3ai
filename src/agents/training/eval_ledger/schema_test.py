"""``eval_ledger.schema`` — the v2 row, its validator, the v1 row and its upgrade on read, the keys, the digest,
the companion records.

Every rule has a VALID row broken in exactly ONE way: a schema that accepts a row it should not is the failure that
matters, so each rule has a row that must be refused for exactly that rule."""
from __future__ import annotations

import copy

import pytest

from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import testkit as K

OPEN = {"request_id": "r1", "kind": "adhoc", "family": None, "ts": "2026-10-03T11:00:00+00:00"}


def test_a_valid_row_passes_and_so_does_a_backfilled_one():
    assert S.validate_row(K.make_row(request=OPEN)) == []
    assert S.validate_row(K.make_row()) == []                 # request None + provenance = backfilled


@pytest.mark.parametrize("name,mutate,needle", [
    ("schema", lambda r: r.update(schema=S.SCHEMA_V1), "schema"),
    ("missing key", lambda r: r.pop("flags"), "missing"),
    ("unknown key", lambda r: r.update(extra=1), "unknown keys"),
    ("purpose outside the closed list", lambda r: r.update(purpose="study"), "purpose"),
    ("ladder is backfill-only", lambda r: r.update(purpose="ladder"), "BACKFILLED"),
    ("player kind", lambda r: r["player"].update(kind="human"), "kind"),
    ("bad sha", lambda r: r["player"].update(sha256="xyz"), "sha256"),
    ("sha null without its flag", lambda r: r["player"].update(sha256=None), "sha_unrecorded"),
    ("regime tampered", lambda r: r["regime"].update(turn_limit=100), "regime_id"),
    ("unknown protocol", lambda r: r.update(regime=S.with_regime_id({**r["regime"], "protocol": "gen3_x"})),
     "protocol"),
    ("seat rule", lambda r: r.update(regime=S.with_regime_id({**r["regime"], "seat_rule": "p2"})), "seat_rule"),
    ("greedy side with a temperature", lambda r: r.update(regime=S.with_regime_id({**r["regime"], "player_temp": 1.0})),
     "greedy"),
    ("team_set not a digest", lambda r: r.update(regime=S.with_regime_id({**r["regime"], "team_set": "pool"})),
     "team_set"),
    ("eval core null without its flag", lambda r: r.update(regime=S.with_regime_id({**r["regime"], "eval_core": None})),
     "eval_core_unrecorded"),
    ("request kind", lambda r: r["request"].update(kind="whatever"), "request.kind"),
    ("request batch", lambda r: r["request"].update(batch=-1), "request.batch"),
    ("request opened after the row", lambda r: r["request"].update(opened="2026-10-04T00:00:00+00:00"), "opened"),
    ("aborted outside [voided, 2 voided]", lambda r: r["counts"].update(aborted=1), "aborted"),
    ("counts negative", lambda r: r["counts"].update(w=-1), "non-negative"),
    ("games != 2 pairs", lambda r: r["counts"].update(w=3), "games"),
    ("half-points != 2W + D", lambda r: r["pairs"].update(counts=[0, 0, 0, 2, 0]), "half-points"),
    ("team wins do not match W", lambda r: r["teams"][S.team_id("team one")].update(p=[4, 1]), "wins"),
    ("teams null without its flag", lambda r: r.update(teams=None), "teams_unrecorded"),
    ("seed null without its flag", lambda r: r.update(seed=None), "seed_unrecorded"),
    ("seed range", lambda r: r["seed"].update(game_hi=9), "game-index range"),
    ("digest missing", lambda r: r["compute"].pop("outcome_digest"), "audit keys"),
    ("digest null without its flag", lambda r: r["compute"].update(outcome_digest=None), "digest_unrecorded"),
    ("digest not hex", lambda r: r["compute"].update(outcome_digest="abc"), "outcome_digest"),
    ("near-tie list unsorted", lambda r: r["compute"].update(near_tie_games=[2, 1]), "near_tie_games"),
    ("near-tie index outside the batch", lambda r: r["compute"].update(near_tie_games=[40]), "outside"),
    ("flags unknown", lambda r: r.update(flags=["made_up"]), "flags"),
    ("flags unsorted", lambda r: r.update(flags=["teams_unrecorded", "digest_unrecorded"]), "flags"),
    ("draws under draws_folded", lambda r: r.update(flags=["draws_folded"]), "draws_folded"),
])
def test_each_rule_refuses_its_own_broken_row(name, mutate, needle):
    row = K.make_row(request=OPEN, pc=(0, 1, 1, 0, 0) if name.startswith("draws") else (0, 0, 2, 0, 0))
    mutate(row)
    problems = S.validate_row(row)
    assert problems, f"{name}: a broken row passed"
    assert any(needle in p for p in problems), (name, problems)


def test_a_request_less_row_is_legal_only_when_upgraded_or_backfilled():
    row = K.make_row()
    row["provenance"] = None
    assert any("request: null" in p for p in S.validate_row(row))


def test_flags_make_their_fields_nullable_and_only_then():
    row = K.make_row(request=OPEN)
    row["flags"] = ["digest_unrecorded", "seed_unrecorded", "teams_unrecorded"]
    row["compute"].update(outcome_digest=None, near_tie_games=None)
    row["seed"] = None
    row["teams"] = None
    assert S.validate_row(row) == []
    row["teams"] = K.make_row()["teams"]
    assert any("must be null under the teams_unrecorded" in p for p in S.validate_row(row))


def test_a_voided_pair_counts_aborted_games_and_the_seed_range_covers_them():
    row = K.make_row(request=OPEN, voided=1, aborted=2)
    assert S.validate_row(row) == []
    assert S.validate_row(K.make_row(request=OPEN, voided=1, aborted=1)) == []


def test_regime_id_digests_the_v2_identity_not_the_label():
    reg = K.regime()
    assert S.regime_id({**reg, "team_source": "another label"}) == reg["regime_id"]
    assert S.regime_id({**reg, "team_set": S.team_set_id({"t": "other"})}) != reg["regime_id"]
    assert S.regime_id({**reg, "protocol": "gen3_eval_protocol_v1_bot_rr"}) != reg["regime_id"]
    assert S.regime_id({**reg, "v1_id": "x"}) == reg["regime_id"]


def test_the_outcome_digest_hashes_the_ordered_vector_and_skips_the_near_ties():
    g = [(0, "W", 30), (1, "L", 31), (2, "D", 250)]
    d = S.outcome_digest(g)
    assert d == S.outcome_digest(list(reversed(g))), "the order is the game index, not the list"
    assert d != S.outcome_digest([(0, "W", 30), (1, "W", 31), (2, "D", 250)])
    assert d != S.outcome_digest([(0, "W", 30), (1, "L", 32), (2, "D", 250)]), "turns are in the vector"
    assert S.outcome_digest(g, near_tie=[1]) == S.outcome_digest([(0, "W", 30), (1, "W", 99), (2, "D", 250)], [1])
    with pytest.raises(ValueError, match="twice"):
        S.outcome_digest(g + [(0, "W", 30)])


def test_batch_key_and_seed_key():
    r = K.make_row(request=OPEN, batch=3)
    assert S.batch_key(r) == ("batch", "r1", 3, K.SHA_A, K.SHA_B, r["regime"]["regime_id"])
    assert S.seed_key(r)[-1] == r["seed"]["cycle_seed"]
    assert S.batch_key(K.make_row()) is None
    rep = K.make_row(request={**OPEN, "kind": "audit_replay"})
    assert S.seed_key(rep) is None, "a replay request deliberately replays a seed block"
    longer = K.make_row(request=OPEN, batch=3, pc=(0, 0, 4, 0, 0))
    assert S.seed_key(longer) == S.seed_key(r), "one cycle seed under another batch length is the same games"


# ------------------------------------------------------------------------------------------------ v1 upgrade
def test_a_v1_row_is_validated_as_v1_and_upgraded_deterministically():
    v1 = K.make_v1_row()
    assert S.validate_row_v1(v1) == []
    up = S.upgrade_v1(v1)
    assert S.validate_row(up) == []
    assert up == S.upgrade_v1(copy.deepcopy(v1)), "deterministic"
    assert v1["schema"] == S.SCHEMA_V1 and "request" not in v1, "the stored row is not touched"
    reg = up["regime"]
    assert (up["request"], up["flags"], up["provenance"]) == (None, ["digest_unrecorded"], None)
    assert up["player"]["kind"] == "checkpoint" and up["counts"]["aborted"] == 0
    assert reg["protocol"] == "gen3_eval_protocol_v1_h2h" and reg["seat_rule"] == "fixed_p1"
    assert reg["player_temp"] is None and reg["opponent_temp"] is None
    assert reg["team_set"] == S.team_set_id({"team_source": "pool"})
    assert reg["v1_id"] == v1["regime"]["regime_id"] and reg["regime_id"] != reg["v1_id"]
    cp = up["compute"]
    assert (cp["near_tie_game_count"], cp["near_tie_game_count_wide"]) == (3, 5)
    assert cp["near_tie_games"] is None and cp["outcome_digest"] is None


def test_a_bot_round_robin_v1_row_upgrades_to_balanced_seats_and_bot_native_temperatures():
    v1 = K.make_v1_row(item_key="botrr:random:heuristic", play="sampled", p_id="bot:random", o_id="bot:heuristic",
                       mirror_rule="gen3_mirrored_pairs_v1+seat_alternating_by_pair", voided=1)
    up = S.upgrade_v1(v1)
    assert up["regime"]["protocol"] == "gen3_eval_protocol_v1_bot_rr" and up["regime"]["seat_rule"] == "balanced"
    assert up["regime"]["player_temp"] == S.BOT_NATIVE_TEMP and up["player"]["kind"] == "bot"
    assert up["counts"]["aborted"] == 2, "aborted = 2 x voided"


def test_an_unknown_v1_writer_or_schema_is_refused_not_guessed():
    with pytest.raises(S.LedgerSchemaError, match="unknown writer"):
        S.upgrade_v1(K.make_v1_row(item_key="cycle:7"))
    with pytest.raises(S.LedgerSchemaError, match="v1 schema"):
        S.upgrade_v1({**K.make_v1_row(), "purpose": "ab"})          # `ab` was not a v1 purpose
    with pytest.raises(S.LedgerSchemaError, match="neither"):
        S.as_v2({**K.make_v1_row(), "schema": "gen3_eval_count_row_v0"})


# ------------------------------------------------------------------------------------------------ companions
def _event(kind, **f):
    e = {"schema": S.EVENT_SCHEMA, "event": kind, "seq": 1, "ts": "2026-10-03T12:00:00+00:00", "writer_id": "w"}
    e.update(f)
    return e


def test_companion_records_validate_and_refuse():
    fam = _event("family", family_id="x5", decision_kind="ab_verdict", rule="x5 §7.4 v1",
                 protocol="gen3_eval_protocol_v1_h2h", commit=None)
    assert S.validate_event(fam) == []
    assert any("group-sequential" in p for p in S.validate_event({**fam, "decision_kind": "promotion"}))
    assert any("not in" in p for p in S.validate_event({**fam, "event": "edit"}))
    d = {"schema": S.DECISION_SCHEMA, "decision_id": "w:d0", "kind": "ab_verdict", "subject": "x5", "request_id": None,
         "family": "x5", "consumed": {"digest": "0" * 64, "count": 1, "row_ids": ["w:0"]},
         "as_of": "2026-10-03T12:00:00+00:00", "rule": "r", "rule_version": "1", "verdict": "CONTINUE",
         "ts": "2026-10-03T12:00:00+00:00", "writer_id": "w"}
    assert S.validate_decision(d) == []
    assert any("count" in p for p in S.validate_decision({**d, "consumed": {**d["consumed"], "count": 2}}))
    assert any("names the request_id or the family" in p for p in S.validate_decision({**d, "family": None}))
    ref = {"schema": S.REFERENCE_SCHEMA, "reference_id": "w:r0", "members": [{"id": "a", "sha256": K.SHA_A},
                                                                             {"id": "b", "sha256": K.SHA_B}],
           "weights": [0.25, 0.75], "solver": "maxent_nash", "solver_version": "1", "draws": 200, "seed": 1,
           "rows_digest": "0" * 64, "created_at": "2026-10-03T12:00:00+00:00", "writer_id": "w"}
    assert S.validate_reference(ref) == []
    assert any("sum" in p for p in S.validate_reference({**ref, "weights": [0.25, 0.7]}))
