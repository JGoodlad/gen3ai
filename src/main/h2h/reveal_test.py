"""The PER-SIDE oracle reveal's rules (``main.h2h.reveal``): which (p1, p2) levels a cell plays at under each mode,
the typed refusals, the row stamp, the protocol per mode, and the spec's per-side form. Pure: no engine, no game
(the real games: ``play_reveal_integration_test.py``; the Rust rows: ``rust_env/tests/oracle_reveal_test.rs``)."""
from __future__ import annotations

import json

import pytest

from agents.training import eval_ledger as L
from main.h2h import play as PL
from main.h2h import reveal as RV
from utils.rust_env import protocol as P

TS = "ts:" + "0" * 16


@pytest.mark.parametrize("mode,player,opponent,want", [
    ("off", "off", "off", ("off", "off")),
    ("one_sided", "species", "off", ("species", "off")),
    ("one_sided", "off", "species", ("off", "species")),
    ("one_sided", "full", "off", ("full", "off")),
    ("one_sided", "off", "full", ("off", "full")),
    ("both_sided", "species", "off", ("species", "species")),
    ("both_sided", "off", "full", ("full", "full")),
    # any information-lever spec the observation honours plays as its recorded LABEL (design_information_levers.md §6)
    ("one_sided", "moves:seen@0.95", "off", ("moves:seen@0.95", "off")),
    ("both_sided", "off", "item:seen@0.95", ("item:seen@0.95", "item:seen@0.95")),
    # withheld: the oracle side plays BELOW its recorded level, by declaration — (off, off) in either seat
    ("withheld", "species", "off", ("off", "off")),
    ("withheld", "off", "full", ("off", "off")),
    ("withheld", "moves:seen@0.95", "off", ("off", "off")),
    ("withheld", "off", "item:seen@0.95", ("off", "off")),
])
def test_the_cell_levels_follow_the_mode_and_the_recorded_levels(mode, player, opponent, want):
    assert RV.side_levels(mode, player, opponent) == want
    RV.check_side_levels(want, mode, player, opponent)          # what the mode requires is accepted


@pytest.mark.parametrize("mode,player,opponent,match", [
    ("off", "species", "off", "one_sided"),             # an oracle checkpoint under off: told which mode to declare
    ("off", "off", "full", "one_sided"),
    ("one_sided", "off", "off", "needs an ORACLE"),     # a reveal mode with no oracle side
    ("both_sided", "off", "off", "needs an ORACLE"),
    ("one_sided", "species", "full", "both sides are oracle"),
    ("both_sided", "full", "full", "both sides are oracle"),
    ("withheld", "off", "off", "needs an ORACLE"),      # nothing to withhold
    ("withheld", "species", "full", "both sides are oracle"),
    ("withheld", "moves:seen@0.95", "moves:seen@0.95", "both sides are oracle"),
])
def test_a_mode_inconsistent_with_the_checkpoints_is_a_typed_refusal(mode, player, opponent, match):
    with pytest.raises(RV.RevealModeError, match=match):
        RV.side_levels(mode, player, opponent)


def test_an_unknown_mode_is_refused():
    with pytest.raises(RV.RevealModeError, match="not one of"):
        RV.side_levels("sideways", "species", "off")


@pytest.mark.parametrize("levels,mode,player,opponent,match", [
    (("full", "off"), "one_sided", "species", "off", "trained under --oracle-reveal species"),   # told more
    (("off", "off"), "one_sided", "species", "off", "trained under --oracle-reveal species"),    # told nothing
    (("off", "species"), "one_sided", "off", "full", "trained under --oracle-reveal full"),       # the p2 oracle
    (("species", "species"), "one_sided", "species", "off", "not what mode one_sided requires"),  # non-oracle told
    (("species", "off"), "both_sided", "species", "off", "not what mode both_sided requires"),    # non-oracle not told
    (("species", "full"), "both_sided", "species", "off", "not what mode both_sided requires"),
])
def test_a_side_at_the_wrong_level_is_refused(levels, mode, player, opponent, match):
    with pytest.raises(RV.RevealLevelMismatch, match=match):
        RV.check_side_levels(levels, mode, player, opponent)


def test_each_mode_is_its_own_closed_list_protocol_and_off_is_the_h2h_protocol_unchanged():
    assert RV.PROTOCOL_OF_MODE["off"] == PL.PROTOCOL == "gen3_eval_protocol_v1_h2h"
    assert set(RV.PROTOCOL_OF_MODE.values()) <= set(L.PROTOCOLS)
    assert len(set(RV.PROTOCOL_OF_MODE.values())) == len(RV.MODES)
    regimes = {m: PL.regime_for(300, TS, mode=m) for m in RV.MODES}
    assert regimes["off"] == PL.regime_for(300, TS), "the default regime IS off's"
    assert len({r["regime_id"] for r in regimes.values()}) == 4, "the modes never pool: four regimes"
    for m, r in regimes.items():
        assert r["protocol"] == RV.PROTOCOL_OF_MODE[m]
        assert {k: v for k, v in r.items() if k not in ("protocol", "regime_id")} == \
            {k: v for k, v in regimes["off"].items() if k not in ("protocol", "regime_id")}, "only the protocol differs"


def test_the_row_stamp_is_absent_under_off_and_names_both_levels_otherwise():
    assert RV.row_block("off", RV.OFF_OFF) is None
    assert RV.row_block("one_sided", ("off", "full")) == {"mode": "one_sided", "player": "off", "opponent": "full"}


def test_the_spec_writes_a_symmetric_reveal_as_the_string_and_a_split_one_as_the_pair():
    kw = dict(n=1, threads=1, teams=["X"], names=("a", "b"), turn_limit=300, refusal_budget=0, bank_dir=None)
    assert json.loads(P.spec_json(**kw))["oracle_reveal"] == "off"
    assert P.spec_json(**kw) == P.spec_json(**kw, oracle_reveal=("off", "off")), "off/off is the spec of before"
    assert json.loads(P.spec_json(**kw, oracle_reveal="species"))["oracle_reveal"] == "species"
    assert json.loads(P.spec_json(**kw, oracle_reveal=("species", "off")))["oracle_reveal"] == ["species", "off"]
    assert json.loads(P.spec_json(**kw, oracle_reveal=("full", "full")))["oracle_reveal"] == "full"
    for bad in ("x", ("off",), ("off", "x"), ("off", "off", "off"), "ki_wrongmoves", ("hp:seen", "off")):
        with pytest.raises(ValueError, match="oracle_reveal"):
            P.spec_json(**kw, oracle_reveal=bad)
    # a spec label round-trips per side; a level given as its clause text is written as the level
    assert json.loads(P.spec_json(**kw, oracle_reveal=("moves:seen@0.95", "off")))["oracle_reveal"] == ["moves:seen@0.95", "off"]
    assert P.spec_json(**kw, oracle_reveal="species:unseen") == P.spec_json(**kw, oracle_reveal="species")


def test_a_recorded_spec_label_is_the_level_the_engine_plays_a_side_at(tmp_path):
    """The h2h engine reads each checkpoint's RECORDED label (`recorded_level`) and plays its side at it: a run that
    recorded a knock-in spec is one_sided-playable at exactly that label."""
    import dataclasses

    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import save_model_snapshot
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    v = ModelVersion.from_layout_and_policy_kwargs(Gen3ObservationEncoder(load_mappings()).get_layout(),
                                                   {"net_arch": [512, 512]})
    (tmp_path / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(tmp_path), dataclasses.replace(v, oracle_reveal="moves:seen@0.95"), git_hash="t")

    class Ref:
        zip_path = str(tmp_path / "final_model.zip")

    rec = RV.recorded_level(Ref())
    assert rec == "moves:seen@0.95"
    assert RV.side_levels("one_sided", rec, "off") == ("moves:seen@0.95", "off")
    with pytest.raises(RV.RevealModeError):
        RV.side_levels("off", rec, "off")


# ---------------------------------------------------------------------------------------------- the withheld mode
#: The knock-in fork's recorded level (``ki_moves``): the cell the honest-play cross plays.
ORACLE = "moves:seen@0.95"


def test_off_still_refuses_an_oracle_checkpoint_and_names_withheld_as_the_declared_exception():
    """The refusal is the DEFAULT and stays: `withheld` is the explicit exception, not a loosening of `off`."""
    for player, opponent in ((ORACLE, "off"), ("off", ORACLE), ("species", "off"), ("off", "full")):
        with pytest.raises(RV.RevealModeError, match="withheld"):
            RV.side_levels("off", player, opponent)
        # ... and the level check refuses it too: (off, off) is NOT accepted for an oracle cell under `off`
        with pytest.raises(RV.RevealModeError, match="never trained on"):
            RV.check_side_levels(RV.OFF_OFF, "off", player, opponent)


def test_withheld_plays_the_oracle_side_at_off_in_either_seat_and_accepts_exactly_that():
    for player, opponent in ((ORACLE, "off"), ("off", ORACLE)):
        assert RV.side_levels("withheld", player, opponent) == RV.OFF_OFF
        RV.check_side_levels(RV.OFF_OFF, "withheld", player, opponent)


def test_off_off_is_accepted_for_an_oracle_cell_only_under_withheld():
    for mode in RV.MODES:
        for player, opponent in ((ORACLE, "off"), ("off", ORACLE)):
            if mode == "withheld":
                RV.check_side_levels(RV.OFF_OFF, mode, player, opponent)
            elif mode == "off":
                with pytest.raises(RV.RevealModeError):
                    RV.check_side_levels(RV.OFF_OFF, mode, player, opponent)
            else:
                with pytest.raises(RV.RevealLevelMismatch, match="would play at 'off'"):
                    RV.check_side_levels(RV.OFF_OFF, mode, player, opponent)


def test_withheld_refuses_a_cell_with_no_oracle_side():
    """Nothing to withhold: the row would carry a withheld stamp over a plain-vs-plain play, indistinguishable by its
    observations from an `off` row yet pooled with no one."""
    with pytest.raises(RV.RevealModeError, match="needs an ORACLE"):
        RV.side_levels("withheld", "off", "off")
    with pytest.raises(RV.RevealModeError, match="needs an ORACLE"):
        RV.check_side_levels(RV.OFF_OFF, "withheld", "off", "off")


def test_withheld_refuses_oracle_against_oracle_and_says_why():
    for player, opponent in ((ORACLE, ORACLE), (ORACLE, "species"), ("full", ORACLE)):
        with pytest.raises(RV.RevealModeError, match="both sides are oracle.*ONE withheld level"):
            RV.side_levels("withheld", player, opponent)


@pytest.mark.parametrize("levels", [(ORACLE, "off"), ("off", ORACLE), ("species", "off"), (ORACLE, ORACLE)])
def test_a_withheld_cell_never_tells_a_side_anything(levels):
    with pytest.raises(RV.RevealLevelMismatch, match="plays EVERY side at off by declaration"):
        RV.check_side_levels(levels, "withheld", ORACLE, "off")


def test_the_withheld_protocol_is_its_own_regime_and_no_other_modes():
    pr = RV.PROTOCOL_OF_MODE["withheld"]
    assert pr == "gen3_eval_protocol_v1_h2h_oracle_withheld" == PL.PROTOCOL_WITHHELD
    assert pr in L.PROTOCOLS
    assert pr not in {v for m, v in RV.PROTOCOL_OF_MODE.items() if m != "withheld"}
    withheld, off = PL.regime_for(300, TS, mode="withheld"), PL.regime_for(300, TS)
    assert withheld["protocol"] == pr and withheld["regime_id"] != off["regime_id"]
    for other in ("one_sided", "both_sided"):
        assert withheld["regime_id"] != PL.regime_for(300, TS, mode=other)["regime_id"]


def _stamp_for():
    """A real diagnostic-fork stamp (``main.train.diagnostic_fork.decide``) for a fork INTO ``ORACLE``."""
    from main.train import diagnostic_fork as DF

    return DF.decide(consent="knock-in: does the network convert told moves?", model_path="/m/rb_parent/final_model.zip",
                     model_dir="/m/rb_child", requested=ORACLE, saved="off", now="2026-10-14T00:00:00+00:00")


def _run(root, name, level, lineage=None, metadata_reveal=None):
    """A run dir with a zip, a ``model_config.json`` recording ``level`` and, with ``lineage``, a ``metadata.json``;
    ``metadata_reveal`` writes metadata's own per-checkpoint ``oracle_reveal`` / ``oracle_reveal_record`` (a fork's
    carries its PARENT's: `off`)."""
    run = root / name
    run.mkdir()
    (run / "final_model.zip").write_bytes(b"zip")
    (run / "model_config.json").write_text(json.dumps({"oracle_reveal": level}))
    meta = {}
    if lineage is not None:
        meta["lineage"] = lineage
    if metadata_reveal is not None:
        meta["oracle_reveal"] = metadata_reveal
        meta["oracle_reveal_record"] = {"label": metadata_reveal, "hash": "stale"}
    if meta:
        (run / "metadata.json").write_text(json.dumps(meta))
    return run


class Ref:
    """What `oracle_side` reads of a `PlayerRef`: the zip path and the id."""

    def __init__(self, run):
        self.zip_path, self.id = str(run / "final_model.zip"), run.name


def test_the_withheld_stamp_carries_the_recorded_level_the_hash_and_the_consent(tmp_path):
    from agents.model.oracle_reveal import record

    stamp = _stamp_for()
    # the fork's metadata.json says `off` (its extractor's mode: the PARENT's); model_config.json says the fork's level
    fork = _run(tmp_path, "rb_hib_moves_s3101", ORACLE,
                lineage={"schema": 1, "role": "fork", "fork_parent": None, "diagnostic_fork": stamp},
                metadata_reveal="off")
    plain = _run(tmp_path, "rb_hib_ctrl_s3101", "off")
    for player, opponent, side in ((Ref(fork), Ref(plain), "player"), (Ref(plain), Ref(fork), "opponent")):
        levels = (RV.recorded_level(player), RV.recorded_level(opponent))
        assert RV.side_levels("withheld", *levels) == RV.OFF_OFF
        oracle = RV.cell_oracle("withheld", player, opponent, *levels)
        assert RV.row_block("withheld", RV.OFF_OFF, oracle) == {
            "mode": "withheld", "player": "off", "opponent": "off",
            "withheld_level": ORACLE,                                    # the level it recorded and did NOT play at
            "oracle": {"side": side, "run": "rb_hib_moves_s3101", "level": ORACLE, "canonical": ORACLE,
                       "reveal_hash": record(ORACLE)["hash"], "diagnostic_fork": True,
                       "fork_consent": {"run": "rb_hib_moves_s3101", "self": True, "schema": stamp["schema"],
                                        "reason": stamp["reason"], "date": stamp["date"],
                                        "parent_spec_hash": stamp["parent_spec"]["hash"],
                                        "child_spec_hash": record(ORACLE)["hash"]}}}
    assert record(ORACLE)["hash"] == "b82fd2fa64fc67e8", "the hash the real knock-in forks' consent records"


def test_a_withheld_stamp_without_the_oracle_provenance_is_refused():
    with pytest.raises(RV.RevealModeError, match="withheld_level"):
        RV.row_block("withheld", RV.OFF_OFF)


def test_one_sided_and_both_sided_rows_carry_the_oracle_provenance_too(tmp_path):
    fresh = _run(tmp_path, "rb_oracle_fresh", "species")            # trained FRESH under the reveal: no fork consent
    plain = _run(tmp_path, "rb_plain", "off")
    for mode, levels in (("one_sided", ("species", "off")), ("both_sided", ("species", "species"))):
        oracle = RV.cell_oracle(mode, Ref(fresh), Ref(plain), "species", "off")
        block = RV.row_block(mode, levels, oracle)
        assert block["mode"] == mode and (block["player"], block["opponent"]) == levels
        assert "withheld_level" not in block, "the key is the withheld mode's alone"
        assert block["oracle"]["level"] == "species" and block["oracle"]["canonical"] == "species:unseen"
        assert block["oracle"]["diagnostic_fork"] is False and block["oracle"]["fork_consent"] is None
    assert RV.cell_oracle("off", Ref(plain), Ref(plain), "off", "off") is None
    assert RV.row_block("one_sided", ("species", "off")) == {"mode": "one_sided", "player": "species",
                                                             "opponent": "off"}, "no provenance given: the old shape"


def test_a_plain_fork_of_a_consented_fork_stamps_the_ancestors_consent(tmp_path):
    from agents.training.lineage import build_lineage

    stamp = _stamp_for()
    parent = _run(tmp_path, "rb_hib_moves_s3101", ORACLE,
                  lineage={"schema": 1, "role": "fork", "fork_parent": None, "diagnostic_fork": stamp})
    block = build_lineage(model_path=str(parent / "final_model.zip"), model_dir=str(tmp_path / "rb_grand"))
    grand = _run(tmp_path, "rb_grand", ORACLE, lineage=block)
    oracle = RV.oracle_side(Ref(grand), "player", ORACLE)
    assert oracle["diagnostic_fork"] is True and oracle["run"] == "rb_grand"
    assert oracle["fork_consent"]["run"] == "rb_hib_moves_s3101" and oracle["fork_consent"]["self"] is False


def test_a_lineage_that_contradicts_the_recorded_level_is_refused(tmp_path):
    stamp = _stamp_for()                                            # consents a fork INTO moves:seen@0.95 ...
    run = _run(tmp_path, "rb_hib_odd", "item:seen@0.95",            # ... but the run records another level
               lineage={"schema": 1, "role": "fork", "fork_parent": None, "diagnostic_fork": stamp})
    with pytest.raises(RV.RevealProvenanceMismatch, match="names the child spec 'moves:seen@0.95'"):
        RV.oracle_side(Ref(run), "player", "item:seen@0.95")
