"""gen3_event_window_v1 (Tier H-B) — the event-window CONTRACT the model reads, and the seat consumer.

The row's column contract (`constants.EventCol`, its one-hot groups, the plain-int mirror), the cant /
status vocabularies the rows carry, and the EventSeats module's build/mask contract (OFF builds
nothing; PAD rows get zero attention weight by key-mask; ON forward reads the block). The Python FOLD
that produced the rows (`EventWindowTracker`) and its tests are deleted with the Python battle layer
(T27 P6 slice 6d-2): the rows are the Rust core's (`src/rust_sim/src/trackers/`, pinned by
`tests/tracker_semantics_test.rs` and `tests/window_record_test.rs`), and `rust_core_obs_layout_test.py`
holds `EventCol` equal to the Rust encoder's columns.
"""
import numpy as np
import pytest
import torch

from agents.observation.constants import (
    EVENT_T_MOVE, EVENT_TOKEN_DIM, EVENT_WINDOW_DIM, EVENT_WINDOW_N, OFFSET_EVENT_WINDOW,
    EVENT_EFF_GROUP, EVENT_OUTCOME_GROUP, EventCol as C,
)


def test_event_seats_off_builds_nothing_on_reads_block():
    pytest.importorskip("sb3_contrib")
    from agents.model.identity_init_test import _build_real_policy

    off, _ = _build_real_policy()
    assert not any("history_events" in k for k in off.policy.state_dict())
    on, enc = _build_real_policy(history_events=True)
    fe = on.policy.features_extractor
    assert any("history_events.proj" in k for k in on.policy.state_dict())
    # PAD rows are key-masked; a real row is not
    ev = torch.zeros(2, EVENT_WINDOW_N, EVENT_TOKEN_DIM)
    ev[:, -1, C.TYPE] = EVENT_T_MOVE
    ev[:, -1, C.VALID] = 1.0
    tokens, pad = fe.history_events(ev, fe.embeddings)
    assert tokens.shape == (2, EVENT_WINDOW_N, tokens.shape[-1])
    assert bool(pad[:, :-1].all()) and not bool(pad[:, -1].any())
    # and the full forward moves when the block content changes
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.random((2, enc.dimension), dtype=np.float32))
    o1 = {"observation": obs.clone(), "action_mask": torch.ones(2, 11)}
    obs2 = obs.clone()
    sl = slice(OFFSET_EVENT_WINDOW, OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM)
    obs2[:, sl] = 0.0
    obs2[:, OFFSET_EVENT_WINDOW + C.VALID] = 1.0     # one valid pad-ish row, different content
    o2 = {"observation": obs2, "action_mask": torch.ones(2, 11)}
    with torch.no_grad():
        pi1, vf1 = fe(o1)
        pi2, vf2 = fe(o2)
    assert not torch.allclose(pi1, pi2) or not torch.allclose(vf1, vf2)


def test_pre_floor_config_is_refused():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90 (ARCH_SIGNATURE bumped), so a
    pre-floor config is REFUSED rather than migrated — the floor's stated purpose: "refuses
    pre-floor configs outright instead of walking dead branches". This asserts the behaviour
    that is now true rather than propping up a branch nothing can reach."""
    from agents.model.model_version import ModelVersionError, _migrate_config
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 80})


# ---------------------------------------------------------------------------
# gen3_frame_deletion_v1 — EVENT_T_CANT, the lag frames' one unsubstituted fact
# ---------------------------------------------------------------------------

def test_cant_reason_ids_are_distinct_and_zero_means_none():
    """Distinct reasons must get distinct ids, and 0 must be reserved for 'not a cant row'.

    A collapse here is silent: the row still exists, the column still reads nonzero, and the model
    simply cannot tell paralysis from sleep. (The sibling status mapping DID collapse this way
    during the frame-deletion work — a Status enum's `.value` is an int, so every status hashed to
    one id — which is why this is asserted rather than assumed.)"""
    from agents.observation.gen3_effects import CANT_REASONS, cant_reason_id
    assert cant_reason_id(None) == 0
    ids = {r: cant_reason_id(r) for r in CANT_REASONS}
    assert 0 not in ids.values(), "0 is reserved for 'no cant' and must not collide with a reason"
    assert len(set(ids.values())) == len(CANT_REASONS), f"reason ids collide: {ids}"


def test_the_status_seat_table_covers_the_producer_vocabulary():
    """The clamp-sweep contract at the sixth site the model-wide sweep did not reach.

    `EventSeats.status_emb` was `Embedding(8)` with a literal `clamp(max=7)` — a WIDTH and a
    CLAMP both spelled as bare numbers, with the producer's vocabulary living in a different
    package. Gen-3's status set is closed, so the spare rows are a genuine no-op safety net;
    what was missing is the relationship that makes a grown vocabulary FAIL instead of clamping
    a new id onto `tox`."""
    from agents.model.team_transformer import EventSeats
    from agents.observation.constants import EVENT_STATUS_IDS, N_EVENT_STATUS

    assert N_EVENT_STATUS == max(EVENT_STATUS_IDS.values()) + 1
    assert N_EVENT_STATUS <= EventSeats._STATUS_ROWS
    seats = EventSeats({"event_window_n": 4, "species_embedding_dim": 8,
                        "move_embedding_dim": 8})
    assert seats.status_emb.num_embeddings == EventSeats._STATUS_ROWS
    # every live id addresses its OWN row — none of them is the clamp target
    assert max(EVENT_STATUS_IDS.values()) < seats.status_emb.num_embeddings - 1


def test_the_archive_cant_vocabulary_is_FROZEN():
    """`CANT_REASONS` sizes TURN_DELTA_DIM (159) — the lag-frame width 79 archived runs recorded.

    The frames are deleted from the live obs, so `TurnDeltaEncoder` survives only as the prober's
    decoder for that archive. Growing `CANT_REASONS` would shift every offset after the cant block
    and make it mis-slice historical data — silently, since it would still return a plausible
    dict. New reasons go in `CANT_REASONS_LIVE`. This test is what makes that split enforced
    rather than merely intended."""
    from agents.observation.gen3_effects import CANT_REASONS, CANT_DIM, CANT_REASONS_LIVE
    from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
    assert CANT_DIM == 12 and TURN_DELTA_DIM == 159, "the ARCHIVE format moved — the prober will mis-slice"
    assert CANT_REASONS_LIVE[:len(CANT_REASONS)] == CANT_REASONS, "live must EXTEND the archive, never reorder it"
    assert "damp" in CANT_REASONS_LIVE and "damp" not in CANT_REASONS


# ---------------------------------------------------------------------------
# gen3_event_col_names_v1 — the 22-column contract as a NAMED declaration
# ---------------------------------------------------------------------------
# `EventCol` is imported by BOTH `state_encoder` (the producer) and
# `team_transformer.EventSeats` (the consumer), plus the feature-coverage probes and every
# oracle. Before it existed the contract was a comment plus ~30 bare integers spread over five
# files, which is the shape the positional-binding sweep convicted five times: two ends bound to
# the same subject by POSITION, with nothing relating them. These tests are what make the
# declaration load-bearing rather than decorative.


def test_event_column_map_tiles_the_token_exactly():
    """The members must TILE `range(EVENT_TOKEN_DIM)` — no gaps, no overlaps, no overflow.

    A gap is a column nobody writes and the consumer still slices (reading a constant 0 as a
    feature); an overlap is two facts sharing an address, which reads as whichever wrote last.
    Neither raises anywhere on its own — the token is a flat float row."""
    values = [int(c) for c in C]
    assert len(set(values)) == len(values), f"EventCol has DUPLICATE column indices: {values}"
    assert sorted(values) == list(range(EVENT_TOKEN_DIM)), (
        f"EventCol must cover 0..{EVENT_TOKEN_DIM - 1} exactly, got {sorted(values)} — "
        "widening the token means adding a member AND bumping EVENT_TOKEN_DIM")


def test_the_two_one_hot_groups_are_contiguous_and_in_order():
    """Both groups are written by INDEXING, so their order and contiguity are the contract.

    `state_encoder` writes the effectiveness one-hot as `EFF_NEUTRAL + eff` where `eff` is the
    TurnDelta effectiveness code (0 neutral / 1 super / 2 resist / 3 immune), and `EventSeats`
    takes the raw scalars as one `MAGNITUDE..WE_FIRST` slice that spans both groups. Reorder a
    member and the producer writes 'resisted' where the consumer reads 'super effective' — a
    pure relabelling with no shape change, so nothing else would notice."""
    for group, name in ((EVENT_OUTCOME_GROUP, "EVENT_OUTCOME_GROUP"),
                        (EVENT_EFF_GROUP, "EVENT_EFF_GROUP")):
        cols = [int(c) for c in group]
        assert cols == list(range(cols[0], cols[0] + len(cols))), (
            f"{name} must be CONTIGUOUS and ascending (it is indexed as base+offset): {cols}")
    assert [int(c) for c in EVENT_EFF_GROUP] == [C.EFF_NEUTRAL, C.EFF_SUPER,
                                                 C.EFF_RESIST, C.EFF_IMMUNE], (
        "the eff order is the TurnDelta effectiveness code order — reordering silently "
        "relabels every historical row")
    # The consumer's scalar run spans MAGNITUDE..WE_FIRST inclusive; both one-hots sit inside it,
    # which is why EventSeats can take three slices instead of thirteen columns.
    assert C.MAGNITUDE < C.OUT_HIT and C.EFF_IMMUNE < C.WE_FIRST


def test_the_consumer_imports_the_SAME_declaration():
    """Not "agree on the numbers" — the SAME object.

    A module holding its own copy of the map would pass every value assertion above right up until
    one of them was edited. The consumer (`team_transformer.EventSeats`) reads the `EVENT_COL` plain-int
    mirror, itself generated from `EventCol` (see
    `test_the_plain_int_mirror_agrees_with_the_enum_member_for_member`). The PRODUCER is the Rust
    encoder since T27 P6 slice 6d-2; its columns are held equal to `EventCol` by
    `rust_core_obs_layout_test.py::test_the_event_columns_are_equal`."""
    from agents.observation.constants import EVENT_COL
    from agents.model import team_transformer as _consumer
    assert _consumer.EVENT_COL is EVENT_COL


def test_event_seats_scalar_count_matches_the_column_map():
    """`EventSeats._N_SCALARS` is a WEIGHT SHAPE (it sizes `proj`'s input), and the column map
    is what it is supposed to count. A column that changes routing — id ↔ raw scalar — moves
    that width, and a stale `_N_SCALARS` builds a Linear of the wrong size against a `torch.cat`
    of the right one, which raises far from the cause."""
    from agents.model.team_transformer import EventSeats
    ids = {C.TYPE, C.ACTOR_SPECIES, C.TARGET_SPECIES, C.MOVE, C.STATUS,
           C.CANT, C.FAINT_CAUSE, C.ITEM_TRANSITION,
           # gen3_event_record_v2 (E12)
           C.REL_SPECIES, C.ENTRY, C.DENIAL, C.CALLER, C.STAT}
    scalars = set(C) - ids - {C.VALID}
    assert EventSeats._N_SCALARS == len(scalars), (
        f"EventSeats._N_SCALARS={EventSeats._N_SCALARS} but the column map says {len(scalars)} "
        f"raw scalars ({sorted(int(c) for c in scalars)})")


def test_the_plain_int_mirror_agrees_with_the_enum_member_for_member():
    """`EVENT_COL` is a generated mirror, so this is a tautology TODAY — its value is that it
    stays one. Both live consumers read the MIRROR, not the enum, for reasons that have nothing
    to do with the contract (`state_encoder` because an enum member's `LOAD_ATTR` costs ~36% of
    the obs write loop; `team_transformer` because `torch.fx` renders an enum member as its repr
    `<EventCol.TYPE: 0>` into generated graph code, which is a SyntaxError at compile). A future
    edit that hand-writes an entry into the mirror instead of deriving it fails here."""
    from agents.observation.constants import EVENT_COL
    mirror = {k: v for k, v in vars(EVENT_COL).items() if not k.startswith("_")}
    assert mirror == {c.name: int(c) for c in C}
    assert all(type(v) is int for v in mirror.values()), (
        "the mirror must hold PLAIN ints — an IntEnum member here defeats both of its purposes")
