"""gen3_extractor_stashes_v1 — the OpStashes recipe applied to `Gen3FeaturesExtractor`.

The bug class this file pins (the v89 lesson made it concrete): phases communicated through
mutable `self.last_*` instance stashes and consumers read them with `getattr`, so nothing
type-level connected producer to consumer — a consumer rewiring silently orphaned five value
routes for two generations. The container gives three structural guarantees, each pinned here:

  * STALE CROSS-BATCH READS ARE UNREPRESENTABLE — `forward_internal` replaces the whole
    `ExtractorStashes` container at entry, so no stash (written or not this forward) can carry
    a previous batch (the `OpStashes` guarantee, extractor half).
  * STRAY WRITES FAIL LOUD — every `last_*` name is a read-only property; assigning it raises
    instead of silently forking the state away from the container.
  * ABSENT `layout` FAILS AT ENTRY — the Optional-only-for-SB3 default raises a named
    ValueError instead of a deep `'NoneType' object is not subscriptable` (task 4b).
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.features_extractor import ExtractorStashes, Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


@pytest.fixture(scope="module")
def fe_and_layout():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(3)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings)
    fe.eval()
    return fe, layout


def _obs(layout, batch=2, seed=11):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(batch, layout["total_dim"], generator=g)}


def test_stale_cross_batch_read_is_unrepresentable(fe_and_layout):
    """Poison the container — including a field this baseline config NEVER writes (alpha is not
    built) — and run a forward: EVERY field must reflect this forward, not the poison. The
    never-written field is the sharp half: the old per-stash reset convention protected only the
    stashes someone remembered to reset (the op's top-K trio famously had none)."""
    fe, layout = fe_and_layout
    sentinel = torch.full((1, 3), 7.0)
    fe.stash.value_pooled = sentinel          # a stale "previous batch" (writes go via stash)
    fe.stash.alpha_logits = sentinel          # a field NO module writes in this config
    fe.stash.belief_supervision["alpha_logits"] = sentinel
    before = fe.stash
    with torch.no_grad():
        fe(_obs(layout))
    assert fe.stash is not before                              # replaced as a UNIT at entry
    assert fe.last_value_pooled is not None and fe.last_value_pooled is not sentinel
    assert fe.last_alpha_logits is None                        # not merely overwritten: RESET
    assert "alpha_logits" not in fe._belief_supervision        # key absent ⇒ head did not run
    # the surface is live: the pointer stash was written this forward and the property sees it
    assert fe.last_pointer_inputs is fe.stash.pointer_inputs is not None


def test_stray_write_to_every_legacy_name_fails_loud(fe_and_layout):
    """Assigning ANY `last_*` name (or the private hand-off names) raises AttributeError —
    the write surface is `fe.stash.<field>` only. The name list is derived from the class's
    own property set, so a stash added tomorrow is covered without editing this test."""
    fe, _ = fe_and_layout
    props = [n for n in dir(type(fe))
             if n.startswith("last_") and isinstance(getattr(type(fe), n), property)]
    assert len(props) >= 18, props            # the swept inventory, not a token few
    for name in props + ["_thresh_probs", "_entity_latent_table", "_belief_supervision"]:
        assert isinstance(getattr(type(fe), name), property), name
        with pytest.raises(AttributeError):
            setattr(fe, name, None)


def test_container_defaults_are_all_none_or_empty():
    """A fresh container IS the reset state — `forward_internal`'s entry replacement relies on
    every field defaulting to None (and the supervision dict to empty), so a field added with a
    non-None default would silently weaken the stale-read guarantee."""
    s = ExtractorStashes()
    for f, v in vars(s).items():
        if f == "belief_supervision":
            assert v == {}
        else:
            assert v is None, f


def test_missing_layout_fails_loud_at_entry():
    """Task 4b: `layout=None` (the SB3-signature default) must raise a NAMED error at entry,
    not a deep TypeError from `Embeddings` indexing None."""
    space = gym.spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32)
    with pytest.raises(ValueError, match="requires layout"):
        Gen3FeaturesExtractor(space)
