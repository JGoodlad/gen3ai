import numpy as np

from agents.inference.belief_decode import decode_species_belief, BELIEF_TOPK


def _names():
    """A 10-species synthetic vocab (num -> name)."""
    return {i: f"sp{i}" for i in range(10)}


def test_decodes_only_believed_slots_topk_sorted():
    logits = np.zeros((6, 10), dtype=np.float32)
    # slot 2 (believed): species 7 >> 3 >> 1
    logits[2, 7] = 5.0; logits[2, 3] = 3.0; logits[2, 1] = 1.0
    # slot 4 (believed): species 9 >> 8
    logits[4, 9] = 4.0; logits[4, 8] = 2.0
    # slot 0 is REVEALED but has a strong logit — must NOT be decoded.
    logits[0, 5] = 9.0
    mask = np.array([False, False, True, False, True, False])

    out = decode_species_belief(logits, mask, top_k=3, num_to_name=_names())

    assert [e["slot"] for e in out] == [2, 4]                      # believed slots only, in order
    assert [t["species"] for t in out[0]["top"]] == ["sp7", "sp3", "sp1"]
    assert len(out[1]["top"]) == 3                                  # top_k entries per slot
    assert [t["species"] for t in out[1]["top"]][:2] == ["sp9", "sp8"]  # the two non-zero rank first

    # Probabilities are a descending softmax posterior, formatted as a percent string.
    p = [float(t["prob"].rstrip("%")) for t in out[0]["top"]]
    assert p == sorted(p, reverse=True)
    assert all(t["prob"].endswith("%") for t in out[0]["top"])


def test_empty_when_nothing_believed():
    logits = np.zeros((6, 10), dtype=np.float32)
    mask = np.zeros(6, dtype=bool)
    assert decode_species_belief(logits, mask, num_to_name=_names()) == []


def test_topk_clamped_to_vocab_and_documented_default():
    logits = np.zeros((6, 2), dtype=np.float32)
    logits[1, 0] = 1.0
    mask = np.array([False, True, False, False, False, False])
    out = decode_species_belief(logits, mask, top_k=5, num_to_name={0: "a", 1: "b"})
    assert len(out[0]["top"]) == 2     # clamped to the vocab size
    assert BELIEF_TOPK == 3            # the documented "2 or 3 most likely" default


def test_unknown_num_falls_back_to_placeholder():
    logits = np.zeros((6, 4), dtype=np.float32)
    logits[0, 3] = 5.0
    mask = np.array([True, False, False, False, False, False])
    out = decode_species_belief(logits, mask, top_k=1, num_to_name={0: "a"})  # 3 not in map
    assert out[0]["top"][0]["species"] == "num_3"


def test_default_num_map_round_trips_real_species():
    """With no injected map, the decoder resolves the real gen3 species id for a num — and that id
    maps back to the same num (the inverse of the labels' species_to_num)."""
    from agents.gen3_data import species

    logits = np.zeros((6, 400), dtype=np.float32)
    logits[2, 1] = 10.0                       # national-dex num 1
    mask = np.array([False, False, True, False, False, False])
    out = decode_species_belief(logits, mask, top_k=1)
    name = out[0]["top"][0]["species"]
    assert species.species_data(name).num == 1
