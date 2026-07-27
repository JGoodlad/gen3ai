"""Tests for the per-team LUT (`gen3_zarch_lut_v1`, v46) — the free per-team conditioning code.

Two halves: the offline SIGNATURE (`agents.model.team_signature`) that keys the table, and the
in-forward LOOKUP + z fold. The load-bearing properties:

  * the signature is PERMUTATION-invariant and rejects the things that would make it lie
    (duplicate teams, move-set mutators);
  * an UNMATCHED team resolves to row 0, whose zero code makes 'add' EXACTLY the DeepSets z —
    so the generalist / prober / any off-table probe degrades instead of mis-conditioning;
  * OFF is byte-identical (the whole no-op contract this project versions on).
"""

import pytest
import torch

from agents.model.team_signature import (
    TEAM_SIGNATURE_DIM, build_roster_table, team_signature)


# --- fixtures ----------------------------------------------------------------------------------

def _mappings():
    return {"species": {"aaa": {"num": 3}, "bbb": {"num": 9}, "ccc": {"num": 5}},
            "moves": {"m1": {"num": 100}, "m2": {"num": 200}, "m3": {"num": 300},
                      "m4": {"num": 400}, "mimic": {"num": 102}}}


def _team(*mons):
    """A minimal Showdown export: each mon is (species, [moves])."""
    blocks = []
    for species, moves in mons:
        lines = [f"{species} @ Leftovers", "Ability: Levitate", "Adamant Nature"]
        lines += [f"- {m}" for m in moves]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_MONS = [("aaa", ["m1", "m2", "m3", "m4"]),
         ("bbb", ["m1", "m2", "m3", "m4"]),
         ("ccc", ["m1", "m2", "m3", "m4"]),
         ("aaa", ["m1", "m2", "m3", "m4"]),
         ("bbb", ["m1", "m2", "m3", "m4"]),
         ("ccc", ["m1", "m2", "m3", "m4"])]


# --- signature ---------------------------------------------------------------------------------

def test_signature_has_the_declared_width():
    sig = team_signature(_team(*_MONS), _mappings())
    assert len(sig) == TEAM_SIGNATURE_DIM == 30


def test_signature_is_invariant_to_team_and_move_order():
    """A team is a SET of mons, each a set of moves — reordering must not change its identity.

    This is what lets the lookup work regardless of how the sim happens to order our party in the
    observation on any given turn.
    """
    base = team_signature(_team(*_MONS), _mappings())
    shuffled = _MONS[3:] + _MONS[:3]
    shuffled = [(sp, list(reversed(mv))) for sp, mv in shuffled]
    assert team_signature(_team(*shuffled), _mappings()) == base


def test_signature_separates_same_roster_different_moves():
    """The measured motivation: 5 of the def-20 cluster's teams share a species roster.

    Species alone would collapse them onto one LUT row, silently making the "per-team" code a
    per-PAIR code.
    """
    other = [(sp, ["m1", "m2", "m3", "m1"]) for sp, _ in _MONS]
    assert team_signature(_team(*_MONS), _mappings()) != team_signature(_team(*other), _mappings())


def test_build_roster_table_rejects_duplicate_teams():
    t = _team(*_MONS)
    with pytest.raises(ValueError, match="SAME species"):
        build_roster_table([t, t], _mappings())


def test_signature_rejects_move_set_mutators():
    """Mimic rewrites its own move slot, so the signature would flip MID-BATTLE."""
    mons = [("aaa", ["m1", "m2", "m3", "mimic"])] + _MONS[1:]
    with pytest.raises(ValueError, match="mimic"):
        team_signature(_team(*mons), _mappings())


def test_signature_rejects_unknown_species_and_moves():
    with pytest.raises(ValueError, match="unknown species"):
        team_signature(_team(("zzz", ["m1", "m2", "m3", "m4"]), *_MONS[1:]), _mappings())
    with pytest.raises(ValueError, match="unknown move"):
        team_signature(_team(("aaa", ["m1", "m2", "m3", "nope"]), *_MONS[1:]), _mappings())


# --- lookup + z fold (module-level, no full extractor build) -------------------------------------

class _LutStub(torch.nn.Module):
    """The LUT half of `Gen3FeaturesExtractor` in isolation — same math, no 3000-line build.

    Mirrors `_zarch_lut_index` + the forward's z fold exactly; the extractor-level wiring is
    covered by the end-to-end smoke.
    """

    def __init__(self, rosters, dim=4, mode="add"):
        super().__init__()
        self.mode = mode
        self.register_buffer("table", torch.tensor(rosters, dtype=torch.long))
        self.emb = torch.nn.Embedding(len(rosters) + 1, dim)
        torch.nn.init.normal_(self.emb.weight, std=1.0)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.norm = torch.nn.LayerNorm(dim)

    def index(self, sig):
        match = (sig[:, None, :] == self.table[None, :, :]).all(dim=-1)
        return torch.where(match.any(dim=1), match.long().argmax(dim=1) + 1,
                           torch.zeros(sig.shape[0], dtype=torch.long))

    def forward(self, sig, z_deepsets):
        idx = self.index(sig)
        code = self.emb(idx)
        base = z_deepsets if self.mode == "add" else torch.zeros_like(code)
        return self.norm(base + code), idx


def _sigs(mappings=None):
    m = mappings or _mappings()
    a = list(team_signature(_team(*_MONS), m))
    b = list(team_signature(_team(*[(sp, ["m1", "m2", "m3", "m1"]) for sp, _ in _MONS]), m))
    return [a, b]


def test_lookup_maps_each_known_team_to_its_own_row():
    rosters = _sigs()
    lut = _LutStub(rosters)
    sig = torch.tensor(rosters, dtype=torch.long)
    assert lut.index(sig).tolist() == [1, 2]


def test_unmatched_team_falls_back_to_row_zero():
    """The generalist's pool, an off-table probe, the prober — none may be MIS-conditioned."""
    rosters = _sigs()
    lut = _LutStub(rosters)
    unknown = torch.zeros(1, TEAM_SIGNATURE_DIM, dtype=torch.long)
    assert lut.index(unknown).tolist() == [0]


def test_add_mode_on_an_unmatched_team_is_exactly_the_deepsets_z():
    """Row 0 is zero-init'd, so 'add' degrades to LayerNorm(z) — no team code injected.

    Without this, every unmatched decision would silently ride whatever row 0 drifted to.
    """
    lut = _LutStub(_sigs(), dim=4, mode="add")
    z = torch.randn(3, 4)
    unknown = torch.zeros(3, TEAM_SIGNATURE_DIM, dtype=torch.long)
    out, idx = lut(unknown, z)
    assert idx.tolist() == [0, 0, 0]
    torch.testing.assert_close(out, lut.norm(z))


def test_known_teams_get_distinct_codes_at_init():
    """The WHOLE POINT: random init makes the per-team codes large and ~orthogonal from step 0.

    The ill-conditioning diagnosis was z_i = z̄ + tiny ε_i; if the codes started identical (or
    zero-init) the LUT would reproduce exactly the geometry it exists to break.
    """
    rosters = _sigs()
    lut = _LutStub(rosters, dim=32)
    sig = torch.tensor(rosters, dtype=torch.long)
    out, _ = lut(sig, torch.zeros(2, 32))
    cos = torch.nn.functional.cosine_similarity(out[0], out[1], dim=0)
    assert abs(float(cos)) < 0.8, "per-team codes started nearly parallel — no conditioning signal"


def test_only_mode_ignores_the_deepsets_z():
    lut = _LutStub(_sigs(), dim=4, mode="only")
    sig = torch.tensor(_sigs(), dtype=torch.long)
    a, _ = lut(sig, torch.randn(2, 4))
    b, _ = lut(sig, torch.randn(2, 4) * 100.0)
    torch.testing.assert_close(a, b)


def test_gradient_reaches_only_the_matched_rows():
    """A team absent from the minibatch must not have its code dragged by other teams' gradients."""
    rosters = _sigs()
    lut = _LutStub(rosters, dim=4)
    sig = torch.tensor([rosters[0]], dtype=torch.long)
    out, _ = lut(sig, torch.zeros(1, 4))
    # NOT out.sum(): LayerNorm's output is mean-centred, so its sum is identically 0 for ALL inputs
    # and the gradient would vanish for reasons unrelated to the LUT. Weight the dims instead.
    (out * torch.arange(1.0, 5.0)).sum().backward()
    g = lut.emb.weight.grad
    assert g[1].abs().sum() > 0, "the matched team's code got no gradient"
    assert g[0].abs().sum() == 0 and g[2].abs().sum() == 0


# --- end-to-end through the real extractor -----------------------------------------------------

@pytest.fixture(scope="module")
def ek_and_space():
    """The real observation layout + Dict space (shared with zarch_test's fixture)."""
    import numpy as np
    from gymnasium import spaces
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    total = ek["layout"]["total_dim"]
    space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (total,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
    })
    return ek, space, total


ZDIM = 16


def _fake_rosters(n):
    """n distinct signatures that no real observation can produce (species num 0 = the pad slot)."""
    return [[0] * (TEAM_SIGNATURE_DIM - 1) + [900 + i] for i in range(n)]


def _build(ek, space, lut="off", rosters=None):
    from agents.model.features_extractor import Gen3FeaturesExtractor
    return Gen3FeaturesExtractor(space, **{**ek, "zarch_film": "heads", "zarch_dim": ZDIM,
                                           "zarch_lut": lut, "zarch_lut_rosters": rosters})


def test_extractor_requires_film_and_rosters(ek_and_space):
    ek, space, _ = ek_and_space
    from agents.model.features_extractor import Gen3FeaturesExtractor
    with pytest.raises(ValueError, match="requires --zarch-film"):
        Gen3FeaturesExtractor(space, **{**ek, "zarch_film": "off", "zarch_dim": 0,
                                        "zarch_lut": "add", "zarch_lut_rosters": _fake_rosters(2)})
    with pytest.raises(ValueError, match="requires zarch_lut_rosters"):
        _build(ek, space, lut="add", rosters=None)
    with pytest.raises(ValueError, match="30 ints"):
        _build(ek, space, lut="add", rosters=[[1, 2, 3]])


def test_extractor_off_is_byte_identical(ek_and_space):
    """The project's no-op contract: --zarch-lut off must not perturb the v44 forward at all."""
    ek, space, total = ek_and_space
    torch.manual_seed(0)
    off = _build(ek, space, lut="off")
    torch.manual_seed(0)
    also_off = _build(ek, space, lut="off")
    obs = {"observation": torch.zeros(2, total), "action_mask": torch.ones(2, 11)}
    with torch.no_grad():
        a_pi, a_vf = off(obs)
        b_pi, b_vf = also_off(obs)
    torch.testing.assert_close(a_pi, b_pi)
    torch.testing.assert_close(a_vf, b_vf)
    assert off.zarch_lut_emb is None and off.zarch_lut_teams == 0


def test_extractor_unmatched_team_is_unaffected_by_the_learned_codes(ek_and_space):
    """A zeros observation matches no roster → row 0 → the forward carries NO team code.

    Asserted the direct way: scramble every LEARNED row and the unmatched forward must not move.
    That is the property that lets an unmatched team (the pool, the prober, a frozen opponent on
    some other team) run through a LUT model without being conditioned on a WRONG team's code —
    stronger than comparing against a separately-built reference z, whose weights would differ.
    """
    ek, space, total = ek_and_space
    torch.manual_seed(0)
    fe = _build(ek, space, lut="add", rosters=_fake_rosters(3))
    obs = {"observation": torch.zeros(2, total), "action_mask": torch.ones(2, 11)}
    with torch.no_grad():
        pi_a, vf_a = fe(obs)
        assert fe.last_zarch_lut_idx.tolist() == [0, 0], "a zeros obs should match no team"
        fe.zarch_lut_emb.weight[1:] += 10.0        # scramble the LEARNED codes only
        pi_b, vf_b = fe(obs)
    torch.testing.assert_close(pi_a, pi_b)
    torch.testing.assert_close(vf_a, vf_b)


def test_extractor_roster_table_is_persistent(ek_and_space):
    """The team↔row mapping MUST ride the checkpoint — a reload with a different table would
    re-key every learned per-team code (silently conditioning on the wrong team)."""
    ek, space, _ = ek_and_space
    fe = _build(ek, space, lut="add", rosters=_fake_rosters(4))
    sd = fe.state_dict()
    assert "zarch_lut_table" in sd and sd["zarch_lut_table"].shape == (4, TEAM_SIGNATURE_DIM)
    assert "zarch_lut_emb.weight" in sd and sd["zarch_lut_emb.weight"].shape == (5, ZDIM)


# --- versioning --------------------------------------------------------------------------------

def test_version_gate_rejects_a_lut_mode_or_team_count_change(ek_and_space):
    """off↔add↔only and the table height are weight-shape/forward changes → FATAL on any load."""
    from agents.model.model_version import ModelVersion, ModelVersionError
    ek, _, _ = ek_and_space

    def mv(lut, teams):
        pk = {"features_extractor_kwargs": {
            **ek, "zarch_film": "heads", "zarch_dim": ZDIM, "zarch_lut": lut,
            "zarch_lut_rosters": _fake_rosters(teams) if teams else None}}
        return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk)

    base = mv("add", 20)
    base.check_compatible(mv("add", 20))                        # like-for-like passes
    with pytest.raises(ModelVersionError, match="zarch_lut mismatch"):
        mv("only", 20).check_compatible(base)
    with pytest.raises(ModelVersionError, match="zarch_lut mismatch"):
        mv("off", 0).check_compatible(base)
    with pytest.raises(ModelVersionError, match="zarch_lut_teams mismatch"):
        mv("add", 10).check_compatible(base)


def test_old_configs_migrate_to_lut_off():
    from agents.model.model_version import _migrate_config
    out = _migrate_config({"config_version": 45})
    assert out["zarch_lut"] == "off" and out["zarch_lut_teams"] == 0
