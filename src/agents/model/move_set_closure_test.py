"""`gen3_move_set_closure_v1` (config v154, `--move-set-closure`) — the FOUR-MOVE FACT in the opponent move belief.

A gen-3 mon has at most four moves, so once four are revealed every other move's presence is 0. X5's fixed-mass
construction (k = 4 − revealed) already gives every consumer of the opponent ACTIVE's moves and the op's per-mon
roster that fact; the one leak (F-X5-33) was MoveBelief's REINJECTION on every other slot, which soft-embeds the
moves by their independent sigmoid inclusion weights — so a benched mon with four revealed moves still reinjected
believed others (a trained screen final carries a summed sigmoid of ~2.4 on them;
`designs/research_state/measurements/belief_closure_2026-10-10/`).

The decisive check is behavioural, on constructed boards (the committed real parity rows with a mon's four move
slots written): bump the published posterior's logit of every UNREVEALED move of a four-revealed mon and read the
policy's masked log-probabilities and the value.

* `on`: bit-identical, for a BENCHED and for the ACTIVE four-revealed mon (reverting the reinjection to the sigmoid
  weights fails the bench case);
* `off` (production): the active case is bit-identical too (the move group), the bench case MOVES — the leak this
  flag closes, pinned so the test cannot pass vacuously (a model whose reinjection reads nothing would pass `on`).

Plus the construction (`slot_move_presence(..., graph=True)`): the same values as the detached default, mass
4 − r, exactly 0 off the revealed moves at r = 4, and a gradient into the move logits (the PPO route the sigmoid
rows had); the build refusal without the belief family; and the versioning (migration, the compat gate).
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Optional, Tuple

import pytest
import torch

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.hypothesis_set import HypothesisBuilder, MOVE_SET_CLOSURE_MODES
from agents.model.hypothesis_tokens import slot_move_presence
from agents.model.parity_probe import perturb_
from agents.model.probe_facts_test import _policy
from agents.observation.constants import TEAM_SIZE
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM

BUMP = 6.0
# A real leak moves the value / log-probabilities by ~1e-2 on these boards (measured 2026-10-10: max |Δv| 1.9e-2 on
# the fixture battles at this perturbation); the sensitivity bar sits far below that and far above fp32 noise.
LEAK_BAR = 1e-4


@pytest.fixture(scope="module")
def layout() -> dict:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


@pytest.fixture(scope="module")
def builder(layout: dict) -> HypothesisBuilder:
    return HypothesisBuilder(layout, global_input_dim=50, n_move_seats=6)


# ----------------------------------------------------------------------------------------------- the construction

def _legal_moves(hb: HypothesisBuilder, species: int, n: int) -> list:
    legal = hb.move_legal[species] & hb.move_valid
    nums = [int(m) for m in torch.nonzero(legal).flatten()
            if int(m) != HIDDEN_POWER_MOVE_NUM and not 355 <= int(m) <= 370]
    assert len(nums) >= n, f"species {species} has {len(nums)} plain legal moves"
    return nums[:n]


def _species(hb: HypothesisBuilder, k: int) -> int:
    """The k-th species num (> 0) with at least 12 plain legal moves."""
    rows = (hb.move_legal & hb.move_valid.unsqueeze(0)).sum(-1)
    nums = [int(s) for s in torch.nonzero(rows >= 12).flatten() if int(s) > 0]
    return nums[k]


def test_the_graph_presence_equals_the_detached_one_and_closes_at_four(builder: HypothesisBuilder) -> None:
    """Same VALUES as the op roster's detached presence; Σ over the unrevealed candidates = 4 − r; exactly 0 on every
    unrevealed move at r = 4; 1 on each revealed move. Reverting the mass to the sigmoid weights fails the sums."""
    hb = builder
    sp = [_species(hb, i) for i in range(6)]
    revealed = torch.zeros(1, TEAM_SIZE, 4, dtype=torch.long)
    for j, r in enumerate((0, 1, 2, 3, 4, 4)):
        if r:
            revealed[0, j, :r] = torch.tensor(_legal_moves(hb, sp[j], r))
    g = torch.Generator().manual_seed(7)
    logits = (torch.randn(1, TEAM_SIZE, hb.n_moves, generator=g) * 2.0).requires_grad_(True)
    species = torch.tensor([sp])
    w_g, rev, _sel = slot_move_presence(hb, logits, species, revealed, graph=True)
    w = w_g.detach()
    w_d, _r, _s = slot_move_presence(hb, logits, species, revealed)
    assert torch.equal(w, w_d), "the graph variant must carry the SAME values"
    num = torch.arange(hb.n_moves)
    unrev = ~rev[0] & (num != HIDDEN_POWER_MOVE_NUM)                                  # [6,M]
    for j, r in enumerate((0, 1, 2, 3, 4, 4)):
        assert float(w[0, j][rev[0, j]].sum()) == float(r)                         # revealed pinned at 1
        mass = float(w[0, j][unrev[j]].sum())
        if r == 4:
            assert mass == 0.0, f"slot {j}: four revealed moves, yet {mass} mass on the others"
        else:
            assert abs(mass - (4 - r)) < 1e-4, f"slot {j}: mass {mass} != 4 - {r}"
    # the gradient reaches the move logits on a live slot (the PPO route the sigmoid rows had), never at r = 4
    (w_g * torch.randn(w_g.shape, generator=g)).sum().backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 0].abs().sum()) > 0.0
    assert float(logits.grad[0, 4][unrev[4]].abs().sum()) == 0.0


# -------------------------------------------------------------------------------------- the behavioural invariance

def _bumper(fe: Any) -> Tuple[Callable[[float], None], Callable[[], None]]:
    """Wrap the instance's `move_logits` so a set bump is added to every UNREVEALED move (num >= 1) of every slot
    with four revealed moves. Returns (set_bump, restore)."""
    mb = fe.move_belief
    orig = mb.move_logits
    state = {"bump": 0.0}

    def bumped(opp_tokens: torch.Tensor, opp_species_ids: Optional[torch.Tensor] = None,
               opp_move_ids: Optional[torch.Tensor] = None, **kw: Any) -> torch.Tensor:
        out: torch.Tensor = orig(opp_tokens, opp_species_ids, opp_move_ids, **kw)
        if state["bump"] == 0.0 or opp_move_ids is None:
            return out
        M = out.shape[-1]
        rev = torch.zeros_like(out, dtype=torch.bool)
        rev.scatter_(-1, opp_move_ids.clamp(0, M - 1), opp_move_ids > 0)
        slot4 = ((opp_move_ids > 0).sum(-1) == 4).unsqueeze(-1)
        hit = slot4 & ~rev & (torch.arange(M) >= 1)
        return torch.where(hit, out + state["bump"], out)

    mb.move_logits = bumped

    def set_bump(b: float) -> None:
        state["bump"] = b

    def restore() -> None:
        del mb.move_logits
    return set_bump, restore


def _board(fe: Any, hb: HypothesisBuilder, layout: dict, x: torch.Tensor,
           active: bool) -> Tuple[torch.Tensor, list]:
    """The parity rows with ONE alive revealed opponent mon per row given four legal moves: the opponent ACTIVE
    (``active``) or a BENCHED one. Rows with no such mon are dropped (for ``active``, also a row where another mon
    already shows four revealed moves); returns the boards and the kept row indices."""
    with torch.no_grad():
        ctx = fe.unpack({"observation": x})
    p = layout["parts"]["opp_team"]
    per = p["reshape"][1]
    pk = layout["pokemon"]
    mo = pk["moves"]["offset"]
    id_off = pk["moves"]["layout"]["slot_layout"]["id"]["offset"]
    slots = [s["offset"] for s in pk["moves"]["layout"]["slots"]]
    out, kept = [], []
    for i in range(x.shape[0]):
        sp = ctx.species_ids[i, TEAM_SIZE:]
        alive = ctx.hp_and_active[i, TEAM_SIZE:, 0] > 0
        known = ~ctx.opp_believed_mask[i].bool() & (sp > 0) & alive
        act = int(ctx.opp_active_local[i])
        cands = [act] if active else [j for j in range(TEAM_SIZE) if j != act]
        js = [j for j in cands if bool(known[j])]
        # the ACTIVE case isolates the active: a row where another opponent mon already shows four revealed moves
        # (the bump would reach that bench mon too) is dropped
        four = (ctx.all_move_ids[i, TEAM_SIZE:] > 0).sum(-1) == 4
        if not js or (active and bool(four[[j for j in range(TEAM_SIZE) if j != act]].any())):
            continue
        j = js[0]
        row = x[i].clone()
        for k, m in enumerate(_legal_moves(hb, int(sp[j]), 4)):
            row[p["start"] + j * per + mo + slots[k] + id_off] = float(m)
        out.append(row)
        kept.append(i)
    assert len(out) >= 8, f"only {len(out)} constructed boards"
    return torch.stack(out), kept


@pytest.fixture(scope="module")
def perturbed() -> Any:
    """The production surface with `--move-set-closure on`, perturbed (name-keyed, the K9 golden's) so the
    zero-init heads — the pointer scorer, the move head's delta — carry signal."""
    pol = copy.deepcopy(_policy(move_set_closure="on"))
    perturb_(pol, seed=1234, scale=0.02, keyed=True)
    pol.set_training_mode(False)
    return pol


def _delta(pol: Any, x: torch.Tensor, mask: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    fe = pol.features_extractor
    fe.move_set_closure = mode
    set_bump, restore = _bumper(fe)
    try:
        o = {"observation": x}
        with torch.no_grad():
            set_bump(0.0)
            v0, l0 = pol.rollout_core(o, mask)
            dv, dl = torch.zeros_like(v0), torch.zeros_like(l0)
            for b in (BUMP, -BUMP):
                set_bump(b)
                v, lp = pol.rollout_core(o, mask)
                dv = torch.maximum(dv, (v - v0).abs())
                dl = torch.maximum(dl, (lp - l0).abs().masked_fill(~mask.bool(), 0.0))
    finally:
        restore()
        fe.move_set_closure = "on"
    return dv, dl


@pytest.fixture(scope="module")
def boards(perturbed: Any, builder: HypothesisBuilder, layout: dict) -> dict:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    obs, mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    x, m = torch.as_tensor(obs), torch.as_tensor(mask)
    fe = perturbed.features_extractor
    out = {}
    for name, active in (("bench", False), ("active", True)):
        b, keep = _board(fe, builder, layout, x, active)
        out[name] = (b, m[keep])
    return out


def test_on_a_benched_four_revealed_mon_reinjects_nothing_else(perturbed: Any, boards: dict) -> None:
    """THE fix: under `on`, the unrevealed moves of a benched mon with four revealed moves reach nothing the policy or
    value reads. Reverting the bench rows' reinjection weights to the sigmoid fails it."""
    x, m = boards["bench"]
    dv, dl = _delta(perturbed, x, m, "on")
    assert float(dv.max()) == 0.0 and float(dl.max()) == 0.0, (float(dv.max()), float(dl.max()))


def test_off_the_bench_leak_is_real(perturbed: Any, boards: dict) -> None:
    """Production (`off`): the same bump MOVES the outputs — the leak `on` closes (F-X5-33). Pins the test's
    sensitivity: were the bench rows unread, the `on` test above would pass vacuously."""
    x, m = boards["bench"]
    dv, dl = _delta(perturbed, x, m, "off")
    assert max(float(dv.max()), float(dl.max())) > LEAK_BAR


@pytest.mark.parametrize("mode", MOVE_SET_CLOSURE_MODES)
def test_the_active_was_already_closed(perturbed: Any, boards: dict, mode: str) -> None:
    """The opponent ACTIVE's moves go through the move group (k = 4 − r) in both modes: E4 / E5, OTHER_move, the op,
    the flat pointer and the active's reinjection read nothing of a four-revealed active's unrevealed moves."""
    x, m = boards["active"]
    dv, dl = _delta(perturbed, x, m, mode)
    assert float(dv.max()) == 0.0 and float(dl.max()) == 0.0, (float(dv.max()), float(dl.max()))


@pytest.fixture(scope="module")
def endstate_arm() -> Any:
    """The END-STATE arm as DECLARED (`main.train.arch_arms`'s `endstate` overlay over the production toggles: the static
    token encoding and every fact lever), perturbed like `perturbed`. Owner 2026-10-10 ("Do A"): the closure is in
    that arm, a pre-data amendment of the closing test (`design_endstate_closing_test.md` Decision record)."""
    from main.train.arch_arms import arm_overlay
    pol = copy.deepcopy(_policy(**arm_overlay("endstate")))
    perturb_(pol, seed=1234, scale=0.02, keyed=True)
    pol.set_training_mode(False)
    return pol


def test_the_endstate_arm_closes_the_bench_leak(endstate_arm: Any, boards: dict) -> None:
    """The arm BUILDS with the closure on (fails if `move_set_closure` leaves the `endstate` overlay), and under the
    arm's own encoding (static tokens) the bench case is closed: bit-identical under the bump. The same arm with the
    flag flipped off MOVES, so the reinjection is read under the static encoding too and the check is not vacuous."""
    fe = endstate_arm.features_extractor
    assert fe.move_set_closure == "on", "the declared `endstate` arm must build with --move-set-closure on"
    x, m = boards["bench"]
    dv, dl = _delta(endstate_arm, x, m, "on")
    assert float(dv.max()) == 0.0 and float(dl.max()) == 0.0, (float(dv.max()), float(dl.max()))
    dv, dl = _delta(endstate_arm, x, m, "off")
    assert max(float(dv.max()), float(dl.max())) > LEAK_BAR


# --------------------------------------------------------------------------------------------- build + versioning

def test_on_needs_the_belief_family() -> None:
    from agents.model.identity_init_test import _build_real_policy
    with pytest.raises(ValueError, match="move_set_closure='on' requires the opponent-belief family"):
        _build_real_policy(move_set_closure="on", opp_belief_slots=False, opp_intent=False)
    with pytest.raises(ValueError, match="move_set_closure must be one of"):
        _policy(move_set_closure="sometimes")


def test_versioning() -> None:
    import dataclasses
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    assert MODEL_CONFIG_VERSION >= 154
    assert _migrate_config({"config_version": 153})["move_set_closure"] == "off"
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    mv = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    assert mv.move_set_closure == "off"
    with pytest.raises(ModelVersionError, match="move_set_closure mismatch"):
        mv.check_compatible(dataclasses.replace(mv, move_set_closure="on"))
    rec = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {"move_set_closure": "on"}})
    assert rec.move_set_closure == "on"
