"""gen3_move_resolution_x5_v1 — the move-resolution family under X5 (production + `--move-resolution on`; X5's
hypothesis tokens are the only belief representation since the version break, config v144).

The family's RULES (`move_facts` / `switch_facts`) hold no belief-mode branch: under fixed_mass they read the flat
opponent pointer's re-expression, with OTHER_move a PRICED seat set (one per priority level) and OTHER_species a PRICED 7th mon (the
OTHER decision, `designs/endstate/design_arch_audit.md` Decision record 2026-10-07; X5 design §3.7 / §9 M3 (c)).
What must hold, each failing on revert:

  * **the same rules on identical inputs** — a one-member OTHER_move tail prices as naming that move as a
    seat, and an OTHER_species column at β = 0 changes nothing, so the X5 read is the blob read plus OTHER's columns;
    duplicated seats / mons at half the mass are invariant (the I2 property of a linear α / β contraction);
  * **OTHER is priced, never excluded** — its KO mass reaches `p_ko_us` / `dbond_p_ko`, its ORDER is the tail's exact
    marginal (not the priority-0 residual, not the tail's mean priority), a Ghost / non-Ghost tail decides
    `p_lands_switch`;
  * **a real fixed_mass forward** runs with the family on, gathers K + 1 seats and 7 mons, and reads each hidden
    slot as its hypothesis and OTHER_species as the renormalised tail;
  * **the one-lever init property under fixed_mass** on a real `MaskablePPO` policy;
  * **composition** with `--token-encoding static` and `--policy-readout trunk`;
  * a CPU `--debug` smoke under `--arch production --move-resolution on` (`slow`).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import torch

from agents import gen3_data
from agents.model.move_resolution import MoveResolutionOps, move_facts, split_other_move, switch_facts
from agents.model.move_resolution_extractor_test import SEVEN, _build, _prod_kwargs
from agents.model.move_resolution_rules import MOVE_RESOLUTION_MOVE_IDX as MI, PRIORITY_MAX, PRIORITY_MIN
from agents.model.move_resolution_tables import build_priority_table
from agents.model.move_resolution_test import FLAG_T, KIND_T, N_MOVES, SEAT_T, T, _move, num, ops
from utils.paths import src_path

PHYS_T = torch.zeros(N_MOVES)
PRIO = torch.zeros(N_MOVES)
for _mid in gen3_data.moves.raw():
    _md = gen3_data.moves.get(_mid)
    if _md is not None and 0 <= int(_md.num) < N_MOVES:
        PHYS_T[int(_md.num)] = _move(_mid)["phys"]
        PRIO[int(_md.num)] = float(_md.priority)
TABLES = {"prio_w": build_priority_table(PRIO), "kind": SEAT_T, "flag": FLAG_T, "phys": PHYS_T,
          "flinch": torch.zeros(N_MOVES)}
L = PRIORITY_MAX - PRIORITY_MIN + 1


def _tail(**mass: float) -> torch.Tensor:
    u = torch.zeros(1, N_MOVES)
    for mid, w in mass.items():
        u[0, num(mid)] = w
    return u


def as_other_move(o: MoveResolutionOps, u: torch.Tensor) -> MoveResolutionOps:
    """``o`` with its LAST seat re-expressed as X5's OTHER_move over the tail ``u`` — through `split_other_move`, the
    function `gather_ops` calls under fixed_mass (one seat per priority level)."""
    return split_other_move(o, u, TABLES, torch.ones(1))


def with_other_species(o: MoveResolutionOps, beta_other: float = 0.0, ghost: bool = False) -> MoveResolutionOps:
    """``o`` with a 7th mon column (X5's OTHER_species) at β mass ``beta_other`` (taken from slot 1)."""
    def add(t: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, row], dim=1)
    imm = torch.zeros(1, 1, 19)
    g = torch.Generator().manual_seed(7)
    if ghost:
        imm[0, 0, T["NORMAL"]] = imm[0, 0, T["FIGHTING"]] = 1.0
    beta = o.beta.clone()
    beta[0, 1] -= beta_other
    return o._replace(
        beta=torch.cat([beta, torch.tensor([[beta_other]])], dim=1),
        imm_dmg=add(o.imm_dmg, imm), chart0=add(o.chart0, imm), p_type=add(o.p_type, torch.rand(1, 1, 19, generator=g)),
        abl_block=add(o.abl_block, torch.zeros(1, 1, o.abl_block.shape[-1])),
        opp_named_abl=add(o.opp_named_abl, torch.zeros(1, 1, o.opp_named_abl.shape[-1])),
        opp_cond=add(o.opp_cond, torch.zeros(1, 1, 7)), opp_rest=add(o.opp_rest, torch.zeros(1, 1)),
        opp_alive=add(o.opp_alive, torch.ones(1, 1)),
        out_cells=add(o.out_cells.transpose(1, 2), torch.rand(1, 1, 4, 5, generator=g)).transpose(1, 2))


def facts(o: MoveResolutionOps):
    return move_facts(o, KIND_T, FLAG_T), switch_facts(o)


def _grid(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(1, 6, 3, 14, generator=g) * 0.5
    p[..., 4] = 0.9
    return p


# ------------------------------------------------------------------------- the same rules, identical inputs
@pytest.mark.parametrize("moves", [("bodyslam", "protect", "counter", "thunderwave"),
                                   ("magiccoat", "focuspunch", "destinybond", "substitute")])
@pytest.mark.parametrize("member", ["protect", "quickattack", "earthquake", "magiccoat", "taunt", "toxic", "roar"])
def test_a_one_member_tail_prices_exactly_as_naming_that_move(moves, member):
    named = ops(moves, seats=("earthquake", "growl", member), alpha=(0.3, 0.2, 0.25), pair_in=_grid(),
                p_out=torch.tensor([[0.4]]))
    other = as_other_move(named, _tail(**{member: 1.0}))
    assert other.alpha.shape[-1] == 2 + L
    for a, b in zip(facts(named), facts(other)):
        assert torch.allclose(a, b, atol=1e-6, rtol=0.0)


def test_an_other_species_column_at_zero_beta_changes_nothing():
    o = ops(("bodyslam", "thunderwave", "rapidspin", "pursuit"), seats=("earthquake", "growl", "protect"),
            alpha=(0.3, 0.2, 0.2), a_switch=0.3, pair_in=_grid(1))
    ref = facts(o)
    got = facts(with_other_species(o, 0.0))
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-6, rtol=0.0)


def test_two_half_copies_of_a_seat_or_a_mon_equal_one_whole():
    """I2 for the family's two contractions: α over the seat axis and β over the mon axis are linear, so a seat (or a
    mon) split into two identical columns at half the mass reads the same — no fixed seat / mon count is assumed."""
    o = ops(("bodyslam", "thunderwave", "counter", "rapidspin"), seats=("earthquake", "growl", "protect"),
            alpha=(0.3, 0.2, 0.2), a_switch=0.3, pair_in=_grid(2))
    o = with_other_species(o, 0.2, ghost=True)
    ref = facts(o)

    def dup(t: torch.Tensor, ax: int, i: int, half: bool = False) -> torch.Tensor:
        col = t.narrow(ax, i, 1)
        if half:
            col = col / 2
            t = torch.cat([t.narrow(ax, 0, i), col, t.narrow(ax, i + 1, t.shape[ax] - i - 1)], dim=ax)
        return torch.cat([t, col], dim=ax)
    s = o._replace(alpha=dup(o.alpha, 1, 0, half=True), seat_nums=dup(o.seat_nums, 1, 0),
                   seat_prio=dup(o.seat_prio, 1, 0), seat_kind=dup(o.seat_kind, 1, 0),
                   seat_flag=dup(o.seat_flag, 1, 0), seat_phys=dup(o.seat_phys, 1, 0),
                   seat_flinch=dup(o.seat_flinch, 1, 0), pair_in=dup(o.pair_in, 2, 0),
                   pair_type_mult=dup(o.pair_type_mult, 2, 0), d_burn_k=dup(o.d_burn_k, 1, 0),
                   d_slp_k=dup(o.d_slp_k, 1, 0))
    m = s._replace(beta=dup(s.beta, 1, 6, half=True), imm_dmg=dup(s.imm_dmg, 1, 6), chart0=dup(s.chart0, 1, 6),
                   p_type=dup(s.p_type, 1, 6), abl_block=dup(s.abl_block, 1, 6),
                   opp_named_abl=dup(s.opp_named_abl, 1, 6), opp_cond=dup(s.opp_cond, 1, 6),
                   opp_rest=dup(s.opp_rest, 1, 6), opp_alive=dup(s.opp_alive, 1, 6),
                   out_cells=dup(s.out_cells, 2, 6))
    for a, b in zip(ref, facts(m)):
        assert torch.allclose(a, b, atol=1e-6, rtol=0.0)


# ------------------------------------------------------------------------------ OTHER is PRICED
def _other_ko(ko: float) -> Dict[str, Any]:
    g = torch.zeros(1, 6, 3, 14)
    g[0, 0, 2, 3] = ko                      # OTHER_move (the last seat) KOs our active with P = ko
    g[0, 0, 2, 4] = 1.0
    g[0, 0, 2, 1] = 1.0
    return {"pair_in": g}


def test_other_moves_ko_mass_reaches_the_destiny_bond_feature():
    o = as_other_move(ops(("destinybond",), seats=("growl", "growl", "growl"), alpha=(0.1, 0.1, 0.4),
                          **_other_ko(1.0)), _tail(earthquake=0.6, surf=0.4))
    assert mf_(o, "dbond_p_ko") == pytest.approx(0.4)
    assert mf_(o, "p_ko_us") == pytest.approx(0.4)


def test_other_moves_order_is_the_tails_exact_expectation():
    """Half the tail is Protect (+3), half Earthquake (0); we outspeed (p_out = 1) with a priority-0 move: OTHER acts
    first with probability exactly 0.5. The priority-0 residual would say 0, the tail's MEAN priority (1.5) 1."""
    o = as_other_move(ops(("bodyslam",), seats=("growl", "growl", "growl"), alpha=(0.1, 0.1, 0.4),
                          p_out=torch.tensor([[1.0]]), **_other_ko(1.0)), _tail(protect=0.5, earthquake=0.5))
    assert mf_(o, "p_ko_first") == pytest.approx(0.4 * 0.5)
    # ORDER × KIND is the exact joint: the tail's Protect member always moves first (stall odds 1), so it blocks our
    # move with probability 0.5 — not E[first] · E[protect] = 0.25 (what one tail-averaged seat would say).
    blk = as_other_move(ops(("bodyslam",), seats=("growl", "growl", "growl"), alpha=(0.0, 0.0, 1.0), a_switch=0.0,
                            p_out=torch.tensor([[1.0]])), _tail(protect=0.5, earthquake=0.5))
    assert mf_(blk, "p_lands_stay") == pytest.approx(0.5)
    # a slower Quick Attack member still moves first; a Counter member never does
    qa = as_other_move(ops(("bodyslam",), seats=("growl", "growl", "growl"), alpha=(0.0, 0.0, 1.0), a_switch=0.0,
                           p_out=torch.tensor([[1.0]]), **_other_ko(1.0)), _tail(quickattack=0.3, counter=0.7))
    assert mf_(qa, "p_ko_first") == pytest.approx(0.3)


def test_other_species_decides_the_switch_branch():
    base = ops(("bodyslam",), alpha=(0.0, 0.0, 0.0), a_switch=1.0)
    for ghost, want in ((False, 1.0), (True, 0.0)):
        o = with_other_species(base, 1.0, ghost=ghost)
        o = o._replace(beta=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
        assert mf_(o, "p_lands_switch") == pytest.approx(want)
        assert mf_(o) == pytest.approx(want)


def mf_(o: MoveResolutionOps, coord: str = "p_resolve", slot: int = 0) -> float:
    return float(move_facts(o, KIND_T, FLAG_T)[0, slot, MI[coord]])


# ------------------------------------------------------------------------- a REAL fixed_mass forward
@pytest.fixture(scope="module")
def obs() -> torch.Tensor:
    return torch.as_tensor(np.load(src_path("agents", "model", "compile_parity_obs.npz"))["obs"])


def _capture(fe: Any, x: torch.Tensor):
    cell = fe.move_resolution_cell
    cap: Dict[str, Any] = {}
    orig = cell.forward

    def hook(o):
        cap["ops"] = o
        return orig(o)
    cell.forward = hook
    try:
        with torch.no_grad():
            fe({"observation": x})
    finally:
        cell.forward = orig
    return cap["ops"]


def test_fixed_mass_forward_runs_with_the_family_on(obs):
    fe = _build(move_resolution="on")
    assert fe.move_resolution_cell is not None and fe.flat_intent_head is not None
    o = _capture(fe, obs)
    pi = fe.last_pointer_inputs
    assert pi.move_cells.shape[2] == fe.pointer_move_cell_dim
    assert pi.switch_cells.shape[2] == fe.pointer_switch_cell_dim
    assert torch.isfinite(pi.move_cells).all() and torch.isfinite(pi.switch_cells).all()
    K = fe.entity_topk_seats
    assert o.alpha.shape[-1] == K + L and o.beta.shape[-1] == 7 and o.out_cells.shape[2] == 7
    assert o.pair_in.shape[2] == K + L and o.seat_kind.shape[1] == K + L
    x5 = fe.stash.flat_consumer_ops
    assert float(o.alpha[:, K:].sum(-1).max()) > 0.0                # OTHER_move carries mass on the bank
    assert torch.equal(o.pair_in[:, :, :K + 1], x5.pair_in)          # the named seats + OTHER's column, as given
    for lvl in range(1, L):                                         # every level shares OTHER's damage column
        assert torch.equal(o.pair_in[:, :, K + lvl], x5.pair_in[:, :, K])
    # α + α_SWITCH + the masked-seat mass is the flat pointer's whole distribution
    assert float((o.alpha.sum(-1) + o.a_switch[:, 0]).max()) <= 1.0 + 1e-6
    # β is the flat pointer's switch distribution INCLUDING OTHER_species (priced, never dropped)
    has = torch.isfinite(x5.beta).any(-1, keepdim=True).float()
    want_beta = torch.softmax(x5.beta.float().clamp(min=-1e9), dim=-1) * has
    assert torch.allclose(o.beta, want_beta, atol=1e-6, rtol=0.0)
    assert float(o.beta[:, 6].max()) > 0.0


def test_fixed_mass_gathers_each_hypothesis_and_the_tail(obs):
    fe = _build(move_resolution="on")
    o = _capture(fe, obs)
    hs = fe.last_hypothesis
    cell = fe.move_resolution_cell
    x5 = fe.stash.flat_consumer_ops
    # OTHER_species: the believed-slot read with the renormalised tail as its species distribution
    assert torch.allclose(o.imm_dmg[:, 6], hs.other_tail_probs @ cell.SPECIES_P_IMM, atol=1e-6, rtol=0.0)
    assert torch.allclose(o.p_type[:, 6], hs.other_tail_probs @ cell.SPECIES_HAS_TYPE, atol=1e-6, rtol=0.0)
    assert float(o.opp_cond[:, 6].abs().max()) == 0.0
    # a hidden slot: its HYPOTHESIS species, exactly (the roster's one-hot), never the blob's usage posterior
    hyp = hs.slot_is_hypothesis
    assert bool(hyp.any())
    want = cell.SPECIES_P_IMM[hs.slot_species]                       # [B,6,19]
    assert torch.allclose(o.imm_dmg[:, :6][hyp], want[hyp], atol=1e-6, rtol=0.0)
    # OTHER_move: its α is split over the priority levels by the tail's mass, and the levels' tables average back
    # to the tail's (the law of total expectation): Σ_l P(l) · E[flag | l] = E_tail[flag]
    K = fe.entity_topk_seats
    M = x5.other_u.shape[-1]
    p_l = x5.other_u @ cell.PRIO_W[:M]                                               # [B,L]
    a_other = torch.softmax(x5.alpha.float(), dim=-1)[:, K] * x5.seat_live[:, K]
    assert torch.allclose(o.alpha[:, K:], a_other[:, None] * p_l, atol=1e-6, rtol=0.0)
    assert torch.equal(o.seat_prio[0, K:], torch.arange(PRIORITY_MIN, PRIORITY_MAX + 1).to(o.seat_prio.dtype))
    assert torch.allclose((p_l[:, :, None] * o.seat_flag[:, K:]).sum(1), x5.other_u @ cell.FLAG[:M],
                          atol=1e-5, rtol=0.0)
    assert torch.allclose((p_l[:, :, None] * o.seat_kind[:, K:]).sum(1), x5.other_u @ cell.SEAT_KIND[:M],
                          atol=1e-5, rtol=0.0)


# --------------------------------------------------------------- the one-lever init + composition (policies)
def _policy(**over: Any):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.identity_init_test import _Env
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    enc = Gen3ObservationEncoder(load_mappings())
    kw = {**enc.get_features_extractor_kwargs(), **_prod_kwargs(**over)}
    torch.manual_seed(0)
    return MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw, "net_arch": dict(pi=[64], vf=[64])}).policy


def test_the_one_lever_init_property_under_fixed_mass():
    off = _policy()
    on = _policy(move_resolution="on")
    fe = on.features_extractor
    assert all(getattr(fe, n) is None for n in SEVEN)
    assert fe.alpha_head is None and fe.beta_head is None           # X5 retires α / β; the family reads the pointer
    for p in fe.move_resolution_cell.parameters():
        assert float(p.detach().abs().max()) == 0.0
    a, b = off.state_dict(), on.state_dict()
    shared = [k for k in b if k in a and not k.startswith("pointer_head.")]
    assert len(shared) > 100
    moved = [k for k in shared if not torch.equal(a[k], b[k])]
    assert not moved, f"the family moved other parameters' initial bytes under fixed_mass: {moved[:5]}"
    only_off = {k.split(".")[1] for k in a if k not in b and k.startswith("features_extractor.")}
    assert only_off == set(SEVEN)
    only_on = {k.split(".")[1] for k in b if k not in a and k.startswith("features_extractor.")}
    assert only_on == {"move_resolution_cell"}
    opt = {id(p) for g in on.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt for p in fe.move_resolution_cell.parameters())


@pytest.mark.parametrize("over", [{"token_encoding": "static"}, {"policy_readout": "trunk"},
                                  {"token_encoding": "static", "policy_readout": "trunk"}],
                         ids=["static", "trunk", "static+trunk"])
def test_composes_with_static_tokens_and_the_trunk_readout(obs, over):
    pol = _policy(move_resolution="on", **over)
    fe = pol.features_extractor
    assert fe.move_resolution_cell is not None and all(getattr(fe, n) is None for n in SEVEN)
    x = {"observation": obs[:16]}
    with torch.no_grad():
        dist = pol.get_distribution(x, action_masks=np.ones((16, 11), dtype=bool))
        v = pol.predict_values(x)
    assert torch.isfinite(dist.distribution.logits).all() and torch.isfinite(v).all()
    o = _capture(fe, obs[:16])
    assert o.alpha.shape[-1] == fe.entity_topk_seats + L and o.beta.shape[-1] == 7
    # the family's cell rides the pointer head at the widened width
    assert fe.last_pointer_inputs.move_cells.shape[2] == pol.pointer_head.move_cell_dim


# ------------------------------------------------------------------------------------- the CPU smoke
@pytest.mark.slow
def test_debug_smoke_under_fixed_mass_with_the_family_on(tmp_path):
    """The root CLAUDE.md smoke on the production architecture's fixed_mass arm with the family on: the Rust core
    collects through the inference service (the family in every decision forward), the learner runs updates (the K6
    freeze, K9's checks, the declared selection sites at run time), the model saves and round-trips.

    `--arch production` brings the production RECIPE, whose update fires at 98,304 completed-game rows (hours on CPU),
    so the smoke types a small shape: `--rollout-target-samples 2304` (a multiple of lcm(batch 384, n_envs) at parse
    time and of the debug quantum 384 at run time), `--batch-size 384`, `--n-epochs 1`, `--steps 4000` — two updates,
    ~6 min on one CPU core (measured 2026-10-07)."""
    from agents.training.tb_relevance_test import _run_smoke
    tags, series = _run_smoke(tmp_path, "mr_x5_smoke",
                              ["--arch", "production", "--move-resolution", "on",
                               "--allow-nonproduction-arch", "--n-epochs", "1", "--batch-size", "384",
                               "--rollout-target-samples", "2304",
                               "--steps", "4000"])          # argparse keeps the LAST --steps (the helper's is 10000)
    cfg = (tmp_path / "models" / "mr_x5_smoke" / "model_config.json").read_text()
    assert '"move_resolution": "on"' in cfg and '"belief_tokens"' not in cfg
    assert any(t.startswith("behaviour/") for t in tags)


def test_split_other_move_compiles_as_one_graph():
    """F-MR-1 (`gen3_move_resolution_traceable_v1`): `split_other_move` runs inside the compiled learner region
    (`fullgraph=True`), so it must trace. ``NamedTuple._replace`` lives in ``collections``, which dynamo SKIPS — the
    CUDA launch of fixed_mass × `--move-resolution on` died on it at startup (2026-10-07). Same values as eager."""
    named = ops(("bodyslam", "protect", "counter", "thunderwave"), seats=("earthquake", "growl", "protect"),
                alpha=(0.3, 0.2, 0.25), pair_in=_grid(), p_out=torch.tensor([[0.4]]))
    u = _tail(protect=0.5, quickattack=0.25, earthquake=0.25)
    want = split_other_move(named, u, TABLES, torch.ones(1))
    torch._dynamo.reset()
    try:
        got = torch.compile(split_other_move, fullgraph=True, backend="eager")(named, u, TABLES, torch.ones(1))
    finally:
        torch._dynamo.reset()
    assert type(got) is MoveResolutionOps and got._fields == want._fields
    for f in want._fields:
        assert torch.equal(getattr(got, f), getattr(want, f)), f
