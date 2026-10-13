"""The MODEL-INPUT injection (consumer (c)) on a production-config extractor and REAL rows of a committed bank battle:
``off`` is bit-identical, each lever reaches the belief channel it names at its mass, revealed facts stay pinned, the
placebo / map targets are what they say, and a planted defect is caught."""
import inspect
import json

import numpy as np
import pytest
import torch

from agents.info_levers import inject as I
from agents.info_levers import spec as S

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def fe():
    import gymnasium as gym

    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.paths import repo_path
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval()


@pytest.fixture(scope="module")
def battle():
    """p1's rows of one committed bank battle and p2's TRUE team (the opponent p1's beliefs are about)."""
    from agents.battle.core_obs import wrap_row
    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1

    bat = B.load_bank(BANK_V1).battles[3]
    res = run_core([bat.recorded()], trackers=True, obs=True)[0]
    rows = np.stack([wrap_row(e["obs"]) for e in res["trackers"][0] if "obs" in e]).astype(np.float32)
    assert len(rows) >= 20, "the fixture battle is long enough to reveal several opponent mons"
    return rows, I.TrueTeam.of(bat.p2["team"])


def _run(fe, rows, spec=None, did="d0", team=None, salt="info_levers_placebo_v1:"):
    x = {"observation": torch.as_tensor(rows)}
    with torch.no_grad():
        if spec is None:
            out = fe(x)
            return out, fe.last_move_belief_logits.clone(), None, {}
        inj = I.Injector(fe, salt=salt)
        with inj.installed():
            inj.set(spec, did, team)
            out = fe(x)
            st = dict(inj.stats)
        item = fe.stash.item_logits.clone() if fe.stash.item_logits is not None else None
        return out, fe.last_move_belief_logits.clone(), item, st


def test_off_and_mass_free_specs_are_bit_identical(fe, battle):
    rows, team = battle
    base, mb, _, _ = _run(fe, rows)
    for spec in ("off", S.InfoReveal()):
        out, mb2, _, _ = _run(fe, rows, spec, team=team)
        assert all(torch.equal(a, b) for a, b in zip(base, out)) and torch.equal(mb, mb2)


def test_true_moves_reach_the_published_posterior_at_their_mass_and_pins_keep_their_bytes(fe, battle):
    rows, team = battle
    _, base, _, _ = _run(fe, rows)
    for p in (0.6, 0.95, 1.0):
        _, mb, _, st = _run(fe, rows, f"TRUE@{p}", team=team)
        m = st["moves"]
        assert m["slots"] > 0 and m["seen_unmatched"] == 0
        q = torch.sigmoid(mb.double())
        n_checked = 0
        for b, row in enumerate(m["targets"]):
            for s, tgt in enumerate(row):
                if not tgt:
                    continue
                n_checked += 1
                # the target channels carry at least the mass (the clip may hold one at 1 - 1e-6)
                assert float(q[b, s, tgt].min()) >= min(p, 1.0 - 1e-6) - 1e-9
        assert n_checked == m["slots"]
    # revealed moves and the bookkeeping 237 keep their exact bytes
    from agents.model.features_extractor import Gen3FeaturesExtractor  # noqa: F401  (import side effects)
    ctx = fe.unpack({"observation": torch.as_tensor(rows)})
    rev = ctx.all_move_ids[:, 6:, :].long()
    _, mb, _, _ = _run(fe, rows, "TRUE@1.0", team=team)
    for b in range(rev.shape[0]):
        for s in range(6):
            pins = [int(x) for x in rev[b, s] if int(x) > 0] + [I.HP_BARE]
            assert torch.equal(mb[b, s, pins], base[b, s, pins])


def test_the_placebo_avoids_the_truth_and_map_is_the_networks_own_top(fe, battle):
    rows, team = battle
    _, base, _, _ = _run(fe, rows)
    _, _, _, st = _run(fe, rows, "PLACEBO@0.95", team=team)
    _, _, _, st2 = _run(fe, rows, "PLACEBO@0.95", team=team)
    _, _, _, st3 = _run(fe, rows, "PLACEBO@0.95", did="d1", team=team)
    assert st["moves"]["targets"] == st2["moves"]["targets"], "deterministic per decision"
    assert st["moves"]["targets"] != st3["moves"]["targets"], "a different decision draws its own wrong set"
    ctx = fe.unpack({"observation": torch.as_tensor(rows)})
    sp = ctx.species_ids[:, 6:].long()
    for b, row in enumerate(st["moves"]["targets"]):
        for s, tgt in enumerate(row):
            if tgt:
                assert not set(tgt) & set(team.mons[int(sp[b, s])].moves), "a placebo move is never a true one"
    _, _, _, sm = _run(fe, rows, "SELFMAP@1.0", team=team)
    q0 = torch.sigmoid(base.double())
    legal = I.Injector(fe).legal
    rev = ctx.all_move_ids[:, 6:, :].long()
    n_map = 0
    for b, row in enumerate(sm["moves"]["targets"]):
        for s, tgt in enumerate(row):
            if not tgt:
                continue
            n_map += 1
            pins = {int(x) for x in rev[b, s] if int(x) > 0} | {I.HP_BARE}
            if any(I.is_hp(n) for n in pins):
                pins |= set(I.TYPED)
            cand = [n for n in np.flatnonzero(legal[int(sp[b, s])]) if int(n) not in pins and int(n) not in tgt]
            assert sum(1 for n in tgt if I.is_hp(n)) <= 1 and len(tgt) <= 4
            # every map target is at least as likely, by the network's OWN base belief, as every candidate left out
            # (a second Hidden Power is skipped by the at-most-one rule, so it may outrank a target)
            worst = float(q0[b, s, tgt].min())
            assert all(float(q0[b, s, n]) <= worst + 1e-12 for n in cand if not I.is_hp(int(n)))
    assert n_map > 0


def test_item_and_spread_levers_reach_their_consumers(fe, battle):
    rows, team = battle
    _, _, item, st = _run(fe, rows, "item:seen,spread:seen", team=team)
    assert st["item"]["slots"] > 0 and st["spread"]["slots"] > 0
    ctx = fe.unpack({"observation": torch.as_tensor(rows)})
    sp = ctx.species_ids[:, 6:].long()
    seen = ~ctx.opp_believed_mask.bool()
    known = ctx.item_ids[:, 6:].long() > 0
    post = torch.softmax(item.double(), dim=-1)
    sb = fe.stash.spread_belief
    for b in range(rows.shape[0]):
        for s in range(6):
            if not seen[b, s]:
                continue
            tm = team.mons[int(sp[b, s])]
            assert [round(float(x)) for x in sb[b, s]] == list(tm.stats)
            if not known[b, s]:
                assert float(post[b, s, tm.item]) > 1.0 - 1e-6


def test_refusals_and_a_planted_off_by_one_target_is_caught(fe, battle):
    rows, team = battle
    inj = I.Injector(fe)
    for bad in ("ability:seen", "species", "hp", "spread:seen~map"):
        with pytest.raises(S.InfoRevealError):
            inj.set(bad, "d", team)
    with pytest.raises(ValueError, match="TRUE team"):
        inj.set("TRUE@0.9", "d", None)
    # TEETH: an injector whose target is one move channel off must fail the reach check above
    orig = I.Injector.move_targets

    def shifted(self, ctx, hs, base_logits):
        tgt, mass, pinned, hp_rev, st = orig(self, ctx, hs, base_logits)
        return np.roll(tgt, 1, axis=-1), mass, pinned, hp_rev, st

    I.Injector.move_targets = shifted
    try:
        _, mb, _, st = _run(fe, rows, "TRUE@1.0", team=team)
    finally:
        I.Injector.move_targets = orig
    q = torch.sigmoid(mb.double())
    miss = sum(float(q[b, s, t].min()) < 0.5 for b, row in enumerate(st["moves"]["targets"])
               for s, t in enumerate(row) if t)
    assert miss > 0, "the planted off-by-one target leaves the true moves unpushed: the reach check sees it"


def test_the_hidden_power_type_is_a_spread_fact_and_its_presence_a_moves_fact(fe, battle):
    """Taxonomy (design §1.1): ``moves`` tells THAT a seen mon holds a Hidden Power, ``spread`` (the IVs) tells its
    TYPE. A moves lever alone leaves the network's type posterior as it was; a spread lever moves only the type."""
    rows, team = battle
    _, base, _, _ = _run(fe, rows)
    T = list(I.TYPED)

    def split(mb):
        qt = torch.sigmoid(mb.double()[..., T])
        pres = qt.sum(-1)
        return pres, qt / pres.unsqueeze(-1)

    p0, t0 = split(base)
    _, mb, _, st = _run(fe, rows, "TRUE@1.0", team=team)
    on = torch.from_numpy(st["moves"]["hp_on"])
    assert int(on.sum()) > 0, "the fixture battle has a seen mon with an unrevealed true Hidden Power"
    p1, t1 = split(mb)
    assert float((p1[on] - 1.0).abs().max()) < 1e-4, "the moves lever pushes the Hidden Power's presence"
    assert float((t1[on] - t0[on]).abs().max()) < 1e-4, "...and leaves its TYPE to the network"
    _, mb2, _, st2 = _run(fe, rows, "spread:seen", team=team)
    ht = torch.from_numpy(st2["moves"]["hp_type"])
    sel = ht >= 0
    assert bool(sel.any()) and not bool(torch.from_numpy(st2["moves"]["hp_on"]).any())
    p2, t2 = split(mb2)
    assert float((p2[sel] - p0[sel]).abs().max()) < 1e-4, "the spread lever leaves the presence"
    assert float(t2[sel].gather(-1, ht[sel].unsqueeze(-1)).min()) > 1.0 - 1e-4, "...and tells the type"
    _, mb3, _, _ = _run(fe, rows, "moves:seen,spread:seen", team=team)
    q3 = torch.sigmoid(mb3.double()[..., T])
    assert float(q3[on].gather(-1, ht[on].unsqueeze(-1)).min()) > 1.0 - 1e-4, "both levers: the true typed id"
