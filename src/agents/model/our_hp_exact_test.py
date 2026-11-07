"""gen3_our_hp_exact_v1 — every HP the model computes for OUR side is an EXACT integer, through one helper family
(`damage_kinds.our_max_hp` / `our_hp_points` / `our_healed_hp`).

The obs carries ``f32(hp / maxhp)`` (`encoder::put` casts the f64 quotient); the op rebuilt the HP as
``hp_frac · maxhp`` in fp32, which misses ``hp`` by an ulp for ~8.5 % of (maxhp ≤ 714, hp) pairs. At
``hp == fixed`` an up-rounding (100 / 714 → 100.0000076: a max-HP Blissey) read Seismic Toss / Night Shade as
NOT KO-ing a mon Showdown faints (`sim/pokemon.ts` ``damage``: ``hp -= d; if (hp <= 0) faint``).

Pins: the helpers EXHAUSTIVELY over every reachable (max HP, HP) pair and heal fraction; the fixed-damage KO end to
end through the op; and a source guard that no forward module re-spells our max HP or multiplies an HP fraction by a
max HP outside the helpers (their side — a percent × a neutral estimate — is `opp_maxhp`, out of the guard's scope).
"""
import io
import re
import tokenize

import numpy as np
import torch

from agents.model.damage_kinds import our_healed_hp, our_hp_points, our_max_hp
from agents.model.damage_op_layout import _DMG_IDX_PHYS_PKO
from agents.model.damage_op_test import _T2I, _believe_op, _fake_ctx, _make_layout
from agents.model.features_extractor import TEAM_SIZE, _DMG_PER_MON, DamageOperator
from agents.observation.constants import POKEMON_SPREAD_OFFSET
from agents.model.selection_sites import FORWARD_MODULES
from utils.paths import src_path

_BLISSEY, _TAUROS = 242, 128
_MAX_HP = 714                                       # every gen-3 L100 max HP is <= 714 (Blissey, 31 IV / 252 EV)


def _obs_frac(hp: np.ndarray, maxhp: int) -> np.ndarray:
    """The obs cell exactly as the Rust encoder writes it: the f64 quotient cast to f32."""
    return (hp / float(maxhp)).astype(np.float32)


def test_our_cur_hp_is_integer_exact_for_every_reachable_pair():
    """EXHAUSTIVE over maxhp 1..714, hp 0..maxhp (256k pairs): the helper returns hp bit-exactly."""
    for maxhp in range(1, _MAX_HP + 1):
        hp = np.arange(0, maxhp + 1)
        got = our_hp_points(torch.from_numpy(_obs_frac(hp, maxhp)), torch.tensor(float(maxhp)))
        assert torch.equal(got, torch.from_numpy(hp.astype(np.float32))), maxhp


def test_the_plain_product_is_not_exact():
    """The input the helper exists for: the naive fp32 product misses 100 at maxhp 714 (stays true while the
    encoder's cast and IEEE fp32 do; if it ever stops, the pin below loses its teeth — fix the fixture)."""
    v = torch.from_numpy(_obs_frac(np.array([100]), _MAX_HP))
    assert (v * float(_MAX_HP)).item() > 100.0


def _incoming_phys_pko(op: DamageOperator, hp: int) -> float:
    """P(KO) on the physical channel of OUR slot 0 (a max-HP Blissey, 714) at `hp`, the opponent believed to
    carry Seismic Toss (fixed 100, accuracy 1)."""
    ctx = _fake_ctx(op, attacker_num=_TAUROS, attacker_t1=_T2I["NORMAL"], attacker_t2=0,
                    defenders=[(_BLISSEY, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5, hp_probs_active=[0.0] * 16)
    ctx.pokemon_part[:, 0, POKEMON_SPREAD_OFFSET + 6] = 1.0          # 252 HP EVs -> max HP 714
    ctx.hp_and_active[:, 0, 0] = float(_obs_frac(np.array([hp]), _MAX_HP)[0])
    pm = op(ctx, _believe_op(op, "seismictoss"))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    return float(pm[0, 0, _DMG_IDX_PHYS_PKO].item())


def test_seismic_toss_kos_our_mon_at_exactly_100_hp():
    """End to end through the op's forward: 100 HP left -> Seismic Toss KOs (Showdown: hp <= 0 faints);
    101 HP -> it does not. Fails on revert to ``hp_frac * maxhp`` (the KO read 0 at 100)."""
    op = DamageOperator(_make_layout())
    ko_100, ko_101 = _incoming_phys_pko(op, 100), _incoming_phys_pko(op, 101)
    assert ko_100 > 0.99, ko_100
    assert ko_101 < 0.01, ko_101                    # (the belief's -12 logit tail on the other moves)


def _kernel_heal_fracs() -> torch.Tensor:
    """The four heal fractions exactly as `_recovery_worlds` builds them in fp32: the table's 0.5 / 1.0 and the
    weather blend ``(2/3)·sun + 0.25·other + 0.5·(1 − sun − other)`` at sun, other weather and clear."""
    sun = torch.tensor([1.0, 0.0, 0.0]); other = torch.tensor([0.0, 1.0, 0.0])
    w = (2.0 / 3.0) * sun + 0.25 * other + 0.5 * (1.0 - sun - other)
    return torch.cat([torch.tensor([0.0, 0.5, 1.0]), w])


def test_healed_hp_is_showdowns_truncated_heal_for_every_reachable_pair():
    """EXHAUSTIVE: maxhp 1..714, hp 0..maxhp, each heal fraction (none, 1/2, Rest's 1, and the weather 2/3, 1/4,
    1/2) -> ``min(maxhp, hp + trunc(frac · maxhp))`` (gen-3 Showdown truncates a heal) in exact integers."""
    fracs = _kernel_heal_fracs()
    exact = [(0, 1), (1, 2), (1, 1), (2, 3), (1, 4), (1, 2)]                    # the same six, as rationals
    for maxhp in range(1, _MAX_HP + 1):
        hp = np.arange(0, maxhp + 1)
        cur = our_hp_points(torch.from_numpy(_obs_frac(hp, maxhp)), torch.tensor(float(maxhp)))
        got = our_healed_hp(cur[:, None], fracs[None, :], torch.tensor(float(maxhp)))
        want = np.minimum(maxhp, hp[:, None] + np.array([maxhp * n // d for n, d in exact])[None, :])
        assert torch.equal(got, torch.from_numpy(want.astype(np.float32))), maxhp


def test_our_max_hp_is_the_l100_formula_for_every_pool_spread():
    """IV 0..31, EV 0..252 in steps of 4 (every pool set), base HP 1..255, read back through the obs cells (f64
    quotient cast to f32, times 31 / 252) -> exactly ``2·B + IV + EV/4 + 110``."""
    b = torch.arange(1, 256, dtype=torch.float32)[:, None, None]
    ivs, evs = np.arange(32), np.arange(0, 253, 4)
    iv = torch.from_numpy((ivs / 31.0).astype(np.float32))[None, :, None] * 31.0
    ev = torch.from_numpy((evs / 252.0).astype(np.float32))[None, None, :] * 252.0
    want = 2 * np.arange(1, 256)[:, None, None] + ivs[None, :, None] + evs[None, None, :] // 4 + 110
    assert torch.equal(our_max_hp(b, iv, ev), torch.from_numpy(want.astype(np.float32)))


#: An HP product or our max-HP formula spelled outside the helpers (comments and strings stripped first).
_FORBIDDEN = (re.compile(r"/\s*4\.0\s*\+\s*110\.0"),                      # our max HP, re-spelled
              re.compile(r"\*\s*(our_)?maxhp\b"), re.compile(r"(?<!\w)(our_)?maxhp(\[[^\]]*\])?\s*\*(?!\*)"))
#: Declared exceptions, (module, stripped line): an HP AMOUNT that is not our current HP and gates no threshold.
_ALLOWED = {
    # Leech Seed's drain as an expected amount for BOTH sides (`min(hp, 1/8)` of max HP, no floor): a continuous
    # cell, never a cut. FINDING (gen3_our_hp_exact_v1): the game drains trunc(maxhp / 8).
    ("eot_residual", "drained = maxhp * torch.minimum(hp, hp.new_full((), 1.0 / 8.0)) * seeded"),
    # THEIR HP's percent bin (HP Percentage Mod): an opponent's HP, never ours.
    ("ko_exact", "lo = cur_hp - (1.0 - full) * (OPP_HP_PERCENT * maxhp) - full * (0.5 * one_hp)"),
}


def _code_lines(path) -> list:
    out, toks = {}, tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    for t in toks:
        if t.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                      tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        out.setdefault(t.start[0], []).append(t.string)
    return [(n, " ".join(v)) for n, v in sorted(out.items())]


def test_no_forward_module_computes_our_hp_outside_the_helpers():
    """Every forward module but `damage_kinds` (which owns the helpers): no inline max-HP formula of the real
    spread, no ``x * maxhp`` product. Fails on any new site, so OUR HP stays integer-exact by construction."""
    hits = []
    for mod in FORWARD_MODULES:
        if mod == "damage_kinds":
            continue
        for n, code in _code_lines(src_path("agents", "model", f"{mod}.py")):
            flat = re.sub(r"\s+", " ", code.replace(" . ", ".").replace(" (", "(").replace("( ", "(")
                          .replace(" )", ")").replace(" [", "[").replace("[ ", "[").replace(" ]", "]")
                          .replace(" ,", ","))
            if any(p.search(flat) for p in _FORBIDDEN):
                hits.append((mod, n, flat))
    allowed = {(m, re.sub(r"\s+", "", a)) for m, a in _ALLOWED}
    bad = [h for h in hits if (h[0], re.sub(r"\s+", "", h[2])) not in allowed]
    assert not bad, bad
    assert len(hits) == len(_ALLOWED), ("a declared exception no longer exists", hits)
