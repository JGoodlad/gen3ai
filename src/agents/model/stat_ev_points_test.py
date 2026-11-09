"""gen3_stat_ev_points_v1 — every stat the model computes from OUR observed spread reads an EV as Showdown does, in
stat POINTS ``tr(ev / 4)`` (`sim/battle.ts` ``statModify``: ``tr(tr(2·base + iv + tr(evs / 4)) · level/100 + 5)``;
the HP branch ``tr(tr(2·base + iv + tr(evs / 4) + 100) · level/100 + 10)``), through ONE decode
(`damage_kinds.our_spread` / `our_stats` / `our_max_hp`).

The op, the end-of-turn rule, the static tokens and the effective-stat facts each re-spelled ``ev / 4`` UNFLOORED:
an 85-EV stat read 21.25 points where the game adds 21 (a max HP up to 0.75 HP high). The committed team files carry
no such EV, but training rollouts do (rb_x60t_s3502's K9 scan rows: a Donphan with 90 Def / 166 SpD), as do 61 of the
4,975 top-25 Smogon spreads. The belief head's EV PRIOR
averaged the raw EVs while its label is the stat-effective ``4·⌊ev/4⌋`` — the prior now lives in the label's space.

gen3_nature_floor_v1 (round 2): the nature step is Showdown's truncation too (`damage_kinds.gen3_nature_stat`: a Timid
359 Speed is 394, not 394.9), our Speed reads the same decode (`damage_op_speed`; `move_order.gen3_speed_stat` is
gone), and our Shedinja's max HP is its species' 1.

Pins: the decode EXHAUSTIVELY over every IV and EV the encoder can write; the stats against `priors.gen3_stat` (the
exact integer reference) at every nature; the EV prior in the label's space; and a source guard that no forward
module computes a stat of our spread OR an end-of-turn residual amount (gen3_residual_exact_v1) outside the exact
helpers.
"""
import io
import re
import tokenize

import numpy as np
import torch

from agents import gen3_data
from agents.model.belief_tables import _SPREAD_EV_IDX, build_species_ev_prior
from agents.model.damage_kinds import gen3_nature_stat, our_max_hp, our_spread, our_stats
from agents.model.selection_sites import FORWARD_MODULES
from utils.paths import src_path


def _spread_block(ivs: np.ndarray, evs: np.ndarray, nature: np.ndarray) -> torch.Tensor:
    """``[N, 18]`` spread blocks exactly as `encoder::slot::spread` writes them (the f64 quotient cast to f32)."""
    n = ivs.shape[0]
    out = np.zeros((n, 18), dtype=np.float32)
    out[:, 0:6] = (ivs / 31.0).astype(np.float32)
    out[:, 6:12] = (evs / 252.0).astype(np.float32)
    out[:, 12] = 1.0
    out[:, 13:18] = nature
    return torch.from_numpy(out)


def test_our_spread_reads_every_iv_and_ev_as_its_integer_and_its_stat_points():
    """EXHAUSTIVE: IV 0..31 and EV 0..255 through the obs cell → the integer IV and ``⌊EV / 4⌋`` bit-exactly
    (85 → 21, 255 → 63)."""
    evs = np.arange(256)
    ivs = np.arange(256) % 32
    blk = _spread_block(np.repeat(ivs[:, None], 6, 1), np.repeat(evs[:, None], 6, 1), np.ones((256, 5)))
    iv, ev_points, nature = our_spread(blk)
    assert torch.equal(iv, torch.from_numpy(np.repeat(ivs[:, None], 6, 1).astype(np.float32)))
    assert torch.equal(ev_points, torch.from_numpy(np.repeat(evs[:, None] // 4, 6, 1).astype(np.float32)))
    assert torch.equal(nature, torch.ones(256, 5))
    assert ev_points[85, 0].item() == 21.0 and ev_points[255, 0].item() == 63.0


def test_our_stats_are_showdowns_integers_at_a_neutral_nature_for_every_base_iv_and_ev():
    """EXHAUSTIVE over base 1..255, IV 0..31, EV 0..255 (one stat column at a time, neutral nature): the five
    non-HP stats equal `priors.gen3_stat` (the integer reference: ``2·B + IV + EV // 4 + 5``) and the max HP
    equals ``2·B + IV + EV // 4 + 110`` (1 at base HP 1, Shedinja's ``species.maxHP``), bit-exactly. Fails on revert
    to ``ev / 4``: every EV % 4 != 0 reads high."""
    b, iv, ev = np.meshgrid(np.arange(1, 256), np.arange(32), np.arange(256), indexing="ij")
    b, iv, ev = b.ravel(), iv.ravel(), ev.ravel()
    blk = _spread_block(np.repeat(iv[:, None], 6, 1), np.repeat(ev[:, None], 6, 1), np.ones((b.size, 5)))
    base = torch.from_numpy(np.repeat(b[:, None], 6, 1).astype(np.float32))
    max_hp, stats = our_stats(base, blk)
    want_hp = np.where(b == 1, 1, 2 * b + iv + ev // 4 + 110)                       # Shedinja's species maxHP 1
    want = 2 * b + iv + ev // 4 + 5
    assert torch.equal(max_hp, torch.from_numpy(want_hp.astype(np.float32)))
    assert torch.equal(stats, torch.from_numpy(np.repeat(want[:, None], 5, 1).astype(np.float32)))
    spot = [(100, 31, 85), (255, 31, 3), (80, 0, 255)]                              # gen3_stat itself, spot-checked
    for sb, siv, sev in spot:
        assert gen3_data.priors.gen3_stat(sb, sev, 1.0, iv=siv) == 2 * sb + siv + sev // 4 + 5


def test_our_stats_truncate_the_nature_as_showdown_does_for_every_raw_stat_and_nature():
    """gen3_nature_floor_v1 — `sim/battle.ts` ``statModify``: ``stat = tr(tr(stat · 110, 16) / 100)`` (``· 90`` for
    the hindered stat). EXHAUSTIVE over base 1..255 × EV 0..255 × IV {0, 31} × the three multipliers, fed exactly
    as the encoder writes the spread (the nature column f32(0.9 / 1.0 / 1.1)): every non-HP stat equals the integer
    reference `priors.gen3_stat` bit-exactly. The old ``(…) × nature`` read a Timid 359 Speed as 394.9, not 394 —
    fails on revert at every boosted / hindered stat whose product is not a multiple of 10."""
    b, ev, iv = np.meshgrid(np.arange(1, 256), np.arange(256), np.array([0, 31]), indexing="ij")
    b, ev, iv = b.ravel(), ev.ravel(), iv.ravel()
    base = torch.from_numpy(np.repeat(b[:, None], 6, 1).astype(np.float32))
    for mult in (0.9, 1.0, 1.1):
        blk = _spread_block(np.repeat(iv[:, None], 6, 1), np.repeat(ev[:, None], 6, 1), np.full((b.size, 5), mult))
        _hp, stats = our_stats(base, blk)
        want = np.array([gen3_data.priors.gen3_stat(int(x), int(e), mult, iv=int(i)) for x, e, i in zip(b, ev, iv)])
        assert torch.equal(stats, torch.from_numpy(np.repeat(want[:, None], 5, 1).astype(np.float32))), mult
    timid = _spread_block(np.full((1, 6), 31), np.full((1, 6), 252), np.array([[0.9, 1.0, 1.0, 1.0, 1.1]]))
    _hp, jolteon = our_stats(torch.tensor([[65.0, 65.0, 60.0, 110.0, 95.0, 130.0]]), timid)
    assert jolteon[0, 4].item() == 394.0 and jolteon[0, 0].item() == 206.0            # 394.9 / 206.1 before


def test_gen3_nature_stat_is_the_one_nature_step():
    """The helper alone, on the integer grid 5..700 × {90, 100, 110}: ``raw · pct // 100``."""
    raw = torch.arange(5, 701, dtype=torch.float32)
    for mult, pct in ((0.9, 90), (1.0, 100), (1.1, 110)):
        got = gen3_nature_stat(raw, torch.full_like(raw, float(np.float32(mult))))
        assert torch.equal(got, (raw.long() * pct // 100).float()), mult


def test_our_shedinja_has_1_max_hp_whatever_its_spread():
    """`sim/pokemon.ts` ``if (this.species.maxHP) stats.hp = this.species.maxHP`` (`data/pokedex.ts` shedinja
    ``maxHP: 1``): the HP formula's 2·1 + 31 + 63 + 110 = 206 is not Shedinja's max HP."""
    blk = _spread_block(np.full((1, 6), 31), np.array([[252, 0, 0, 0, 0, 0]]), np.ones((1, 5)))
    iv, ev_points, _n = our_spread(blk)
    assert our_max_hp(torch.tensor([1.0]), iv[:, 0], ev_points[:, 0]).item() == 1.0
    sp = gen3_data.species.get("shedinja")
    assert sp.base_stats["hp"] == 1 and all(gen3_data.species.get(s).base_stats["hp"] > 1
                                           for s in gen3_data.species.base_form_ids() if s != "shedinja")


def test_an_85_hp_ev_blissey_has_672_max_hp_not_672_25():
    """The brief's Showdown-verified case: Blissey (base HP 255), 31 IV, 85 HP EVs → ``tr(510 + 31 + tr(85/4) + 100)
    + 10`` = 672 (the old ``ev / 4`` read 672.25)."""
    blk = _spread_block(np.full((1, 6), 31), np.array([[85, 0, 0, 0, 0, 0]]), np.ones((1, 5)))
    iv, ev_points, _n = our_spread(blk)
    assert our_max_hp(torch.tensor([255.0]), iv[:, 0], ev_points[:, 0]).item() == 672.0


def test_the_ev_prior_lives_in_the_labels_stat_effective_space():
    """The nature/EV head's prior is the usage mean of ``4·⌊ev/4⌋`` (its label's space): for every species whose
    top-25 spreads carry an EV that is not a multiple of 4, the prior equals that mean (fp32 tolerance), and it
    differs from the raw-EV mean on at least one (else this pin has lost its teeth — fix the fixture)."""
    prior = build_species_ev_prior(600)                                          # > every gen-3 species num
    checked = moved = 0
    for sid in gen3_data.species.base_form_ids():
        rows = gen3_data.priors.spreads(sid)
        if not rows or not any(int(e) % 4 for _n, evs, _w in rows for e in evs):
            continue
        snum = gen3_data.species.get(sid).num
        w = np.array([float(r[2]) for r in rows])
        evs = np.array([[int(r[1][k]) for k in _SPREAD_EV_IDX] for r in rows])
        eff = (w[:, None] * (evs // 4 * 4)).sum(0) / w.sum()
        raw = (w[:, None] * evs).sum(0) / w.sum()
        assert np.allclose(prior[snum].numpy(), eff, atol=1e-3), sid
        moved += int(np.abs(raw - eff).max() > 1e-2)
        checked += 1
    assert checked > 0 and moved > 0, (checked, moved)


#: The forward-module spellings that bypass the exact helpers (comments / strings stripped first), each with the
#: helper that owns it:
#: * our spread's decode (`damage_kinds.our_spread` / `our_stats`): de-normalising the EV column (``* 252``), an
#:   ``ev / 4``, reading the nature columns (``13 + _NAT…``, ``spread[..., 13…``, ``_SPREAD_NATURE``) or multiplying a
#:   stat by a nature multiplier (gen3_stat_ev_points_v1, gen3_nature_floor_v1);
#: * an end-of-turn amount (`damage_kinds.residual_points`): a residual's ``/ 16`` / ``/ 8`` fraction, or any amount
#:   derived as ``maxhp / …`` (gen3_residual_exact_v1).
_FORBIDDEN = {
    "ev*252": re.compile(r"\*\s*252(\.0)?\b"),
    "ev/4": re.compile(r"(?<![\w.])ev\w*(\[[^\]]*\])?\s*/\s*4(\.0)?\b"),
    "nature column": re.compile(r"13\s*\+\s*_NAT|\[\s*\.\.\.\s*,\s*13\b|_SPREAD_NATURE"),
    "* nature": re.compile(r"\*\s*\(?\s*(nat\w*|e_mult)\b"),
    "residual 1/16, 1/8": re.compile(r"/\s*(16|8)(\.0?)?\b"),
    "maxhp /": re.compile(r"max_?hp\w*\s*/"),
}
#: Declared exceptions, (module, stripped line).
_ALLOWED = {
    # THEIR believed stat: an EXPECTATION over stat-effective EVs and the nature posterior, E[⌊EV/4⌋] and E[mult] by
    # linearity (a floor kills the gradient).
    ("belief_heads", "believed = ((2.0 * base + 31.0 + ev / 4.0 + 5.0) * e_mult).clamp(min=1.0)"),
    # the status consequence of LANDING a status on THEIR mon: the nominal tick of an ESTIMATED max HP (E13).
    ("damage_op_pairwise", "d_sched = (-(1.0 / 8.0) * (is_brn + is_psn) - (1.0 / 16.0) * is_tox)"),
    # the Spikes ENTRY chip (a switch-in amount, not a residual): nominal (FINDING, gen3_residual_exact_v1's report).
    ("damage_op_pairwise", "chip_table = torch.tensor([0.0, 1.0 / 8, 1.0 / 6, 1.0 / 4], device=ctx.device)"),
    # Substitute's cost ⌊maxhp / 4⌋ of OUR exact max HP: already Showdown's integer (no minimum of 1 applies).
    ("move_resolution", "sub_frac = torch.floor(o.our_maxhp / 4.0) / o.our_maxhp"),
    # the crit-stage PROBABILITY table, not an HP amount.
    ("ko_exact", "CRIT_P_BY_RATIO: Dict[int, float] = {1: 1.0 / 16.0, 2: 1.0 / 8.0, 3: 1.0 / 4.0, 4: 1.0 / 3.0, "
                 "5: 1.0 / 2.0}"),
}


def _code_lines(path) -> list:
    out = {}
    for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if t.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                      tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        out.setdefault(t.start[0], []).append(t.string)
    return [(n, " ".join(v)) for n, v in sorted(out.items())]


def _flat(code: str) -> str:
    return re.sub(r"\s+", " ", code.replace(" . ", ".").replace(" (", "(").replace("( ", "(").replace(" )", ")")
                  .replace(" [", "[").replace("[ ", "[").replace(" ]", "]").replace(" ,", ","))


def test_no_forward_module_computes_our_stat_or_a_residual_outside_the_exact_helpers():
    """Every forward module but `damage_kinds` (which owns the helpers): none of `_FORBIDDEN`'s spellings, so every
    stat of our spread floors the EV and the nature, and every end-of-turn amount is Showdown's integer, BY
    CONSTRUCTION — a new stat or residual site that re-spells either fails here. The declared exceptions must each
    still exist (a stale one fails too)."""
    hits = {}
    for mod in FORWARD_MODULES:
        if mod == "damage_kinds":
            continue
        for n, code in _code_lines(src_path("agents", "model", f"{mod}.py")):
            flat = _flat(code)
            names = [k for k, p in _FORBIDDEN.items() if p.search(flat)]
            if names:
                hits[(mod, n)] = (flat, names)
    allowed = {(m, re.sub(r"\s+", "", _flat(a))) for m, a in _ALLOWED}
    seen = {(mod, re.sub(r"\s+", "", flat)) for (mod, _n), (flat, _k) in hits.items()}
    bad = [(mod, n, flat, k) for (mod, n), (flat, k) in hits.items() if (mod, re.sub(r"\s+", "", flat)) not in allowed]
    assert not bad, bad
    assert allowed <= seen, ("a declared exception no longer exists", sorted(allowed - seen))


def test_the_guard_catches_each_bypass_it_names():
    """The guard's teeth: one representative bypass per pattern (as `_flat` prints it) matches, so a regex that
    rotted into matching nothing fails here rather than passing every module vacuously."""
    samples = {
        "ev*252": "ev = spread[..., 6] * 252.0",
        "ev/4": "pts = ev / 4.0",
        "nature column": "nat = spread[..., 13 + _NAT_SPE]",
        "* nature": "stat = inner * nature_mult",
        "residual 1/16, 1/8": "leftovers = p_lo / 16.0",
        "maxhp /": "chip = maxhp / 16.0",
    }
    assert set(samples) == set(_FORBIDDEN)
    for name, code in samples.items():
        assert _FORBIDDEN[name].search(_flat(code)), name
