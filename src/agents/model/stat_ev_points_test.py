"""gen3_stat_ev_points_v1 — every stat the model computes from OUR observed spread reads an EV as Showdown does, in
stat POINTS ``tr(ev / 4)`` (`sim/battle.ts` ``statModify``: ``tr(tr(2·base + iv + tr(evs / 4)) · level/100 + 5)``;
the HP branch ``tr(tr(2·base + iv + tr(evs / 4) + 100) · level/100 + 10)``), through ONE decode
(`damage_kinds.our_spread` / `our_stats` / `our_max_hp`).

The op, the end-of-turn rule, the static tokens and the effective-stat facts each re-spelled ``ev / 4`` UNFLOORED:
an 85-EV stat read 21.25 points where the game adds 21 (a max HP up to 0.75 HP high). The committed team files carry
no such EV, but training rollouts do (rb_x60t_s3502's K9 scan rows: a Donphan with 90 Def / 166 SpD), as do 61 of the
4,975 top-25 Smogon spreads. The belief head's EV PRIOR
averaged the raw EVs while its label is the stat-effective ``4·⌊ev/4⌋`` — the prior now lives in the label's space.

Pins: the decode EXHAUSTIVELY over every IV and EV the encoder can write; the stats against `priors.gen3_stat` (the
exact integer reference) at a neutral nature; the EV prior in the label's space; and a source guard that no forward
module de-normalises the EV column or divides an EV by 4 outside the helpers.
"""
import io
import re
import tokenize

import numpy as np
import torch

from agents import gen3_data
from agents.model.belief_tables import _SPREAD_EV_IDX, build_species_ev_prior
from agents.model.damage_kinds import our_max_hp, our_spread, our_stats
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
    equals ``2·B + IV + EV // 4 + 110``, bit-exactly. Fails on revert to ``ev / 4``: every EV % 4 != 0 reads high."""
    b, iv, ev = np.meshgrid(np.arange(1, 256), np.arange(32), np.arange(256), indexing="ij")
    b, iv, ev = b.ravel(), iv.ravel(), ev.ravel()
    blk = _spread_block(np.repeat(iv[:, None], 6, 1), np.repeat(ev[:, None], 6, 1), np.ones((b.size, 5)))
    base = torch.from_numpy(np.repeat(b[:, None], 6, 1).astype(np.float32))
    max_hp, stats = our_stats(base, blk)
    want_hp = 2 * b + iv + ev // 4 + 110
    want = 2 * b + iv + ev // 4 + 5
    assert torch.equal(max_hp, torch.from_numpy(want_hp.astype(np.float32)))
    assert torch.equal(stats, torch.from_numpy(np.repeat(want[:, None], 5, 1).astype(np.float32)))
    spot = [(100, 31, 85), (255, 31, 3), (80, 0, 255)]                              # gen3_stat itself, spot-checked
    for sb, siv, sev in spot:
        assert gen3_data.priors.gen3_stat(sb, sev, 1.0, iv=siv) == 2 * sb + siv + sev // 4 + 5


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


#: De-normalising the observation's EV column, or dividing an EV by 4, outside the helpers (comments / strings
#: stripped first).
_FORBIDDEN = (re.compile(r"\*\s*252(\.0)?\b"), re.compile(r"(?<![\w.])ev\w*(\[[^\]]*\])?\s*/\s*4(\.0)?\b"))
#: Declared exceptions, (module, stripped line).
_ALLOWED = {
    # our Speed: `move_order.gen3_speed_stat` rounds the EV and floors ev / 4 (and the nature) itself.
    ("damage_op_speed", "stat = gen3_speed_stat(base, spread[..., _BS_SPE] * 31.0, spread[..., 6 + _BS_SPE] * 252.0,"),
    # THEIR believed stat: an EXPECTATION over stat-effective EVs, E[⌊EV/4⌋] by linearity (a floor kills the gradient).
    ("belief_heads", "believed = ((2.0 * base + 31.0 + ev / 4.0 + 5.0) * e_mult).clamp(min=1.0)"),
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


def test_no_forward_module_decodes_an_ev_outside_the_helpers():
    """Every forward module but `damage_kinds` (which owns the decode): no ``* 252`` de-normalisation of the EV
    column and no ``ev / 4``. Fails on any new site, so every stat of our spread floors the EV by construction."""
    hits = []
    for mod in FORWARD_MODULES:
        if mod == "damage_kinds":
            continue
        for n, code in _code_lines(src_path("agents", "model", f"{mod}.py")):
            flat = _flat(code)
            if any(p.search(flat) for p in _FORBIDDEN):
                hits.append((mod, n, flat))
    allowed = {(m, re.sub(r"\s+", "", _flat(a))) for m, a in _ALLOWED}
    bad = [h for h in hits if (h[0], re.sub(r"\s+", "", h[2])) not in allowed]
    assert not bad, bad
    assert len(hits) == len(_ALLOWED), ("a declared exception no longer exists", hits)
