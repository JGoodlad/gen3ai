"""The cost ablation's harness (`ablate.py`, unchanged) plus ONE measurement-only arm:

  fm_perrow   the BEFORE arm: `gathered_hypothesis_tokens` replaced in this process by the per-row pass
              it replaced (`pokemon_encoder(hypothesis_ctx(...))[:, 6:12]`, the real pass's move tokens
              restored) — the graph the parent commit compiles.

Usage: ablate2.py --arm {blob,fm,fm_perrow} ... (every other flag is ablate.py's)."""
import sys

import ablate

if "fm_perrow" in sys.argv:
    import agents.model.extractor_forward as EF
    from agents.model.hypothesis_tokens import hypothesis_ctx
    from agents.observation.constants import TEAM_SIZE

    real_build = EF.ExtractorForward._build_hypothesis_species
    _hs_box = {}

    def build_hs(self, ctx, role):
        hs = real_build(self, ctx, role)
        _hs_box["hs"] = hs
        return hs

    def perrow_tokens(pe, embeddings, ctx, slot_species, dex_rows):
        hs = _hs_box["hs"]
        hctx = hypothesis_ctx(ctx, hs, pe.layout)
        keep = pe.last_move_tokens
        tok = pe(hctx, embeddings)
        pe.last_move_tokens = keep
        return tok[:, TEAM_SIZE:2 * TEAM_SIZE]

    EF.ExtractorForward._build_hypothesis_species = build_hs
    EF.gathered_hypothesis_tokens = perrow_tokens
    sys.argv[sys.argv.index("fm_perrow")] = "fm"

if "fm_nohyp" in sys.argv:
    # the FLOOR: the hypothesis tokens are DEAD (splice is fed the real pass's opponent tokens), so Inductor
    # drops the whole hypothesis encoding and its backward — fm − fm_nohyp is that encoding's cost.
    import agents.model.extractor_forward as EF
    from agents.model.hypothesis_tokens import splice_hypothesis_tokens as _orig_splice
    EF.splice_hypothesis_tokens = lambda rt, oh, hs, mk: _orig_splice(rt, rt[:, 6:12], hs, mk)
    sys.argv[sys.argv.index("fm_nohyp")] = "fm"

ablate.main()
