# Model leaf — HISTORY

**This file is HISTORY. It is ADDITIVE ONLY — do not update it, and do not re-derive a plan from
it.** Passages lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08 whose standing RULE is stated
in that leaf or in a `designs/model/` topic doc, and whose remaining content is the dated record of
how it was found.

## The SimSiam latent predictor on `BeliefHead` — DELETED at v75

(`BeliefHead` also carried an asymmetric SimSiam **latent** predictor until v75, regressing each
believed slot toward the stop-grad `pokemon_encoder` role-token of the true hidden mon. It is DELETED —
it was never fed forward, its own role-geometry probe concluded decodable != helps, and it cost ~13% of
the train step. Predicting the opponent's unrevealed mons is unaffected: the species CE, the moves BCE
and the T0 species prior all remain. See `designs/CHANGELOG.md` for how these landed.)

## The two refactor-proof bundles — the op split, and the 2026-08-23 class split

The standing rule (*a refactor claiming to change nothing is proved, not reviewed*) is in the leaf;
the file-layout detail is in `designs/model/file_layout.md`. These are the bundles that were run.

**The gate for a refactor claiming to change nothing is proof, not review:** byte-identity on pi/vf +
the raw op block (`tmp/damage_op_equiv_probe.py`), unchanged `state_dict` keys, the constructed-scenario
physics oracle (`damage_op_probe_fuzz_test.py`, 22/22), and the full suite. All four held.

The 2026-08-23 class split was held to the same standard, escalated once more (the proofs are in the
commit message): a **line-coverage splitter** reading the pre-split text from the COMMIT assigned all
2,280 original lines to exactly one target (2,270 assigned + 10 verified-blank) before writing
anything; **44/44 class members are source-hash identical** with `forward` / `forward_internal` /
`__init__` additionally executable-AST identical; the `state_dict` KEY sha and the whole-`state_dict`
TENSOR sha are unchanged on a seeded production-config build (236 keys); and pi/vf plus every
`ExtractorStashes` field and every `last_*` property are bit-identical through both `fe(obs)` and
`type(fe).forward(fe, obs)`. Exactly ONE deliberate edit: the class's base, `torch.nn.Module` →
`ExtractorForward`.

## M1 — what SB3 ortho-init had silently falsified, measured

The rule and the guard are in the leaf under *Identity-at-init is NOT free*.

Until 2026-08-01 this silently falsified the identity-at-init contract for **13** Linears in every
real training run — the zero-init physics projections (`prefuse_proj` and, in the configs of the
day, the between-layers refine loop's), `film_pi`/`film_vf`, plus the belief heads
(`MoveBelief.move_head`, `SpreadBelief.*`, `HPTypeBelief.type_head`) whose zero-init is what makes
the **cold-start posterior equal the Smogon prior**. Measured max|W| before the fix: 0.19–0.47. See `designs/research_state/ledger.md` → **M1**
for the standing caveat this puts on the K10 and D4 result families.
