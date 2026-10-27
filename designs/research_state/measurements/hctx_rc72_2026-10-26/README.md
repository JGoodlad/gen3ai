# H-CTX: P\*'s home-team edge does NOT survive new teams built from its familiar sets (2026-10-26 / 27)

Registered in [`PREREG.md`](PREREG.md) (§1–7, then Amendment 1, the engine-identity ruling, and Amendment 2, the
rc72b redraw; each written before any rc72 / rc72b game counted). Spec [`read_spec_am2.json`](read_spec_am2.json)
(pin `905c29f5b`, sha256 `84901059…`), rule `premium_recovery`. Record: [`read_am2/record.json`](read_am2/record.json)
(with `cells.json`, `inputs.json`, `decision.json`).

Abbreviations: pp = percentage points; P\* = `rb_x26_s2601` @280,000,170 (X26's plateau); Kakuna = Metamon's
Kakuna, checkpoint 34; F = `forum72` (human-built forum teams, unfamiliar); a SET = one mon's species plus its sorted
moves; the POOL = the 719 teams P\* trained on; premium = home72 − F; recovery = rc72b − F.

## Verdict

**`hctx_rc72_am2_2026-10-26: H-CTX SUPPORTED`**: rc72b recovers **−0.34 [−2.05, +0.20]** of the home premium.

| panel (P\* vs Kakuna, symmetric, greedy) | units | W / n | score [Wilson 95 %] |
|---|---:|---|---|
| **home72** (P\*'s own pool teams) | 40 / 40 | 492 / 800 | **0.615 [0.581, 0.648]** |
| **F** `forum72` (unfamiliar human-built teams) | 40 / 40 | 432 / 800 | **0.540 [0.505, 0.574]** |
| **rc72b** (NEW sixes of verbatim pool sets) | 60 / 60 | 617 / 1,200 | **0.514 [0.486, 0.542]** |

- **Premium** home72 − F: **+7.5 pp [+2.9, +12.1]**, so it is resolved on this engine.
- **Recovery** rc72b − F: **−2.6 pp [−7.2, +2.0]**.
- **rc72b − home72: −10.1 pp [−14.2, −6.0].**
- Every interval is a percentile bootstrap over 20-game units, B = 10,000, seed 7273.
- **The registered rule:** KILLED at a recovered share ≥ 0.75, SUPPORTED at ≤ 0.25. The share is −0.34, so SUPPORTED.
  The share's upper bound is +0.20, so even the most favourable end of the interval is inside the SUPPORTED band.
- **Secondary, cross-engine (read by no verdict):** rc72b 0.514 ≤ 0.52, also SUPPORTED by the original absolute bars
  (X51: home72 0.598, U 0.487).

**Reading.** Give P\* six sets it has trained on hundreds of times, in combinations it never trained on, and it plays
them like unfamiliar human teams. The point estimate is even slightly worse than F; the interval does not exclude
"the same as F". Its edge over Kakuna on home72 (+11.5 pp over 0.50 here) belongs to the particular pool TEAMS, not to
the sets they are made of. This is the forensics' per-mon result ("a familiar set on an unfamiliar team plays like
Kakuna's", `c758fcb82`) confirmed by a registered intervention, with the team context removed and set familiarity
kept.

**What it implies for the levers.** The team-context account (H-CTX) stands: breadth must be breadth of TEAMS
(contexts), not of sets.
- **X62**, recombined-team training, is no longer shelved; its gate reads SUPPORTED, which builds its corpus.
- D3 (new sets inside old teams) targets the smaller channel.
- N1-P's procedural teams and X63's union corpus supply new contexts. Whether contexts learnt from recombined or
  procedural sixes transfer to human-built ones is the open question they test.

## The run

- **One engine:** `port=bdc5d2be3af5e1fb/94;env=95b45f104119bdb2/129`. Every unit was played at `905c29f5b` from a
  clean detached worktree; the record carries the engine.
- **Units:** 140 of 140 counted on their first attempt, 2,800 games, 2026-10-27 01:43 → 14:55 PT (`--concurrency 2`,
  rule 5b's cores, short-lane units). Regimes: one id per group, greedy throughout.
- **Interleaving:** rounds of 3 rc72b + 2 home72 + 2 F units, seeds outermost.
- **Dispersion:** the unit-level SD is binomial (home72 0.098 against 0.109 expected; rc72b 0.114 against 0.112;
  F 0.116 against 0.111), so the unit bootstrap and the Wilson intervals agree.
- **Halves** (seed order, descriptive): home72 0.600 / 0.630, rc72b 0.498 / 0.530, F 0.530 / 0.550. rc72b sits
  below both controls in both halves.
- **X51 replay (descriptive).** Of the control units at X51's seeds 0–19:
  - **home72:** 19 of 20 replay X51's games exactly;
  - **F:** 17 of 20 replay exactly.

  The engine moved between X51's E0 and this pin, and a few games differ. This is why Amendment 1 put the controls
  on this read's own engine. The two superseded guard units (home72 and H, seed 15, played at `b044d214b`) both
  replayed X51 game for game.
- **Packs.** Kakuna's peer packs under the LEGACY regime here: the pin predates `b0441623e`. That regime touches F's
  one Frustration team on Kakuna's side, about 1 team in 72, the same for every unit. rc72b and home72 hold no
  Frustration. Our side's packer is post-fix (`d8d674b99` is in the pin).

## The panel

- **rc72b** (`teams_rc72b/`, `rc72b_manifest.json`, `scripts/build_rc72.py --panel rc72b`, seed 7272) is a DIAGNOSTIC
  panel in `designs/research_state/diagnostic_panels.json`.
- **Sets:** every mon block is a verbatim pool occurrence.
- **Species:** 34 species drawn by home72's profile with Smogon teammate lift, total variation 0.116 from home72.
- **Distance from training teams:** exactly 2 sets shared with its nearest team of the pool, `d3_setmut_v1` and
  `w_forum_v1`; ≤ 2 with N1-P's corpus.
- **Distance from held-out and own teams:** ≤ 2 with every held-out panel and `n1p72`; ≤ 3 within the panel.
- **Legality:** validator, level 100, every Hidden Power playing its named type as written.
- **rc72**, the first draw, is RETIRED. Its looser Hidden Power rule passed 71 sets with no IV line, which the live
  server rejects. It was replaced before any game counted (Amendment 2).

## Confounds (from the PREREG, unchanged by the result)

1. **Symmetric cells.** Kakuna pilots rc72b too. If Kakuna pilots recombined sixes unusually WELL, rc72b's score would
   fall for a reason that is not P\*'s. No Kakuna-vs-Kakuna cell rates rc72b's strength (the open assumption A-K).
   Against this: rc72b is drawn by home72's species profile from the pool's most repeated sets, so its teams are
   not weak or strange in the way P72's are.
2. **One checkpoint, one draw** (P\*, seed 7272).
3. **rc72b excludes the pool's no-IV-line Hidden Power sets** (737 of 1,912), as home72 and F do (FINDING 1).
4. **Recombination breaks the sixes' synergy.** A recombined team is coherent by Smogon teammate lift, not built by a
   player. A team plan P\* could play well may simply not exist on some rc72b teams. The rule cannot separate "P\*
   memorised its teams' plans" from "P\* can only play teams that HAVE a plan it learnt". Both say team-level
   breadth.

## FINDINGS

1. **737 of the pool's 1,912 Hidden Power sets** name a type and carry no IV line. The live server rejects them, and
   Showdown's batch validator only rewrites them. This read did not check whether training plays them as the named
   type or as the default-IV type (the heldout72 "HP-as-filled Dark" class). It needs an owner.
2. **Three registrations, one result.**
   - The first lane (pin `b044d214b`) was held for the engine-identity ruling after two guard units, and no rc72
     unit.
   - The Amendment-1 lane (pin `aaf7ced86`) was held when rc72 failed at team validation (0 rc72 games).
   - Neither state is recorded. About 1.5 h of slot time was spent on them.
3. **The pack regime is legacy on Kakuna's side** (pin before `b0441623e`). A post-fix read of these three panels is
   its own regime and would not pool with this one.
4. **Not registered, not done:** the forensics' secondary H-CTX prediction (the seen-set per-mon edge ≈ 0 on
   rc72b). Every unit's capture is kept, so it can be read post hoc.
5. **`w_forum_v1` does not reproduce** from its source on main (`build_w_forum --check`, seen while gating this unit;
   unrelated to this read).

## Files

- **This folder:**
  - `PREREG.md`;
  - `read_spec.json` and `read_spec_am1.json` (superseded) and `read_spec_am2.json` (played);
  - `scripts/build_rc72.py`;
  - the two panels with their manifests and anchors configs;
  - `read_am2/` (record, cells, inputs, decision).
- **State:** `~/gen3ai_archive/hctx_rc72_am2_read/` (captures of every game). The superseded states are
  `hctx_rc72_read` and `hctx_rc72_am1_read`.
