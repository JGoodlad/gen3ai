# Surprise-OHKO / unrevealed-threat belief coverage

**Status:** 🔬 open · **Ledger id:** H3

One-line claim: *the incoming-KO belief under-reads threats it hasn't fully seen, so our healthy mons
stay in and die to OHKOs the model never priced.*

## Known (established, cleared the honesty gates)
- On **52% of lethal healthy-stay deaths in losses**, the belief read **pko < 0.3** at the lethal
  decision (median 0.30). The machinery exists + fires on clearly-revealed threats — it's UNDER-firing,
  not blind. (Measured: 731 healthy stay-deaths over 438 mature loss battles; provenance tracked via the
  opponent's revealed mon/move set.)
- Provenance of the surprise deaths (loss battles): **36%** killer just-switched / first-appearance
  (unknown moveset), **42%** known mon but this MOVE never seen (priors gap), **22%** mon+move BOTH
  already seen yet still under-read (a pure **calibration bug**). So ~78% is "doesn't price unrevealed
  moves," ~22% is "miscalculates known damage."
- The losses skew toward the fully-known calibration bug (22%) vs wins (12%).

## Not-known (the open questions — what would resolve this)
- **RECOVERABILITY (the decisive open question):** on these death turns, did a safe switch actually
  avoid the death, or was the mon already committed/lost? → `models/saved_work/surprise_death_recoverability.py`
  (falsify anchored on the death decisions). **Running / pending as of 2026-06-12.**
- Would a priors-priced belief (account for a revealed mon's likely unseen moves) actually raise pko on
  the just-switched / new-move deaths — and by enough to change the policy's action?
- Does the policy even ACT on a higher pko (the under-switching question)?

## Pros (the upside / why it might be the lever)
- Big surface: 52% of lethal healthy deaths; ~64% of those are fixable (priors + calibration).
- Convergence: this ground-up forensic re-derived the SAME lever the top-down frontier audits flagged
  (surprise-OHKO, opponent-blind obs, under-switching). Two independent methods agreeing.
- The 22% fully-known calibration bug is a clean, fully-fixable target (the belief had all the info).

## Cons (the caveats / why it might not work or not matter)
- **Pervasive in wins too** (56% vs 52%) → surprise-OHKO is a feature of the game, NOT clearly
  loss-causal. The only loss-skewed signal is the calibration-bug share.
- The census found the falsifier often finds **no better action** on death turns — they're frequently
  downstream symptoms of an earlier misplay (committed by the lethal turn). → see the multi-ply frontier item.
- Even a perfect belief needs the policy to ACT (under-switching).
- Surprise from a genuinely-unseen moveset is partly **irreducible** — the lever is *calibrated caution*
  (an epistemic / threat-provenance signal), not perfect threat prediction.

## Next test
- **Land the recoverability number first** (running). If a switch materially avoids the death in a
  meaningful fraction → build the priors-priced-coverage belief + a caution signal. If mostly NEUTRAL
  (committed) → the lever is UPSTREAM (the earlier decision), not the belief — pivot to the multi-ply
  frontier item. Kill the obs-feature build if recoverability < ~20%.
