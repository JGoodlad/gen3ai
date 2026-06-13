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
- **RECOVERABILITY — RESOLVED** (n=381, all surprise-deaths falsified, 0 errors, stable from n≈60):
  **42% are AVOIDABLE** (a materially-better action existed; **64% of those a SWITCH** to a bulky wall —
  Swampert/Jirachi/Snorlax/Gyarados — median gain ~1.0 mon). **33% are LUCK** (the belief was RIGHT —
  low pko — and the mon died to a crit/high-roll; NOT a coverage failure, irreducible). **25% are
  NEUTRAL** (no better action — committed/lost upstream). So the belief's true recoverable headroom is
  the **~42% avoidable**, NOT the full 52% surprise rate — and it's mostly a SWITCH-to-a-wall (this lever
  is half-belief-coverage, half-under-switching).

## Not-known (the open questions — what would resolve this)
- Would a priors-priced belief (account for a revealed/just-switched mon's likely unseen moves) actually
  raise pko on the avoidable deaths — and by enough to change the policy's action?
- Does the policy even ACT on a higher pko (the under-switching question)? 64% of the avoidable escapes
  are switches, so this is load-bearing.
- Pre-build: are the 64%-switch escapes to a bench mon the model could KNOW survives (a per-slot belief),
  not hindsight?

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
- Even a perfect belief needs the policy to ACT (under-switching) — 64% of the avoidable escapes ARE
  switches, so this lever is half-belief, half-under-switching.
- **33% of "surprise deaths" are LUCK** — the belief was RIGHT (low pko) and the mon died to a crit/roll;
  a better belief can't help those (they inflate the apparent "under-firing" — the true headroom is 42%,
  not 52%).
- Surprise from a genuinely-unseen moveset is partly **irreducible** — the lever is *calibrated caution*
  (an epistemic / threat-provenance signal), not perfect threat prediction.

## Next test
- Recoverability LANDED at **42% > the 20% kill bar → GO (gated).** Build the priors-priced threat
  coverage (price a revealed/just-switched mon's LIKELY unseen moves) + a caution/threat-provenance
  signal — BUT it only pays if the policy then SWITCHES, so **pair it with the under-switching lever**
  (`--switch-bias-weight`) and validate BOTH: the death-turn pko must rise AND the switch-to-a-wall rate
  must rise. Cheap pre-build oracle: confirm the 64%-switch escapes are to a bench mon the model could
  KNOW survives (a per-slot belief), not hindsight. The 25% NEUTRAL + 33% LUCK are out of scope here
  (multi-ply / aleatoric).
