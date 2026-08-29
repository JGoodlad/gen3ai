# TRANSFER-COEFFICIENT CELL — analysis plan, written BEFORE the main run

Written 2026-08-29 ~10:00 PDT, after a 36-battle pilot (18 base + 8 honest orientation-games over
game indices 0–1 of all 9 roster opponents) and BEFORE any A−B contrast existed. The pilot was
read for COST and for the defensive rate table only; both arms won every pilot game, so no
outcome difference had been observed when this was written.

## Primary (registered in ledger 2af60c2, scored never adjusted)

`A − B` = paired win-rate difference over the unit `(opponent, game_index)`, arm A =
defensive search (iter-2 configuration verbatim), arm B = `--arm base` (the same network,
search structurally off, greedy argmax), on the FULL 9-bot eval roster
(`eval_opponent_names()` — the battery's default `--opponents`).

## Two things declared here that the ledger's registration could not know

1. **THE ROSTER HAS A CEILING, and it makes the registered "+5–12 pp" reading arithmetically
   unreachable.** The subject's own recorded `latest_eval` at step 24 000 000 gives
   `win_rate_vs_bots = 0.9162`, so the maximum attainable A−B on this population is +8.4 pp and
   the *realistic* maximum is far less. The registered band is therefore scored as
   **unreachable-by-construction**, not as missed. The transfer coefficient — A−B divided by the
   naive additive expectation computed on THIS population's own measured overrule rate — is the
   reading that survives, and it is reported as primary.

2. **A pre-declared hard-subset split.** Declared from the subject's own `latest_eval` per-opponent
   win rates (external to this cell's data): the FOUR opponents at ≤0.91 —
   `aggressive_v2` (0.88), `setup_sweep_v2` (0.88), `heuristic` (0.89), `heuristic2` (0.91) —
   form the `hard4` stratum; `random` (1.00) is reported separately as the ZERO-HEADROOM cell.
   `hard4` is a secondary, not a substitute for the primary.

## Naive additive expectation

`E_naive = (overrules per game, MEASURED on this roster) × (+0.0474 per-decision win-prob gain,
probe K)`. Transfer coefficient `τ = (A−B) / E_naive`. Both the numerator and the denominator's
first factor come from this cell; only +0.0474 is imported.

## Addendum, written 10:02 PDT — still before any main-run A−B was inspected

Two more readouts, declared now:

3. **Conditioning on arm A's overrule count.** In a unit where A never overruled, the two arms are
   the SAME trajectory and `d_i` must be exactly 0. That gives a free internal-validity check —
   a non-zero delta in the zero-overrule bucket would prove the pairing is broken — and it makes
   the diluted primary interpretable: `delta | n_overrules >= 1` is the effect size to set against
   a per-decision gain. **The conditioning is legitimate rather than outcome-selection**: whether
   A's FIRST overrule happens is a function of the prefix the two arms share exactly, so it is a
   pre-treatment variable, not a post-treatment one. Reported as a secondary with the same CI
   machinery, and `tau` is reported on both the diluted and the conditional row.
4. **Delta stratified by overrule count** (0 / 1 / 2 / 3+), for the same reason.

## Estimator

Paired differences `d_i = score_A(i) − score_B(i)` over units where both arms finished, score
1 / 0.5 / 0 for win / tie / loss. CI = normal 95% on the mean of `d_i`. McNemar discordant counts
reported alongside. Per-opponent and per-stratum splits reported. A timed-out or unfinished game
is its own bucket and is excluded from both arms of its pair, never scored.

## Cuts

Sized from the pilot: honest 8.15 s/game, base 0.95 s/game, 9 opponents ⇒ ~82 s per game index
across both arms. Three shards over disjoint index windows [0,150) [150,300) [300,450),
`nice 15`, BLAS pinned, ≈3 cores, estimated ~3.4 h.
