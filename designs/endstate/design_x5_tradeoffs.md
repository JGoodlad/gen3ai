# X5 trade-offs: what the chosen semantics give up, and when to revisit

**Status:** ALWAYS-CURRENT (an end-state doc). Owner decisions 2026-10-03. Companion to
[`design_x5_belief_tokens.md`](design_x5_belief_tokens.md) §9, which holds the option tables, measurements and citations.
This doc exists so that, once X5 has run, we can judge from evidence whether a richer choice is worth its cost
(owner: "pick something reasonable with respect to richness and training budget … think about other, better options
once we understand the limitations").

## 0. The decisions

| choice | chosen | one-line reason |
|---|---|---|
| M2: what a worst case means over uncertain threats | **C: the presence-scaled max, max_m (π_m · v_m)**, which is today's formula with new weights | No second lever on the A/B; zero new code paths; the blob is its special case |
| M3: what OTHER is to the damage physics | **(c) the hybrid: OTHER gets today's AVERAGED attacker and defender, computed on the renormalised TAIL only** | X5 becomes a strict refinement of the blob: it can only add information, for one token |
| The A/B margin (orchestrator, under the owner's "reasonable" mindset) | **δ = 3.5 pp if P0 measures σ_h ≤ 2.5; otherwise δ = 4.5 pp at the same ≤ 41 GPU-h budget** | 4.5 pp (≈ 31 Elo) keeps power ≥ 0.76 at σ 3.43 without raising the budget. Fixed BEFORE P0 reports, so the choice can't be fitted to the number |

## 1. M2 = C: the presence-scaled max

**What it computes.** For each worst-case site (14 of the op's 41 reductions over opponents), the threat of each
candidate is discounted by its presence, then the largest is taken.

**What it gets right.**
- A 0 %-present candidate contributes nothing (I1 holds exactly).
- A certain, revealed threat counts in full.
- It is continuous in π, so there is no rounding boundary (rule 8).
- The blob is the special case with no hypotheses, so the A/B tests ONE change: the tokens.

**What it gives up: the known limitations.**
1. **Threats don't accumulate.**
   - Two separate 50 % chances of an OHKO read as ONE 50 % threat. The true chance that at least one is present is 75 % (the noisy-OR).
   - Example: two plausible Spikes-plus-Earthquake answers on their bench under-read the danger of staying in.
   - **Under-states the threat; the bias grows with the number of medium-probability candidates.**
2. **It is not an expectation or a quantile**, so its number has no clean probabilistic reading. A head reading it must learn its quirks.
3. **I2 is false by declaration.** Splitting one candidate into two half-tokens halves its contribution. Harmless today, because hypothesis tokens are never split, but it constrains any future design that merges or splits tokens.

**How we will know it is binding.** These reads are planned in U7 / the A/B readers.
- **Threat mis-pricing by candidate count.** Bucket decisions by how many unrevealed candidates carry π ≥ some share of the max threat. If the policy's loss rate after staying in rises with that count, beyond what the single-max explains, accumulation matters.
- **Intent and purpose reads.** If opponent-intent log loss on switch-ins to UNSEEN mons does not improve with X5 while revealed-move intent does, the threat channel for hidden mons is a suspect.

**The richer alternative, when warranted: option A, the expected max (noisy-OR) under independent presence.**
- It is principled and consistent with X5's belief, which is a product of Bernoullis.
- Cost: a sort plus a cumulative product per site; about +0.5 agent-day.
- It changes the revealed-move channel too, so it runs as **its own lever after X26**, never bundled.
- Option D (a soft worst case with a temperature) is a further alternative, but it adds a hyperparameter.

## 2. M3 = (c): OTHER priced as today's averaged tail

**What it computes.** OTHER's attacker and defender rows are the blob's own construction (`P @ tables`) applied to the
tail distribution: every species not already named, renormalised.

**What it gets right.**
- Every hidden species is priced somewhere: named guesses get exact physics, and the tail gets today's.
- No damage cell sees OTHER as "immune".
- It costs one matrix multiply per row and no new tokens.
- It is a strict refinement: X5 with zero named guesses IS the blob.

**What it gives up: the known limitations.**
1. **The averaging error (the Jensen gap) stays on the tail's share,** and at a 1:1 budget the tail is LARGE.
   - OTHER holds about **57–74 % of hidden mass** under the Smogon prior and pool-memorising proxies.
   - The named list contains only **34–52 % of the true hidden mons** (`measurements/x5_revision_2026-10-03/`).
   - So for most hidden mass, the physics is still a phantom average: a Blissey and a Skarmory blend into a mediocre mon that resembles neither.
   - The trained head may be sharper on pool teams (UNVERIFIED), which would shrink this.
2. **The tail's average carries no either/or structure.** "It is either a physical wall or a special sweeper", which demands opposite plays, is invisible inside OTHER. Only a named guess carries it.
3. **Two physics semantics coexist** (exact for the named, averaged for the tail), and heads must learn to read both.

**How we will know it is binding.** U8, U7 and the A/B log these.
- **OTHER's mass and attention share over training,** on pool and off pool. If OTHER still holds a large share of hidden mass late in training, and attention keeps leaning on it, the budget is binding.
- **Named-list recall of the true hidden mons, by number of reveals.** Recall stuck below ~0.6 means most of the opponent is still averaged.
- **The intent miss rate (`opp_intent/other_label_rate`, U4).** A high rate of "switched to something in OTHER" says the list is too short.

**The richer alternatives, when warranted.**
- **(a) A larger hypothesis budget** (the named follow-up lever).
  - +6 tokens cut OTHER's share to about **25–41 %** with recall about **0.69–0.77**; a budget of 18 gives about **6–24 %**.
  - Cost: about +7 % forward compute per +6 tokens, damage rows in proportion, +0.5 agent-day.
  - Combine with (c); never replace it.
- **Type-bucketed OTHER.** Split the tail into a few Smogon-derived buckets (e.g. by defensive profile), each averaged separately. The Jensen gap gets finer for a few tokens.
- **Rejected:**
  - (b), a pessimistic worst case over the tail: expensive and biased.
  - (d), no physics: a regression against today.
  - Sampled tail representatives: nondeterministic, against rule 8.

## 3. How these choices fit the budget

The owner's frame is **richness against training budget**. Both choices are the CHEAPEST option that is not a regression.

| | added compute | added levers on the A/B | regression risk vs today |
|---|---|---|---|
| M2 = C | 0 | 0 | none (the same formula) |
| M3 = (c) | ≈ 1 matmul per row | 0 | none (a strict refinement) |
| Upgrade: M2 → A | sort + cumprod per site | its own A/B | changes the revealed channel |
| Upgrade: M3 → (a) at +6 | ≈ +7 % forward | its own A/B | none if combined with (c) |

**The rule for upgrading:** each upgrade is ONE lever, tested from a plateaued parent with paired controls, per the
plateau-first rule in `era_plan_post_m5.md`. It is promoted only when its §1 / §2 diagnostic shows the limitation
binding AND the upgrade's strength-per-GPU-hour is non-inferior.


## 4. Evidence: X5 Tier 0, the physics-channel oracle (2026-10-03, `cda60edb`)

`measurements/x5_tier0_oracle_2026-10-03/`; the ledger entry of the same date. Cold-start Smogon beliefs, the Lane S bank, the production op.
- **M3 (c) at 1:1 is CONFIRMED.** It captures **76–100 %** of what any budget reaches on the threat reads. It halves the expected-threat error against the blob (24.1 → 12.0 pp of HP) and halves the KO-race flips (18.8 → 9.4 %). It is never worse on the defender side, where no budget helps.
- **The budget upgrade (a) is real but small.** At k+6: race flips −0.97 pp [−1.34, −0.65], expected threat −2.9 pp. Defender reads get slightly WORSE (+0.3–0.8 pp).
- **Under M2 = C the worst-case channel ignores the budget** (22.9 pp at every budget). Option A (noisy-OR) cuts it to 16.1 pp at 1:1. So **after X26, M2 option A outranks the budget**.
- **Most of the "perfect OTHER" ceiling is information, not pricing.** A perfect OTHER reaches 4.7 pp (expected threat) and 3.3 % (race flips). Closing that needs a SHARPER BELIEF, not a richer OTHER.
- On real states the 1:1 list recalls only 14–33 % of the true hidden mons, lower than the random-reveal estimate.
- **Open for U3 (F4):** how OTHER's averaged attacker is built; how OTHER's mass, which exceeds 1 on 73.6 % of decisions at 1:1, enters a max-type site (Tier 0 capped its presence at 1).
- **Follow-up lever ranking:** belief sharpness > M2 option A > budget +6.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-03 | M2 worst-case semantics (OWNER) | C, the presence-scaled max | A noisy-OR expected max (a second lever; queued after X26); B threshold (rule 8); D soft max (a hyperparameter) | `design_x5_belief_tokens.md` §9 |
| 2026-10-03 | M3 OTHER physics (OWNER) | (c) the averaged tail (hybrid) | (a) +6 budget (the follow-up lever); (b) a pessimistic tail; (d) no physics (a regression) | §9; `measurements/x5_revision_2026-10-03/` |
| 2026-10-03 | A/B margin (ORCHESTRATOR, under the owner's "reasonable richness vs budget") | 3.5 pp if P0's σ_h ≤ 2.5, else 4.5 pp at ≤ 41 GPU-h; fixed before P0 reports | 3.5 pp at power ≈ 0.57; raising the budget | §9 power table |
| 2026-10-03 | Tier 0 read (ORCHESTRATOR) | M3 (c) at 1:1 CONFIRMED; follow-ups ranked belief sharpness > M2 option A > budget +6 | budget +6 now (≈ 1 pp of race flips for ≈ +7 % FLOPs) | §4; `cda60edb` |
