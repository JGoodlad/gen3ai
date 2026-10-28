# Pre-commitment memo: what each read of 10-27 → 10-30 will make us do, written before any of them is seen

**Status: PRE-COMMITMENT, written 2026-10-27, BLIND.** I have seen no result of any read below. I read only:
- their PREREGs, amendments and reviews;
- the designs they feed;
- [`../STATUS.md`](../STATUS.md), which gives unit COUNTS only (for example "124 of 140 units").

I opened no read state, record, `inputs.json`, `units.jsonl`, cells file or summary. Two committed artefacts were
visible in the documents I had to read; neither is an outcome of a read below:
- X59-S's floor references, its FR-2 class and its re-encode share;
- S-U stage 0's one searched game, a home72 game of a read that is not on this list.

**How to use it.** When a read lands:
1. find its class in §2;
2. do what the row says;
3. tell the owner the row's sentence;
4. check the traps before writing anything into `UNDERSTANDING.md` or the ledger.

A departure from a branch is allowed only by the orchestrator, and it is recorded in §6 with its reason. The branch
itself is never edited after its read lands. Where the registered documents leave a gap, §4 proposes a ruling,
marked MY JUDGEMENT; the orchestrator should accept or change each one **before** the read it concerns lands.

Abbreviations:
- P\* is our plateau checkpoint (X26 @280M). C0 is the plain 20M continuation of P\*, the control of every
  plateau fork. Kakuna is Metamon's strongest public model.
- home72 is our 72 sample training teams. F (`forum72`) is 72 human-built forum teams P\* never saw. U is three
  unfamiliar panels pooled, H, F and P72.
- S is in-house strength against a frozen reference population. pp is a percentage point of win rate.
- LB / UB are the lower / upper bound of the registered interval.
- Goal (b) is beating Kakuna on unfamiliar teams (P\* 0.487 on U; it needs ≈ 0.53).

---

## 0. One screen

| read | lands (ESTIMATE) | its likeliest class (the registered prediction) | the one decision it moves |
|---|---|---|---|
| **H-CTX rc72b** | 10-27 | SUPPORTED or a low UNDECIDED | X62 is shelved (KILLED) or its corpus is built (else); X63's union keeps or drops the recombined third |
| **LR-F look 1** | 10-28 | UNDECIDED (Δ̂ 0.0 pp predicted) | look 2, and whether every capacity read (X48, N4) must re-run with a learning-rate floor |
| **N1-P look 1** | ≈ 10-29 | UNDECIDED / CONTINUE (P 0.45) | X53-N1 is funded (GAIN), waits (CONTINUE), or X62 registers (NULL) |
| **X59-S floor gate** | ≈ 10-29 to 10-30 | unknown; students far below P\* | whether the human-data Kakuna read continues at all |
| **D3 look 2** | ≈ 10-30 | UNDECIDED ± ≈ 3 pp (P 0.93–0.97) | set mutation retired as a standalone lever; d3 stays in or leaves X63's union |
| **X48** | after 10-30 (needs the `n4_screen` rule and 26–38 Kakuna lane-h) | NO-GO 0.36–0.50, GO 0.28–0.36 | N4-P (GO) or the N4 toy / N4-P-direct choice (else) |
| **S-U gate** | not registered yet | STOP (P(GO) ≈ 0.06 if U behaves like home72) | whether search spends ≈ 450–650 CPU-h (V-RANK + S-U) on goal (b) |

**The bias this memo guards against.** All seven reads are low-powered, and most will land UNDECIDED. The tempting
error is to line up several positive point estimates and act as if they were a GAIN. **No point estimate from these
reads is pooled with another after the fact.** The registered place for a +1–2 pp family effect is PR-1′, on new
games.

---

## 1. The plan these reads feed (as of 10-27, `../STATUS.md` §3)

- **GPU:** LR-F s3503 → LR-F s3501 → X48 s3501–s3503 → N4.
- **The ranked plan:**
  1. X48 (production capacity);
  2. X63 (dose: 100M on a union corpus);
  3. PR-1′ (a zero-GPU precision read of D3 and N1-P against C0 on F; information only).
- **Gated:** X62 (on rc72b, then PR-1′, then N1-P); X53-N1 (on N1-P's GO and a σ gate); S-U (on a sign-accuracy
  gate).

Nothing below reorders the top three except where a row says so.

---

## 2. Each read

### 2.1 H-CTX on rc72b: does P\*'s home edge belong to its TEAMS or to its SETS?

**Rule** (`premium_recovery`; [PREREG](../research_state/measurements/hctx_rc72_2026-10-26/PREREG.md) Amendments 1–2):
- premium P = home72 − F; recovery R = rc72b − F, every arm on one engine;
- **INCONCLUSIVE** unless P's LB > 0 (P ≈ 0.80 at X51's 6.8 pp);
- else **KILLED** if R ≥ 0.75 P, **SUPPORTED** if R ≤ 0.25 P, else **UNDECIDED**;
- **VOID** if an arm counts < 75 % of its units, or two engines appear.

| class | we do next | the owner is told |
|---|---|---|
| **KILLED** | **SHELVE X62** (its gate G0, [`design_x62_team_recombination.md`](design_x62_team_recombination.md) §6). X63's union drops the recombined third and becomes U2 (procedural + set-mutated teams). If N1-P also reads LOSS, X63 is not run ([`design_x63_long_breadth_continuation.md`](design_x63_long_breadth_continuation.md) §8). If R / P's interval reaches below 0.5, record a FINDING; the verdict stands. STATUS goal 3 reads "set-level". | "Our model keeps most of its home advantage when its familiar Pokémon sets are shuffled into new teams. What it learned is mostly about the individual sets, not the exact six-mon teams, so training on reshuffled teams (X62) is shelved. Teaching it more sets stays the route." |
| **SUPPORTED** | **BUILD `x62_recomb_v1`** (≈ 1 agent-day, 1–3 CPU-h), then X62's G1 (PR-1′) and G2 (N1-P). X63's union is U3. Order of X62 vs X63: §4 item 5. STATUS goal 3 reads "team-level". | "Familiar sets in new teams play like unfamiliar teams: the edge is tied to the specific teams it practised. That is the case for X62, training on new combinations of the sets it already knows." |
| **UNDECIDED** | **Exactly as SUPPORTED** (X62's design routes them alike, by design: the read cannot split them where the truth is predicted). The prior is lower (F +1.0 pp vs +1.5). Goal 3 stays OPEN. | "Somewhere in between. Both readings lead to the same next step, so we build the reshuffled-team corpus; we do not yet know which half of the edge it can reach." |
| **INCONCLUSIVE** | **HOLD** X62 and X63's composition. A top-up of home72 / F units counts only as a new registration with a stated second-look α, written before any top-up game (§4 item 7). Else X62 waits for PR-1′ / N1-P. | "The home-versus-forum gap did not show clearly enough in this sample to measure the new teams against; we hold." |
| **VOID** | Fix the cause, re-run. Nothing routes. | "The run broke a registered check; it is being re-run." |

**Traps: what NOT to conclude.**
- **KILLED ≠ "team context never matters".** It says P\*'s CURRENT premium survives recombination. Two other
  causes are not separated:
  - the cell is symmetric, so Kakuna also pilots rc72b, and Kakuna's own response is in the score (confound 2);
  - rc72b draws the pool's most repeated sets, so its teams may be STRONGER (confound 4).
- **SUPPORTED ≠ "training on recombined teams will help".** It locates the premium; whether training can move it
  is X62's own read.
- **Never split SUPPORTED from UNDECIDED in action after seeing which one fired.** They route identically by
  registration.
- **rc72b plays at `905c29f5b`, before the Kakuna peer pack changed (`b0441623e`).** Its F is the legacy-pack
  regime. Compare R and P only inside this read, never with a post-`b0441623e` F (X48's K, PR-1′, X63).
- **N2 (compositional encoding) does not decide this read**, in either direction. N2 measured how the policy
  shifts under a set re-draw, not win rate.

### 2.2 LR-F look 1: is the plateau the learning-rate controller's halving?

**Rule** (X35 R1; [`../research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md)
§4.1.2, [`design_compute_efficiency.md`](design_compute_efficiency.md) §5.6.4):
- Δ_pop = LR-F − C0 on S, seeds s3501–s3503, Welch at 97.5 %;
- **GAIN** = interval wholly above +2.0 pp; **NULL** = inside ±2.0; **LOSS** = wholly below −2.0; else UNDECIDED;
- an UNDECIDED with Δ̂ > 0 → **CONTINUE** to look 2 (s3504–s3505, + 8.6 GPU-h); with Δ̂ ≤ 0 it is final;
- guards: G-F, G-C, G-K; G-E only on a GAIN;
- **VOID** a seed whose trailing-2M approx KL exceeds 0.08.

At three seeds a GAIN needs the whole 97.5 % interval above +2 pp, so look 1 is built to end CONTINUE, UNDECIDED or
NULL. The registered prediction is Δ̂ 0.0, modal UNDECIDED.

| class | we do next | the owner is told |
|---|---|---|
| **GAIN** (guards pass: ADOPT-CANDIDATE) | (1) the learning-rate floor becomes a recipe question (`design_learner_recipe.md` §3.7); nothing is adopted on one read; (2) **every capacity read is re-designed with the floor on BOTH arms before any further width read** (X48 PREREG §9's last row; N4 review M4): N4 and N4-P re-register; (3) X48, already training or trained, finishes and is read as registered (§4 item 3); (4) every X35 fork's NULL / UNDECIDED (A1, SPRT, D1, D3, N1-P) is labelled "at the halved rate". It is NOT re-opened by re-interpretation; a breadth + floor arm is its own registration; (5) X63 is unchanged (its control X26c has no floor; one lever per A/B). | "Holding the learning rate up made the plateau network stronger in-house by more than 2 points. The plateau is at least partly the training schedule, not the network's size. Every 'bigger network' test now runs with the rate held up too." |
| **CONTINUE** | Look 2 (8.6 GPU-h) enters the GPU queue **after X48's arms and before N4 launches** (§4 item 2). Nothing else changes: a CONTINUE is not a GAIN. | "Leaning positive but not decided; two more seeds settle it." |
| **NULL** | The controller's halving is not the plateau's cause at this floor and dose. Production LR-F closes. The caveat "NULL at the halved rate" is downgraded for the X35 forks. N4 / N4-P proceed as the N4 review's branch table says for "not GAIN". | "Keeping the learning rate high changed nothing measurable (within ±2 points). The plateau is not the schedule." |
| **LOSS** | The controller's cuts were protective. No floor anywhere. The X48 review's M3 worry (an LR-elevated GO) becomes a reported risk on X48's record. | "A higher learning rate made it worse; the automatic cuts were helping." |
| **UNDECIDED, final** (Δ̂ ≤ 0) | Nothing licensed. Capacity reads proceed as for "not GAIN". | "No sign it helps; not enough seeds to rule out a small effect." |
| **TRADE** (a guard FAILs) | The orchestrator rules. | the guard named, in words. |

**Traps.**
- **R1 is in-house S only.** An LR-F GAIN says nothing about goal (b). G-K FAILs only on a DETECTED loss on home72 or
  `away`; it does not read F.
- **A CONTINUE must not move X48's or N4's interpretation.** The X48 / N4 branch tables key on GAIN, never on a
  positive Δ̂.
- **Production C0 cuts to 1.0047e-4 within ≈ 2–3M**, so the contrast is ≈ ×1.73, not ×1.44. A GAIN measures that
  dose of step size, not "the rate two rungs up".
- **The toy LR-F never ran**, so there is no calibration in either direction. A production NULL is not "toy-confirmed".

### 2.3 N1-P look 1 (X61P): does breadth at 10⁴ procedural teams lift F at the plateau?

**Rule** (D1's rule on F alone; [PREREG](../research_state/measurements/x61p_n1_production_2026-10-26/PREREG.md) §4–§6):
- Δ_F = N1-P − C0, Newcombe 95 % widened by √φ̂, sign rule ≥ 2 of 3;
- **GAIN** iff LB > 0 and the sign rule; **LOSS** iff UB < 0; **NULL** iff inside ±3.0; else UNDECIDED;
- UNDECIDED with Δ̂ > 0 → CONTINUE; with Δ̂ ≤ 0 it is FINAL;
- co-primary P: S NON-INFERIOR iff LB > −2.0 pp;
- guards G-F, G-E, G-C read only on an F GAIN.

| class | we do next | the owner is told |
|---|---|---|
| **ADOPT-CANDIDATE** (F GAIN, P non-inferior, guards pass) | **GO X53-N1** (fresh production run on 10⁴ teams): its zero-GPU σ gate first, then its PREREG; its cloud-GPU question is the owner's (`../STATUS.md` §6, [`design_x53_n1_fresh_breadth.md`](design_x53_n1_fresh_breadth.md)). **X62 becomes REDUNDANT as a standalone fork** (G2); its corpus goes to X63 / X53. n1 stays in X63. The G-E reader (a 5M-step best responder of N1-P's final) is GPU work: it queues after X48's arms. | "Training on 12,000 generated teams made the plateau model better at piloting human-built teams it had never seen (forum teams, against Kakuna), without costing in-house strength. We fund the from-scratch version." |
| **TRADE** (F GAIN, in-house loss not excluded, or a guard FAILs) | GO X53-N1 as above, with P as its co-primary; the exchange rate Δ̂_F / −Δ̂_P goes to the owner. | "Better on unfamiliar teams, possibly at some cost on our own; here is the exchange rate. Your call whether that trade is worth it." |
| **CONTINUE** | Look 2 (s3504–s3505, 8.2 GPU-h) in the GPU tail. X62's G2 waits. **X63 treats n1 as IN** without waiting (§4 item 1). | "Leaning positive, not decided; two more seeds." |
| **NULL** | Rules out **"N1 at the plateau, share 0.5, 20M" only**. X62's G2 → **REGISTER X62** (if G0 allowed it and PR-1′ has not KILLED). n1 stays in X63 (a 20M NULL is weak evidence at a larger dose). X53-N1 is not triggered. | "Generated teams at the plateau did not move forum-team play by 3 points or more. This does not rule out training on them from scratch or for longer." |
| **LOSS** | No N1 follow-up at the plateau. **n1 leaves X63.** If H-CTX is KILLED as well, X63 is not run. Report Δ_H − Δ_F (incoherence vs breadth). | "The generated teams hurt: their sets are not coherent human sets. We stop this family at the plateau." |
| **UNDECIDED, final** (Δ̂ ≤ 0) | As NULL for X62 (register) and X63 (n1 in). Nothing licensed. | "No sign of a gain; not enough data to rule out a small one." |
| **INCONCLUSIVE / VOID** | Re-play or re-run the named seed. | "A registered check failed; re-running." |

**Traps.**
- **Never read an N1-trained model's U, H or P72 as goal (b)** (PREREG §8: 62 % and 56.5 % of H's and P72's full
  sets are in the corpus). A gain concentrated on H / P72 over F is generator familiarity.
- **A GAIN does not say WHICH of new contexts or new sets did it.** 72 % of N1's slots are new sets, and 99.7 % of
  its teams are new sixes. It cannot confirm or refute H-CTX (§3, row J1).
- **NULL ≠ "breadth does not work"** (dose: ≈ 24 pilots per corpus team), and NULL ≠ "plasticity is lost" (the
  probe read NO-LOSS).
- **False GAIN is 2.4 % at look 1 and 3.5–7.4 % over two looks** (PREREG FINDING 1). A look-1 GAIN with
  φ̂ near its widening threshold is reported with that number.
- **The in-house S co-primary is already played** (STATUS). Its number must not colour how F is read: F decides,
  and P only classifies.
- **N1-P's F is the legacy-pack regime** (before `b0441623e`). Never set it against a post-`b0441623e` F.

### 2.4 X59-S: is human data better than our own self-play data? The floor gate first, then the full read

**What lands in this window is most likely only the GATE** (Amendment 5, rule `x59s_gate`).
- **The gate.** One boolean per contrast: UNINFORMATIVE iff BOTH arms' pilot means are ≤ 0.0958 (the max-damage bot
  + 5 pp). The rest of the Kakuna read plays iff Δ_data or Δ_labels is informative.
- **The full read.** After 720 more units, at the pilot's rate, it is weeks out.
  - Δ_data = S-H − S-S: HUMAN-DATA-BETTER / SELF-PLAY-DATA-BETTER / NO-DIFFERENCE-AT-4PP / NOT-RESOLVED.
  - Δ_labels = S-H − S-P: HUMAN-LABELS-BETTER / OWN-LABELS-BETTER / NO-DIFFERENCE-AT-4PP / UNCLEAR.
  - The guards, from the PREREG ([PREREG](../research_state/measurements/x59s_matched_students_2026-10-25/PREREG.md)
    §5, A2.4–A2.10, A5.1–A5.8, A6.1):
    - KAKUNA-SPECIFIC: the in-house h2h sign disagrees;
    - POPULATION-SPECIFIC: F alone disagrees with a human win;
    - the fit gate;
    - the closure is pending until Δ_restriction = S-P0 − S-P is read, and blocked if its LB > 0.

| outcome | we do next | the owner is told |
|---|---|---|
| **Gate: both UNINFORMATIVE** | **Stop the Kakuna read**; report both contrasts' Kakuna and in-house numbers, "routes nothing". The registered X60 route is **X60-T** (H-TEAMS against D1), but X60 is pending its review and takes no GPU slot from this. The ≥ 1500 follow-up does NOT fire. | "Students trained only on replays, or only on our own games, were too weak against Kakuna to tell apart: about as weak as a bot that always picks the hardest-hitting move. The question is unanswered, not answered no." |
| **Gate: one informative** | That contrast's full read plays; the other is reported UNINFORMATIVE. | "One of the two comparisons is readable; it continues." |
| **HUMAN-DATA-BETTER** (not KAKUNA- or POPULATION-SPECIFIC) | The human-data levers stay open; X60's clause follows Δ_labels (below). The full GPU queue is unchanged until X60 is reviewed and registered. | "At this volume, students taught from human replays beat students taught from our model's own games, against Kakuna and in-house." |
| **SELF-PLAY-DATA-BETTER, before Δ_restriction** | Report it as **"closure pending Δ_restriction"**. S-P0's U units play next (≈ 25–35 lane-h). Nothing closes. | "Our own games trained the better student, but one known artefact of how the human rows are built is not yet ruled out. No decision yet." |
| **SELF-PLAY-DATA-BETTER, Δ_restriction LB > 0** | **Closes NOTHING.** The ≥ 1500 follow-up fires (Δ_data is neither HUMAN-DATA-BETTER nor UNINFORMATIVE). X60-T is the route. | "The self-play student won, but the human rows carry a fingerprint of how we fill in hidden information, and that fingerprint alone costs a student about as much. We cannot blame the humans' play for the loss, so we close nothing." |
| **SELF-PLAY-DATA-BETTER, every guard clear** (Δ_labels not HUMAN-LABELS-BETTER, Δ_states LB ≤ 0, fit gate, in-house agrees, Δ_restriction LB ≤ 0) | **Close "1300+ ladder data in the panel-distant, forfeit-heavy slice, at this mix and volume, as a behaviour-cloning target for a fresh network at 3 epochs"**: no H-TEAMS, no H-BC; the owner decides about more data. | the closure text verbatim, plus: "This is about this slice of ladder data, not human data in general, and not about adding human data to our trained model." |
| **NO-DIFFERENCE-AT-4PP** (Kakuna AND in-house inside ±4) | Closes nothing; the ≥ 1500 follow-up fires; X60-T. | "Human data is neither better nor worse than our self-play data by more than 4 points at this volume." |
| **NOT-RESOLVED** | Follow-up fires; X60-T. | "Not resolved at this size." |
| **Δ_labels HUMAN-LABELS-BETTER** (S-P learned its teacher by the midpoint test; not KAKUNA- or POPULATION-SPECIFIC) | **X60-B** (H-BC against a P\*-label placebo). | "Human choices were a better teacher than our model's choices restricted to what the human could have done." |
| **Δ_labels OWN-LABELS-BETTER** (not KAKUNA-SPECIFIC) | H-BC dropped; X60-T. | "Our model's choices were the better teacher; we drop the human-imitation lever." |

**Traps.**
- **SELF-PLAY-DATA-BETTER with Δ_restriction blocking** is the expected trap. S-H's and S-P's rows carry C-GAME's
  drawn-move signature, and S-S's do not. Δ_restriction > 0 means that signature costs a student by itself. Never
  tell the owner "human data is worse" on this branch.
- **S-S is "our plateau policy's own games and choices"** (a distillation of P\*), not the rollout data PPO trains
  on (G-m8).
- **The gate's booleans are not arm rates.** No arm rate is read or quoted before the full record (m3).
- **The student-to-P\* gap (Q-H10) is cross-engine and secondary.** It never answers "would a human-only model
  beat ours"; that is already answered for Metamon's human-only models (P\* beats all three).
- **Kakuna is human-pretrained.** A human-data win that only Kakuna sees is KAKUNA-SPECIFIC by registration; do not
  argue around it.
- **X59-S's F numbers are the legacy-pack regime** (m13).

### 2.5 D3 look 2 (AM-D3-L2): set mutation, the TERMINAL read

**Rule** (D1's rule on H = the U mean, five seeds, all on E1;
[PREREG](../research_state/measurements/x41_d3_setmut_2026-10-25/PREREG.md) §3–§6 and AM-D3-L2):
- **GAIN** iff LB > 0 and ≥ 3 of 5 seeds; **LOSS** iff UB < 0; **NULL** iff inside ±3.0 (**unreachable**: the
  half-width is ≥ 3.0 pp); else UNDECIDED;
- P co-primary NON-INFERIOR iff LB > −2.0 pp;
- the interval is reported as a bound; an UNDECIDED inside ±3.1 is reported as "no effect beyond ≈ 3 pp";
- **TERMINAL**: no look 3, no pooling of these rows with any later D3 data;
- P(GAIN | what is known) ≈ 3–7 % ([review](../research_state/measurements/x41_d3_setmut_2026-10-25/REVIEW_look2.md) M2).

| class | we do next | the owner is told |
|---|---|---|
| **ADOPT-CANDIDATE** (H GAIN, P non-inferior, guards pass) | D3 joins X53-N1's recipe (beat-Metamon §6.5); d3 stays in X63. The G-E reader (`ge_d3`, GPU) is the orchestrator's call; until it plays, the verdict reads INCONCLUSIVE. | "Mutating the sets inside our own teams made the model better on unfamiliar teams. It becomes part of the from-scratch recipe." |
| **TRADE** | The owner gets the rate Δ̂_H / −Δ̂_P. D3 + D1 in one arm is NOT licensed. | "A gain on unfamiliar teams, possibly at a cost on ours." |
| **UNDECIDED, bound inside ±3.1** (the expected class) | **D3 is retired as a standalone lever.** No more D3 seeds or units. d3 stays in X63's union (8.6 % of its draws) and in PR-1′ (new games, a new registration). | "Set mutation does not move unfamiliar-team play by more than about 3 points either way. We stop testing it on its own; it stays as a small ingredient of the big mix (X63)." |
| **UNDECIDED, bound wider than ±3.1** | As above, with the bound printed. | as above, with the bound. |
| **LOSS** | Stop the family. **d3 leaves X63.** | "It hurt; we drop it." |
| **P inferiority not excluded** (whatever H reads) | d3's place in X63 follows §4 item 4. | — |
| **INCONCLUSIVE / VOID** | Re-play the named units at the pin (m7 pre-approves a resume). | "A registered check failed; re-running." |

**Traps.**
- **The bound is not a NULL.** "No effect beyond ≈ 3 pp" leaves a +1–2 pp contributor entirely possible. That is
  exactly the size PR-1′ and X63 are built to see. Never write "D3 does nothing".
- **Never re-read look 1's E2 rows beside look 2.** The descriptive E2 → E1 rows (m6) decide nothing.
- **PR-1′ re-reads D3 on NEW unit seeds at a post-`b0441623e` pin.** It must never import look 2's games.
- **X63's gate says "F LOSS"**, but D3's registered primary is H (the three-panel mean), and F alone is a reported
  line. Use H's class (§4 item 4).

### 2.6 X48: per-token-kind FFN experts at the plateau, under the `n4_screen` rule

**Rule** ([PREREG](../research_state/measurements/x48_kffn_production_2026-10-27/PREREG.md) §5.3, one look):
- **GO-S** iff LB_S > 0; **NO-GO-S** iff UB_S < +2.0 (90 % Welch, S at 2,000 pairs);
- **GO-K** iff LB_K > 0; **NO-GO-K** iff UB_K < +1.5 (F only, 1,000 games a final);
- **MIXED** (classed UNDECIDED): GO on one meter with the other's UB < 0;
- **GO** iff GO-S or GO-K;
- **NO-GO** iff NO-GO-S and not GO-K, labelled "LR-confounded" if X48's mean LR < 0.9× C0's, and "experts not
  engaged" if §8.7's D_k / null ≤ 1.0 (NOTE 1, `80973fd75`);
- a GO is labelled "LR-elevated" if X48's mean LR > 1.1× C0's (NOTE 1);
- the family false GO is ≈ 0.09.

| class | we do next | the owner is told |
|---|---|---|
| **GO** (any label) | **Skip the N4 toy.** N4-P (production width at X48's parameter count) at `N4_PIN` with a fresh production C0′, read by the PR rule. Nothing joins the recipe. If LR-F has a final GAIN, N4-P registers with the floor on both arms. | "Giving each kind of input token its own feed-forward block helped at the plateau. It is a screen, about a 1-in-11 false-alarm rate, so we re-test capacity properly next." |
| **NO-GO** | Closes **per-kind FFN untying at the plateau, within 20M**, nothing wider. N4 (with the review's fixes) or N4-P-direct is the orchestrator's choice, unless LR-F is GAIN. A NO-GO labelled "LR-confounded" is a weak NO-GO. | "No in-house gain of 2 points or more from per-kind experts within 20M steps. This does not say the network is big enough." |
| **MIXED / UNDECIDED** | Nothing licensed; both intervals go to the orchestrator and the owner. No look 2. | "Unclear; here are both numbers." |
| **INCONCLUSIVE** | Fix the missing input (an S cell short, a meter on two engines, a VOID seed). | — |

**Traps.**
- **NO-GO ≠ "capacity does not bind".** It does not close FFN width, `D_MODEL` or from-scratch capacity.
- **The three labels are labels, not classes** (NOTE 1, registered before the first GO). A weak NO-GO ("LR-confounded",
  "experts not engaged") still closes per-kind untying only weakly; a weak GO ("LR-elevated") may be partly LR-F's
  lever. No further label is added after any X48 number is seen.
- **A GO-K from 1,000 F games a final is a weak meter** (GO-K at a true +3 pp ≈ 0.6). Never quote X48's K as goal
  (b) progress.
- **K plays the post-`b0441623e` pack.** It is not comparable with X51's U 0.487 or any legacy-pack F (N1-P, D3,
  H-CTX).

### 2.7 The S-U gate: do one-ply search's deviations beat chance on unfamiliar teams?

**Not yet registered.** It is proposed by the [S-U review](../research_state/measurements/su_headroom_2026-10-27/REVIEW.md)
("The gate test"). This memo pre-commits to that proposal's rule and routes. A registration that changes the rule
must say so, and must land before any gate state is scored.
- **The measure.** Sign accuracy of S's deviations from π under the self-referee, on ≈ 1,200 U states (turns 10–30)
  from P\* vs Kakuna games at unit seeds 100–109.
- **GO** iff LB(pooled-U sign accuracy) > 0.50; else **STOP**.
- **The GIGO tooth.** The scorer must first reproduce the review's banked numbers exactly: sign accuracy 0.513 / 0.580
  and IMP +0.0001 / +0.0193.

| class | we do next | the owner is told |
|---|---|---|
| **GO** | V-RANK's live branches next ([`design_v_rank_on_u.md`](design_v_rank_on_u.md)), then the amended S-U, if V-RANK does not turn it OFF. The amendment must carry the review's four B1 items, M3 (I reported, phase 2 dropped) and M4 (failed deals bounded both ways). Search unpauses for this question only. | "On unfamiliar teams, the search's changes to the model's moves are right more often than chance, by its own judgement. That earns a real test against Kakuna, not a claim of gain." |
| **STOP** | **No live S-U read, and no V-RANK as its filter.** Record: "S at P\* is a chance-level perturbation of π on U, as on home72, under its own model". Search stays paused for goal (b). | "When the search overrules the model on unfamiliar teams, it is right about half the time even by its own judgement, so it cannot help there yet. We do not spend the ≈ 500 CPU-hours." |

**Traps.**
- **GO is graded by the self-referee**, the best case for S. It licenses the next stage only, never a gain (E1-0's
  direction: gains shrink against a real bot).
- **STOP does NOT claim DNC-B**, and it does not say search cannot help on U. Another λ, another leaf or opponent
  model, or more worlds stay open.
- **Home72's sign accuracy and U − home72 are reported, sign only.** They are not a verdict on H-MID's mechanism.
  I (the U − home72 difference) is leak-confounded with unknown sign (review M1).

---

## 3. The joint outcomes

Each row is a combination that will plausibly occur, with what we do and what we must NOT conclude. A combination not
listed routes by its reads' own rows, independently. "Any" means the other read does not change the row.

| # | combination | we do | do NOT conclude |
|---|---|---|---|
| J1 | **H-CTX KILLED + N1-P GAIN** | X62 shelved (G0 fires first). X53-N1 GO. X63 runs as U2 (n1 + d3). | Not a contradiction: N1's teams are mostly NEW SETS, which is what KILLED says matters. Not that N1 "worked through sets" either: N1-P cannot attribute. Not that X62 "would have worked". |
| J2 | **H-CTX KILLED + N1-P NULL / UNDECIDED-final** | X62 shelved. X63 runs as U2 (n1 in). X53-N1 not triggered. | Not "breadth is dead": X63 is the dose read and is untouched. Not that the premium is unreachable: KILLED says sets carry it, and set breadth has been read only at 20M. |
| J3 | **H-CTX KILLED + N1-P LOSS** | X62 shelved and n1 out, so **X63 is not run** (only d3 would remain; X63 §8). PR-1′ still runs as information. | Not that "breadth hurts": LOSS is incoherent procedural sets. Not a reason to re-open X62 against its G0. |
| J4 | **H-CTX SUPPORTED / UNDECIDED + N1-P GAIN** | Build the X62 corpus. X62's fork is REDUNDANT; the corpus goes to X53-N1 and X63 (U3). X53-N1 GO. | Not "contexts did it": N1-P confounds contexts with sets. Not a reason to run X62 as well: a ±3 pp read cannot separate two positive arms. |
| J5 | **H-CTX SUPPORTED / UNDECIDED + N1-P NULL / LOSS / UNDECIDED-final** | Build the corpus. X62 registers via G2, unless PR-1′ has already read KILL. X63 runs as U3 (n1 out on LOSS). Order: X63 Stage A first (§4 item 5). | Not "context breadth fails": N1-P's contexts came with incoherent new sets at ≈ 24 pilots a team, and this is the branch X62 exists for. |
| J6 | **H-CTX any + N1-P CONTINUE** | X62's G2 waits for look 2. X63's PREREG does not wait: n1 is in (§4 item 1). | Not a positive result: CONTINUE is Δ̂ > 0 inside an UNDECIDED. |
| J7 | **H-CTX INCONCLUSIVE + anything** | X62 and X63's composition HOLD. X48, LR-F and PR-1′ are untouched. | Not "no premium": the gate is about resolution. X51's home72 − F is 6.8 pp on another engine. |
| J8 | **D3 UNDECIDED (bound) + N1-P NULL / UNDECIDED-final + X59-S not HUMAN-DATA-BETTER**, the strategy doc's "all levers null" premise | The re-ranked plan holds: X48 → X63 → PR-1′, X62 by its gates. **No new 20M plateau breadth fork, and no "levers exhausted" escalation to the owner**: REVIEW_2 M3 marked that trigger text false while X62 and X63 stand. | Not "breadth is exhausted", and not "the plateau is the ceiling": capacity (X48), the schedule (LR-F) and dose (X63) are all unread. Never pool D1 + D3 + N1-P point estimates into a family claim; that is PR-1′'s job on new games. |
| J9 | **D3 ADOPT-CANDIDATE + N1-P ADOPT-CANDIDATE** | X53-N1's recipe takes n1 and d3. Both stay in X63. | Do not add the two Δ̂s into a goal (b) forecast. Neither read's U is goal (b): N1-P's is hazarded (§2.3), and the claim is X54's on `forum216b`. |
| J10 | **D3 GAIN + N1-P NULL** (or the reverse) | Each by its own row. | Not "set mutation works and generated teams don't": two reads at ≈ ±3 pp cannot rank two arms. |
| J11 | **LR-F GAIN (final) + X48 GO** | N4-P registers with the floor on BOTH arms. The recipe question on the floor goes to the owner. X48's GO carries "LR-elevated" if its mean LR > 1.1× C0's (NOTE 1). | Not that capacity helped: X48's experts may run one controller rung higher (review M3), so its GO may be partly LR, labelled or not. |
| J12 | **LR-F GAIN (final) + X48 NO-GO** | X48's NO-GO is labelled "at the controller's learning rate" and closes nothing. N4 is redesigned with the floor. | Not that "capacity does not bind": capacity was tested behind a binding learning rate. |
| J13 | **LR-F NULL / UNDECIDED-final / LOSS + X48 GO** | N4-P directly, skipping the N4 toy (X48 PREREG §9). | Not adoption. The family false GO is 0.09, and GO buys a re-read. |
| J14 | **LR-F NULL + X48 NO-GO** | The N4 toy with the review's fixes, or N4-P-direct: the orchestrator's choice. X63 is unaffected. | Not "the architecture is at its ceiling": X48's NO-GO is per-kind untying at the same width over 20M. Width, `D_MODEL` and dose are open. |
| J15 | **LR-F CONTINUE + X48 anything** | X48 is read as registered. LR-F look 2 runs before N4 launches (§4 item 2). | Do not label X48 "LR-confounded" (or anything else) on a CONTINUE. |
| J16 | **LR-F GAIN + the breadth reads** (D3, N1-P, X63) | Breadth NULL / UNDECIDED rows are labelled "at the halved rate". A breadth + floor arm is its own registration. X63 keeps X26c's learning-rate path. | Not that the breadth reads are void, or re-opened by re-interpretation. Do not add the floor to X63: two levers, and X26c has no floor. |
| J17 | **X59-S HUMAN-DATA-BETTER + breadth NULLs** | X60's clause by its code. X60 waits for its review; it does not jump X48 or X63. | Not that "human teams are the missing breadth": X60-T reads that against D1 directly. X59-S is about fresh students, not about P\*. |
| J18 | **X59-S both UNINFORMATIVE + anything** | Stop X59-S's Kakuna read. Every other row is unchanged. | Not "human data is useless". Do not switch to the in-house head-to-heads as the decider: Amendment 5 deleted that route. |
| J19 | **S-U gate GO + H-CTX SUPPORTED** | Each by its own row. V-RANK and S-U compete with X62 / X63 / PR-1′ for the Kakuna lane: the orchestrator orders them. | Not that search "fixes the team-memorisation premium": I is leak-confounded with unknown sign, and the gate is self-refereed. |
| J20 | **Several UNDECIDEDs with positive Δ̂** (D3, N1-P, LR-F, X48) | Nothing beyond each row. | Never "the evidence leans positive across the board". Unregistered pooling of seen reads is the bias this memo exists to prevent. |

**UNDECIDED combinations, named.** No registered rule decides these. The orchestrator rules, and the ruling goes in §6:
- **U1. N1-P TRADE + H-CTX KILLED.** Fund X53-N1 at an in-house price, or not? The exchange rate goes to the owner.
  This memo does not decide it.
- **U2. X48 MIXED.** GO on one meter with harm on the other. No route is registered.
- **U3. LR-F TRADE** (a guard FAILs).
- **U4. D3 or N1-P G-E unplayed** (both need a GPU best responder). The class reads INCONCLUSIVE until the
  orchestrator spends the GPU.
- **U5. X59-S Δ_data and Δ_labels pointing different ways**: for example SELF-PLAY-DATA-BETTER with
  HUMAN-LABELS-BETTER. The code routes Δ_labels to X60-B, and the closure is barred by its own condition. This is
  coherent, but the owner's message must carry both.

---

## 4. Gaps in the registered documents: rulings proposed before the reads land (MY JUDGEMENT)

Each is a place where two documents disagree or neither says. The orchestrator accepts or changes each one, before
the read it concerns lands.

**The orchestrator ACCEPTED rulings 1–7 as written on 2026-10-27, before any of the concerned reads landed.** Each is propagated as one line into the document it governs (ruling 8 was already closed).

1. **N1-P CONTINUE and X63's gate.** X63 §8 waits on "N1-P look 1 (+ look 2 on CONTINUE)", but only a LOSS removes
   n1. **Ruled (accepted 2026-10-27):** on CONTINUE, n1 is IN and X63's PREREG does not wait for look 2. A CONTINUE requires Δ̂ > 0, and
   a look-2 LOSS would need the interval to move wholly below zero.
2. **LR-F CONTINUE and N4.** The N4 review requires "LR-F look 1's class on record" and branches on GAIN. A CONTINUE
   can still end GAIN. **Ruled (accepted 2026-10-27):** LR-F look 2 (8.6 GPU-h) runs after X48's arms and **before N4 launches**. N4 is
   ≈ 30 GPU-h, and a GAIN would force its re-registration; N4 waits on T69 in any case.
3. **LR-F GAIN while X48 is training or trained.** X48 PREREG §9 says the capacity reads are re-designed "before any
   FURTHER width read". **Ruled (accepted 2026-10-27):** X48 is not stopped. It finishes and is read as registered. Its record carries
   "read at the controller's learning rate", and N4 / N4-P re-register with the floor.
4. **X63's d3 gate.** X63 says "F LOSS, or P INFERIOR" ⇒ d3 out. But D3's primary is H, and its rule defines only
   "non-inferior" (LB > −2.0). **Ruled (accepted 2026-10-27):** d3 leaves X63 iff D3's registered H class is LOSS, or Δ_P's 97.5 % UB
   < −2.0 pp (a detected in-pool loss beyond the margin). "Non-inferiority not shown" alone keeps d3 in: it is
   8.6 % of X63's draws, and X63 carries its own in-pool price.
5. **X62 vs X63 order on SUPPORTED.** STATUS ranks X62 below X63 per resource-day; REVIEW_2 §6 says X62 "enters
   level with" X63 when rc72b supports H-CTX. **Ruled (accepted 2026-10-27):** X63 Stage A first in both SUPPORTED and UNDECIDED. Its
   lane cost is 66–95 lane-h against X62's 238–342, and it consumes X62's corpus, so the corpus build is shared.
   X62's fork follows by G1 / G2. A rented CPU box (owner question 2) lets them run side by side.
6. **The S-U gate's rule.** **Ruled (accepted 2026-10-27):** register the review's rule verbatim (GO iff LB > 0.50 on pooled U; the
   GIGO tooth first), so this memo's rows apply. Any change goes in the registration, before any gate state is
   scored.
7. **H-CTX INCONCLUSIVE top-up.** X62's tree names "more home72 / F control units" as the cheap remedy. Adding units
   after seeing an INCONCLUSIVE is a second look. **Ruled (accepted 2026-10-27):** allowed only as a dated registration naming its α,
   the units, and the pooled rule, before any top-up game. Otherwise the HOLD stands and X62 waits for PR-1′ or
   N1-P.
8. **X48's review labels: CLOSED before any ruling.** "Experts not engaged" (M1) and "LR-elevated" (M3) were
   registered as X48 PREREG NOTE 1 (`80973fd75`) before the first GO, while this memo was being written.

---

## 5. FINDINGS (standing rule 7)

1. **X63's gate text names "F LOSS" for D3**, whose registered primary is H. This is a mismatch between two
   documents, proposed for a ruling in §4 item 4. It is not edited in X63's doc here (rule 9).
2. **STATUS and REVIEW_2 disagree on X62 vs X63 under SUPPORTED** (§4 item 5).
3. **H-CTX's pin `905c29f5b` predates the peer-pack split `b0441623e`.** Its F is the legacy-pack regime, like
   N1-P's, D3's and X59-S's. X48, PR-1′ and X63 play the new pack. Legacy-pack F numbers and new-pack F numbers are
   never set side by side as one series.
4. **The `n4_screen` rule is not built** (X48 PREREG FINDING 1; `src/main/reads/` has no such rule at `e52dc800e`).
   X48's read cannot start until it is, so X48 is unlikely to land inside this window.
5. **The S-U gate has no registration or scorer yet.** §2.7 pre-commits to an unregistered proposal; §4 item 6
   closes that gap only if the orchestrator registers it.
6. **LR-F's licence table is written for the toy.** STAGED_PLAN §4.1.2 says production's predictions are §4.1's,
   but production's class consequences are taken here from the toy table plus the X48 and N4 documents. The toy arm
   never ran.
7. **What I saw.** I saw no in-flight result. Visible in documents I had to read:
   - X59-S's committed artefacts (floor references, FR-2 NO-ARTEFACT, the re-encode share 0.60 %);
   - S-U stage 0's one game (a loss at turn 66), in its amendment draft.

   Neither is an outcome of a read in §2.

---

## 6. Decision record

- **2026-10-27: written, blind, before any of the seven reads landed.** It holds:
  - one decision tree per read (§2);
  - 20 joint rows and five named UNDECIDED combinations (§3);
  - eight gaps between documents (§4): seven proposed for a ruling, none ruled yet; the eighth (X48's labels) was
    closed by X48's NOTE 1 (`80973fd75`) before this memo landed.
- **Rule for later entries.** When a read lands, add a dated line here: the class, the row that fired, and any
  departure with the orchestrator's reason. A §4 ruling, once made, is recorded here too. The branches above are
  never edited after their read lands.
- **2026-10-28: LR-F look 1 landed: UNDECIDED, FINAL** (Δ̂ −1.13 pp, Welch 97.5 % [−2.45, +0.18], 3 seeds; record
  `x35_breakers_stage_2026-10-17/reads/lrfloor_look1.json`). The §2.2 row that fired is **UNDECIDED, final (Δ̂ ≤ 0)**:
  nothing is licensed, look 2 does not run (`kit stage-check` REFUSES it), and the capacity reads proceed as for "not
  GAIN". With X48: J13 (X48 GO) names LR-F UNDECIDED-final; J14 (X48 NO-GO) names only an LR-F NULL, so on a NO-GO
  the §2.2 row governs ("as for not GAIN"). Either way X48 carries no "at the controller's learning rate" label. One departure, registered before any game (AM-LRF-L1, `f7fddc39b`; accepted by
  the orchestrator): the guards were registered as not read on a non-qualifying R1, so no G-K Kakuna lane played.
