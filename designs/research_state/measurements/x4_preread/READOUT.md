# X4 pre-read — does the critic see the moves the policy starves? (2026-09-30)

**The question (owner).** Lane S (`a41dcfca`) found that the ai_v14 policy STARVES near-best moves:
on about half of the decisive turns it puts < 1 % on an action whose ground-truth value is within
ε = 0.1 of the best. Does the CRITIC see those moves? One-ply counterfactual Q (X4, the Q-head
experiment, `designs/endstate/design_q_head.md`):

> Q̂_V(s, a) = mean over S shared dice seeds of [the terminal reward if the game ends before our
> next decision, else 2·V(s′_a) − 1], with V the checkpoint's own WIN-PROB critic
> (`--critic winprob`: V = P(win) ∈ [0, 1]) read on the trainee's own observation of the successor.

If V ranks a starved near-best action near the top, one-ply labels hand the policy a gradient it
never gets today. If not, only depth, playouts or a better V (X6) help.

## Verdict — by USE, not one yes / no

**V IS BLIND to the starved moves, and "blind" here means V shares the policy's blind spot.** The
teacher it makes is a better PICKER than the policy, but not a RESCUER of the moves the policy
starves.

Ranges are over the three checkpoints K2 final (G0′), N0 final (75.0M) and C_fix final (83.1M), each
read against its own continuation's truth, variant M (defined below).

| use of one-ply labels | fine enough? | the number that decides it |
|---|---|---|
| **Top-k rescue / anti-starvation floor** (put the starved near-best actions back into a candidate set) | **NO — no better than chance, and no better than a uniform floor** | Q̂'s top-2 holds a starved near-best action **0.20–0.25**, against **0.26–0.27** by chance (2 / n). The same share for the near-best actions the policy FEEDS is **0.44–0.48**. As a target, softmax(Q̂/τ) puts **0.12–0.19** of its mass on the starved actions (variant R: 0.08–0.17), against **0.17–0.18** for UNIFORM. Anything Q̂ feeds them, an entropy floor feeds them too, and Q̂ does not pick out which ones |
| **Softmax distillation** | **only at a SHARP temperature: τ ≤ 0.03 on the ±1 scale (≤ 0.015 in win-prob units); τ = 0.1 is neutral, τ = 0.3 hurts** | Regret gain against the policy's own distribution, on decisive turns: τ = 0.03 ≈ the argmax gain below; τ = 0.1 is **−0.002 to +0.010** (all intervals straddle 0); τ = 0.3 is WORSE than the policy (regret 0.36–0.38 vs 0.31–0.35). The signal lives in Q̂'s fine ordering at the top, not in a coarse ordering |
| **Argmax teaching** | **yes, modestly — IF the opponent is modelled the way the truth models it** | Q̂'s argmax has lower truth regret than the policy's argmax: **+0.038 [+0.017, +0.060] / +0.056 [+0.033, +0.079] / +0.039 [+0.016, +0.062]** on decisive turns (K2 / N0 / C_fix), and **+0.064 to +0.073** on turns where the policy starves a near-best move (all three intervals exclude 0). With the opponent held at its RECORDED action (variant R), the gain shrinks to **−0.007 to +0.020** on decisive turns (not detected) |

**By category (variant M, decisive turns): no category is on the "sees it" side.** The share of
starved near-best actions that land in Q̂'s top-2 (chance ≈ 0.27):
- **switch** (n = 272–430): 0.21–0.23;
- **setup, the owner's delayed-payoff case** (n = 11–20): 0.18–0.25, and Q̂-near only **0.27–0.31** against ~0.5 chance, so V is blind here too. The sample is small and every interval is wide;
- **status**: 0.17–0.29;
- **attack** is the least blind: 0.23–0.35 (K2 0.347 [0.256, 0.445], C_fix 0.327 [0.239, 0.410]; each interval straddles chance);
- **hazard / recovery**: n ≤ 7, no read.

**Tag: MEASUREMENT · V BLIND to starvation (3/3 checkpoints, both opponent variants, every S) · a
one-ply ARGMAX teacher beats the policy's argmax (3/3, variant M only).**

What it means for X4: one-ply labels from this V cannot fix starvation. What they carry is a small
argmax-level improvement at a sharp temperature. That improvement sharpens the policy further, and
sharpening is the direction that produced the starvation. An anti-starvation mechanism needs
something that knows WHICH starved moves are good: depth, playouts, or a better V (X6). Or it can
be a plain entropy / uniform floor, which feeds them exactly as much as Q̂ does.

## The key numbers (95 % intervals: percentile bootstrap resampling battles, ≤ 4 turns per battle; 1,000 draws)

Variant M = the opponent's open root decision answered by the checkpoint's own greedy choice. This
is the ground truth's own construction, so it is the like-for-like read of V. Variant R = the
opponent held at the action it actually played.

| | K2 final (M) | N0 final (M) | C_fix final (M) | range under R |
|---|---|---|---|---|
| turns / decisive | 1,600 / 932 | 1,600 / 943 | 1,600 / 934 | same |
| (a) within-turn Spearman ρ | 0.230 [0.205, 0.255] | 0.226 [0.201, 0.253] | 0.243 [0.220, 0.268] | 0.166–0.175 |
| (a) Q̂ argmax is near-best | 0.588 [0.558, 0.617] | 0.567 [0.539, 0.593] | 0.596 [0.569, 0.622] | 0.532–0.566 |
| (1) Q̂ top-2 holds a near-best | 0.747 [0.721, 0.770] | 0.733 [0.708, 0.755] | 0.754 [0.732, 0.776] | 0.698–0.715 |
| (1) Q̂ top-3 holds a near-best | 0.834 [0.813, 0.853] | 0.821 [0.800, 0.840] | 0.840 [0.821, 0.858] | 0.797–0.804 |
| **(b) STARVED near-best: Q̂-near** (chance) | **0.522 [0.474, 0.567]** (0.519) | **0.453 [0.391, 0.507]** (0.516) | **0.506 [0.458, 0.557]** (0.509) | 0.37–0.43 (chance 0.49) |
| (b) STARVED near-best: in Q̂ top-2 (chance) | 0.251 [0.215, 0.289] (0.266) | 0.204 [0.164, 0.245] (0.268) | 0.245 [0.207, 0.283] (0.264) | 0.16–0.18 |
| (b) FED near-best: Q̂-near / top-2 | 0.825 / 0.469 | 0.769 / 0.436 | 0.833 / 0.481 | 0.75–0.82 / 0.41–0.45 |
| (c) DOMINATED: Q̂-near (false positive) / top-2 | 0.404 / 0.192 | 0.432 / 0.195 | 0.415 / 0.195 | 0.42–0.44 / 0.21–0.23 |
| starved − dominated, Q̂-near | +0.118 [+0.069, +0.162] | +0.021 [−0.044, +0.076] | +0.092 [+0.039, +0.148] | −0.071 to +0.012 |
| (3) truth gap > 0.4 inside Q̂ top-2 (chance) | 0.190 (0.271) | 0.200 (0.273) | 0.193 (0.271) | 0.217–0.226 |
| (3) truth gap 0.1–0.2 inside Q̂ top-2 (chance) | 0.275 (0.277) | 0.281 (0.271) | 0.285 (0.276) | at chance |
| (2) regret on decisive turns: Q̂ argmax / top-2 uniform / top-3 uniform | 0.274 / 0.299 / 0.325 | 0.289 / 0.315 / 0.343 | 0.260 / 0.292 / 0.320 | |
| (2) … softmax τ = 0.01 / 0.03 / 0.1 / 0.3 | 0.273 / 0.281 / 0.317 / 0.362 | 0.293 / 0.304 / 0.342 / 0.382 | 0.264 / 0.275 / 0.311 / 0.356 | |
| (2) … policy argmax / policy distribution | 0.312 / 0.316 | 0.345 / 0.351 | 0.299 / 0.311 | |
| **(e) teacher gain on STARVED turns: Q̂ argmax vs policy argmax** | **+0.064 [+0.035, +0.093]** | **+0.072 [+0.037, +0.111]** | **+0.073 [+0.042, +0.105]** | +0.019 to +0.036 (N0 alone detected) |

- **(4) Paired vs absolute error.** Mean |Q̂ − truth| is **0.31–0.35**; the mean SIGNED error is
  **+0.14 to +0.19**, so V is optimistic against the greedy-continuation truth. V's errors are mostly
  COMMON-MODE: 68–71 % of their variance is a per-turn offset, with a within-turn correlation of
  0.63–0.66. The difference Q̂(a) − Q̂(b) is therefore better than the levels: RMS pair error is
  0.34–0.36, against 0.57–0.63 if the errors were independent. But the residual is still **~2.7×
  the noise floor (0.13)** and far above ε = 0.1, so V's DIFFERENCES do not resolve near-equal moves.
- **(5) Terminal / V mixing.** Actions with a game-ending branch carry a signed error of +0.04 to
  +0.08; the all-V actions on the same turns carry +0.01 to +0.03. The difference, **+0.02 to +0.05**,
  is not detected in any checkpoint (every interval straddles 0). Paired seed by seed with the exact
  successor's truth outcome, V-scored branches are nearly unbiased on the turns where some branch
  ends the game (+0.03 to +0.07). They are strongly optimistic on every other turn (+0.14 to +0.19).
  So the mix does not systematically favour game-ending moves at this resolution. V's optimism sits
  on the undecided turns, not the decided ones.
- **Calibration.** On the exact successors, 720k V-scored branches per checkpoint: mean V is
  0.66–0.68 against a win rate of 0.58–0.59; Brier 0.163–0.167 (a constant scores 0.241); ECE
  0.07–0.09. V is over-confident in every decile from 0.1 to 0.9, most in the middle: at V ≈ 0.55 the
  win rate is 0.38–0.40. The top decile is calibrated.
- **Does S matter? No.** Q̂ on 1 seed reads almost the same as on 32. Against the held-out truth on
  seeds 32–63, K2 goes ρ 0.206 → 0.226, top-2 recall 0.723 → 0.739, regret 0.313 → 0.305, and
  starved Q̂-near 0.48 → 0.50. N0 and C_fix are the same. V's own error dominates the one-turn
  dice noise, so S = 1–4 is enough for X4's labels.
- **Cross-continuation.** Every checkpoint's Q̂ against each of the other two continuations' truth
  gives the same picture: starved Q̂-near 0.41–0.51, dominated 0.41–0.45, and Q̂'s argmax regret
  below the policy's on starved turns in all six cross pairs (`readout_tables.md`).

## Cost per labelled decision (the X4 cost model)

A single-worker benchmark: 200 subset turns, S = 8, K2, CPU, nice 19, box load 1.0–3.8 (warn,
never stretch). Rows: `~/gen3ai_archive/x4_preread/bench_K2_S8_t{1,4}.jsonl`.

| | 1 torch thread | 4 torch threads |
|---|---|---|
| successor (sim + fold + encode, the playout step path) | **133 µs** | 135 µs |
| V forward per successor row | **946 µs** | 549 µs |
| all-in per successor (incl. the root replay and the opponent's greedy forwards) | 1.76 ms | 1.17 ms |
| **one labelled decision** (every legal action × 8 seeds, ~7.0 actions ≈ 56 successors, one opponent variant) | **~98 ms** | **~65 ms** |

The successor itself costs less than Lane I's 232 µs (that was the search TREE's step-built rows).
**The critic forward is ~7× the sim**, so X4's label cost is a forward-pass budget. Batching labels
across roots on the GPU is the lever, not the sim. Since S = 1 reads like S = 32, a label needs
~7 successors, not 56: **~12 ms at 1 thread** on this CPU.

## How it was measured

- **Reader:** `src/main/policy_spectrum/qhat.py` (branching, durable rows) and `qhat_report.py`
  (every read). Tests: `qhat_report_test.py` (a Q̂ equal to the truth sees every starved move, a
  reversed one sees none; rule regrets; the joint bootstrap) and `qhat_integration_test.py` (every
  action × seed scored; CRN-deterministic; R closes the open root; the TEETH below).
- **Branching:** Lane I's playout core (`utils.rust_env.successors`), driven one decision at a time.
  Each branch advances only to OUR next decision, or to the end. The captured row is the training
  path's row for that decision.
- **Seeds:** Q̂ uses Lane S truth's own 64 seeds (`truth.turn_seeds`), so seed k rolls the same
  first-turn dice as truth playout k. ⚠️ Comparing Q̂ on seeds 0..S−1 against a truth that INCLUDES
  them shares noise, so every read uses a HELD-OUT truth: Q̂ on seeds 0–7 against the truth on
  seeds 8–63, and the S-curve against seeds 32–63. The leak check (Lane S's exact 64-seed sets)
  moves each read by ≤ 0.03 (top-2 recall rises most), so the leak is small.
- **Opponent at the root (both variants reported).** On **1,441 of the 1,600 turns** the
  opponent's root request is still OPEN at the branch point (it answered after us in the log). The
  ground truth answered it with the continuation's GREEDY choice there, not the recorded one.
  - **M** reproduces that construction (the checkpoint's greedy), so it is the like-for-like read of V.
  - **R** holds the recorded action, as the brief asked. Its recorded answer is moved in front of
    ours in the log, and one turn needed a rejected-then-retried pair moved (a hidden trap, below).
  R reads lower on every measure. Part of R's disagreement with the truth is that R and the truth
  answer different opponent moves on 90 % of the turns.
- **Sanity (the teeth).** The played action under the battle's OWN dice, with the opponent on its
  recorded answers, reproduces the recorded next observation **byte for byte on 1,548 of 1,548
  turns**, all three checkpoints; 52 turns have no next decision. V there equals V on the recorded
  row to 3e-7. Q̂_8(played) against V(recorded next): mean difference +0.002 to +0.003, mean
  absolute 0.03–0.04, correlation 0.98.
- **Branch facts per checkpoint:**
  - 731,008 branches (M rows for 1,441 turns on top);
  - 11,648 ended before our next decision (terminal reward);
  - 51,605 had an opponent-only decision (its replacement) answered by greedy before capture;
  - 118.6k were captured at a SWITCH-ONLY decision (our mon fainted mid-turn or a forced switch);
  - 85 turns mix game-ending and V-scored branches;
  - 0 refused after the repair (one refusal before the trap fix was re-run).
- **Scale:** Q̂ on ±1 = 2V − 1. A tie scores 0 (truth's convention). The winprob critic was trained
  with ties as losses (0), a ≤ 0.5-point difference on the rare tie.

## Turn mechanics (verified in `deps/pokemon-showdown` @ `e0551883f` before branching)

See [`mechanics.md`](mechanics.md): file:line for every answer, and how each decision type holds or
models the opponent. In short:
- **after a faint, only the fainted side chooses**, and in gen ≤ 3 it does so **mid-turn**, after
  every action (`sim/battle.ts:2861–2864`, `:2933`);
- **a double faint is SIMULTANEOUS** (one `makeRequest('switch')` to both sides, resolved at
  `allChoicesDone`);
- **Baton Pass leaves the opponent's move LOCKED** in the saved queue (`:3021–3040`);
- **the next turn's choice is made after seeing the switch-in** (`endTurn` → `makeRequest('move')`,
  `:1797`).

The X4 spec's §5.1 claim ("sequential after a faint; the opponent's reply must be modelled") is
**partly confirmed** (single faint; the next-turn reply) and **partly contradicted**: a double
faint is simultaneous, and a mid-turn replacement can face an opponent move that is already locked,
not a reply.

## Honest limits

- **The truth is conditional on a GREEDY continuation**, both sides, not Nash and not the recorded
  opponent. V was trained to predict P(win) under stochastic self-play and pool opponents, so some
  of its "error" is a target mismatch. The mismatch shows as V's optimism (+0.14 to +0.19), which is
  mostly common-mode and cancels in the within-turn reads that decide the verdict.
- **The winner's curse:** V* is a max of 56 noisy means and biased upward. That inflates every
  rule's regret EQUALLY, so rule-vs-rule differences are unaffected. It also widens gaps to the best
  action, which makes "near-best" slightly conservative.
- **ε on Q̂'s scale:** "Q̂-near" uses ε = 0.1 on Q̂'s own spread, which is compressed relative to the
  truth's, so Q̂-near rates are high for every group. The chance column and the rank reads (top-k,
  regret) are scale-free, and the verdict rests on them.
- **Selection:** the subset over-samples non-attacking categories. Per-category rows condition on
  the category. Setup / hazard / recovery have ≤ 20 starved actions each, so their rows are
  anecdotes.
- **Policy used for "starved":** each checkpoint's own probabilities from the Lane S baseline read
  (`~/gen3ai_archive/policy_spectrum/baseline_2026-09-29/<label>.probs.npz`).
- Every N0 checkpoint was trained under the compile miscompile (Lane S's caveat).
- The bank's bot-opponent states predate the F-LF-1 bot fix (Lane S's caveat).

## Files

- `READOUT.md` (this), `mechanics.md`, `readout_tables.md`: every table, both variants, every
  checkpoint, the cross-continuation, S-curve, leak, sanity and branch facts.
- `readout.json.gz`: the full readout, `gen3_x4_preread_readout_v1`.
- Rows (archive, not committed, ~20 MB each). sha256:

  | file | sha256 |
  |---|---|
  | `~/gen3ai_archive/x4_preread/qhat_K2final_S64.jsonl` | `fe04d4b588786ea8…` |
  | `qhat_N0final75M_S64.jsonl` | `c8b2b0b124382f92…` |
  | `qhat_C_fixfinal_S64.jsonl` | `4a4abe9ca5cf9ca0…` |

  Plus `sanity_<label>.jsonl` and `bench_K2_S8_t{1,4}.jsonl`. The rows regenerate
  deterministically (CRN, fixed seeds) in ~20 min per checkpoint on 3 CPU workers.

To reproduce:

```bash
export PYTHONPATH=$PYTHONPATH:src
B=designs/research_state/measurements/m5_laneS
python -m main.policy_spectrum.qhat run --bank $B/bank_v1 --subset $B/truth_v2/gt_subset_v2.json \
    --ckpt models/ai_v14_07_g0p_k2/final_model.zip=K2final --out <dir>/qhat_K2final_S64.jsonl --seeds 64 --workers 3
python -m main.policy_spectrum.qhat sanity --bank $B/bank_v1 --subset $B/truth_v2/gt_subset_v2.json \
    --ckpt models/ai_v14_07_g0p_k2/final_model.zip=K2final --out <dir>/sanity_K2final.jsonl
python -m main.policy_spectrum.qhat_report --bank $B/bank_v1 --truth-dir $B/truth_v2 --qhat-dir <dir> \
    --probs-dir ~/gen3ai_archive/policy_spectrum/baseline_2026-09-29 --out <dir>/readout.json --md <dir>/readout_tables.md
```

(the release cdylib: `cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo build --release --lib`)
