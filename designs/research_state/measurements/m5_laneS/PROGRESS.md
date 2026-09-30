# M5 Lane S — the POLICY-SPECTRUM instrument: PROGRESS (resume point)

Lane S of `designs/endstate/program_rust_core.md` §2 M5 (owner, 2026-09-29). Owns
`src/main/policy_spectrum/` (bank builder, replay, categories, spectrum, reader, report, CLI, tests)
and this directory (`bank_v1/`, the baseline read, this file).

**The owner's intent:** save ≥ 10,000 turns ONCE, then compare the SAME turns across policies
(checkpoints over training, future architectures, search, Q-derived mixes): how much probability goes
to the nth-best choice. Later, with branch ground truth: starvation over the WHOLE move space
(near-best moves losing mass) vs healthy sharpening (dominated moves losing mass).

Gates: ① re-encodable bank (byte-equal re-encode at the recording commit); ② ≥ 10,000 decisions,
stratified and stamped; ③ the reader (any checkpoint, forward passes only, rank-mass spectrum +
entropy per stratum); ④ ground truth on a ~1–2k-turn subset via Lane I; ⑤ the trend with ground
truth.

## How to run

```bash
export PYTHONPATH=$PYTHONPATH:src
B=designs/research_state/measurements/m5_laneS/bank_v1
python -m main.policy_spectrum gate --bank $B                       # gate ① at this checkout (~45 s)
python -m main.policy_spectrum read --bank $B --out <dir outside models/> \
    --ckpt models/<run>/checkpoints/<ckpt>.zip=<label> [...]         # ~45 s re-encode + ~11 s / ckpt, CPU
python -m main.policy_spectrum report --bank $B --reads <dir> --labels L1,L2 \
    [--pair A:B ...] --out <report.md>
python3 -m pytest src/main/policy_spectrum -q                       # routine (unit + gate ① slice + reader teeth)
python3 -m pytest src/main/policy_spectrum -q -m slow               # gate ① over the whole bank
```

`build` exists (`python -m main.policy_spectrum build --out <dir>`) but **the v1 bank is written
once** — a second build refuses without `--force`. It is byte-deterministic (two builds: identical
`.gz` files and manifests).

## Units

| # | unit | status |
|---|---|---|
| 1 | the BANK (`bank.py`, `replay.py`, `categories.py`) + `bank_v1/` + gates ① and ② | LANDED (git log: `M5 Lane S unit 1+2`) |
| 2 | the READER (`reader.py`, `spectrum.py`, `report.py`, CLI) + gate ③ | LANDED (same commit) |
| 3 | the BASELINE READ over the fixed lineage (`baseline_2026-09-29/`) | LANDED (same commit) |
| 4 | gate ④: ground truth via Lane I on a first ~200-turn subset | CODE LANDED (`truth.py`, `python -m main.policy_spectrum.truth select\|run\|read`; tests `truth_test.py`, `truth_integration_test.py`) + the subset `truth_v1/gt_subset_v1.json` (236 turns: 216 free + 20 forced); the first run (K2 final greedy continuation, S = 16) IN PROGRESS → `truth_v1/rows_K2final_S16.jsonl` |
| 5 | gate ⑤: the trend with ground truth | after ④ (spec below) |

## The bank (`bank_v1/`, content sha `8ca1bfa544bf…`)

**Source: the lineage's own EVAL TRACES.** Why: each trace carries the sim's reconstruction record
(seed + both packed teams + the command log — exactly the core input log), AND the obs / logits /
mask / action the recording policy saw at every decision. So gate ① is checkable on every recorded
decision with no new games, and the eval roster already spans the opponent classes. Freshly
recorded games through the core were the alternative; the scripted bots are not in the Rust env
yet (Lane F), so bot games would still have come through the old path, with no recorded obs to
check against. Traces are also what the prober reads, so a banked turn can be inspected there.

- **20,712 decisions in 580 battles**, 146 distinct teams for our side. Decisions with a single
  legal action are dropped (1 legal action has no spectrum; counted in the manifest).
- **Sources** (per eval cycle × roster opponent: 2 wins + 2 losses + 1 draw, drawn in
  `sha256(bank_seed:prefix)` order, a short bucket filled loss → win → draw): N0 @ 36M / 46M / 56M /
  66M / 74M, C_fix @ 82M, K2 @ 90M, K3 @ 98M (the eval snapshot of that cycle played our side); and
  **K2 final against the round-0 exploiter A′** (A′'s 20 eval games vs its target: the traced side
  is A′, the bank takes the UNTRACED side = K2 final's own decisions).
- **Strata stamped on every decision:** `kind` (free 18,907 / forced switch 1,805; `switch_only`
  exists as a stamp, 0 in v1), `phase` (opening = turn ≤ 3: 1,943; endgame = either side ≤ 2 mons:
  5,644; midgame 13,125), `n_legal` + bucket (2–3: 1,154; 4–6: 6,335; 7+: 13,223), `opp_class`
  (bot 11,109 over the 9 roster bots; pool snapshot 8,855 over sentinels 0–4; exploiter 748),
  `opp_name`, our `team` hash, the battle `outcome` (win 9,700 / loss 9,147 / draw 1,865), the
  `source` cycle, and every legal action's token and move CATEGORY — attack 17,538 / status 11,669 /
  setup 3,984 / hazard 1,642 / recovery 5,258 / switch 19,533 decisions where legal (status moves also
  carry a subtype: inflict / protect / phaze / cure / other). Rules: `categories.py`'s docstring;
  the 0-base-power damaging set is pinned against Showdown's own `category` by test.
- **What of the recording is kept:** the sha256 of the recorded obs row, the recorded action, and
  the recording policy's 11 raw logits (the shortest decimal that round-trips each float32). No obs
  vector is stored.
- **Selection bias (honest limit):** the trace quota is LOSS-ENRICHED by design; the per-outcome draw
  balances it inside each (cycle, opponent), but the bank is not a random sample of play — a
  stratum's SHARE of the bank is not its share of games. Every policy is also read partly OFF its
  own state distribution (the turns were visited by N0 36M–75M, C_fix, K2, K3).

**Gate ① — PASS.** Every recorded decision of every banked battle's traced side (21,087 decisions,
incl. the exploiter side of A′'s games and the single-legal decisions later dropped) re-encodes
BYTE-EQUAL through `core_events --trackers --obs` at HEAD, with an equal mask and the recorded action
equal to the token the log played. The recordings span four commits (`8d07051a`, `6521f420`,
`a5b2fdba`, `13cc85ec`); all reproduce at the build commit, so no pinned re-encode was needed.
A request the log never answered (the 250-turn stall FORFEIT's `forcelose`) is not a decision: the
replay drops it, and only the final request may be unanswered (refused otherwise). Routine test:
a slice (4 battles per source) + teeth (a flipped obs byte, a dropped decision, a wrong action each
fail); `slow`: the whole bank. The byte check is REQUIRED while the encoder identity stamped in the
manifest (obs-golden sha, obs width, arch signature) equals the checkout's; after a deliberate
encoder change the rows belong to the recording commit, and the routine test checks only that every
banked decision replays to the same tokens and choice.

**Gate ② — PASS** (`bank_test.py` on the committed bank): ≥ 10,000 decisions, every stamp in
vocabulary, every category legal on ≥ 1,000 decisions, all three opponent classes, all 9 bots,
≥ 50 teams, the manifest's counts equal a recount, the files unedited (content sha).

## The reader (gate ③ — PASS)

Re-encodes the bank once per invocation (~45 s, 2 core processes), then per checkpoint: CPU,
eval mode, fp32, fixed threads, batch 256, `load_foreign_opponent` (arch signature verified; a
bare run dir is refused). Output: `<label>.json` (every stratum: rank-mass spectrum ranks 1–6 + tail,
entropy, normalised entropy, effective choices, top-1 ≥ 0.9 / 0.99; per category: mass, share of
turns with < 1 % mass, share top-1, best rank; the 1st/2nd/3rd choice by category; battle-clustered
95 % bootstrap intervals, fixed seed) + `<label>.probs.npz` (per-decision probabilities and logits,
bank order, keyed by the decision-id hash) for paired comparisons and gate ④.
- **Teeth:** reading an eval snapshot reproduces the probabilities it RECORDED on its own banked
  decisions: N0@74M's eval snapshot, 2,748 decisions, max |Δp| 4.1e-6, argmax agreement 1.000.
- **Deterministic:** two reads give identical probabilities and JSON (test).
- **Every read states** whether it ran on the recorded observations (`reencode.obs_as_recorded`).

## The baseline read (`baseline_2026-09-29/REPORT.md`; per-read JSONs beside it; the probs npz in `~/gen3ai_archive/policy_spectrum/baseline_2026-09-29/`)

13 checkpoints: N0 @ 2.4 / 9.5 / 21.7 / 30.1 / 39.8 / 48.2 / 60.1 / 69.9 / 75.0M, C_fix @ 83.1M, K2 @
91.1M (G0′), K3 @ 99.2M, and A′ @ 92.1M (**INTERRUPTED**, SIGTERM ~1M steps after its fork — not a
final exploiter). ⚠️ Every N0 checkpoint was TRAINED under the CUDA compile miscompile; C_fix is the
first block on the fixed learner. All six runs' checkpoints carry `ent_coef` 0.05.

Headline (all 20,712 turns; rank-1 / rank-2 / rank-3 mass, entropy in nats):

| policy | rank 1 | rank 2 | rank 3 | entropy |
|---|---|---|---|---|
| N0 @ 2.4M | 0.474 | 0.216 | 0.133 | 1.376 |
| N0 @ 9.5M | 0.586 | 0.193 | 0.096 | 1.107 |
| N0 @ 30–70M (range) | 0.550–0.580 | 0.186–0.197 | 0.095–0.105 | 1.12–1.20 |
| N0 @ 75.0M | 0.559 | 0.193 | 0.103 | 1.161 |
| C_fix @ 83.1M | 0.616 | 0.180 | 0.086 | 1.006 |
| K2 @ 91.1M | 0.629 | 0.177 | 0.082 | 0.976 |
| K3 @ 99.2M | 0.621 | 0.179 | 0.085 | 0.993 |

- **N0 sharpened by 9.5M and then stopped** — on free turns it drifted slightly BACK over 9.5M → 75M
  (rank 1 −0.033 [−0.041, −0.023], entropy +0.065), while forced switches kept sharpening (+0.033).
- **The one large sharpening step is N0 75M → C_fix** (+8M on the fixed learner): rank 1 +0.057
  [+0.052, +0.062], rank 2 −0.013, rank 3 −0.017, entropy −0.155 — rank 1 rises in 49 of the 50
  reported strata (every kind, phase, legal bucket, opponent class, source and outcome; the one
  exception is turns against the `random` bot, −0.001). C_fix → K2 adds +0.013; K2 → K3 gives back −0.008. The
  association with the compile fix is not a cause: C_fix is also 8M more steps (confounded).
- **By category (free turns where the category is legal):** the MEAN mass on status / setup /
  recovery barely moved after the fix (status 0.249 → 0.230, setup 0.202 → 0.188, recovery 0.220 →
  0.198 from N0 75M to K3), but the share of turns where a legal category holds < 1 % DOUBLED: setup
  0.165 → 0.307, status 0.091 → 0.183, recovery 0.060 → 0.158, switch 0.150 → 0.256. The fixed-learner
  policy is more decisive about non-attacking moves — near-all or near-nothing. Whether the
  near-nothing turns are starvation (the move was near-best) or correct avoidance is exactly gate ④'s
  question; this read cannot tell.
- Over N0's own 9.5M → 75M the policy moved mass TOWARD status and setup (status 0.205 → 0.249,
  setup 0.132 → 0.202, attack 0.559 → 0.495; the attack share of the top choice 0.632 → 0.544).
- Cross-check with the ledger (X22(f)): N0 final's mean rank-1 mass here is 0.559 (free turns 0.546)
  — the probability that a T = 1 sample equals the argmax; X22(f) counted 62.8 % own-argmax play
  against Kakuna on Kakuna games' states. Same order; different states.
- A′ (interrupted) is BROADER than its parent K2 (rank 1 −0.044 [−0.050, −0.038]) — consistent with a
  fresh fork training against one target at `ent_coef` 0.05, but it is 1M steps old.

## Gate ④ — the SPEC for the next unit (ground truth via Lane I, landed `2c84729e`)

API (`src/utils/rust_env/successors.py`): `play_out(log, at, side, policy=greedy(scorer),
seeds=[...], actions=None, stall="production", max_turns=999)` → `res.values()` = {action: mean value
over seeds}; `record_to_log(record)` turns a reconstruction record into the log. Seeds are SHARED by
every action (common random numbers). ~170 playouts/s per thread with a trivial scorer; the network
forward dominates with a real one; one handle per thread.

1. **The subset.** A fixed, stratified draw from `bank_v1` (a `gt_subset_v1.json` of decision ids,
   drawn by hash with a declared seed): first ~200 FREE turns, then grown to ~1–2k; proportional
   over phase × opponent class, with every category represented (≥ 30 turns where hazard / setup /
   recovery is legal). Forced switches as their own ~10 %.
2. **Mapping.** A bank decision `(battle, side, n)` → Lane I's `at` = the index in the log's command
   list of that side's n-th CHOOSE. VERIFY per turn: the root's `tokens` must equal the bank's
   `tokens` (refuse otherwise) — the n-th answered decision is assumed to be the n-th command of
   that side.
3. **The continuation policy** (both sides, greedy): K2 final (G0′), forward on CPU through the
   reader's `policy_logits`. Every action's value is then "the value of this action followed by
   G0′'s greedy play on both sides" — a DECLARED continuation, not a Nash value; the report says
   so. A second continuation (a scripted bot, or N0 final) as a sensitivity row on a slice.
4. **Seeds.** S dice seeds per turn (start S = 16; the CI of each action's value is a Wilson/Bernoulli
   interval over S; grow S until the near-best set is stable). Values: +1 / −1 / tie 0 / truncated 0.
5. **Readouts per policy on the subset:** best true value V*; NEAR-BEST set = actions with value ≥
   V* − ε (ε = 0.1 on the ±1 scale, and a CI-aware variant: not separable from the best at 95 %);
   DOMINATED = the rest. Mass on near-best vs dominated; the STARVATION rate = turns where some
   near-best action gets < 1 % mass, per category of that action; regret = V* − Σ π(a) v(a). At
   genuine GUESS turns (≥ 2 near-best actions of different categories) near-best mass should stay
   split.
6. **Caveats to carry:** F-LI-4 (every failure is `SuccessorsError`: a bad request cannot yet be told
   from a port refusal mid-branch — count and report refused turns, never drop silently); F-LI-6
   (`max_turns` must stay < 1000); F-LI-1 (replacement-switch expansion in the search TREE — not the
   playout path, but check no refused branch clusters on forced switches). CPU cost: ~200 turns ×
   ~7 actions × 16 seeds ≈ 22k playouts of ~40 decisions each — CPU hours with a real network; run
   detached, incremental (one durable row per turn), nice 19.

## Gate ⑤ — the SPEC

Read every snapshot of a lineage on the ground-truth subset (the reader's probs + ④'s values):
per snapshot, mass on NEAR-BEST vs DOMINATED, the starvation rate per category, regret. Healthy
sharpening = dominated mass falls while near-best mass (and its split at guess turns) holds;
starvation = near-best mass also falls. The first series: N0 2.4 → 75M, C_fix, K2, K3 (the reads in
`baseline_2026-09-29/` already hold every probability needed; only ④'s values are missing).

## Open findings

- **F-LS-1: `ent_coef` is 0.05, not 0.02.** All six checkpoints read here (N0 2.4M and final, C_fix,
  K2, K3, A′) store `ent_coef` 0.05 in their SB3 `data`; EXPERIMENT_BACKLOG X23 describes the arm as
  "`--ent-coef` 0.02 → 0.005". The arm's baseline must be read from the checkpoint before it runs.
- **F-LS-2: the exploiter class is thin** — 748 decisions from 20 battles of ONE exploiter, and that
  exploiter (A′) was interrupted at +1M steps. Grow it when round-0's exploiters finish (a `bank_v2`
  that adds sources; `bank_v1` never changes).
- **F-LS-3: N0's early cycles are not on disk.** N0's eval traces start at 36M, so the bank has no
  turns visited by the early (< 36M) policy; early checkpoints are read on later-policy states.
- **F-LS-4: the bank inherits the eval roster's teams.** Our side's teams are the eval draw (146
  distinct); the ladder campaign's ~5 iconic Smogon sample teams are not specially represented.
- **F-LS-5: `core_events` is the replay path** (the port's M1/M4 tool, whose rows slice O pins to
  the Python encoder and Lane 0's gate ① pins to `sim_bridge`), not the Lane 0 env core itself: the
  env core stages episodes by team index + a numeric seed and cannot replay a "sodium,<hex>" eval
  seed or an arbitrary command log. Lane I's `play_out` takes the same input log, so ④ needs no
  conversion beyond `record_to_log`.
