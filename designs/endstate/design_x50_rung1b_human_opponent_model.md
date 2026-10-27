# X50 rung 1b: what to do with the human labels — a human opponent model for search, and the intent head offline (design + pre-registration)

**Status (2026-10-23):** DESIGN + PRE-REGISTRATION (§8). **Step 0 BUILT** (C-GAME, §2 "As built", `d66a74ec1`).
**R0 READ: FIXED**, P_c +0.012 nats [+0.005, +0.020]; R1's gap re-read: **GAP, +0.253 [+0.232, +0.276]**
([the read](../research_state/measurements/x50_r1b_r0_2026-10-23/README.md)). The owner approved (b) on 2026-10-23.
**(b)'s HOM BUILT and FROZEN (2026-10-24):** 1,762,236 training rows (19 % of training battles removed by the
team-memory rule); validation NLL 1.766 (P\*) → 1.156; sha256 `76e29a01…b36f`
([the build](../research_state/measurements/x50_r1b_hom_2026-10-23/README.md)). **R-b1 READ: LEARNED**, Δ NLL −0.582 nats [−0.599, −0.565]
(voluntary switches −0.874). **R-b2 READ: HOM-NONE**: bot AUC 0.459 [0.406, 0.510] (NO-EVIDENCE), ΔAUC vs P\* −0.013
[−0.068, +0.045]; the lever is CLOSED for search at P\* (§6). R-a1 is not read.
No GPU. This is the
training-use design that R1's GAP verdict asked for
([`design_x50_rung1_intent_labels.md`](design_x50_rung1_intent_labels.md) §5;
[the R1 read](../research_state/measurements/x50_r1_intent_read_2026-10-20/README.md), `6d10e6f9`).
**Owner grant:** rung 1 only, i.e. opponent-action labels for the intent head (2026-10-20). Rungs 2+ stay closed.
**Building use (b) needs the owner's explicit yes** (§1.3, F-1b-1).

## 0. The answer in one screen

**Recommendation: both uses, in this order. Step 0 runs first either way.**

1. **Step 0: fix F-R1-1.** Complete the viewer's own side from Smogon sets and re-measure the pipeline's cost (R0).
   Today that cost is 0.23 nats, as large as R1's gap. Both uses read the same completed rows. One extraction gives
   two labels per row: the opponent's action for (a), the viewer's own action for (b).
2. **(b) first in priority: a separate HUMAN OPPONENT MODEL (HOM).** It is X26's network, fine-tuned from P\* on
   ≈ 2M choices by human players rated ≥ 1500. It is used ONLY as search's opponent-choice likelihood judge and never
   read by the policy. It is the lever the off-pool L3 read points at: with our own network as the judge, the
   opponent's choices carry evidence only when the opponent is our own network (bots 0.472, NO-EVIDENCE;
   snapshots 0.794). Its decisive read, **R-b2**, re-scores the frozen off-pool bank with the HOM as the judge.
   **PASS = the bot stratum's AUC reads WORKS (lower bound > 0.555) AND beats P\* in a paired comparison.** A pass
   re-opens search's opponent-choice route for opponents that are not our network.
3. **(a) second, OFFLINE only: refit the intent head on human + self-play labels (R-a1).** The trunk stays frozen,
   which is all the intent labels can change in production (the head is `detached`). The read compares the refit
   head with X26's head on held-out humans (switches above all) and checks that it does not regress on self-play. No
   GPU training use is recommended now: we play no humans, so a human-fit intent prediction has no consumer in play,
   and the policy would read a head pulled away from its training opponents. A GPU arm is registered separately,
   only if R-a1 reads MIXABLE and a consumer is named. Metamon is the only candidate: it learned from human replays.

**Is either one a prior the network reads at play?** (§1)
- **(a): no.** It is rung 1 as granted. The human data enters as labels in a loss. The row the network reads is
  unchanged.
- **(b): no for the policy, which never reads it.** At play, search would consult a learned likelihood trained on
  sanctioned labels and Smogon-completed inputs. The priors rule allows that.
- **But (b) sits at the edge of the grant.** The grant says "for the intent head", and (b) reads human choices from
  the human's own seat with a completed own side. That is the partial-information nuance behind the owner's
  2026-08-18 punt.

**Cost (§7):**
- CPU: ≈ 15–18 CPU-hours on the agent cores (3 physical cores), in ≤ 45-min resumable chunks.
- GPU: 0 GPU-hours (≈ 0.2 GPU-hours optional, for the HOM's training, if a lease window is idle).
- Builds: ≈ 3–4 agent-days (opus-high).

## 1. The two uses against the rules

### 1.1 The rules

- **The priors rule** (owner 2026-08-15; root `CLAUDE.md` § Paths, the model, data). The line is INPUT vs EXPERIENCE.
  Anything the network READS (obs features, tables, `data/` priors) must trace to Smogon, ground-truth labels or
  ladder replays. The team pool only measures structure and never ships. Ladder-replay actions are an allowed source
  of training labels (ai_v11 D2: "doubly sanctioned").
- **The rung-1 grant** (2026-10-20): opponent-action labels for the intent head. Behaviour cloning of our policy
  (rung 3) and anything beyond labels (rung 2's outcome labels, a human prior) stay closed.

### 1.2 (a) The intent head: ALLOWED. It is rung 1 itself.

- **No input changes.** The row is the production encoder's reading of the log's public lines. The viewer's own side
  is completed from Smogon draws (§2), never from the pool and never from a later reveal of the OPPONENT.
- **The human data is experience.** It enters only as labels in the intent loss. Production runs that head
  `detached` (`opp_intent_grad_mode`), so the labels train only the head's own parameters and cannot reshape the
  trunk.
- **At play the policy reads the head's output.** α feeds the move-resolution family and the pair-outcome op
  (`ARCHITECTURE.md` §3), so a human-fit α would steer play. That is a learned output trained on sanctioned labels,
  not a prior: no number derived from the replays is fed in as a feature, a table or an initialisation.
- **The condition:** the completed own-side sets are an INPUT during the human-label passes, so they must be
  Smogon-drawn (§2). They are.

### 1.3 (b) The human opponent model: ALLOWED by the priors rule; the grant's scope needs the owner's word

- **The policy never reads it.** It is a separate network file. Only the opponent-choice likelihood
  (`main.value_search.opp_likelihood`, its `Forward`) calls it, for search's world weights. The policy's inputs, its
  weights and its training are untouched.
- **What it is trained on is sanctioned.** Labels: human choices from ladder replays. Inputs: the encoder's rows,
  with the acting human's own side completed from Smogon draws plus the facts the log reveals.
- **What reaches play is a likelihood, not a prior.** If search ever runs live, the only human-derived thing in the
  loop is the HOM's estimate of how likely the opponent's observed choices were in each candidate world. That is a
  model's output conditioned on the state. The world sampler's prior over teams stays Smogon's. The rule's "ladder
  replays" clause covers it.
- **The grant's words are narrower.** "Opponent-action labels for the intent head."
  - For: the HOM's labels ARE human opponents' actions. The recommendation the owner approved named the purpose:
    "human opponent actions as intent labels, offline, feeding search's opponent model"
    ([`design_bot_ideas_screen.md`](design_bot_ideas_screen.md) §5). R1's own rule kept the door open: "unless
    search's opponent model wants them".
  - Against: the HOM reads each choice from the human's OWN seat with a completed own side. That is the
    partial-information nuance the owner punted on 2026-08-18. The rung-1 doc's status line also lists "behaviour
    cloning" among the closed rungs. There the phrase means rung 3 (a warm start of our policy), which (b) is not:
    no policy initialisation, no policy loss, nothing the policy reads.
  - So step 0, R0 and R-a1 run under the grant as it stands. **The HOM's build waits for the owner's yes**
    (F-1b-1). The registration below is fixed now either way.

### 1.4 Rejected forms

- **The intent head as search's judge.** It predicts the opponent's choice from OUR view, so it does not depend on
  the candidate world and cannot rank worlds. Forcing its belief inputs to a world's facts feeds it hard beliefs it
  never trained on.
- **A fresh HOM from random weights.** About 2M rows is small for a 5.6M-parameter network. It also adds a second
  lever: R-b2 must compare a human fine-tune with its own starting point, P\*.
- **The HOM as the opponent's policy in playouts.** That is a different use, with its own read. Not in this unit.

## 2. Step 0: fix F-R1-1 (the viewer's own side, completed from Smogon sets)

**The problem.** R1's spectator rows cost X26's head 0.23 nats on X26's own battles
(S − S-full +0.229 [+0.206, +0.254]). The cost has three sources:
- the viewer's own side is public-only (what its own mons revealed so far);
- the reader fabricates the own spread (IVs 31, EVs 0, nature serious; F-X50-4);
- the decision's legality is synthesised from revealed moves only, with a Struggle placeholder.

(b) cannot run that way at all. Its label is the viewer's own action, and a first-use move is not legal in a
revealed-moves-only request.

**The fix: C-GAME.** For each seat of each battle, ONE completed packed team, drawn once (seeded by a hash of the
battle and the seat):
- **every own fact the log reveals by the end of the battle is kept:** species, moves, items, abilities. The player
  knew all of them at every decision;
- **every other fact is drawn from the Smogon prior**, through the `agents.gen3_data` facade (`world_draws.Prior`,
  the honest world builder's draws):
  - a never-seen species from the Smogon teammates prior, given the revealed ones;
  - moves from the move marginals, given the revealed moves (Smogon has no within-species move joints);
  - item, ability, nature, EVs and IVs from the Smogon spreads;
  - Hidden Power's type through the HP repair. Its IVs carry the type;
- **the team must be legal:** Showdown's `TeamValidator('gen3ou')`, with a redraw on rejection (as
  `world_legality` does);
- **the team must explain the log:** every labelled action of that seat must be legal under the decision built from
  this team. Otherwise the team is redrawn. A seat with no consistent team after 32 draws is excluded and counted.
  This catches, for example, a Choice item drawn for a mon that changed moves without switching.

The reader is OPENED with this team, as the S-full control opens with the trainee's packed team.
**As built (`src/main/x50_r1/own.py`, `complete.py`; 2026-10-23).** The reader learns its own roster only from
requests, so each decision's request is a REAL one, synthesised as the simulator writes it:
- the whole roster (details, condition, stats, moves, ability, the current item) in the simulator's slot order (a
  switch swaps the incoming mon into slot 1), Return / Frustration with their power;
- the active's four moves at their tracked PP (Pressure deducts 2: the foe's revealed ability, else its species'
  Smogon majority, since gen 3 never announces it; a non-Ghost's Curse targets itself; a called move pays nothing);
- the legality the public lines determine: a Choice Band lock, Encore, Disable, Taunt, Torment, 0 PP, Struggle when
  nothing is left, the trap flags. A locked turn (a `[from] lockedmove` continuation, a charge's release, a
  recharge) lists that move alone, as the real request did. The lock is read off the decision's own window and
  handed to the prefix guard as an input, like the team;
- the own stream's private lines a spectator log lacks: the own Pressure announce (`[silent]`) and Struggle's
  `-activate`;
- **Mimic's slot swap** (R0 FINDING 1's 14 refusals; built 2026-10-23 before the training-date extraction): gen 3
  takes gen 4's `onHit`, so the copied move sits in Mimic's slot (PP min(5, base), max PP with Mimic's 3 PP Ups, no
  target) until the mon leaves, and the label maps onto it. All 14 refused held-out seats now complete. A mimicked
  Hidden Power (the user's type) is not modelled and refuses, counted.

Every public line naming an own mon is fed with its HP exact: the MIDDLE integer of the percent display's preimage
(±1 HP on a 3-wide preimage; F-1b-9). **The fidelity gate:** with the trainee's TRUE team and the own stream's
exact HP, the completed rows equal the rows the trainee played on byte for byte (4,581 of 4,582 rows on 150 traces
of 380M; `complete_test.py` holds the golden trace), and the own-action labels are its real choices. On the
public log only the HP cells move.

The C-GAME draws: the reveals are `agents.info_levers.revealed`'s, read from the other seat. The packed order is
the lead, then the seen mons by first appearance, then the drawn ones. A never-seen gender-ratio species draws M / F
uniformly; a Frustration set gets happiness 0. A REDRAW draws only the mons the log refuted (an action's legality
depends on the acting mon's set alone, so this samples the same conditional as a whole-team redraw), and a refuted
Choice Band (the mon broke its lock without switching) leaves that mon's item draw: the exact conditioning. A seat
whose species has no Smogon prior is excluded and counted. The extraction emits BOTH labels per row (owner's yes to
(b), 2026-10-23): `label` (a) and `own_action` (b's 11-way index, −1 none) with `n_legal`; a row with either is kept.

**Why C-GAME and not prefix-only.** A prefix-only completion keeps only what is revealed before each decision and
redraws the rest. It has no future information. But it needs one completed team per reveal epoch, and so a fresh
reader each time, about 20 per seat: ≈ 50 CPU-hours over the training dates against ≈ 5 for C-GAME. It also puts a
first-use move outside the input, which makes (b)'s label illegal.

**C-GAME's declared leak (F-1b-4).** Real facts revealed later sit beside Smogon-drawn ones. A model TRAINED on these
rows could learn real-vs-drawn signatures. The census's leakage table names the same class of leak. Three things
bound it:
- every decisive read of a trained model runs on rows whose own side is TRUE: R-b2's bank and R-a1's S-full;
- the trace twin (§8.5) measures what completion costs where the truth is known;
- the frozen X26 head never trained on these rows, so it cannot exploit them, and R0 reads it alone.

**The guards (a failure is a refusal, never a skip):** R1's prefix guard, re-run with the completed team; the label
legality check above; the team validator. Counts of refusals and exclusions go in the extraction manifest.

## 3. The data

- **Population:** the spectator manifest only (`replays/showdown/gen3ou/SPECTATOR_MANIFEST_2026-09-23.txt`, 376,410
  logs), read path by path from the manifest. The corpus folder now also holds the archive pulls (06-16..06-20,
  08-03..08-17, 09-24 onward). Those are UPLOADED battles, skewed: +35 to +128 Elo, 26–41 % longer, fewer forfeits
  ([`replay_archive_bias_2026-10-21`](../research_state/measurements/replay_archive_bias_2026-10-21/README.md),
  `fd7f4624`). They never enter.
- **Split:**
  - TRAIN on the dates 2026-05-18 → 2026-09-20;
  - HOLD OUT 2026-09-21, 09-22 and 09-23, untouched. They are read once per registered read below and never used to
    select a model, an epoch or a setting. R1 has already read them with X26's head; reading them again for
    evaluation is allowed by the rung-1 doc's reservation, which forbids only training use (F-1b-11);
  - VALIDATION (early stopping only): 2 % of the TRAIN battles, by `sha256(battle id) mod 50 == 0`.
- **Rating:** rated logs only.
  - For (a), the labelled OPPONENT's pre-battle rating is ≥ 1500 (R1's primary).
  - For (b), the acting human's own rating is ≥ 1500.
  - Both ratings ride every row (`rating_labelled`, `rating_viewer`, as R1's parts carry them).
- **Rows:** one per viewer decision (turn and forced; R1's decision points), both seats of every log, the viewer's
  side completed (§2). Two labels per row:
  - **(a)'s label:** R1's production intent label (the opponent's action in the window the decision opens), resolved
    by `flat_intent_targets`. Hidden Power is excluded as in R1.
  - **(b)'s label:** the viewer's OWN choice at that decision. It is mapped to the 11-way action under the completed
    team's slot order, the action token `opp_likelihood` scores.
    - A move is labelled by the slot of the move it used. For a move called by another move, the label is the
      CALLER's slot.
    - A switch is labelled by the target's bench slot. That includes forced replacements, which are real choices of
      the actor and which `opp_likelihood` scores.
    - A hidden choice (a `cant`: sleep, paralysis, flinch, freeze) or no action carries no label.
    - A decision with one legal action is kept but carries no information. It is counted, not read.
- **Team-memory exclusion (F-1b-2).** The off-pool bank's opponents pilot `heldout72_hpfix`. Those are 72 teams from
  Metamon's `hl_05_26`, which was built from human ladder replays. A human in the training window may have piloted
  one of them. Any training battle in which some side has ≥ 4 mons whose revealed facts all fit the same heldout72
  team's sets is EXCLUDED, both seats. The count is reported. Without the exclusion, R-b2 could pass on memory, as the
  pool-bank L3 read did.
  **As built** (`src/main/x50_r1/memory.py`, `run.py --exclude-teams`): a mon fits a set of its species
  (`team_overlap`'s forme rule) that holds every revealed move (a bare Hidden Power fits any typed one) and its
  revealed original item and ability. A mon that revealed only its species fits a set of that species: the rule
  counts what the log cannot tell apart, so it excludes more than true copies (a dev day, 07-01: 585 of 3,301 logs,
  63 of the 72 teams hit). The excluded battles and their teams ride each part's JSON.
- **Expected size (ESTIMATED):** the census counted 2.10M observable turn choices at ≥ 1500 over the whole corpus
  ([census](../research_state/measurements/x50_replay_census_2026-10-20/README.md)). Less the held-out days, plus
  forced replacements, that is **N ≈ 2.0–2.4M** rows for (b). R1's held-out yield (48,579 supervised ≥ 1500 rows in
  3 days, ×≈ 50) gives the same order for (a). The extraction manifest records the real counts.
- **Self-play rows (R-a1 only):** X26's eval-trace full rows, the training row as the trainee saw it, from cycles
  below 360M. Cycles ≥ 360M stay R1's S and B and are evaluation only.
- **Output:** `~/gen3ai_archive/x50_r1b/`, never `data/` (frozen) and never `models/`, which is read-only. Compressed
  `npz` parts (R1's part format, two label blocks) and a `manifest.json`: commit, binary hash, counts, exclusions.

## 4. (b) The human opponent model

- **What it is:** X26's architecture, **5,624,179 parameters** (2,028,828 in the features extractor; counted from the
  checkpoint 2026-10-23), initialised from **P\*** (`rb_x26_s2601` @ 280,000,170). P\* is the judge of the off-pool
  read, so R-b2 isolates one lever, human fine-tuning.
- **Input:** the acting seat's observation row, the encoder's own layout. The HOM is never given a domain bit.
- **Output:** the 11 policy logits. Its likelihood is the masked softmax at T = 1, in float64, exactly as
  `opp_likelihood` computes it for P\*.
- **Loss:** cross-entropy of the masked policy softmax at the human's action.
  - Every parameter the policy logits depend on trains.
  - The value head gets no gradient.
  - The internal belief and intent heads are not kept calibrated. Nothing may read them from the HOM (F-1b-13).
- **Training (fixed now):**
  - AdamW, learning rate 1e-4 cosine-decayed to 1e-5 over 3 epochs, weight decay 1e-5, batch 512, gradient-norm
    clip 0.5, fp32 on the CPU, seed 51.
  - Validation NLL every quarter epoch. The kept model is the one with the lowest validation NLL.
  - Stop early after 2 consecutive rises.
- **Storage:** P\*'s checkpoint format with the policy weights replaced. Its sidecar records the lineage: init P\*
  with its sha256, the rows manifest's sha256, the seed and the commit. It lives under the archive. It is never a
  run, never a baseline and never in `models/`.
- **The encoder gate (G-ENC, F-1b-6):** the training rows must be encoded byte for byte as X26's pin encodes them.
  - R1's full reads matched the trainee's played rows on 404 of 405 traces at R1's commit.
  - The build re-asserts it at its own commit on R1's 2,306 traces before extracting anything
    (`python -m main.x50_r1.genc`: the full read against `*_states.npz`'s rows and masks, bit for bit).
  - The bank's rows are already byte-identical at X26's pin (the off-pool gate G4).
  - Since `d8d674b99` (`gen3_volatile_silent_ends_v1`) the reader clears seven silently ending volatiles X26's
    traces recorded set, so G-ENC FAILS on 11 of the 2,306 traces. `genc --exclude` takes X59-S Amendment 4c's
    list (`python -m main.x50_r1.fr2_exclude`), and passes with it.

## 5. (a) The intent head, offline (R-a1)

- **Model:** X26 @ 380,000,195, R1's checkpoint. The trunk is frozen. Only the flat intent head's own parameters
  train, starting from X26's head: exactly what the intent loss can move in production.
- **Labels:** each batch is half human rows ((a)'s label, ≥ 1500 labelled, training dates) and half self-play rows
  (§3). One epoch over the human rows.
  - AdamW, learning rate 1e-4, weight decay 1e-5, batch 512, seed 52.
  - The kept head is the one with the best validation NLL on the 50 / 50 mix.
- **What this does and does not stand for.** In a training run the trunk would move under PPO beside the head. So
  R-a1 previews the head's side of a training use, not its strength. The self-play rows are true rows and the human
  rows are completed rows, the same contrast a training use would have. That is why R-a1 carries the KEYED check
  (§8.3).

## 6. What each result unlocks

| result | unlocks |
|---|---|
| R0 FIXED or REDUCED | R-a1, and the HOM's training rows |
| R0 NOT-FIXED | (a) STOPS: a refit would fit the artifact. (b) continues: R-b2 reads true rows. The FINDING travels with every human number |
| R0's re-read of R1 reads NO GAP | (a) STOPS: R1's gap was the pipeline's |
| R-a1 MIXABLE | a GPU training-use arm MAY be registered separately, with a named consumer (the beat-Metamon path). Nothing is queued here |
| R-a1 TRADE / NO-GAIN / KEYED / INCONCLUSIVE | (a) closes at rung 1. One head cannot serve both populations without a domain signal, which ai_v11 D1 advises against |
| R-b1 LEARNED | R-b2 |
| R-b1 not LEARNED | (b) STOPS before R-b2. This HOM is no better a human predictor than P\*. The lever is untested, not closed |
| **R-b2 HOM-EVIDENCE** | **search's opponent-choice route re-opens for non-self opponents.** Register stage 1c's phase A on the off-pool bank with the HOM judging the bot stratum, then phase B; register L3-H (§8.6); propose to the orchestrator that search be un-paused. The second lever, public-team proposals, pairs with it: the HOM judges, the proposals supply |
| R-b2 HOM-WEAK | stage 1c's phase A only (zero playouts) on the off-pool bank |
| R-b2 HOM-NONE | the human-opponent-model lever is CLOSED for search at P\*. Public-team proposals remain; search stays paused |
| R-b2 INCONCLUSIVE | ONE extension, registered in its own pre-registration before any battle: a loss-enriched bot bank (the off-pool read's FINDING 1, ≈ 8 CPU-hours of battles), frozen before any HOM likelihood is computed. R-b2's rule is re-read once on the pooled bot strata, and that verdict is final |

## 7. Cost

**Measured 2026-10-23 (feasibility, < 1 CPU-minute):**
- X26 @ 280M on the CPU, 6 threads on the agent cores (`taskset -c 5-7,13-15`, nice 19), while `rb_x35_c0_s3504`
  trained;
- rows from R1's `humans_0000.npz`, after one warm-up batch;
- **forward 794 rows/s** (the feature pass, batch 512); **forward + backward 373 rows/s** (policy-logit
  cross-entropy, batch 256); peak 1.8 GB;
- one 24-batch timing per mode (F-1b-8).

| step | CPU-hours | GPU-hours | build |
|---|---|---|---|
| step 0 build: C-GAME completion, the opened reader, the two labels, the guards, G-ENC | — | 0 | opus-high ≈ 1.5–2 agent-days |
| R0: re-extract R1's populations completed, and read | ≈ 1 | 0 | R1's `read.py` + a part flag |
| training-date extraction (≈ 369k logs; R1 took 3 min per 7.4k logs, ×≈ 2 for completion) | ≈ 5 (ESTIMATED) | 0 | — |
| R-a1: head refit (≈ 2 × 2.2M rows through the frozen trunk at ≈ 790 rows/s) + read | ≈ 2 | 0 | opus-medium ≈ 0.5 agent-day |
| HOM training (2.2M rows × ≤ 3 epochs at 373 rows/s) | ≈ 5 | 0 (≈ 0.2 optional) | opus-high ≈ 1 agent-day (trainer + sidecar) |
| R-b1 (≈ 50–100k held-out rows × 2 models) | ≈ 0.2 | 0 | — |
| R-b2 (318 states × 2 judges at ≈ 9 s a state, plus G-REPRO) | ≈ 1.5 | 0 | a driver over the bank's `offpool_run.py` |
| **total** | **≈ 15–18** | **0** | **≈ 3–4 agent-days** |

**Running it:** every heavy step under `scripts/ops/mem_cap.sh` in a heavy slot, at rule 5b's priority, in ≤ 45-min
resumable chunks (R1's `run.py` pattern). Wall time is ≈ 1–2 days with the slot queue. **CPU is preferred:** the
GPU belongs to X35, and on the CPU nobody waits for the HOM. If a lease window is idle, its training moves to the GPU
unchanged.

## 8. PRE-REGISTRATION (2026-10-23, before any row is extracted, any model trained or any number read)

**Common to every read.**
- 95 % CIs come from a battle-cluster bootstrap: 2,000 resamples, seed 50 (R1's).
- Paired contrasts are resampled jointly.
- **Rule 8:** a CI end within 0.002 nats, or 0.002 AUC, of a bar takes the more conservative label. L3's AUC bars
  already carry their ±0.005 band.
- Every row of a population is read; nothing is capped.

### 8.1 R0: is F-R1-1 fixed? (CPU, X26 @ 380M, frozen)

- **Populations:** R1's H, S and B re-extracted with the completed own side (H_c, S_c, B_c), and R1's S-full and
  B-full unchanged. R1's supervised-row rules apply; the placeholder no longer exists.
- **Primary:** the penalty P_c = NLL(S_c) − NLL(S-full).
  - **FIXED:** P_c's CI upper < +0.08 nats.
  - **REDUCED:** not FIXED, and the paired reduction NLL(S) − NLL(S_c) has a CI lower > +0.05.
  - **NOT-FIXED:** otherwise.
- **Re-read beside it, with R1's own rule:** Δ NLL(H_c − S_c) (GAP / NO GAP / INCONCLUSIVE at ±0.05), and its move
  and switch strata.
- **Descriptive:** S-own-true (the trace with the trainee's TRUE packed team and synthesised decisions). It splits
  the remaining penalty into completion error (S_c − S-own-true) and synthesised legality (S-own-true − S-full).

### 8.2 R-b1: does the HOM out-predict P\* on held-out humans? (a GATE for R-b2)

- **Rows:** the held-out dates; acting human ≥ 1500; (b)'s label; decisions with ≥ 2 legal actions.
- **Models:** the HOM and P\*, both through the masked 11-way softmax at T = 1.
- **Primary:** Δ = NLL(HOM) − NLL(P\*).
  - **LEARNED:** Δ's CI upper < −0.05 nats.
  - Otherwise: NOT LEARNED, and (b) stops (§6).
- **Registered secondary (reported; it decides nothing):** the same Δ on voluntary switches (a switch chosen at a
  non-forced decision), and on moves. Forced replacements are reported apart.
- **Also reported:** top-1 for both models; the all-ratings band; the rating bands.
- **Reference only, never subtracted:** X26's intent head on the same dates (R0's H_c). It predicts from OUR view
  without the human's team, so it is a different conditioning, not a baseline.
- **Why it is only a gate (F-1b-5):** both models see completed rows here, and only the HOM was trained on them.

### 8.3 R-a1: the refit intent head against X26's head

- **Rows:**
  - H_c: the held-out dates, labelled ≥ 1500, completed;
  - S-full: cycles ≥ 360M, the true own side;
  - S_c: the same battles, completed.
- **Contrasts (refit − X26 head, paired):** Δ_H on H_c, Δ_S on S-full, and the KEYED statistic
  K = Δ on S_c − Δ_S.
- **Rule:**
  - **MIXABLE:** Δ_H's CI upper < −0.05 nats, AND Δ_S's CI upper < +0.03, AND K's CI lower ≤ +0.05.
  - **KEYED:** the first two hold and K's CI lower > +0.05. The head learned the completion artifact as a domain
    signal.
  - **TRADE:** Δ_H's CI upper < −0.05 and Δ_S's CI upper ≥ +0.03.
  - **NO-GAIN:** Δ_H's CI lower > −0.05.
  - **INCONCLUSIVE:** otherwise.
- **Registered secondary:** Δ_H on switch labels, and whether the refit beats the presence-only floor on human
  switches. R1's head did not: 2.508 against 2.208.

### 8.4 R-b2: THE DECISIVE READ. The off-pool L3 likelihood with the HOM as the judge

- **Bank:** the frozen off-pool bank
  ([`search_offpool_bank_2026-10-20`](../research_state/measurements/search_offpool_bank_2026-10-20/README.md):
  `states_offpool.json`, the 318 read states, 159 bot / 159 snapshot). Its opponents pilot heldout72 ladder teams.
- **Worlds:** the truth plus 8 honest worlds per state, rebuilt deterministically at L3's code commit `8fe4fb89`.
  The scored steps are the same.
- **Judges:** P\* and the HOM, in ONE process. Only the `Forward` changes.
- **G-REPRO (a gate):** P\*'s re-score must equal `rows.jsonl`'s per-world log-likelihoods (|Δ| ≤ 1e-9) on every
  state. Otherwise the worlds differ: a defect, and no read.
- **PRIMARY:** the bot stratum's AUC, truth vs honest, under the HOM. L3's bars, verbatim: WORKS lo > 0.555;
  WEAK lo > 0.505; REVERSED hi < 0.495; NO-EVIDENCE hi < 0.545; else NOT-DETECTED.
- **CO-PRIMARY:** the paired ΔAUC (HOM − P\*) on the bot stratum, per state, battle-clustered. BETTER if its CI
  lower > 0.
- **VERDICT:**
  - **HOM-EVIDENCE:** WORKS and BETTER.
  - **HOM-WEAK:** WEAK and BETTER, or WORKS without BETTER.
  - **HOM-NONE:** NO-EVIDENCE or REVERSED.
  - **INCONCLUSIVE:** otherwise.
- **Reported beside it, never deciding:**
  - the snapshot stratum and the pooled AUC under the HOM, with ΔAUC against P\*. P\* stays the judge for our own
    network;
  - the HOM's familiarity control `AUC_conf`. A bot-stratum CI lower > 0.555 attaches a FAMILIARITY flag to the
    verdict;
  - top1, p_truth and the lifts, by L3's read;
  - by P\*'s outcome (the bot stratum is 96 % P\* wins);
  - the read restricted to states whose opponent team had a near match removed from training (§3).
- **Power, declared (F-1b-7):** at 159 states the CI half-width is ≈ 0.06. WORKS needs an AUC of about 0.62 or more;
  a smaller real effect reads WEAK or NOT-DETECTED.

### 8.5 Descriptive: the trace twin (no verdict)

Scope: R1's S and B traces, read from the OPPONENT's seat (the sentinel or the bot as the viewer and actor).

- **Two kinds of row** for the same decisions, both labelled with the bot's or sentinel's own action:
  - C-GAME-completed rows;
  - the TRUE team with real requests.
- **Read:** NLL of the HOM and of P\* on both.
- **The twin gap:** NLL on completed rows minus NLL on true rows, for each model.
- **Reported:** the twin gap of the HOM minus that of P\*. It shows whether the HOM learned completion artifacts.
  R-b2 reads true rows, so an artifact can only make R-b2 fail, never pass.

### 8.6 Not registered here

- **L3-H**, the ladder's own version of R-b2. At a held-out human decision, does the HOM's likelihood rank a world
  holding a fact revealed LATER in that replay above one that lacks it? It needs a replay world builder: a replay
  has no seed, so its worlds cannot pass the byte gate. It is the next registration after HOM-EVIDENCE.
- **Any GPU training use of (a)**, as §6 says.
- **The HOM as a playout opponent** (§1.4).

### 8.7 Stop rules (each is a STOP and a report, never a skip)

- **A guard fails:** G-ENC, G-REPRO, the prefix guard, the label legality check, or the validator on the TRUE
  trace teams. It is a defect: fix it, re-run, and read nothing in between.
- **Refusals:** more than 1 % of seats refused by the extraction stops it for a root cause.
- **The held-out dates:** touched by any selection, or read before the model under test is frozen and hashed, voids
  that read.
- **The team-memory exclusion** is not applied before the HOM's training: R-b2 is void.
- **The chain:** R0 → R-a1; R-b1 → R-b2; and R-b2's single extension, as in §6. No other extension, re-cut or
  threshold change after a number has been read.

## 9. Risks

- **The grant (F-1b-1).** (b) is designed under a reading of the grant that the owner has not confirmed.
- **The proxy (F-1b-3).** The bots switch on 6 % of labels (F-R1-3), while R1 found the human gap in switches. R-b2
  on bots may not exercise the HOM's main gain. L3-H is the ladder's read, and it is unbuilt.
- **Imprinting human error.** The median player at ≥ 1500 is ≈ 1,560. That is acceptable for a judge (it should model
  the players we would meet) and would be wrong for a policy, which (b) never trains.
- **Meta drift.** The training dates span four months; the held-out days are the last three. A drift penalty lands on
  R-b1 and R-a1 as a pessimistic bias.

## FINDINGS (standing rule 7)

- **F-1b-1 (the grant's scope).** The owner granted "opponent-action labels for the intent head". (b) uses those
  labels for a separate, search-only model, read from the human's own seat with a completed own side. The priors rule
  allows it (§1.3). Whether the grant covers it is the owner's call. **Building the HOM waits for an explicit yes.**
- **F-1b-2 (team memory).** heldout72 comes from Metamon's `hl_05_26` human-ladder teams, so humans in the training
  window may have piloted them. Without §3's exclusion, R-b2 could pass on memory, as the pool-bank L3 read did
  (0.588 → 0.472 off-pool). The exclusion uses set-consistent matching on revealed facts; near copies below
  4 matching mons remain (UNVERIFIED size).
- **F-1b-3 (the bot proxy).** This is §9's proxy risk: the HOM's expected gain is in switches, and the bots rarely
  switch.
- **F-1b-4 (C-GAME's leak).** Facts revealed later sit beside drawn ones (§2). The decisive reads use true rows; the
  human-row reads (R0's H_c, R-b1, R-a1's Δ_H) do not and cannot, since no truth exists for a human's team.
- **F-1b-5 (R-b1's home advantage).** Only the HOM trained on completed rows, so R-b1 is a gate, never the decisive
  read.
- **F-1b-6 (the encoder).** The HOM must read the bank's encoding. HEAD's reader matched X26's played rows at R1's
  commit; the gate re-asserts it at the build's commit.
- **F-1b-7 (power).** R-b2's bot stratum is 159 states. WORKS needs an AUC of about 0.62 or more.
- **F-1b-8 (the cost figures are estimates).**
  - The row counts are extrapolated from the census and from R1's held-out yield.
  - The throughput is one 24-batch timing per mode, taken during a live training run.
  - The extraction cost assumes completion doubles R1's rate.
- **F-1b-9 (the own exact HP).** It is derived from the log's percent and the completed max HP: ±1 HP off what the
  player saw.
- **F-1b-10 (engine-banned moves).** A revealed own move the engine refuses (`BANNED_MOVES`, e.g. Metronome, Bide)
  can make the opened reader refuse that seat. Such seats are counted, and the 1 % refusal stop covers them.
- **F-1b-11 (held-out reuse).** 09-21..09-23 were read by R1. This design reads them again for evaluation only,
  which the reservation allows; no selection may touch them.
- **F-1b-12 (the bank's truths).** Their hidden parts are partly reconstructions (Metamon, the HP repair; the
  off-pool read's hazard 1). That is conservative for R-b2.
- **F-1b-13 (the HOM's interior).** Its belief and intent heads drift uncalibrated under fine-tuning. Nothing may
  read them. Its value head is P\*'s, untouched and meaningless for the HOM.

## Decision record

| date | what | rejected | evidence |
|---|---|---|---|
| 2026-10-23 | **CREATED + PRE-REGISTERED** (research-design worker, orchestrator brief after R1's GAP and the off-pool L3 read). Both uses, in order: step 0 (C-GAME completion of the viewer's own side, one extraction giving two labels) → (b) the HOM, X26's network fine-tuned from P\* on ≈ 2M human ≥ 1500 choices, judge only, its build waiting on the owner's scope ruling → (a) offline only (R-a1, the head refit on a frozen trunk). Reads R0, R-a1, R-b1 (a gate), and R-b2, the decisive read: the off-pool bank's bot-stratum AUC with the HOM as the judge, PASS = WORKS and BETTER than P\* paired. CPU only, ≈ 15–18 CPU-hours | a GPU training-use arm of (a) now (no in-play consumer: we play no humans); the intent head as the judge (it cannot rank worlds); a fresh HOM (data-small, a second lever); prefix-only completion (≈ 50 CPU-hours, and the label becomes illegal); the archive pulls (uploaded, skewed); training without the heldout72 exclusion (memory) | R1's README; the off-pool README §2–§6; `opp_likelihood.py`; `world_draws.py`; the census; `fd7f4624`; the feasibility timing (§7) |
| 2026-10-23 | **STEP 0 BUILT** (X50 rung-1b worker). The synthesised request is a REAL request carrying the completed roster and the tracker's state, not a legality-only spectator request: the reader learns its own roster only from requests. HP is the preimage's middle; Pressure from the revealed ability, else the Smogon majority; locks read off the decision's window and passed to the guard as inputs; the own private lines re-inserted; per-mon redraws with a refuted Choice Band removed from the draw; both labels emitted (the owner approved (b) the same day) | a legality-only request on a team reading (the bench would stay unknown, the fix undone); `pct × max` rounded (a larger expected error); a whole-team redraw (the same distribution, a 25 % failure rate on Dugtrio's 96 % Band prior) | `own.py`, `complete.py`, `complete_test.py` (the byte-equal golden trace) |
| 2026-10-23 | **R0 READ: FIXED** under §8.1: P_c = S_c − S-full +0.012 nats [+0.005, +0.020] (the reduction from R1's pipeline: +0.217). Beside it, R1's rule on the completed rows: Δ NLL(H_c − S_c) +0.253 [+0.232, +0.276], GAP (moves +0.160, switches +0.411). §6: R0 FIXED unlocks R-a1 and the HOM's training rows; the gap stands, so (a) does not stop. 14 human seats refused (Mimic, 0.09 %), 56 excluded (47 with no Smogon prior); 0.47 % in all, under the 1 % stop | reading the gap's growth as the head's fault (the completion error on human teams is UNVERIFIED) | `x50_r1b_r0_2026-10-23/` |
| 2026-10-24 | **(b) BUILT: the HOM, trained and frozen** (X50 rung-1b worker). Mimic's slot swap fixed first (`3f5d4b4da`); the training-date extraction with the team-memory exclusion, G-ENC and completed-only rows (`2bbdffa7d`); the registered recipe unchanged, all 10,346 steps, kept = last (validation 1.766 → 1.156). As built: a species-only reveal FITS a heldout set, so the exclusion removed 70,010 battles (19 %, all 72 teams) rather than only copies; the trainer's thread count dropped 6 → 3 mid-run for speed under shared cores | a looser fit rule (the registration says revealed facts, which a species is); the optional GPU route (not needed) | `x50_r1b_hom_2026-10-23/` |
| 2026-10-24 | **R-b1 READ: LEARNED** under §8.2: Δ NLL(HOM − P\*) −0.582 nats [−0.599, −0.565] on 69,171 held-out rows (actor ≥ 1500); voluntary switches −0.874, moves −0.490, forced −0.537; every rating band −0.58 to −0.61. §6: R-b2 is unlocked | reading R-b1 as evidence for search (F-1b-5: a gate on completed rows only the HOM trained on) | `x50_r1b_hom_2026-10-23/` README §R-b1, `rb1_read.json` |
| 2026-10-24 | **R-b2 READ: HOM-NONE** under §8.4. G-REPRO: P\*'s re-score equals the bank's on all 320 states (|Δ| = 0) and reproduces READ 2. Bot stratum under the HOM 0.459 [0.406, 0.510] NO-EVIDENCE; paired ΔAUC −0.013 [−0.068, +0.045], not BETTER. Descriptive: snapshots 0.706 (P\* 0.794, ΔAUC −0.088 [−0.153, −0.027]); pooled 0.583 WEAK; AUC_conf 0.490 (no FAMILIARITY flag); P\*'s 6 bot losses 0.688 vs 0.854. The near-match split is degenerate: every bank team had training battles removed. §6: the human-opponent-model lever is CLOSED for search at P\*; public-team proposals remain; search stays paused; the INCONCLUSIVE extension does not apply | reading HOM-NONE as closing human evidence on the ladder (F-1b-3: the bank's non-self opponents are scripted bots; L3-H, unregistered, is that read) | `x50_r1b_hom_2026-10-23/` README §R-b2, `rb2_read.json` |
