# Human replays as a TOGGLEABLE training lever: the levers, ranked, and a pre-registered plan (X59, X60)

**Status (2026-10-25): DRAFT for adversarial review; EXPLICIT-ONLY (the plan and its pre-registration; the always-current architecture is [`human_data_architecture.md`](human_data_architecture.md)) (two reviews follow: data / bias / leakage, and experimental /
causal). Nothing is built, trained or launched; no GPU was used.** The data card (§3) is MEASURED on CPU
([`measurements/human_replay_quality_2026-10-25/`](../research_state/measurements/human_replay_quality_2026-10-25/README.md));
everything in §4–§6 is design. Every assumption a review should attack carries an id (**A-1 … A-n**, §8) and a
falsifier. Explicit-only: it is updated when a read or ruling changes the plan, and the architecture doc carries
what is current.

**The grant (owner, 2026-10-25).** Human replays MAY train the POLICY, under two hard conditions: (1) every human-data
path is a TOGGLEABLE lever, OFF by default, so the system always trains and plays with no human data and nothing
depends on it silently; (2) data-quality analysis: is the data good, and does it help? This supersedes the 10-23 scope
(the human model as search's judge only, [`design_x50_rung1b_human_opponent_model.md`](design_x50_rung1b_human_opponent_model.md) §1.3).
**The priors rule is unchanged**: what the network READS (obs features, tables, `data/` priors) traces to Smogon, ground
truth or replays as before, and no human-derived number is fed in as a feature, a table or a prior. Human data enters
only as a TRAINING SIGNAL (labels in a loss, or teams to play), recorded and switchable.

Abbreviations: P\* = X26's plateau (`rb_x26_s2601` @ 280,000,170); HOM = the human opponent model (P\* fine-tuned on
human choices, rung 1b); U = the pooled unfamiliar panels (heldout72_hpfix = H, forum72 = F, P72) vs Kakuna; home72 =
our sample teams (in the pool); C-GAME = rung 1b's own-side completion (the log's reveals + Smogon draws); BC =
behaviour cloning (supervised imitation of the chosen action); pp = percentage points.

---

## 0. One screen

- **The data is usable, by stratum, with filters.** On `hr_spec_v1` (376,410 spectator logs; 0 parse errors; one rule set) the actors rated ≥ 1500 give **1.80M informative rows** (the chosen action known, ≥ 2 legal). Verdicts: ≥ 1500 USE-WITH-FILTER (drop the ≈ 6 % suspected-bot sides, walk-overs, duplicates and every panel-near battle); 1300–1499 for the human-only comparator only; < 1300, unrated and the SKEWED archive DROP for training.
- **Humans cover the unfamiliar region, partly.** At least 15.9 % of ≥ 1500 human sides sit where the unfamiliar panels sit (≤ 2 mons fitting any pool team's sets; self-play: 0 %), and human cores lie near most heldout72 and forum72 teams. That region is exactly what the panel exclusion must remove (card §D). **But P\* predicts human choices no worse on unfamiliar teams than on pool-like ones** (1.762 vs 1.738 nats): the offline proxy gives NO support to H-BC's mechanism (A-1), so X59 tests it before anything is built.
- **The ranked levers for goal (b)** (Kakuna on unfamiliar teams, ≈ 4 pp short at P\*):
  1. **H-BC: an auxiliary BC loss on human rows, mixed into PPO** (`--human-data bc:<snapshot>@<coef>`). The only
     lever whose gradient lands on the states the deficit is about: humans piloting thousands of teams we never trained (4,810 distinct species sextets at ≥ 1500, 41,238 at all ratings, against the pool's 719).
  2. **H-TEAMS: human ladder teams in the self-play pool** (`--human-data teams:<snapshot>@0.5`). D1's machinery with
     a replay-derived corpus; D1's forum version read +2.1 pp [−0.9, +5.1].
  3. **A human-only comparator HB0** (BC from scratch, then optionally offline RL): not a lever, a registered baseline
     (Q-H10).
  4. H-KL (piKL-style KL to the HOM on self-play states) and a BC warm start of a fresh run rank low for this goal (§4).
- **The first thing to RUN costs no GPU and no build: X59 = the HOM itself against Kakuna on U and home72** (≈ 4.5 CPU
  lane-h). The HOM IS P\* behaviour-cloned on humans, so it tests H-BC's central assumption (A-1: human labels carry
  unfamiliar-team piloting skill our policy lacks) before any GPU is spent.
- **The first thing to BUILD: the toggle + the snapshot check** (≈ 1 agent-day; §7.1–7.2; the snapshots themselves are frozen and registered), then H-BC (≈ 2.5 agent-days)
  if X59 SUPPORTS A-1, else H-TEAMS (≈ 1.5 agent-days; §5.1's action table; at X59's power the likely route is H-TEAMS first). The A/B (X60) is D1's frame: 3 + 2 seeds × 20M
  from P\*, ≈ 16 GPU-h per arm, the control reused only if bit-identity holds (§5.4).
- **Complexity budget (§6):** three registered levers at most, one flag (`--human-data`), one snapshot id per run, one
  lever per A/B, one shared control.

---

## 1. The architecture this plan builds on

The sources, their information levels and selection, the snapshots, the one switch, where each source enters, the
named questions Q-H1..Q-H12 with their current answers and the drift policy are the ALWAYS-CURRENT
[`human_data_architecture.md`](human_data_architecture.md). This doc is the explicit-only plan: why each lever, which
first, and the pre-registered reads. It does not restate the architecture.

## 2. The questions this plan answers

Q-H9 (better than self-play?), Q-H10 (a human-only model) and Q-H11 (does it help?) are unmeasured; §2.1, §4.3 and §5
are their designs. Q-H12 (toggle integrity) is §7.1's build.

### 2.1 Q-H9: what "better than self-play" means, and how it is measured

"Better" is per goal. **For goal (b):** a human row is better than a self-play row if, at a matched GPU budget, spending
a share of the update on human rows moves U against Kakuna more than spending it on self-play does. Three offline
proxies come first, each with a falsifier:

1. **Coverage** (Q-H6): the share of human teams in the region the panels occupy (pool set fits ≤ 2), against
   self-play's 0 % (self-play only ever pilots pool teams). *Falsified for goal (b)* if humans put < 10 % of their sides
   there. **Measured: ≥ 15.9 % (a lower bound; card §D): not falsified.**
2. **Learnability / noise floor** (Q-H7): the HOM's held-out NLL is the floor a 5.6M network reaches on human choices
   (1.148 nats at ≥ 1500). Our policy's own entropy on self-play states is the matching number for self-play labels;
   a high human floor means much of a human label is NOISE for a policy trying to win (humans err). *Measured:* the
   floor; *unmeasured:* how much of it is error vs style.
3. **Familiarity gap on human choices** (card §E): P\* predicts humans worse on teams far from the pool than near it.
   *If not,* nothing in the human rows is specific to unfamiliar teams, and H-BC's mechanism (A-1) loses its basis.
   **Measured (post hoc, card §E): no gradient** (P\* 1.762 [1.706, 1.826] nats on the panels' region vs 1.738
   [1.709, 1.767] on pool-like teams). The proxy does not support A-1; it cannot refute it (imitation is not winning).

The decisive read is X59 (the HOM vs Kakuna) and then X60 (the A/B), which reports gain per GPU-hour against the
control's own steps.

---

## 3. The data card

The headline card of `hr_spec_v1` is [`human_data_architecture.md`](human_data_architecture.md) §5.1; the full card,
its registration and the code are
[`measurements/human_replay_quality_2026-10-25/`](../research_state/measurements/human_replay_quality_2026-10-25/README.md).
The verdicts this plan reads from it: ≥ 1500 USE-WITH-FILTER (the lever rows); 1300–1499 USE-WITH-FILTER for HB0 only; < 1300, unrated, suspected bots, walk-overs, duplicates, panel-near battles and the archive population DROP for training. Two card facts shape this plan: coverage of the unfamiliar region is real but sits inside the panel exclusion (card §D), and P\*'s human-prediction NLL shows no familiarity gradient (card §E), which is why X59 runs before H-BC is built.

---

## 4. The lever options, ranked for goal (b)

The ranking criterion: expected pp on U vs Kakuna per GPU-hour, at an acceptable risk to home72 (the co-primary).
Every expected pp below is a **GUESS**; the measured anchors are D1 (+2.1 pp [−0.9, +5.1], breadth via forum teams,
P −1.6) and X52 (practice +3.2 [−0.2, +6.8]).

| rank | lever (flag value) | what changes | why for goal (b) | expected pp (b), GUESS | GPU cost | main risk |
|---|---|---|---|---|---|---|
| **1** | **H-BC** `bc:<snap>@<coef>` | the LOSS: + coef(t) · CE(masked policy softmax, the human's action) on a human minibatch per update | the only option whose gradient lands on states of teams we never trained (4,810 distinct ≥ 1500 species sextets); AlphaStar's KL-to-supervised and Metamon's human pretraining both keep a human term | +1 to +4 | ≈ +20 % learner time (a 25 % human minibatch); ≈ 16 GPU-h per 3-seed arm | pulls the policy toward 1500-level play: an in-pool cost (P); the C-GAME signature (F-1b-4) |
| **2** | **H-TEAMS** `teams:<snap>@<share>` | the POOL: completed human ladder teams join the self-play team draw on both sides | breadth of practice on the ladder population (Kakuna's population: hl_05_26 is replay-derived) | +0 to +4 | none extra; ≈ 13 GPU-h per 3-seed arm | D1 already measured the breadth family small; panel memory (H is replay-derived) |
| 3 | **HB0** (a model, not a flag) | a separate network: BC from scratch on human rows, optionally offline RL (Metamon's recipe) | answers Q-H10; the warm-start candidate | n/a (a comparator) | BC ≈ 1 GPU-h; offline RL ≈ 3–5 GPU-h (ESTIMATE) | none to production |
| 4 | H-KL `kl:<snap>@<coef>` | the LOSS: + coef(t) · KL(π ‖ π_HOM) on SELF-PLAY states (piKL) | in Diplomacy piKL bought human-compatibility, not strength against a fixed agent; our self-play states hold pool teams, so the KL acts where we are already strong | +0 to +1 | the HOM forward per minibatch, ≈ +15 % | pulls in-pool play toward the HOM for no unfamiliar-team gain |
| 5 | BC warm start of a FRESH run | the INIT: HB0's weights start a fresh RL run (AlphaStar's supervised init) | a fresh run reaches the plateau's strength only after ≈ 100M+ steps (X37b), and RL without a human term forgets the init (AlphaStar kept a KL) | unknown | ≥ 42 GPU-h for a matched pair to 100M | the cost; nothing comparable before 100M |

**Rejected:** human game OUTCOMES as critic labels (off-policy: a human state's value under human play is not its
value under ours; it biases the critic exactly on the unfamiliar teams); the HOM as a self-play OPPONENT (it targets
FACING, which costs ≈ 0: FACE_c +0.017); an advantage-weighted BC variant (AWR on outcomes) as a separate lever now
(the budget: it is H-BC's variant, registered only if H-BC reads GAIN or a clear NULL).

**Each lever against the rules.**
- **Priors rule:** none changes what the network reads. H-BC and H-KL are loss terms (experience); H-TEAMS changes
  which teams are PLAYED, as D1's forum corpus does (allowed: the pool measures structure and never ships as a prior).
  The completed rows' own-side sets are Smogon-drawn (C-GAME), never pool-drawn.
- **Search:** the HOM as search's judge is the same toggle's fourth value, `judge:<snap>` (R-b2 read it HOM-NONE; it is
  closed and stays off).
- **Literature** (the papers' claims are cited from memory and are **UNVERIFIED** in this unit; the Metamon facts
  are from its local checkout's README, `~/dev/metamon`):
  - **Metamon** (Grigsby et al. 2025, RLC): IL and offline-RL ("actor-critic") variants trained on the same 1M human
    battles at 15M / 50M / 200M parameters (README model list); the paper reports the RL variants above the IL ones.
    **Kakuna** = Superkazam (11M self-play + 4M human replays) fine-tuned on 7M more self-play battles, with the human
    replays' sampling weight REDUCED (README): human data as the STARTING POINT, self-play as the bulk.
  - **piKL** (Jacob et al. 2022, ICML): search / RL regularised by a KL toward a BC anchor policy, λ controlling the
    trade; it kept strength while predicting and cooperating with humans much better. Its win was human compatibility.
  - **AlphaStar** (Vinyals et al. 2019, Nature): a supervised (human-replay) initialisation, then league RL with a
    KL penalty toward the supervised policy, which the paper credits with keeping human-like strategies in play.

### 4.1 H-BC, specified

- **Rows:** `hr_spec_v1`'s TRAIN dates (≤ 09-20), C-GAME completed, own-action label, ≥ 2 legal actions, the actor
  rated ≥ 1500, not a flagged bot, panel-excluded (every battle where a side has ≥ 4 mons fitting a team of ANY
  registered held-out panel, both seats; F-1b-2's rule widened to all panels). That is the HOM's row set less the bot
  and the wider panel exclusion: ≈ 1.7M rows (ESTIMATE).
- **Loss:** coef(t) · mean CE over a human minibatch of 25 % of the PPO minibatch's size, the masked policy softmax at
  T = 1 (the HOM's loss). coef(0) = 0.05 × the policy loss's running scale (ESTIMATE, to be fixed by a CPU calibration
  before registration closes: §5.6), cosine-decayed to 0 at the fork's end.
- **The weights it touches:** the trunk and the policy head. The value head, the win-prob head and the belief / intent
  heads get NO gradient from it (their stashes are untouched: the fold runs last, §7.3).
- **Selection hazard:** no row is chosen by our model's own judgement (no advantage weighting) in the first arm.

### 4.2 H-TEAMS, specified

- **Teams:** the C-GAME completed team of every ≥ 1500, non-bot, full-six-revealed side of the TRAIN dates, validated
  (`TeamValidator('gen3ou')`), deduplicated by packed text, capped at 5 teams per account (the archive lesson), and
  excluded against every registered panel and its near neighbours (`main.heldout_panels.contamination`, 0 hits at
  startup). Hidden facts are Smogon draws, so each team is a Smogon-plausible completion of a real human core.
- **Mix:** `--team-mix`'s machinery at D1's share (0.5) on both sides; the eval stays on the pool.

### 4.3 HB0, the human-only comparator (Q-H10)

- **HB0-BC:** our architecture from random weights, BC on ALL rated TRAIN rows of `hr_spec_v1` less the card's DROP
  filters other than the rating floor (suspected bots, walk-overs, duplicates, panel-near battles) (≈ 13–14M own-action
  labels, ESTIMATE; all ratings, since a from-scratch model needs the volume; the rung-1b design rejected a fresh HOM on 2M rows
  for that reason), 3 epochs, the HOM's recipe otherwise. GPU ≈ 1 h (ESTIMATE from the HOM's 373 fwd+bwd rows / s on
  CPU and ≈ 100× on the GPU; CPU alone would take ≈ 33 h).
- **HB0-ORL (optional):** an offline actor-critic on the same rows with the game outcome as the reward (Metamon's
  recipe class). Registered separately, only after HB0-BC's read.
- **Reads:** vs P\* in-house on home72 and U (`main.h2h`, CPU), and vs Kakuna on home72 and U (`main.anchors`, ≈ 4.5
  lane-h). Prediction (GUESS): HB0-BC loses to P\* by ≥ 20 pp in-house and scores 0.30–0.45 vs Kakuna on U.

---

## 5. The pre-registered A/B: X59 first, then X60

### 5.1 X59: the HOM against Kakuna (CPU, no build, no GPU) — the A-1 test

- **Arms:** the frozen HOM (sha256 `76e29a01…b36f`) and P\* at the same harness (`63f97a57`, X51's), greedy vs Kakuna
  ckpt 34 greedy. P\*'s X51 cells are REUSED if the harness, seeds 0–19 and units match (else re-run).
- **Cells:** home72, H, F, P72; 400 games each for the HOM (1,600 games ≈ 4.5 lane-h).
- **Statistics:** Δ_U = HOM_U − P\*_U (pooled equal-weight, paired seeds); the PREMIUM contrast
  Δ_prem = (HOM_home72 − HOM_U) − (P\*_home72 − P\*_U) (95 % paired bootstrap over 20-game units).
- **Predictions (GUESS):** HOM_home72 0.52 [0.45, 0.58] (it imitates 1500 play, so it loses part of P\*'s home edge);
  Δ_U −1 pp [−6, +4]; Δ_prem −5 pp [−12, +2].
- **Rule:** SUPPORTS A-1 iff Δ_prem's upper bound < 0 AND Δ_U's lower bound > −3 pp; CONTRADICTS iff Δ_prem's lower
  bound > −2 pp AND Δ_U's upper bound < 0; else UNCLEAR. A bound within 1e-9 of a bar takes the conservative class.
- **Power (ESTIMATE):** Δ_prem differences four cells (two of 400 games, two pooled U of 1,200), so its 95 % half-width
  is ≈ 8 pp; at the predicted −5 pp, P(SUPPORTS) ≈ 0.25. X59 is therefore mostly a screen against CONTRADICTS (an
  HOM clearly worse on U with no premium change); UNCLEAR is the likely class and routes to H-TEAMS first. Doubling
  the HOM's cells (3,200 games, ≈ 9 lane-h) narrows the half-width to ≈ 6.5 pp only, because P\*'s X51 cells are fixed.
- **Action:** SUPPORTS → build H-BC first; CONTRADICTS → H-TEAMS first, H-BC dropped from X60; UNCLEAR → H-TEAMS first
  (cheaper build, D1's measured machinery), H-BC second.
- **Panel memory (a declared confound):** the HOM's training removed only heldout72-near battles (F-1b-2), not
  forum72- or P72-near ones, and humans pilot cores near most forum72 teams (card §D). So H is X59's clean cell;
  F and P72 are reported with that caveat, and the rule is also read on H alone (reported, not deciding).
- **VOID:** the HOM fails to load in the anchors harness, or a 20-game smoke differs from a seeded repeat (the harness
  must be deterministic), or > 1 uncounted unit per cell.

### 5.2 X60: the lever A/B (GPU; D1's frame)

| item | registration |
|---|---|
| arms | ON = P\* forked +20M with `--human-data <lever>:hr_spec_v1@<param>` (H-BC or H-TEAMS by X59's action; ONE lever per arm); OFF = the same fork with `--human-data off` |
| matched control | X35's C0 (P\*, +20M, seeds 3501–3505) ONLY IF bit-identity holds at the arm's pin (§5.4); else a fresh C0h at the arm's pin |
| seeds | 3501–3503 (look 1), 3504–3505 (look 2); shared across arms |
| dose | 20M steps per seed; reported in GPU-h and in human rows consumed |
| **PRIMARY H** | Δ_U = ON − OFF vs Kakuna, pooled equal-weight over H, F, P72, AM-5's widened Newcombe (`main.x35.d1`), 140 games per panel per seed final |
| **CO-PRIMARY P** | Δ_P = ON − OFF on home72 in-house (`main.h2h play-many --panel home72`), Welch 97.5 %, non-inferiority bar −2.0 pp |
| registered secondary | Δ on F + P72 only (populations no human lever trains on; H is the replay population: "learned the population" vs "generalises"); Kakuna on home72 |
| predictions, H-BC (GUESS) | Δ_U +2.5 pp, 80 % predictive [−1, +6]; Δ_P −1.0 [−3, +1] |
| predictions, H-TEAMS (GUESS) | Δ_U +2.0 [−1, +5]; Δ_P −1.5 [−3, 0] |
| verdicts | D1's: ADOPT-CANDIDATE (H GAIN, P non-inferior, guards pass), TRADE (H GAIN, P inferior: report the exchange rate), NULL ([−3, +3]), LOSS, UNDECIDED; look 2 iff UNDECIDED with Δ̂ > 0. Nothing is adopted by the rule alone |
| guards | X35's G-E, G-C, G-F, as D1 |
| VOID | an ON arm's startup stamp (`[HUMAN-DATA] lever=… snapshot=hr_spec_v1 rows_sha=…`) missing, or an OFF arm's stamp not `off`; the rows' encoder layout hash ≠ the trainee's (G-ENC); the realised human share outside ±10 % of the registered one; H-BC's BC loss not falling by ≥ 10 % in the first 1M steps (the term trains nothing: the `label_only` class); a panel contamination hit; K9(b) FAIL on any seed |

### 5.3 What X60 cannot say

- It reads one dose at one plateau checkpoint. A NULL does not rule out a fresh-run use (the warm start, HB0's path).
- H is replay-derived (Metamon's hl_05_26): a gain on H alone, with F + P72 flat, says the lever learned the ladder
  population, which may still be what Kakuna's cells reward. The confirmation is X54's sealed forum216b.

### 5.4 The control: reuse only under bit-identity

C0 ran at D1's pin. The arm runs at a newer pin (it carries the lever). C0 is the control only if, at the arm's pin
with `--human-data off`: `learner_golden_test.py` and `played_bits_sentinel_test.py` hold their committed bytes
unchanged from C0's pin, the fork base and every other argv token are C0's, and the seeds match. Otherwise a fresh
C0h (3 seeds, ≈ 13 GPU-h) runs at the arm's pin. The check is recorded in the X60 sheet before launch.

### 5.5 Cost

| step | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| X59 (the HOM vs Kakuna) | 0 | ≈ 4.5 | ≈ 0.3 |
| the toggle + snapshot check (§7.1–7.2) | 0 | — | ≈ 1 |
| H-BC build (loader, fold, gates) | 0 (≈ 0.2 smoke) | ≈ 1 | ≈ 2.5 |
| or H-TEAMS build (team export, corpus, guards) | 0 | ≈ 5 (C-GAME teams) | ≈ 1.5 |
| X60 one arm, look 1 (+ look 2) | ≈ 16 (+ 11) | ≈ 6.3 (+ 4.2) | ≈ 0.5 |
| fresh control C0h if reuse fails | ≈ 13 | ≈ 6.3 | — |
| HB0-BC (Q-H10) | ≈ 1 | ≈ 6 | ≈ 1 |

### 5.6 Open before registration closes

- H-BC's coef(0) is fixed by a CPU calibration on P\*: the ratio of the BC gradient norm to the PPO gradient norm on 20
  minibatches, set so the BC term's gradient is ≈ 10 % of the policy term's at the start (A-5).
- The human minibatch share (25 %) and the rating floor (≥ 1500) are registered now and not tuned.

---

## 6. The complexity budget

- **One flag:** `--human-data` takes `off` or ONE `<lever>:<snapshot>@<param>` clause. Two levers in one run are
  refused unless a registration names the combination and its reason (`--human-data-combined-consent`).
- **At most three registered levers** (`bc`, `teams`, `kl`) plus the search judge `judge`. A new one replaces a row of
  §4 with a decision-record entry; it is not added beside them.
- **One snapshot per run**, recorded by id. At most TWO snapshots are live: the frozen reference (`hr_spec_v1`) and the
  next candidate. A new snapshot is cut only on a trigger: the drift alarm (§3.9) or +50 % usable volume.
- **One lever per A/B, one shared control, one dose.** No factorial over coef × share × rating floor: a variant is
  registered only after its parent reads, and only one at a time.

---

## 7. Appendix: the implementation (what to build)

### 7.1 The toggle (`agents.info_levers.human_data`, NEW)

- **A spec like the reveal spec:** canonical text `off` | `bc:<snap>@<coef>` | `teams:<snap>@<share>` |
  `kl:<snap>@<coef>` | `judge:<snap>`; hash = the first 16 hex of `sha256("human_data_v1:" + canonical)`; a `record()`
  block {label, canonical, hash, snapshot id, snapshot manifest sha256, rows manifest sha256}.
- **Consumers honour or refuse by name** (`Consumer.validate`, as §3 of the reveal spec): training honours `bc`,
  `teams`, `kl`; search honours `judge`; eval and play honour only `off` (they never read human data).
- **Provenance:** the resolved config carries `human_data` (default `off`); `model_config.json`, `metadata.json` and
  the immutable lineage block record it; a fork inherits `off` unless typed (a lever is never inherited silently: a
  resume of an ON run without the flag refuses, as `--oracle-reveal`'s resume gate does); `main.lineage` shows the
  first ancestor that trained with human data.
- **The off-path gates (Q-H12):** (1) with `off`, a routine test runs the training smoke under an `open()` audit that
  fails on any path under the corpus root, the human-row archive or a snapshot directory; (2) `learner_golden` and the
  played-bits sentinel are unchanged by the build; (3) `main.baselines set` refuses a run with a human-data ancestor as
  a baseline unless `--reason` says so, and the registry row records it.

### 7.2 The snapshot registry

- **A snapshot is a manifest:** (relative path, sha256, bytes) per log, gzip'd, in `~/gen3ai_archive/human_replay_snapshots/`
  (never the repo: 376k lines), plus a committed row in `designs/research_state/human_replay_snapshots.json` (DONE
  2026-10-25 for `hr_spec_v1` and `hr_arch_2026-10-24`, by `hrq.py snapshot`). To build: `agents.info_levers.human_data`
  reads the registry and re-hashes a snapshot's files before any consumer uses it.
- **Immutability:** the corpus is append-only on disk (`sync.py` never rewrites an id), so a snapshot is verified by
  re-hashing its files (`hrq.py snapshot`); a mismatch refuses every consumer. A zstd-compressed tar of the snapshot's
  logs in the archive makes it independent of the live corpus (≈ 0.6 GB, ESTIMATE).
- **Derived rows** (C-GAME parts, H-BC shards, H-TEAMS corpora) carry the snapshot id, the extraction commit and the
  encoder layout hash; a consumer whose layout differs REFUSES (an obs change re-extracts, ≈ 5 CPU-h).
- **Citation:** every result, run config and data card names its snapshot id, or `none`.

### 7.3 H-BC in the learner

- A fold in the EAGER TAIL, after every stash reader (it runs its own extractor forward on human rows, which would
  clobber the minibatch's stashes otherwise; `ppo_step.md`), guarded `if human_bc_on:`, off contributes nothing.
- The loader: shards streamed from the rows archive, one shard (≈ 200 MB) resident, a keyed draw per update
  (reproducible), built at startup (K6), the next shard prefetched on the host.
- Its loss joins the K9 non-finite check; its share and coefficient are logged (`human/bc_loss`, `human/bc_coef`,
  `human/rows_seen`).

### 7.4 H-TEAMS

- Export C-GAME's completed packed team per seat (today's extraction keeps rows, not teams: re-run `complete.py`'s team
  draw only, ≈ 5 CPU-h), then a `utils/team_corpora` builder with D1's filters, the per-account cap and the panel guard.

---

## 8. Assumptions, each with its falsifier (for the reviews)

| id | assumption | falsified by |
|---|---|---|
| A-1 | Human choices carry unfamiliar-team piloting skill that P\* lacks | X59 CONTRADICTS. (The offline proxy, a familiarity gradient in P\*'s human-prediction NLL, was measured and is ABSENT: card §E. A-1 now rests on X59 alone) |
| A-2 | The C-GAME completion does not teach a policy something false (a drawn set it then "expects") | R0's completion cost on our own teams (+0.019 nats) does not carry to human teams (UNVERIFIED there); a trace-twin read on human-like teams would test it |
| A-3 | ≥ 1500 humans are good enough teachers that a small-coef BC does not cost home72 | X60's Δ_P < −2 pp |
| A-4 | The spectator stream is representative of the ladder's finished games | an archive-independent census (the live stream, revived) disagrees at the same dates |
| A-5 | A BC gradient at ≈ 10 % of the policy gradient is large enough to move the trunk, small enough not to dominate | the BC loss does not fall (VOID) or Δ_P breaks the bar |
| A-6 | The bot rule catches the bots that matter (name, volume, hours) | a hand audit of 50 random ≥ 1500 accounts finds a bot the rule missed |
| A-7 | The panel exclusion removes team memory of H / F / P72 | an ON arm's H gain concentrates on H's teams with the most human near-copies |
| A-8 | 20M steps from a plateau is enough plasticity to show a lever | D1's NULL-from-plasticity risk; X60 NULL with HB0 strong would point here |

---

## 9. FINDINGS (standing rule 7)

1. **The offline evidence for H-BC's mechanism is absent** (card §E): P\* predicts humans equally well on unfamiliar and
   pool-like teams. H-BC stays rank 1 on mechanism (it is the only lever whose gradient reaches unfamiliar-team states),
   but its build is gated on X59.
2. **The panel exclusion removes the best-covered region** (card §D): humans pilot cores near most H and F teams
   hundreds of times, so the "unfamiliar" data a lever can keep is the part far from every panel. X60 must report the
   excluded share per panel; if it exceeds half the ≥ 1500 rows, the lever's dose is re-stated before launch.
3. **heldout72_hpfix is a replay-derived panel** (Metamon's hl_05_26): a human-data lever trains on its POPULATION.
   X60's registered secondary (F + P72 only) and X54's sealed forum216b are the clean reads; a gain on H alone is
   "learned the ladder population".
4. **The bot rule is a heuristic** (card FINDING 4): ≈ 6 % of sides, including heavy human grinders it cannot tell
   apart; A-6's hand audit is unrun.
5. **The fill error on HUMAN teams is unmeasurable today**: no full-information human source exists (architecture
   doc §2, source (c-human)). A-2 rests on R0's own-team number.
6. **Experiment ids X59 / X60 are named here only**, not yet in `EXPERIMENT_BACKLOG.md`; the orchestrator assigns
   them (a collision is possible).
7. **Reviews pending.** This is a DRAFT; the orchestrator runs two adversarial reviews before any build. Nothing in it
   is authorised to launch.
8. **Process.** CPU only; no GPU, no launch, no download; `models/`, `data/`, the corpus and the X50 archive were only
   read.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **This doc CREATED (DRAFT; explicit-only; the architecture split out to `human_data_architecture.md` on the owner's requirement)** on the owner's grant (human replays may train the policy, toggleable, with data-quality analysis) and the orchestrator's scope addition (mental model, named questions, snapshots, drift, "better than self-play", a human-only comparator, a complexity budget) | the data card on `hr_spec_v1` (§3); levers ranked H-BC > H-TEAMS > HB0 > H-KL > warm start; X59 (the HOM vs Kakuna, CPU) first; the toggle + snapshot check the first build; X60 in D1's frame | building H-BC before A-1 is tested (X59 is free); a factorial over coef × share × floor (the budget); pooling the archive with the spectator stream (SKEWED); human outcomes as critic labels (off-policy) | §3; fd7f4624; R-b1, R-b2; D1 look 2; X52 |
