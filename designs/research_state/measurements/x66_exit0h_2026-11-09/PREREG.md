# X66 ExIt-0H: can P\* learn the HONEST search's choices on teams it never saw, and does it then beat P\* against the real Kakuna? — PREREG (2026-11-09)

Written and committed BEFORE any student trains on a label of this bank, before any student logit exists and before
any game of this read is played. It is the roadmap's MF-1 (`../../EXPERIMENT_BACKLOG.md` §2.0) and the rung the X65
design names as ExIt-0H ([`design_x65_exit0_search_distillation.md`](../../../endstate/design_x65_exit0_search_distillation.md)
§4.4): X65's machinery with the label source changed, at the recipe the X65 learnability probe showed can move the
network, so its leaky-label comparator is the probe's `curve` arms at that same recipe (one lever: the label source).

**Status: REGISTERED 2026-11-09.** The question, the classes, the folds, the recipe, the gate, the deals, the
statistic and every bar are frozen here. Two blind amendments may follow, and neither may move a bar, a fold, a deal,
a seed, z, β, the learning rate, the epoch budget, the pick rule or the class rule:
- **Amendment 1 (the bank):** the bank's manifest sha256, its counts and drops, and a CPU smoke of the train command
  on it, written after the bank is built (outcome-blind) and before any student trains;
- **Amendment 2 (the students):** the exported zips and their sha256s in the read specs, written after the delivery
  gate and before any Kakuna game (only on DELIVERED).

**Abbreviations.**
- **P\*** = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`), greedy.
- **S′h** = `r0s4_honest_inc@10`, the search S′-HONEST played: 8 worlds from the INCREMENTAL leak-honest builder, fresh
  dice, 4-turn playouts (8 per action, P\*'s critic at the leaf), piKL at λ = 0.015, every eligible decision from turn
  10. It reads only what the live player knows.
- **S′-HONEST** = [`../sprime_honest_play_2026-11-06/`](../sprime_honest_play_2026-11-06/README.md): GO-HONEST, P\* + S′h
  beats P\* against Kakuna on U by **+2.00 pp [+0.34, +3.51]** on 5,097 paired fresh deals (`331f19cfe`); state
  `~/gen3ai_archive/sprime_honest_play_2026-11-06/`, spec sha256 `9222393a…8afa`, play pin `e46e3ddf8`, engine E6.
- **U** = H (`heldout72_hpfix`), F (`forum72`), P72 (`p72`). **E6** = `port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130`.
- **Kakuna** = Metamon's Kakuna, checkpoint 34, greedy, peer pack `gen3_peer_showdown_pack_v1`.
- **π′** = softmax over the legal actions of (log π_P\* + Q̂ / λ), λ = 0.015: the distribution whose mode S′h played.
- **A departure** = a searched decision where S′h's pick (`played`) is not P\*'s greedy choice (`pi_argmax`).
- **Δagree** = mean over a panel's label rows of 1[student's argmax = `played`] − 1[`pi_argmax` = `played`].
- y = a game's result for us (1 / ½ / 0); D = y(student) − y(P\* raw) on one deal. GUESS = unmeasured; ESTIMATE = a
  cost from measured rates.
- **X65** = [`../x65_exit0_2026-11-05/`](../x65_exit0_2026-11-05/README.md) (ExIt-0 on S′-LIVE's leaky labels:
  NOT-DELIVERED, Δagree −1.26 pp [−1.40, −1.11]); **the probe** =
  [`../x65_learnability_probe_2026-11-06/`](../x65_learnability_probe_2026-11-06/README.md) (DESCRIPTIVE: reading 2,
  "(U) unlearnable at this scale").

## 0. Why this read, and what each answer steers

The owner's direction of 2026-11-09 is model-first: improving the network is the main lever, including as the driver
of search. Search is the one lever with a measured game gain against Kakuna on unfamiliar teams (+2.00 pp, honest).
Distilling it into the network is the classic route from "search helps" to "the network is stronger". X65 tried it on
S′-LIVE's leak-inclusive labels and its probe read "(U) unlearnable", but the probe could not split three causes: the
leak, S′'s key noise, and lookahead a reactive network cannot amortise. Honest labels remove the first by
construction. So:

- **DELIVERED** (the student agrees with S′h on panels it never saw more than P\* does) says honest search's choices
  carry signal a network of P\*'s size can read from its own observation. It opens the Kakuna read below.
- **NOT-DELIVERED** says this recipe, on these labels, does not move P\* toward S′h's choices on unseen teams.

**The orchestrator's two pre-commitments (2026-11-09, from an independent review, before registration; §8.3 is
binding):**
1. **A NOT-DELIVERED or a NULL from X66 is NOT evidence for the scale bet (MF-5).** Its labels are only about a third
   reproducible (§8.2), and the gain it distils exists on one panel: S′-HONEST's pooled +2.00 pp is P72 +4.69 pp
   [+2.00, +7.37], H +1.32 [−1.39, +4.00] (spanning 0) and F −0.005 [−2.98, +2.71] (`331f19cfe`'s record). A failure
   to distil it cannot tell "the network is too small" from "the target is mostly noise, and its signal lives on one
   panel".
2. **MF-1b (ExIt-1) is NOT planned on the strength of X66 alone,** whatever the class: any follow-up needs its own case
   (its own evidence and its own registration).

## 1. The question and the classes

**Does a P\* student, fine-tuned offline toward S′h's honest piKL targets on two panels' banked decisions (with a weak
KL anchor to P\*), agree with S′h on the third panel more often than P\* does, and does it then win more games than P\*
against Kakuna on that panel's deals, whose teams it never saw?** The student plays greedy with no search.

| class | rule | licenses | does NOT license |
|---|---|---|---|
| **NOT-DELIVERED** | the delivery gate's LB(Δagree) ≤ 0 (§4); no Kakuna game is played | "At this recipe and ≈ 113k honest labels a fold, a P\* student does not learn S′h's choices on unseen teams." the §7 readings say which pattern | **evidence for the scale bet (MF-5)** (pre-commitment 1); anything about larger networks, more labels, another recipe or in-loop distillation |
| **GO** | DELIVERED and LB(G_X) > 0: at look 1 with z = 2.576, or at look 2 with z = 2.0 (§6) | "A network distilled from honest search's choices beats P\* against Kakuna on unseen teams, with no search at play time." Naming the students an X54 candidate on X54's own terms; the placebo attribution (§7) | **planning MF-1b (ExIt-1) on X66 alone** (pre-commitment 2: it needs its own case); "goal (b) met"; adoption as production; pooling with S′-HONEST's G_U; a claim against other opponents; a per-panel claim |
| **NULL** | DELIVERED, not GO, and UB(G_X, 95 %) < +2.0 pp at look 1 or look 2 | "The distilled student keeps less than S′h's whole measured gain (+2.00 pp) on unseen teams." | **evidence for the scale bet (MF-5)** (pre-commitment 1); planning MF-1b (ExIt-1) either way; closing ExIt (more labels, an in-loop form, a larger student) |
| **UNDECIDED** | otherwise, after look 2 | nothing; no extension without a new registration | evidence for MF-5; planning MF-1b |

- Look 1 decides GO or NULL if either holds; otherwise look 2 plays. GO is checked before NULL.
- **A bound within 0.0005 of its bar is never a crossing** (root rule 8); the record is flagged `boundary`.
- **The NULL bar is S′h's own point gain**, so NULL means "less than all of search's gain", not "no gain". The read is
  powered for ≈ +2 pp, not for a fraction of it (§9): a small true effect most likely reads UNDECIDED, a power
  statement.
- **Not a use of S′-HONEST's licence.** S′-HONEST licensed search at play time, no training change. This is its own
  registration under the orchestrator's brief of 2026-11-09; it trains research students and changes no production
  training, baseline or recipe.

## 2. The label bank (outcome-blind; `python -m main.x65.bank build --source sprime_honest`)

- **Source.** S′-HONEST's COUNTED units (its `record.json`, final attempts only, never a `.partial.<epoch>` dir): 10,197
  units (5,099 searched, 5,098 raw; 1,699–1,700 a panel and arm). The builder reads `search_decisions.jsonl` and the
  capture's START and commands alone; `games.jsonl`, `summary.json`, the capture's chunks and the record's outcomes are
  refused by name (`bank.FORBIDDEN`).
- **The tool reads the honest records unchanged (roadmap FINDING 6, verified 2026-11-09).** S′-HONEST's 188,429
  searched and 66,596 unsearched records (all final dirs) share S′-LIVE's schema `gen3_live_search_decision_v1` and
  every field the bank reads; the S′-HONEST record names its search and engine (`expected_search`
  `r0s4_honest_inc@10`, `engine_registered` E6). Two changes were needed, both landed with this registration:
  - **the engine:** the builder refused anything but S′-LIVE's E4. A bank now has a SOURCE (`bank.SOURCES`:
    `sprime_live` on E4, `sprime_honest` on E6); the checkout's engine and the record's own `expected_search` /
    `engine_registered` must be the source's, and the trainer's x66 recipe refuses a bank of any other source;
  - **the honesty guard (`bank.label_view`).** On an honest source every field of every decision record must be
    classified: `RECORD_FIELDS` may shape a row; `CHECK_FIELDS` and `IGNORED_FIELDS` (telemetry, the builder's working
    state) never do; `HIDDEN_INFO_FIELDS` are computed against the truth and are never read. `species_hit` (each
    world's species overlap with the TRUE opponent team, a diagnostic every searched record carries) is the one such
    field today. An unclassified field, a searched record off `r0s4_honest_inc@10`, a leak block whose `builder` is not
    `honest` / `honest_inc` or whose `partial` / `residual` list is non-empty, a revealed opponent or another λ stops
    the build (`HiddenInfoError`; never a dropped row). Tests that fail on revert: `bank_test.py`'s honest-source
    group (a hidden field made a bank field, an unclassified field, seven leaky edits, a moved `species_hit` leaving
    every row bit for bit, S′-LIVE's leaky fixture refused).
- **Every label's state is one the network saw at play, our side's information only.** The row is rebuilt from the
  capture in `honest_log`'s (update, side, arrival) order through `prepare_root`, and **a label row is kept only if its
  rebuilt row hashes to the recorded `frame_sha256`**, the sha256 of the live core row P\* read when S′h decided. The
  row is our side's observation by construction (the encoder writes one side), and the hash makes it the network's
  live input bit for bit. The label (π′ from `q` and `log_pi`) is S′h's, whose worlds, dice and leak block the guard
  above holds to the honest builder. A mismatch is dropped and counted; > 0.5 % of a panel's label rows dropped VOIDs
  the bank.
- **Exclusions are never labels.** A record with `searched: false` past turn 10 (`why: excluded`, 11,071 counted;
  H 2.6 %, F 10.0 %, P72 3.3 % of search-path decisions) or `one_legal` carries no `q`; it is an ANCHOR row (teacher
  P\*), and an excluded record's `frame_sha256` is the anchor rows' path check (a mismatch drops that row). Measured on
  the fixture game (8 searched, 5 excluded): 8 label rows, 23 rows, 13 frames verified, 0 drops.
- **Anchor rows.** Every decision of our side in both arms' counted games, rebuilt the same way; a raw decision has no
  digest and must pass the two-ordering cross-check and P\*'s greedy-action check (X65 §2, unchanged).
- **The identity of the bank.** One `.npz` per arm, panel and unit seed; `bank_manifest.json` names the source, the
  engine (E6), the side pin's commit (clean), the record's sha256 and every shard's sha256. Its sha256 is registered by
  Amendment 1, and every trainer, scorer and gate refuses any other (`bank.check`).
- **Where it is built.** At the SIDE PIN (§5.1), under `mem_cap` on the agent cores (`bank_build.sh`), in ≤ 44-min
  resumable chunks. ESTIMATE ≈ 3.5 h of wall at 3 workers (X65's bank: 6,351 s for 5,398 units).

## 3. The folds, the students, the recipe (`python -m main.x65.student train --recipe x66`)

| student | trains on (label + anchor rows) | read panel (scored, never batched) | seed |
|---|---|---|---|
| S_H | F + P72 | H | 6611 |
| S_F | H + P72 | F | 6612 |
| S_P | H + F | P72 | 6613 |

- **No student ever sees a read deal.** Both sides of a U deal draw from one panel, so a fold holds out a whole panel:
  every row (label or anchor) of the read panel is excluded from training and validation, and `student.split` VOIDs a
  student whose training or inner-validation rows hold a read-panel row or deal. The gate (§4) scores each student
  only on its read panel's label rows, and the Kakuna read (§5) plays it only on its read panel's deals, so **no deal
  a student is scored or played on was ever one of its training labels.** (Those rows are labels for the OTHER two
  students only.) Panel adjacency (P72's sets are Smogon usage draws; H's hidden sets are usage-filled) is reported,
  not removed, as in X65.
- **The recipe** (registered in `student.RECIPES["x66"]`; nothing is a knob):
  - start from P\*'s weights; the loss is KL(π′ ‖ π_θ) on label rows + β · KL(π_P\* ‖ π_θ) on anchor rows, **β = 0.1**,
    over the legal actions; only the policy-logit path trains (the value head's own parameters are frozen);
  - 256 label + 256 anchor rows a step, each drawn without replacement per epoch from a plan keyed by the seed; AdamW
    (weight decay 1e-5, eps 1e-5), a 200-step linear warm-up then **lr 1e-4** constant; gradient-norm clip 0.5; fp32,
    TF32 off; eval mode;
  - **12 epochs, a checkpoint every half epoch (24)**; at step 0 and every checkpoint the learnability dumps
    (`train` / `val` / `read` argmax and log-softmax) are written;
  - **the pick:** the checkpoint after step 0 with the highest inner-validation Δagree (`val`: the training panels'
    deals whose `deal` sha256 first byte ≡ 0 mod 10, withheld from training), ties to the earlier. No `read` number
    enters the pick (`student.pick_val` records only `val` numbers; `learnability.report_arm`'s `selected_on_val`).
- **Why this recipe, and which of X65's arms are kept** (the probe's results decide):
  - **X65's registered recipe (lr ≤ 3e-5, β = 1, 4 epochs, pick on inner-validation L) is DROPPED.** It barely moved
    the network: on its OWN training rows its Δagree was −0.10 pp and it recovered 6.9 % of S′'s departures (the
    probe's R0_H). A NOT-DELIVERED at that recipe would say nothing about the labels (X65-R2).
  - **The `curve` recipe is KEPT, as THE registered recipe.** It moves the network (by its last checkpoint curve_h left
    P\*'s choice on ≈ 20 % of the read panel's keep rows: K 0.798) and it is the one arm whose result on leaky labels (curve_h −4.79 pp, curve_p −5.16 pp on `read`)
    the honest students compare with at the same recipe: the lever is the label source.
  - **The control (`ctrl_h`) is NOT re-run.** It showed the trainer and the observation can learn a same-departure-rate
    target and transfer it to unseen teams (+11.70 pp [+11.05, +12.32] against its own target) at exactly this recipe,
    on the same panels and teams. That is a property of the recipe and the observation, not of S′'s labels, so the
    probe's reading 3a ("nothing transfers") is already excluded; re-running it would spend ≈ 0.75 GPU-h to re-measure
    it on other deals.
  - **The capacity arm (`cap_h`) is NOT re-run.** It showed P\*'s network memorises S′'s departures in-sample (`train`
    D 0.688); the students' own `train` curves report the same thing here.
- **The export** (`main.x65.export write`): P\*'s zip with the picked policy weights, `x66_S_<X>.zip`, its logits
  bit-identical on the 64-row fixture between the trainer (the side pin) and the PLAY PIN `e46e3ddf8`'s bare loader;
  anything else is VOID.
- **Device:** CUDA under the training session's lease (`gpu_students.sh`, §10). ESTIMATE ≈ 45 min a student
  (≈ 5,300 steps at X65's measured 0.33 s a step, plus 25 dumps of ≈ 190k rows), ≈ 2.25 GPU-h in all.

## 4. The delivery gate (decisive; zero Kakuna CPU; read before any game)

- **The rows.** For S_X, every label row of panel X (S′h's verified searched decisions on X's deals 700–784).
- **The statistic.** a(student) = 1[argmax of the exported zip's masked logits = `played`]; a(P\*) = 1[`pi_argmax` =
  `played`] (the recorded greedy choice). Δagree_X = mean of a(student) − a(P\*); the pooled Δ = the mean of the three
  panels' Δagree.
- **The interval.** A bootstrap of deals within each panel (one game a deal), B = 4,000, `default_rng(20261105)`,
  panels drawn in (H, F, P72) order; LB = the 2.5 % percentile.
- **DELIVERED iff LB > 0**; a LB within 0.0005 of 0 is not a crossing (`boundary`).
- **Computed by** `main.x65.delivery score` / `gate` (X65's code, unchanged) through `gate.sh`, and recomputed by
  [`delivery_by_hand.py`](delivery_by_hand.py) with loops of its own; any disagreement is VOID.
- **Its sensitivity is not a concern.** X65's gate on ≈ 32k rows a panel had a 95 % interval of ± 0.15 pp; this
  bank's panels hold ≈ 63k label rows each.

## 5. The deals, the arms, the engine (played only on DELIVERED)

### 5.1 The two pins

- **The PLAY PIN is `e46e3ddf8`**, S′-HONEST's own: every Kakuna game of this read plays there, on E6, so S′-HONEST's
  banked P\* raw units are this read's control. Its `main.reads` plans the three specs: `check-spec --pin-pending`
  exits 0 on all three, and a stand-in copy of look 1's spec with every PENDING value filled checks with
  `play_path_differs_at_head` 0.
- **The SIDE PIN is `c0b95914e239ff8c66da5a06afe9b42a43d41cf6`** (tag `x66-pin-c0b95914e`) = `e46e3ddf8` + this
  registration's `src/main/x65` change, one cherry-picked commit (main's `src/main/x65` differs from it only in the
  package docstring). The bank, the students, the export and the gate run there; its worktree is
  `/home/goodlad/dev/gen3ai-reads/x66_pin` (the kit re-creates it from the tag if it is gone).
- **Both worktrees must stamp E6, a REFUSING precondition** (FINDING X66-F5): `gpu_students.sh` checks the side pin's
  and `gate.sh` both worktrees' `main.anchors.server.engine_stamp()` against E6 and HOLDs on anything else, before any
  student trains, is scored or is exported; `bank.build` refuses a non-E6 checkout in code. The read operator checks
  the play pin the same way before any game (the read's own engine VOID backs it). **They cannot run at
  main:** main's model is config v166 and later (`7a0153b46`: our stats floor the EV, Leech Seed drains an integer; v167 `566d878b2`: the nature floor, the end-of-turn integers), a forward
  change to P\*'s own logits on some rows. A student trained at main would be fine-tuned under a different function
  from the one that plays at `e46e3ddf8` (FINDING X66-F1).

### 5.2 Look 1: 2,700 deals, the banked control

- **The deals.** Each student plays its read panel's 900 S′-HONEST deals at U unit seeds **700–744** (20 a seed),
  each a lone one-game unit against Kakuna, greedy, no search, at the play pin on E6.
- **The control.** S′-HONEST's banked `pstar` raw unit on the same deal (imported).
- **G-BANK (before look 1 counts).** Re-play 135 banked raw units at the play pin (per panel and unit seed, game
  `(seed − 700) mod 20`). At least 133 must be byte-identical (START and both sides' protocol, `|t:|` removed);
  otherwise the banked control is VOID and look 1 plays a fresh raw arm on the same deals
  ([`read_spec_look1_fresh.json`](read_spec_look1_fresh.json)).

### 5.3 Look 2: the rest of S′-HONEST's deals

- **The deals.** Unit seeds **745–784** on each panel (800 a panel, 2,400 in all), the students played fresh, the
  control still S′-HONEST's banked raw arm (one G-BANK covers both looks: same pin, engine and arm). Look 2 reads the
  cumulative 1,700 deals a panel (5,100 in all), one bootstrap pool of 85 seeds a panel.
- These seeds are S′-HONEST's: no other read may claim U unit seeds 700–784 for a student of this read's kind.

### 5.4 Every look

- Every counted student unit must be on E6, greedy, unsearched and on its registered zip's sha256, or the read is
  VOID; every imported raw unit carries S′-HONEST's checks.
- **Drops.** A deal with one side missing after both attempts is imputed at its worst case for each class (GO:
  student 0, raw 1; NULL: student 1, raw 0). > 0.5 % of a panel's deals dropped at a look VOIDs that look.
  S′-HONEST's two raw-side drops on these seeds (`pstar__heldout72_hpfix__762__g14`, `pstar__p72__780__g10`) enter
  as drops at look 2.

## 6. The statistic and the rule `distill_play` (X65's rule, unchanged)

- Ĝ_X = the mean of D over panel X's deals of the look; Ĝ = (Ĝ_H + Ĝ_F + Ĝ_P) / 3.
- SE = the sample SD of Ĝ* over B = 4,000 replicates resampling, within each panel, the unit seeds with replacement
  (their deals whole), `default_rng(20261109)`.
- GO: LB = Ĝ_go − z·SE_go > 0 (z = 2.576 at look 1, 2.0 at look 2). NULL: UB = Ĝ_null + 1.96·SE_null < +2.0 pp. Each
  under the imputation adverse to its class; exact fractions; root rule 8.
- The specs: [`read_spec_look1.json`](read_spec_look1.json), [`read_spec_look1_fresh.json`](read_spec_look1_fresh.json),
  [`read_spec_look2.json`](read_spec_look2.json), written by [`make_specs.py`](make_specs.py). PENDING today: the
  students' zips and sha256s, the pin (added with them), the bank sha256 and the earlier looks' spec sha256s.
- **By hand:** X65's [`by_hand.py`](../x65_exit0_2026-11-05/by_hand.py) (`python3 by_hand.py <state>/inputs.json
  <read_spec.json>`) recomputes every panel fraction, bound and the class from the read's own inputs and spec with
  loops of its own; it holds no X65 constant, and `src/main/reads/` holds it to the rule on constructed cases.

## 7. Reported, never decided

- **The learning curves** (`learnability_report.json`, from the students' own dumps): per student and checkpoint,
  D / K / Δagree / the student's departure rate / precision on `train`, `val` and `read`; at the selected, the last and
  the best-on-read (OPTIMISTIC) checkpoints, deal-bootstrap intervals and the splits by S′h's q margin (< 0.03,
  0.03–0.10, ≥ 0.10). The probe's readings table (its 1, 2, 3b, 3c, 3d, 3e; 3a is excluded by `ctrl_h`) is applied to
  them and stated, on any class.
- **Honest against leaky, at one recipe** (DESCRIPTIVE, unpaired: other deals, ≈ 2× the label rows): S_H's and S_P's
  `read` Δagree, D and precision beside the probe's curve_h (−4.79 pp, D 0.119, precision 0.23) and curve_p (−5.16 pp).
- **The noise ceiling** (§8.2): each student's `read` agreement beside S′h's own self-agreement under another key.
- **Each arm's pooled-U score, the discordance, game length, the students' V drift** (mean |V_student − V_P\*| on the
  read panel's label rows), the students' agreement with P\* before turn 10.
- **On DELIVERED only:** a DOSE-MATCHED student (S_H on a deal-hash half of its training rows, ≈ the probe's 58.6k
  label rows), REPORTED, so a DELIVERED can be told apart from "more labels" (≈ 0.4 GPU-h; its build is owed, §12).
- **On GO only** (X65 Amendment 1, carried over): three PLACEBO students (the same recipe, rows and seeds, each label
  row's Q̂ permuted among its legal actions by `default_rng([6699, row index])` before π′ is formed) play look 1's deals;
  SEARCH-INFO iff the student − placebo LB (95 %, z 1.96, imputation adverse) > 0. SEARCH-INFO is a necessary, never
  a sufficient, condition for any ExIt-1 case (pre-commitment 2). Goal (a)'s floor: each student and P\* on home72, 300 deals a student (unit seeds 600–614).

## 8. The three questions the brief asks

### 8.1 The seed leak is gone

S′-LIVE's leaky builder played every playout on the TRUE battle seed's dice (`dice: rolls`, L1: every decision
exposed) and built worlds that read the true team (`attribution_miss`, `rewritten_moves`, the `partial` / `residual`
lists). S′h draws fresh dice (`dice: fresh_inc`), never reveals the opponent (`reveal: off`) and builds its worlds
from the information set alone (S′-HONEST's identity guard rebuilt 600 / 600 sampled decisions from I, 0 mismatches).
The bank's honesty guard (§2) holds every one of this bank's labels to that, record by record, and refuses the one
truth-scored field the records carry. The search's RNG key is a SHA-256 digest of the deal (`search_side.deal_key`:
the seed and both packed teams); it seeds S′h's own draws and carries no information about the battle's dice or the
hidden team. So a DELIVERED here cannot be the leak, unlike any reading of X65.

### 8.2 Search noise: does the design need repeated labels?

A label is ONE sample of a stochastic search. Measured (zero CPU, [`noise_ceiling.py`](noise_ceiling.py) →
[`noise_ceiling.json`](noise_ceiling.json), stage 1b's 157 replicate pairs of `r0s4_honest_inc@10` under another key):
S′h keeps its own pick **61.8 % [54.0, 69.0]** of the time (S′-LIVE's leaky `rolls` builder: 72.3 %, stage 1), it
departs from P\* on 36.9 %, and **both runs depart to the same action on only 12.1 % [7.9, 18.1] of decisions: about a
third (32.8 % [22.1, 45.6]) of S′h's departures reproduce.** Honest labels are NOISIER than X65's.

**The design does not need repeated labels, for three reasons:**
1. **The target stays unbiased.** The minimiser of the expected KL(π′ ‖ π_θ) over noisy samples is E[π′ | observation],
   S′h's own choice distribution: the policy that actually played and won +2.00 pp. Noise costs efficiency, not
   direction, and the student's greedy argmax is the MODE of that distribution (it keeps P\*'s choice where S′h's
   departures are noise, and learns only the departures that reproduce).
2. **The gate stays unbiased.** On a held-out row the label is an independent draw, so E[Δagree] = E[p_student −
   p_P\*] over S′h's choice distribution p; noise widens the interval (≈ ± 0.15 pp at X65's size, smaller here), it
   does not move the sign. **But it caps the size:** a perfect student gains only where S′h's mode differs from P\*'s
   choice, so the ceiling is a few pp (UNVERIFIED), never 37 pp; the reported comparison is the student's `read`
   agreement against the 61.8 % self-agreement.
3. **Repeats are not affordable.** A second honest sample of a banked decision needs its incremental carry rebuilt:
   S′-HONEST's REBUILD re-score cost 25.8 CPU-h for 600 decisions (≈ 155 CPU-s each), so one repeat of the 69,988
   departures is ≈ 3,000 CPU-h of the agent cores X60-T's look 2 holds.

What noise does change is the reading of a NOT-DELIVERED: two thirds of the departures are unlearnable by any
observation-function student, so the signal is the reproducible third (≈ 12 % of decisions): ESTIMATE ≈ 13.7k
reproducible departures in a training fold of ≈ 113k label rows. (X65's leaky labels have no measured equivalent:
stage 1's noise rows hold S′'s picks without P\*'s.) X65L-3's noise ceiling on the bank's own rows stays unbuilt
(§12).

### 8.3 What each class means for MF-5 (the scale bet) and MF-1b (ExIt-1): the pre-commitments, binding

- **No class of X66 is evidence for or against the scale bet (MF-5)** (pre-commitment 1, §0). The two confounds are
  measured, not hypothetical: S′h's labels are only about a third reproducible under another key (§8.2), and the gain
  being distilled is one panel's (P72 +4.69 pp; H +1.32 and F 0.00, both intervals spanning 0). A NOT-DELIVERED or a
  NULL cannot separate "P\*'s network is too small to absorb honest lookahead" from "the target is mostly key noise,
  and its signal lives on one panel". The §7 readings (which pattern the curves show) are reported for what they say
  about THIS recipe and THESE labels, never quoted as a capacity result.
- **No class of X66 plans MF-1b (ExIt-1)** (pre-commitment 2). A GO with the placebo's SEARCH-INFO makes an ExIt-1 case
  possible, never automatic: any follow-up needs its own case, its own evidence and its own registration.
- **What X66 CAN say:** DELIVERED or not, at this recipe, on S′-HONEST's honest labels; on DELIVERED, whether the
  distilled student beats P\* against Kakuna on the held-out panels with no search at play time.

### 8.4 Known limitation: the source gain is one panel's

S′-HONEST's per-panel G (REPORTED there, never decided; `331f19cfe`): **P72 +4.69 pp [+2.00, +7.37], H +1.32 pp
[−1.39, +4.00], F −0.005 pp [−2.98, +2.71]**. The pooled +2.00 pp that motivates this read is therefore carried by P72;
on H and F, S′h's play is not shown to beat P\* at all. Consequences for X66:
- the students' targets on H and F may carry little game value even where they are learnable; the gate measures
  agreement, not value, so a DELIVERED on those panels does not imply a game gain there;
- X66's Kakuna classes are pooled over U and make no per-panel claim; its per-panel G_X is REPORTED only, beside
  S′-HONEST's per-panel G on the same deals;
- the S_P student (read on P72, trained on H + F labels) learns from the two panels where S′h's gain is not shown.

## 9. Power and cost

**The gate** has no power problem (§4). **The Kakuna read** ([`power.py`](power.py) → [`power.txt`](power.txt): a
two-look normal Monte Carlo, 200,000 draws; look 1 2,700 deals, look 2 5,100; d = the share of deals where the arms
differ, S′-HONEST's measured 0.350 as a floor and X65's 0.45–0.50, since a student departs from turn 1):

| true Δ | 0 | +0.5 | +1.0 | +1.5 | +2.0 | +3.0 |
|---|---|---|---|---|---|---|
| P(GO) | 0.024–0.025 | 0.07–0.09 | 0.17–0.22 | 0.32–0.43 | 0.52–0.67 | 0.85–0.95 |
| P(GO at look 1) | 0.005 | 0.014–0.016 | 0.03–0.05 | 0.07–0.10 | 0.14–0.21 | 0.36–0.53 |
| P(NULL) | 0.56–0.71 | 0.37–0.48 | 0.21–0.27 | 0.10–0.12 | 0.04 | 0.003–0.004 |

The read is powered for S′h's whole gain, not for half of it; the banked control halves its lane cost, and more
deals would need a fresh raw arm on seeds no read has claimed.

**Cost:**

| item | basis | ESTIMATE |
|---|---|---|
| bank | X65's 6,351 s for 5,398 units, 3 workers | ≈ 3.5 h wall, ≈ 8 CPU-h |
| students (GPU) | X65's 0.33 s a step; ≈ 5,300 steps + 25 dumps a student | **≈ 2.25 GPU-h** |
| gate (CPU) | 3 × (pick + export + score over ≈ 63k rows × 2 models) + the report | ≈ 1–2 CPU-h |
| G-BANK + look 1 | S′-LIVE's raw units ≈ 29 CPU-s; 2,835 units | ≈ 23 CPU-h (only on DELIVERED) |
| look 2 | 2,400 units | ≈ 20 CPU-h |
| dose-matched student (on DELIVERED) | half a student | ≈ 0.4 GPU-h |
| placebo + home72 (on GO) | 2,700 + 1,800 units | ≈ 37 CPU-h |

## 10. The launch (the GPU step is not a `main.launcher` run)

- **No launcher argv exists to check.** The GPU step is `main.x65.student train`, an offline fine-tune, as X65's
  was; `main.checkargs` and `main.launcher --dry-run` validate only launcher runs (FINDING X66-F2). Its own checks are
  the kit's preconditions and the trainer's refusals (the bank sha256 and source, P\*'s bytes, the recipe's learning
  rate, a resume under other knobs), and the CPU smoke of Amendment 1.
- **The GPU kit** ([`gpu_students.sh`](gpu_students.sh)), run by the lease holder once X60-T's GPU work ends:
  ```bash
  GEN3AI_GPU_LEASE_TOKEN_FILE=<lease token file> \
    designs/research_state/measurements/x66_exit0h_2026-11-09/gpu_students.sh      # S_H, S_F, S_P in order; resumable
  ```
  Preconditions first (each a HOLD): no `NO_GO`, no PENDING value, the side-pin worktree at the pin and clean
  (created from the tag and bootstrapped if absent), P\*'s sha256, `bank check` against the registered sha256. Then
  per fold, ≤ 6 chunks of ≤ 50 min:
  ```bash
  env -u CUDA_VISIBLE_DEVICES -u GEN3AI_TEST_ALLOW_GPU taskset -c 0-4,8-12 scripts/ops/gpu_lock.sh \
    "$PY" -m main.x65.student train --recipe x66 --bank ~/gen3ai_archive/x66_exit0h_2026-11-09/bank \
    --bank-sha256 <Amendment 1> --fold heldout72_hpfix --lr 1e-4 \
    --init /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip \
    --out ~/gen3ai_archive/x66_exit0h_2026-11-09/students/s_h --device cuda --budget-min 50
  ```
  (exit 3 = the chunk's budget: re-run; any other non-zero exit HOLDs). Outputs stay under
  `~/gen3ai_archive/x66_exit0h_2026-11-09/`; nothing is written under `models/`.
- **The CPU gate** ([`gate.sh`](gate.sh)), the read operator's, rule 5b: per fold `student pick --recipe x66`,
  `export write --pin-root <play-pin worktree>`, `delivery score`; then `delivery gate`, `delivery_by_hand.py` and
  `learnability report`. Its output `delivery.json` is what the read specs name.

## 11. VOID (any one voids the read or the look named)

- The bank's drop share above 0.5 % on any panel; a `HiddenInfoError` (the bank does not exist); a bank or student
  off the registered sha256.
- A student whose training or inner-validation rows hold a read-panel row or deal.
- A pick that reads any `read` number (by construction: `pick_val` reads `val` alone).
- The export failing bit-identity at the play pin.
- The gate's file disagreeing with `delivery_by_hand.py`.
- A counted unit off E6 or off its registered sha256; drops above 0.5 % of a panel at a look; an unfinished deal at a
  look's close; `check-record` not OK; the game `by_hand.py` disagreeing with the rule.

## 12. Builds

| build | state |
|---|---|
| the bank's honest source and honesty guard; the x66 recipe, its val-Δagree pick and its export pin | **BUILT** with this registration (`src/main/x65/`; tests that fail on revert in `bank_test.py`, `student_test.py`, `export_test.py`) |
| the specs, `check-spec` | **BUILT** (`--pin-pending` exit 0 ×3 at `e46e3ddf8`; a stand-in-filled look 1 checks clean) |
| the GPU kit, the CPU gate, the gate's by-hand check | **BUILT** (`gpu_students.sh`, `gate.sh`, `delivery_by_hand.py`) |
| the game statistic's by-hand check | X65's `by_hand.py`, generic in its inputs and spec (§6) |
| the dose-matched student (a deal-hash half of a fold's training rows) | owed, built only on DELIVERED |
| the placebo permutation and its attribution rule (X65 Amendment 1's, never built) | owed, built only on GO |
| X65L-3's noise ceiling on the bank's own rows | not built (≈ 3,000 CPU-h at REBUILD cost, §8.2) |

## 13. GUESSES (unmeasured; for the record)

- P(DELIVERED) ≈ 0.3: honest labels remove the leak, but they are noisier (61.8 % vs 72.3 % self-agreement), and at
  this exact recipe the leaky labels read ≤ 0 at every checkpoint.
- Given DELIVERED: pooled Δagree +0.5 pp (0, +2); G_X +0.5 pp (−1, +2); P(GO) ≈ 0.15, P(NULL) ≈ 0.4, P(UNDECIDED)
  ≈ 0.45.

## FINDINGs

- **X66-F1 (main cannot train or score this read's students).** Main's model is config v166 (`7a0153b46`) and since v167 (`566d878b2`), a forward
  change to P\*'s logits (our stats floor the EV; Leech Seed drains an integer); S′-HONEST played at v165
  (`e46e3ddf8`). Every step that runs P\*'s function (the bank's P\* check, the trainer's teacher and student, the
  export's trainer side, the gate's scores) runs at the side pin. The export's bit-identity on a 64-row fixture would
  not have caught a main-trained student: the fixture need not hold a row the v166 change moves.
- **X66-F2 (no launcher argv).** `main.checkargs` / `main.launcher --dry-run` do not apply to an offline fine-tune;
  the kit's preconditions and a CPU smoke of the train command stand in for them.
- **X66-F3 (honest labels are noisier).** S′h's self-agreement under another key is 61.8 % against S′-LIVE's leaky
  72.3 %; only a third of S′h's departures reproduce (§8.2). Any later honest-label distillation (ExIt-1's label
  factory) inherits it; a de-noised target (λ, or several keys a decision) is a separate lever, not this read's.
- **X66-F5 (a fresh worktree's engine stamp; the coordinator's hazard from 90ed5fb5a FINDING 8, checked
  2026-11-09).** At 10:35 PST the play-pin worktree `x66_play_e46e3ddf8`, created for `check-spec` but not yet
  bootstrapped, stamped `port=2bbfe333c0a29c2d/94;env=bb63cbaba0c633d8/129` (no `src/rust_env/Cargo.lock`, no release
  lib). `bootstrap.sh --skip-env --with-rust` then produced both (the lock byte-identical to main's) and it stamps E6.
  The side pin `x66_pin` stamped E6 from its own bootstrap at 10:00 (lock and release lib present), before the bank
  build started; main's worktree stamps E6. Nothing was scored or played on the wrong engine: only `check-spec` ran
  there, which loads no engine. The kits now refuse a non-E6 worktree (§5.1); the check was exercised on both
  worktrees (PASS) and on X65's E4 pin worktree (HOLD).
- **X66-F4 (the search key hashes the true seed and teams).** It is an RNG seed with no usable information (§8.1);
  stated so no later reader mistakes it for a channel.
