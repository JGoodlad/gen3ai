# X72 — the swap forks carried toward their plateau, then read on unfamiliar teams: PRE-REGISTRATION (2026-11-25)

**Status: REGISTERED 2026-11-25, blind: before any X72 training step, any X72 game and any read. NOTHING HAS
TRAINED and no launch is authorised by this file.** The launch is the orchestrator's GO (§4.4), through a window
chain the Training session builds (§16). Nothing below changes after the first fork launches except by a dated
amendment that says why, written before the game it governs. One amendment is planned and bounded: §8.2's.

**This is a NEW registration, resting on the orchestrator's ruling of 2026-11-25, and not on any banked class.**
X70's read ([`../x70_ent0_control_2026-11-13/PREREG.md`](../x70_ent0_control_2026-11-13/PREREG.md) §15) was
`ANCHOR-ADDS` with the labels `ENT-BELOW-SWAP` and `LR-MOVED`, and its §15.4 says: "The longer-horizon or
fresh-run magnet arm is NOT licensed by this read." The ruling (the post-Segment-B card proposal's item C1,
"ENT-PLAT", `~/.cache/gen3ai/handoffs/post_segb_queue_proposal_2026-11-25.md`) queues it on other grounds: four of
four entropy-family head-to-head reads are positive, the swap is the one training lever with a positive Kakuna
read, and its forks' entropy was still falling when they stopped. X72 is the next free X number
(`git grep -Iw X72 -- designs` was empty at the registration).

**The question.** Do the three banked swap forks keep gaining when they train on under their own recipe, and is
the network they end at better than the production recipe's on teams it never trained on, against Kakuna?

**The tier is FULL** (`../../README.md` § Registration tiers): its Kakuna class is the input to a baseline
nomination (§12), and its arm commits up to ≈ 55 GPU-h.

**Abbreviations.** P\* = `rb_x26_s2601` at step 280,000,170, the banked production-recipe plateau checkpoint every
X35 fork starts from. C0 = `rb_x35_c0_s3501`–`s3508`, P\* + 20M at the unchanged production recipe (`ent_coef` 0.05,
pin `f4b25238`), the banked control. The swap = CE-3d's arm `rb_x35_ce3dswap_s7321`–`s7323`, P\* + 20M with
`--ent-coef 0 --magnet-kl-coef 0.05`: the entropy bonus replaced by a pull (a Kullback–Leibler term, weight β = 0.05)
toward a trailing copy of the policy refreshed every 2,000,000 steps, "the magnet"
([`../ce3d_magnet_screen_2026-11-11/PREREG.md`](../ce3d_magnet_screen_2026-11-11/PREREG.md)). X67 = the
`ent_coef` 0.0125 screen. X70 = `--ent-coef 0` with no magnet. X71 = the exploitability probe of those arms
([`../x71_exploitability_probe_2026-11-16/PREREG.md`](../x71_exploitability_probe_2026-11-16/PREREG.md)).
`ent_coef` = the weight of the entropy bonus in the PPO loss. Kakuna = Metamon's strongest published policy
(`metamon:Kakuna`), our external opponent. F / H / P = the three 72-team panels of unfamiliar teams, `forum72`,
`heldout72_hpfix`, `p72`; U = the three at equal weight. pp = percentage points of win rate. A "pin" is the commit
a run's code is frozen at. A "lineage" here is one swap fork and its continuation. K9(b) = the behaviour-policy
consistency check every X35 fork is read against. GPU-h / lane-h = hours of the one card / of one CPU read lane.

**In one paragraph.** One arm, no newly trained control. The three swap finals continue in ROUNDS of +20M each, at
the pin and recipe they trained at, at most FOUR rounds (to P\* + 100M). After every round the three new finals
play the three finals of the round before (3 × 3 cells × 4,000 mirrored pairs, on the card, ≈ 0.4 GPU-h): the arm
CONTINUES only on a `GAIN`, and stops on `FLAT` (`PLATEAU-AT-REACH`), on `LOSS` (`DOSE-HARM`, the round before is
the one that is read), at the cap (`NOT-AT-PLATEAU-AT-CAP`), or when any lineage's policy entropy falls below
0.2419 nats (`ENT-FLOOR`). The finals the stop rule names are then read against Kakuna on F, H and P with the
eight C0 finals and the three banked swap finals beside them, rule `b1_kakuna`, 1,200 games a final a panel,
50,400 games: **the deciding contrast is `plat_U`, the read finals minus C0 on U, `POSITIVE` / `SPANS-ZERO` /
`NEGATIVE` by a 90 % interval.** An X71-style exploitability probe of one read final (two responders, ≈ 2 GPU-h)
is a registered veto on the licence, never on the class. Price: 12.75 GPU-h a round measured, **≤ 55.4 GPU-h in
all** (≈ 17 with no gain to ≈ 54 at the cap); **350 lane-h of CPU at the registered rate, ≈ 50 h of wall at the
realised one** (§14). The honest size of it: three lineages of ONE parent; the stop rule speaks about six files, not about the
recipe (§6.2); the Kakuna read has power 0.90–0.97 at +3 pp and 0.62–0.71 at +2 pp (§8.4); and a `POSITIVE` does
not show that the EXTRA dose added anything on Kakuna, because the banked swap finals already read +2.61 pp on F
and H (§8.3).

---

## 1. Why now, and what this can and cannot say

**What is banked** (the ledger wins any disagreement with this summary).

| read | result | record |
|---|---|---|
| swap vs C0, head to head | D +3.97 pp, 90 % [+3.31, +4.63], `PASS` | CE-3d |
| swap vs X70 (ent 0, no magnet) | D +0.73 pp [+0.22, +1.23], `ANCHOR-ADDS`, `ENT-BELOW-SWAP`, `LR-MOVED` | X70 §15 |
| X70 vs C0 (reported) | +3.00 pp [+2.48, +3.51] | X70 §15 |
| swap vs C0 **on Kakuna, F and H** | 0.5287 (3,807 / 7,200) vs 0.5026 (9,650 / 19,200), **+2.61 pp [+0.95, +4.28]**, `POSITIVE`, `KAKUNA-CONFIRMED` | ledger, 2026-11-18 |
| X71, exploitability of the swap's s7321 final | D −0.59 pp [−4.12, +2.95], `NOT-MORE-BY-BAR`: "not a clean bill: one final, one responder recipe, 5M" | X71 §13 |
| the swap forks' entropy | 0.484 / 0.496 / 0.492 nats at the end, against C0's ≈ 1.01; 5M-window means ≈ 0.77, 0.60, 0.54, 0.50: still falling | CE-3d `record/diag9_ce3d.json` |

**What X72 can say.** (1) Whether THESE three networks kept getting stronger against their own earlier selves,
round by round, at a reach of ±0.31 pp (§6). (2) Whether the networks the arm ends at beat the production control's
on the full unfamiliar set U against Kakuna, the first read of any entropy-family final on P (§8). (3) Whether one
of them is more exploitable by a fresh responder than C0's final, at X71's reach (§10).

**What it cannot say.**

1. **Nothing at the lineage level.** All three forks descend from one parent, P\*. A lever is replicated only
   across independent parents (UNDERSTANDING §7); that is the proposal's item C2, not this.
2. **Not "the swap recipe beats production at matched dose".** The control is C0 at P\* + 20M; the arm ends at
   P\* + 40M to + 100M. §2 says why no matched control is trained and what that costs.
3. **Not which half of the lever works.** X70's question stays where X70 left it.
4. **Not a plateau in `main.plateau`'s sense.** That meter compares a run with its own node 50M earlier; for a
   fork the node is its parent's, so X72 registers its own rule and its own label (§6).
5. **Not goal (b)'s level.** The read finals' rate on U against Kakuna is reported with no class (§9).
6. **Not exploitability in general**: one final, one responder recipe, 5M (§10).

## 2. The arm, and the control that is NOT trained

**The arm.** Three lineages, each a banked swap final continued at the swap's own recipe:

| lineage | starts from (round 0) | sha256 of that final | run |
|---|---|---|---|
| 1 | `rb_x35_ce3dswap_s7321/final_model.zip`, step 300,061,272 | `read_spec_kakuna.json`, `s0_s7321` | `rb_x72_swapplat_s7341` |
| 2 | `rb_x35_ce3dswap_s7322/final_model.zip`, step 300,060,944 | `s0_s7322` | `rb_x72_swapplat_s7342` |
| 3 | `rb_x35_ce3dswap_s7323/final_model.zip`, step 300,061,282 | `s0_s7323` | `rb_x72_swapplat_s7343` |

**One lever against the Kakuna control: the swap recipe AND its dose, together.** That is a compound, and it is
registered as one: the read asks about a network ("is the network this arm ends at better than the production
control's"), which is what a baseline nomination needs.

**Why no matched continuation control is trained.** A matched control would continue three C0 forks the same
rounds at the production recipe: + 12.75 GPU-h a round, up to + 51 GPU-h, doubling the arm. It would license one
extra sentence: "at equal dose the swap recipe beats production". No decision this registration feeds needs that
sentence. The nomination compares networks; the dose question is answered inside the arm, round against round
(§6), with the banked swap finals in the Kakuna read (`dose_U`, §8.3). What the choice costs is limit 2 of §1, and
it rests on one stated assumption: **the production line gains nothing from more steps at this stage.** Its
evidence is one run: X26's continuation at 480M against P\* read +0.08 pp [−3.90, +4.06] on U against Kakuna
(ledger, 2026-10-25, "FLAT"; an interval that wide also holds a gain of several pp), and its in-house meter did
not rise from 380M to 480M (−0.01 pp [−1.54, +1.52]). If that assumption is wrong, a `POSITIVE` here is partly
dose. A matched control is a new registration if a later decision needs the recipe claim.

**Declined as Kakuna columns:** X70's four ent-0 forks (a different lever; the proposal's lane item L1 may play
them on its own registration) and X67's finals. **Kept in the read:** the three banked swap finals (arm `s0`),
because without them a difference from the banked +2.61 pp could not be told from a difference of day and pin.

## 3. Rounds, seeds, pin, dose

- **A round** is three launches, one lineage at a time in lineage order, each to the round's target step.

  | round | form | `--model` | `--steps` | dose a lineage |
  |---|---|---|---|---|
  | 1 | FORK into a new run dir | the banked swap final | 320,000,170 | +19,938,898 / +19,939,226 / +19,938,888 |
  | 2 | RESTART of the same run | `models/rb_x72_swapplat_s734N/final_model.zip` | 340,000,170 | +20M less the final's overshoot |
  | 3 | RESTART | the same | 360,000,170 | the same |
  | 4 | RESTART | the same | 380,000,170 | the same |

  Round 1's dose is 61 thousand steps short of 20M because a final lands a few rollouts past its target; the
  targets are fixed steps, so the lineages stay within one rollout of each other at every round end.
- **Seeds 7341 / 7342 / 7343**, typed on round 1 and inherited after (a new seed per lineage, as every X35 fork of
  a banked file carries its own).
- **Pin `9e97cad5709da21a1f9d4d8846096c87eb17b1d0`** (tag `ce3d-pin-9e97cad5`, off main), the swap forks' own
  training pin, on every launch of every round. Nothing is typed but the five flags of §4.1: the recipe is the
  parent's record.
- **The banked swap run dirs are never written.** Round 1 is a fork so that they stay as banked.
- **After every round, before anything else is launched,** each lineage's `final_model.zip`,
  `final_model.magnet.pt`, `model_config.json` and `metadata.json` are COPIED to
  `/home/goodlad/gen3ai_archive/x72_swap_plateau/finals/r<k>/rb_x72_swapplat_s734N/` and their sha256 recorded
  (`finals.json` beside them): the next round overwrites the run dir's final, and every cross and read of X72
  plays the archive copy, never a run dir's live file. ≈ 0.25 GiB a round.
- **No interval restart falls inside a launch** (`--restart-interval-hours 24`; a launch is ≈ 4.25 h).

**One asymmetry, stated as a limit.** A fork re-seeds the opponent pool (the dry run: "pool snapshots … cap 20,
fork re-seeds the pool"), as every X35 fork did; a restart keeps the pool it has. Round 1 therefore starts from a
fresh pool and rounds 2–4 from an inherited one. Forking every round would remove the asymmetry at the price of a
pool that never ages past 20M, which is further from "the run trained on". Rounds are compared only with the
round before, never pooled.

## 4. The launch commands, their validation, and the GO

### 4.1 The commands

```bash
# round 1, lineage N (1..3)
python -m main.launcher --restart-interval-hours 24 --pin-commit 9e97cad5709da21a1f9d4d8846096c87eb17b1d0 \
  --model /home/goodlad/dev/gen3ai/models/rb_x35_ce3dswap_s732N/final_model.zip \
  --steps 320000170 --device cuda --seed 734N --run-name rb_x72_swapplat_s734N
# rounds 2, 3, 4 (TARGET = 340000170 | 360000170 | 380000170)
python -m main.launcher --restart-interval-hours 24 --pin-commit 9e97cad5709da21a1f9d4d8846096c87eb17b1d0 \
  --model /home/goodlad/dev/gen3ai/models/rb_x72_swapplat_s734N/final_model.zip --steps TARGET --device cuda
```

The X35 kit (`launch_x35.sh <arm> <seed>`) has no X72 arm; adding one is a code unit and out of this
registration's scope, so the direct command is the registered form, as X71's was.

### 4.2 What was validated, on CPU, at the registration

Run from a worktree at `4fe7ef436` with `CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores; nothing was created
under `models/`.

- **`python -m main.checkargs --argv "<round-1 argv>" --pin 9e97cad5…`: exit 0, all three**
  ([`checkargs_x72_r1_s7341.txt`](checkargs_x72_r1_s7341.txt), `…s7342`, `…s7343`; the argvs are
  [`argv_x72_r1_s7341.txt`](argv_x72_r1_s7341.txt) and its two siblings): "argv validated against PINNED parser
  @9e97cad5 … 390 options".
- **`python -m main.launcher --dry-run …`: exit 3, all three** ([`dryrun_x72_r1_s7341.txt`](dryrun_x72_r1_s7341.txt)
  and siblings), stopping at exactly the registering shell's own two refusals (the hidden card; nice 19). Each
  resolves: role `FORK of …/rb_x35_ce3dswap_s732N`; pin `9e97cad5…` (source: `pin_commit`); 41 untyped knobs
  restored from the PARENT's record and passed typed, among them **`ent_coef 0.0` (production: 0.05)**,
  **`magnet_kl_coef 0.05` (inherited; production: 0.0)**, `n_epochs` 10, `behaviour_check` `warn`,
  `checkpoint_every_steps` 1,000,000; `--steps 320,000,170 vs checkpoint at 300,061,272 (sidecar) → +19,938,898`
  (lineage 1); disk 44.01 GiB free against 17.57 GiB REQUIRED.
- **The restart form** cannot be dry-run on a run that does not exist. Its shape was run against the banked
  `rb_x35_ce3dswap_s7321` instead ([`dryrun_resume_form_on_banked_s7321.txt`](dryrun_resume_form_on_banked_s7321.txt);
  a dry run creates nothing): role `RESTART of …`, the same pin, `ent_coef 0.0` and `magnet_kl_coef 0.05` from the
  run's OWN record, 16.84 GiB REQUIRED.

**UNVERIFIED, each with the check that covers it:**

1. **A magnet start from a sidecar on the card.** CE-3d §4.4 item 4: "a resume with the sidecar was NOT exercised:
   no fork restarted" (the CPU integration test passes). X72's first launch is the first. Covered by §11's banner
   check and `magnet/refreshes`; a lever-own FATAL on the first launch STOPS everything.
2. **Which file a same-run restart loads.** The restart dry run prints "RESUME role — --model → the run's latest
   checkpoint" and prices disk from `checkpoint_300000383_steps.zip`, while its steps line reads the final's
   300,061,272. The chain records, from the launcher log of every round-2+ launch, the file and step actually
   loaded. Loading the newest periodic checkpoint instead of the final replays ≤ 1,000,000 steps and is a
   recorded deviation of the dose, not a VOID; loading anything older is a HOLD.
3. **The rate and its controller across the fork.** `fork_lr None (untyped default)`: whether the fork starts at
   the parent's last rate (1.0047e-04; 8.3724e-05 for s7322) or elsewhere was not exercised. Reported (§9), with
   no bar.
4. **The dry run from a nice-0 shell with the card visible** (no lease in this session): GO condition G3.

### 4.3 What the first launch must show (the chain's banner check; CE-3d's, verbatim)

```json
{"expect": "beside the entropy bonus (ent_coef 0; the KL's own entropy part adds 0.05 to that weight)",
 "void": ["ent_coef   | 0.05 ", "ent_coef   | 0.1 ", "ent_coef   | 0.0125 ", "[ModelVersion] FATAL"],
 "within_min": 30}
```

The expected text is the line the pin's `magnet.attach` prints. It is checked on EVERY launch of every round: a
restart that came up without the magnet, or with an entropy bonus, is stopped by the chain and is VOID.

### 4.4 The GO conditions (the orchestrator's, with the Training session)

| | condition |
|---|---|
| G1 | Segment B's chain has written its end line, the card is empty, and the launching session holds the GPU lease |
| G2 | the three banked swap finals hash to `read_spec_kakuna.json`'s `s0_*` sha256 |
| G3 | the launcher's dry run of the next launch exits 0 with the card visible from a nice-0 shell, and prints the role, pin, `ent_coef 0.0` and `magnet_kl_coef 0.05` of §4.2 |
| G4 | free disk ≥ that dry run's REQUIRED figure (§13). Short ⇒ HOLD; the chain deletes nothing |
| G5 | the play checkout at `333a73dd` (§5) and the off-main pin tag `ce3d-pin-9e97cad5` exist and verify |
| G6 | the round cross's driver, plan and reader, and the floor's reader, are built and checked (§15); until then no launch |

## 5. The round cross (the meter of the stop rule)

After round k (1 to 4), once the three round-k finals pass §11 and are archived:

- **rows** = the three round-k finals; **columns** = the three round-(k−1) finals (round 0 = the banked swap
  finals); every row against every column, **9 cells × 4,000 mirrored pairs = 72,000 games**.
- **Protocol: X70's deciding cross, unchanged** (its §5.1): each file's network, both sides greedy, batches of 500
  pairs, schedule seed 0, the players' eval team builder (no panel), `--oracle-reveal-mode off`, purpose `ab`,
  label `x72_swap_plateau`, family `x72_round_h2h`, request `x72_r<k>`.
- **Play pin `333a73ddda7829b9ccdf9e17ef04624eb40edeba`**, the pin CE-3d's and X70's crosses were played at (the
  training pin is that commit plus three paths no play code imports). The banked checkout
  `/home/goodlad/dev/gen3ai-reads/ce3d_play_333a73dd` is used read-only if it verifies (`rev-parse`, clean `src`
  and `data`); otherwise a fresh detached checkout.
- **Venue: the card** (`cuda`, under the lease), ≈ 0.40 GPU-h (X70 played 96,000 games in 32 min). One venue a
  cross (X70 §14.2): a cross is finished on the device its first batch was played on.
- **`h[i][j]`** = row i's score against column j in pp, as X70's reader computes it.
- **`INCONCLUSIVE`** iff any cell is missing, not `OK`, or holds other than exactly 4,000 completed pairs after the
  driver's resume. Then no statistic is computed and the chain HOLDs.
- **One look a round.** A round's cross is played once; no top-up, no replay.

## 6. The stop rule ([`x72_rule.py`](x72_rule.py); self-check [`selfcheck.log`](selfcheck.log))

### 6.1 The statistic and the class

D = mean of the nine `h` − 50. SE = √(0.3125 / 9) pp, where 0.3125 pp² = 2500 / (2 × 4,000) is the binomial
bound on a cell's variance at 50 % (banked 1,000-pair cells measured 0.80–1.10 pp against the bound's 1.118).
The **90 % interval is D ± 1.6449 × SE = D ± 0.3065 pp.**

- **`GAIN`** iff its lower end > 0; **`LOSS`** iff its upper end < 0; **`FLAT`** otherwise.
- **Rule 8:** an end within 1e-9 pp of 0 fires nothing (the class is `FLAT`) and is recorded.

### 6.2 Why a fixed-finals interval, and the claim it does not make

The interval treats the six files as GIVEN: game noise is its only noise. That is the question a stop rule asks
("did these three networks get stronger?"), and it is how `main.plateau`'s Tier 1 treats two nodes of one run.
**It makes no claim about the recipe.** "More dose under the swap helps a fork" would need the lineages treated as
a sample of forks, which is X70's `rect_stat`; `decide-round` REPORTS that interval beside the class and it never
decides. At three lineages it is 2 to 4 times wider (median half-width 0.75–1.32 pp at §6.4's scenarios, against
0.31), and a rule built on it would call a true +0.5 pp round gain `FLAT` at least three times in four. The cost of the choice: when forks
differ by luck, a `GAIN` or `LOSS` at zero recipe gain is a TRUE statement about these files and happens 16–25 % of
the time each way (§6.4). A chain that continues on such a `GAIN` spends a round on luck; one that stops on such a
`LOSS` reads the round before.

**Against `main.plateau`:** that meter needs two consecutive FLAT ticks before it says plateau; X72 stops at the
first `FLAT`. The label is therefore `PLATEAU-AT-REACH`, never "plateau", and its reach is ±0.31 pp on one look.

### 6.3 The next action (a pure function; `x72_rule.next_action`)

| after round k | action | arm label | the round that is READ |
|---|---|---|---|
| the entropy floor fired in round k (§7) | STOP; the cross of round k is not played | `ENT-FLOOR` | k − 1 |
| cross `INCONCLUSIVE` | HOLD: nothing is launched, the orchestrator rules | — | — |
| `LOSS` | STOP | `DOSE-HARM` | k − 1 |
| `FLAT` | STOP | `PLATEAU-AT-REACH` | k |
| `GAIN`, k < 4 | CONTINUE to round k + 1 | — | — |
| `GAIN`, k = 4 | STOP | `NOT-AT-PLATEAU-AT-CAP` | 4 |

**"Read round 0" means the banked swap finals: no new Kakuna read and no probe is played under this registration**
(§8.6). The cap is four rounds because that is the card's budget for this item (≤ 51 GPU-h of training); it is not
a claim that 80M suffices.

### 6.4 Operating characteristics ([`power.log`](power.log): `x72_rule.py simulate`, 50,000 reps, seed 20261125)

sig_F = the SD of a fork's own luck in a round (real strength of that file), sig_L = a lineage's persistent
offset, both in pp. Neither is measured for a continuation: X70's registration used 0.3–0.5 pp as its scenarios,
and its realised half-width (0.51 pp) sits between its own rows for 0 and for 0.3.

| sig_L / sig_F | P(`GAIN`) at a true round gain of 0 | +0.25 | +0.5 | +0.75 | +1 | +1.5 | P(`LOSS`) at 0 | at −0.5 | at −1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 / 0 (game noise only) | 0.040 | 0.371 | 0.867 | 0.995 | 1.000 | 1.000 | 0.040 | 0.866 | 1.000 |
| 0 / 0.3 | 0.157 | 0.428 | 0.739 | 0.930 | 0.989 | 1.000 | 0.153 | 0.737 | 0.989 |
| 0 / 0.5 | 0.245 | 0.450 | 0.667 | 0.840 | 0.941 | 0.997 | 0.246 | 0.668 | 0.940 |
| 0.5 / 0.5 | 0.244 | 0.447 | 0.668 | 0.840 | 0.940 | 0.996 | 0.246 | 0.669 | 0.941 |
| 0 / 1.0 | 0.358 | 0.474 | 0.592 | 0.704 | 0.798 | 0.922 | 0.355 | 0.589 | 0.797 |

A lineage's persistent offset cancels (rows and columns are the same lineages). **The rule resolves a round gain
of +0.75 pp or more (0.84–0.99) and of +0.5 pp two times in three; it does not resolve +0.25 pp.**

The chain as a whole (at most four rounds, 12.75 GPU-h a round), at sig_L / sig_F = 0 / 0.3 and 0.5 / 0.5:

| true gain by round | P(`PLATEAU-AT-REACH`) | P(`DOSE-HARM`) | P(cap) | mean rounds | mean training GPU-h |
|---|---|---|---|---|---|
| none | 0.795 / 0.635 | 0.205 / 0.365 | 0.000 / 0.000 | 1.16 / 1.27 | 14.8 / 16.2 |
| +1, then none | 0.798 / 0.648 | 0.202 / 0.351 | 0.000 / 0.001 | 2.15 / 2.17 | 27.4 / 27.7 |
| +1, +0.5, +0.25, 0 | 0.882 / 0.772 | 0.097 / 0.206 | 0.021 / 0.022 | 2.98 / 2.76 | 38.0 / 35.2 |
| +2, +1, +0.5, +0.25 | 0.698 / 0.661 | 0.033 / 0.125 | 0.269 / 0.213 | 3.72 / 3.55 | 47.5 / 45.3 |
| +1 every round | 0.041 / 0.218 | 0.000 / 0.006 | 0.959 / 0.777 | 3.94 / 3.65 | 50.2 / 46.6 |
| +1, then harm (`power.log`'s "gain, then harm") | 0.019 / 0.104 | 0.981 / 0.896 | 0.000 / 0.000 | 1.99 / 1.94 | 25.4 / 24.7 |

**Read plainly:** with no real gain the arm ends after round 1 three times in four or more (0.85 / 0.75), and
one time in five to one in three its label is `DOSE-HARM`. A `LOSS` in round 1 itself (0.15–0.25 with no real
gain) reads round 0 and plays no new Kakuna game. That is the registered outcome of a null, not a failure of the
chain.

## 7. The entropy floor

- **The floor: 0.5 × the lowest banked swap end entropy = 0.5 × 0.4837687760591507 = 0.24188438802957535 nats**,
  read by `x72_rule.floor_nats` from CE-3d's banked `record/diag9_ce3d.json`, never retyped. It is X70's
  registered `ENT-COLLAPSED` bar. **It is a judgment, not a measured cliff:** about a quarter of C0's ≈ 1.01 nats;
  no banked run of this family has trained below 0.46, so nothing shows where a policy this sharp stops being
  trainable or becomes exploitable. Its job is to stop spending card hours on a policy collapsing toward
  deterministic play, and to put a hard number in the way of a run nobody is watching.
- **The statistic:** the mean of a run's last 20 points of −`train/entropy_loss` at or before a MARK (CE-3d's and
  X67's end-entropy definition; 20 points ≈ 2M steps; read through `main.ops.run_ref.scalar_events`, a run's own
  points only, an abandoned attempt's points past a resume step dropped as `diag9_ce3d.py` drops them).
- **The marks:** for each lineage in each round, the steps round start + 5M, + 10M, + 15M (round start = the
  previous target; 300,000,170 in round 1) and the round's end. The statistic is a function of the mark, not of
  when it is read; the chain evaluates each mark within 30 min of the run passing it.
- **It FIRES** iff any lineage's statistic at any mark is below the floor by more than 1e-9 nats
  (`x72_rule.py floor <e1> [<e2> <e3>]`); a value within 1e-9 fires nothing and is recorded (rule 8).
- **On firing:** the chain stops the run it started (SIGTERM to its own launcher's PID, the Training Run SOP §4
  procedure), launches nothing further, and the arm STOPs with `ENT-FLOOR`; the round before is the one read.
  The interrupted run dir is kept as it is.
- **Expectation, not a bar:** the banked 5M-window means fell 0.77 → 0.60 → 0.54 → 0.50. A geometric extrapolation
  settles near 0.41 and never fires; a linear one at the last window's −0.041 nats per 5M crosses the floor about
  30M past the banked finals, in round 2. Both are GUESSes.

## 8. The decisive read: Kakuna, on unfamiliar teams

### 8.1 What is played ([`read_spec_kakuna.json`](read_spec_kakuna.json), checked: [`check_spec.txt`](check_spec.txt))

`python -m main.reads` ([`../../../tools/reads.md`](../../../tools/reads.md)), read `x72_kakuna_2026-11-25`,
rule `b1_kakuna`, opponent `metamon:Kakuna`.

| arm | finals | games |
|---|---|---|
| `plat` | the 3 finals the stop rule names (§6.3), from their archive copies | 3 × 3,600 |
| `c0` | the 8 C0 finals (sha256 pinned, equal to the 11-16 read's) | 8 × 3,600 |
| `s0` | the 3 banked swap finals (sha256 pinned, equal to the 11-16 read's) | 3 × 3,600 |

Panels `forum72`, `heldout72_hpfix`, `p72`; unit seeds 0–59; 1,200 games a final a panel; **50,400 games**, CPU
only. `min_games_frac` 0.95; `allow_partial_at_stop` true; bootstrap 10,000 draws, seed 20261125 (reported).
Everything is played after the stop, at one pin, in one read: `b1_kakuna` imports no banked cell.

### 8.2 The one planned amendment

The spec as registered is NOT playable: the three `plat` refs are the literal `PENDING` and it carries no `pin`
(`check-spec --pin-pending` lists exactly those six fields; every other load refuses). After the arm STOPs and
BEFORE any Kakuna game, a dated amendment (appended to this file as its §18) fills: the three `plat` paths and sha256 (the read
round's archive copies, from `finals.json`), and the pin. **The pin is `15136b61634b667bb9da5ccc6647ff9265238829`**,
the 11-16 entropy read's and BOUNDARY-1's, unless the amendment records why that pin cannot play these files. It
changes nothing else: no panel, count, contrast, level or class. It is written knowing the in-house crosses and
blind to every Kakuna game.

### 8.3 The statistic and the classes

`b1_kakuna`'s registered contrast of arm means with its analytic two-way (run and team) interval at level 0.90,
the rule's own code. **The deciding contrast is `plat_U` = mean(`plat`) − mean(`c0`) on F, H and P at equal
weight:**

- **`POSITIVE`** iff the interval's lower end > 0; **`NEGATIVE`** iff its upper end < 0; **`SPANS-ZERO`** otherwise.
- **Rule 8, the rule's own:** an end within 0.005 pp of 0 is AT-BOUNDARY and takes `SPANS-ZERO`
  (`main.b1.kakuna_rule.EPS_PP`).
- A final with fewer than 95 % of a panel's 1,200 games counted (or more than 1,200) ⇒ the rule's own
  `INCONCLUSIVE`, with its reasons; nothing is classed.

**REPORTED, never deciding:** `plat_FH` (the banked +2.61 pp's own panels), `plat_F`, `plat_H`, `plat_P`;
**`dose_U` = `plat` − `s0`** (did the continuation add anything on Kakuna); `s0_U`, `s0_FH`, `s0_P` (the banked
swap finals against C0, re-played). `s0_FH` is expected to reproduce the banked counts (3,807 / 7,200 against
9,650 / 19,200) if play is deterministic at this pin: **UNVERIFIED**; a mismatch is a FINDING in the read's record,
never a VOID.

**What `POSITIVE` does and does not mean.** The banked swap finals already read +2.61 pp on F and H. `plat_U`
`POSITIVE` says the networks the arm ended at beat the production control's on U; it does NOT say the extra
rounds helped on Kakuna. Only `dose_U` speaks to that, and it is underpowered (§8.4).

### 8.4 Power ([`kakuna_power.log`](kakuna_power.log): [`kakuna_power.py`](kakuna_power.py), through the rule's own `contrast` / `classify`; 2,000 simulated reads a cell)

BOUNDARY-1's ASSUMED team model (our-team SD 10 pp, their-team SD 9 pp, team-by-arm interaction 5 pp), by the SD
of a final's own offset ("seed SD"):

| contrast | seed SD | median half-width | false `POSITIVE` at 0 | P(`POSITIVE`) at +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|
| **`plat_U`** | 0 | 1.57 pp | 0.019 | 0.22 | 0.71 | 0.97 | 1.00 |
| | 0.5 | 1.58 | 0.024 | 0.23 | 0.70 | 0.95 | 1.00 |
| | 1.0 | 1.65 | 0.046 | 0.25 | 0.62 | 0.90 | 0.98 |
| | 1.5 | 2.06 | 0.061 | 0.21 | 0.46 | 0.74 | 0.93 |
| `plat_FH` | 0.5 | 1.94 | 0.022 | 0.16 | 0.53 | 0.86 | 0.98 |
| `dose_U` | 0.5 | 2.23 | 0.008 | 0.09 | 0.39 | 0.79 | 0.96 |
| | 1.0 | 2.30 | 0.024 | 0.12 | 0.35 | 0.67 | 0.88 |

With no team-by-arm interaction the half-widths are slightly narrower (`plat_U` 1.52 pp against 1.57) and the
pattern is the same. **The read resolves a `plat_U` of +3 pp (0.90–0.97 at a seed SD up to 1 pp) and is a coin
flip to two-in-three at +2 pp.**
The proposal's GUESS for it is +3.3 pp (+1, +6). At a seed SD of 1.5 pp the interval under-covers (0.88 against
the nominal 0.90; false `POSITIVE` 6.1 %): three finals cannot estimate a spread that large. `dose_U` resolves
+4 pp and little below.

### 8.5 Price, at both lane rates (the proposal's FINDING F5)

| rate | provenance | 50,400 games |
|---|---|---|
| 144 games a lane-h | BOUNDARY-1's PREREG §12, the registered price | **350 lane-h**; ≈ 117 h of wall on three lanes |
| ≈ 1,012 games an hour of wall | the 11-16 to 11-18 read: 33,600 games in 33.2 h (≈ 337 a lane-h IF it had three lanes, which the proposal did not verify) | **≈ 50 h of wall**; ≈ 150 lane-h on that assumption |

39,600 of the 50,400 games are the control's and `s0`'s, which do not depend on the arm. They cannot be played
during the rounds: a spec with a `PENDING` ref refuses to load, and the rule imports nothing. A rule that imported
a same-pin banked cell would let the CPU lanes play them while the card trains and cut the wall after the stop to
≈ 11 h. That is a code unit, not built and not registered here (FINDING, §15).

### 8.6 Limits of the read

1. **When the read round is 0** (`DOSE-HARM` or `ENT-FLOOR` in round 1) **no Kakuna game and no probe is played
   under X72.** The banked +2.61 pp stands as it is; P for the banked finals is the proposal's lane item L1, its
   own registration.
2. **P's team SDs are ASSUMED equal to F's and H's** in §8.4: no banked P game of these finals exists to measure
   them.
3. **The read finals play at the read pin's game and observation, not their training pin's** (as the banked 11-16
   read's swap finals did). That an ARCHIVE COPY loads at the read pin is **UNVERIFIED**: at HEAD,
   `main.reads.models` takes an absolute path as written and needs `model_config.json` beside the zip, which the
   copy has; the amendment's `check-spec` at the pin is the check, and a refusal is a HOLD.
4. One opponent, three panels, three finals of one parent.

## 9. REPORTED (decides nothing)

- **Two end crosses on the card**, X70's protocol, at the play pin, once each after the STOP, when the read round
  is ≥ 1: the read finals × the three banked swap finals, 9 cells × 4,000 pairs (it IS round 1's cross when the
  read round is 1; otherwise ≈ 0.40 GPU-h); the read finals × the eight C0 finals, 24 cells × 1,000 pairs
  (≈ 0.27 GPU-h), beside the banked +3.97 pp. X70's `rect_stat` interval, no class.
- **The lineages-as-a-sample interval** of every round cross (§6.2).
- **Per lineage per round:** the entropy curve in 5M windows and at every mark, the learning rate's median, its
  changes and its first value against the previous round's last (§4.2 item 3), `train/approx_kl`,
  `train/magnet_kl`, `magnet/refreshes`, `magnet/forward_ms`, `time/fps`
  (`python -m main.ops.tb_read <run> --tag …`); the file and step each restart loaded.
- **Label `LR-MOVED`** (X70's bounds): a lineage whose round `lr_median` is below 0.0000905 or above 0.0001105.
  Reported beside the class, never changing it; s7322's lineage ended its banked 20M at 0.0000837, below the
  lower bound, and carries the label from round 1 if the fork inherits the rate.
- **The read finals' own rate on U against Kakuna**, per panel and pooled, as the rule prints it: a description,
  no class. Goal (b)'s level is not read here.

## 10. The exploitability probe at the end

**What.** X71's design, unchanged, on ONE new target: the read final of lineage 1 (the lowest seed, chosen now, by
rule). Two fresh responders, seeds 7111 and 7112, runs `rb_x35_x71_plat_s7111` / `s7112`, at pin `9e97cad5`:

```
--model <T> --exploiter <T> --no-self-play --trainee-teams <X71's five offense teams>
--fork-lr 5.6e-05 --fork-lr-freeze --behaviour-check warn --steps <T's step + 5,000,000>
--checkpoint-every-steps 1000000 --device cuda --ent-coef 0.05 --magnet-kl-coef 0 --seed <7111|7112>
```

T is the archive copy. They train right after the STOP, before any Kakuna game (the card is free then and the
probe is blind to the Kakuna class). Read: X71's reader regime (the Rust eval core on CPU, greedy against greedy,
seed 20261020; the target against itself 8,000 games, each responder's final against the target 4,000: 16,000
games). L = the responders' mean rate against the target − the target's own rate against itself.
**D = L(`plat`) − L(`c0`)**, with X71's banked cells of `c0`, `x67`, `swap` and `x70` reused and never re-played
(C0's responders learned +5.66 pp); the responder variance is pooled over the FIVE targets, multiplier
t(0.975, 5) = 2.571 (X71's intervals are 95 %, and so is this one). Classes, X71's: `MORE-EXPLOITABLE` (lower end
> +0.05 pp), `LESS-EXPLOITABLE` (upper end < −0.05 pp), `NOT-MORE-BY-BAR` (upper end < +5 − 0.05 pp), `UNRESOLVED`,
`INCONCLUSIVE`; rule 8 at 0.05 pp. Also reported, no class: L(`plat`) − L(`swap`).

**Power** ([`probe_power.log`](probe_power.log): X71's own `simulate` at five targets, 20,000 reads a row):

| responder spread | half-width | true D 0 → `NOT-MORE-BY-BAR` | true D +5 → `MORE-EXPLOITABLE` | true D +2.5 → `UNRESOLVED` |
|---|---|---|---|---|
| 0 (what X71's two reads pooled to) | 2.86 pp | 0.953 | 0.956 | 0.372 |
| 1 pp | 3.65 pp | 0.749 | 0.751 | 0.567 |
| 2 pp | 5.5 pp | 0.383 | 0.406 | 0.756 |

**Its role: a registered VETO on the licence, never on the class** (as X71 was for the entropy levers).
`MORE-EXPLOITABLE` withholds §12's nomination; no other class adds anything to it. **Why it does not decide more:**
one final, one responder recipe, 5M steps; X71's own words for a pass are "not a clean bill". A probe that cannot
clear a network should not be able to promote one. Why it is run at all: the arm pushes entropy further down than
any banked final, which is exactly the direction X70's §10 warned "may win head to head and be MORE exploitable".

**Price:** ≈ 0.93 GPU-h a responder realised in X71 (1.05 registered): **≈ 1.9–2.1 GPU-h**; 7.91 GiB of disk
REQUIRED a responder by the launcher's guard; ≈ 1.6 lane-h of CPU. **UNVERIFIED tool path:** `x71_rule.py` and
`read_x71.py` freeze X71's `TARGETS` and `READS`; a copy that adds the fifth target is NOT built (§15).

## 11. VOID guards, HOLDs, the canary

**VOID** (any one ⇒ that launch, and with it the lineage's round, is VOID; CE-3d §8's list, per round):

- not finished: rc 0, `Last step` ≥ the round's target, `final_model.zip`, the newest checkpoint within 98,304
  steps of it;
- K9(b) not PASS over the round's own rows (ceilings 1e-3 max, 1e-4 p99; the forks run `--behaviour-check warn`,
  so violation files alone are not a VOID). **UNVERIFIED:** that `python -m main.x35.kit k9b --run-dir` reads a
  restart's rows by round;
- the launcher log names another pin than `9e97cad5`;
- the wrong parent: round 1, the GO dry run's role line, `metadata.json`'s lineage (the parent's sha256 and step)
  and `python -m main.sidecar_audit`; rounds 2–4, a role other than RESTART of the lineage's own run;
- round 1: the post-spawn pool-seed check fails;
- either half of the lever missing from the record: §4.3's banner; `ent_coef` 0.0 in `metadata.json` and its
  `cli_args`; `magnet_kl_coef` 0.05 with `magnet_refresh_steps` 2,000,000 in `model_config.json`;
  `final_model.magnet.pt` beside the final; `python -m main.dose` reading 10 epochs and an effective batch of
  65,536;
- a magnet that never refreshed: `python -m main.ops.tb_read <run> --tag magnet/refreshes` short of the nine
  refreshes a 20M round owes.

**A VOID is a HOLD.** The design is paired by lineage, so a round is read only with all three. A relaunch of a
VOID launch from the lineage's previous archive copy, or dropping to two lineages, is a dated amendment written
before any cell of that round's cross is played; the chain does neither by itself.

**Also a HOLD, never resolved by the chain:** a lever-own FATAL on the first launch (it STOPS everything); a
launch failure; disk short of the guard (§13); a cross `INCONCLUSIVE`; a restart that loaded an older file than
its round's newest periodic checkpoint. **A PAGE that keeps training:** throughput below 1,000 steps/s in a
launch's first hour (the banked forks' median was 1,372–1,398). **The canary:** no family bar exists at this pin
with the magnet live; a canary FATAL on a lineage HOLDs ("no family bar") and goes to the orchestrator.

## 12. What each outcome licenses, and who decides

The verdict line carries three things: the arm label (§6.3), the read round, and the Kakuna class of `plat_U`.

| Kakuna class | licenses | does not license |
|---|---|---|
| **`POSITIVE`**, probe not `MORE-EXPLOITABLE` | the sentence "the networks X72 ended at beat the P\* + 20M production control on U against Kakuna, by `plat_U` and its interval"; NOMINATING the read finals to the orchestrator and the owner as the in-house baseline candidate; registering the lineage-level replication (the proposal's C2) and a fresh-run swap arm | any change by itself (below); "the recipe beats production at matched dose"; "more dose helps on Kakuna" (only `dose_U`, reported) |
| **`POSITIVE`**, probe `MORE-EXPLOITABLE` | the sentence; the probe's class beside it | the nomination, until a registered follow-up of the exploitability |
| **`SPANS-ZERO`** | "not separated from the control on U at this read's reach (§8.4)". The banked +2.61 pp on F and H stands as banked; `s0_U` and `dose_U` say, descriptively, whether P or the extra dose moved it | a nomination; a claim of harm |
| **`NEGATIVE`** | "the continued swap networks are worse than the production control's on U". More swap dose closes at this spelling; with `s0_U` beside it the read says whether the banked finals share the result | reading it back onto the +20M swap finals without `s0_U` |
| no read (read round 0) | the arm label alone: "the first extra round did not help these networks (`DOSE-HARM`)" or "the entropy floor stopped it (`ENT-FLOOR`)" | anything about Kakuna |

The arm label licenses, by itself: `NOT-AT-PLATEAU-AT-CAP` ⇒ a further continuation may be REGISTERED (a new
registration; this one ends at four rounds). `PLATEAU-AT-REACH` and `DOSE-HARM` ⇒ no further dose of this arm.
`ENT-FLOOR` ⇒ the floor and what lies under it become a question for a registration of their own.

**Adoption is not this registration's to grant.** A named baseline changes only by
`python -m main.baselines set <name> <run>/<file>.zip --reason "<ledger title>"`
([`../../../baselines.json`](../../../baselines.json)), and the production recipe (`--arch production`'s
`ent_coef`) only in code: both are the orchestrator's proposal and **the owner's decision**. No X72 outcome moves
a recipe, a baseline or a default.

## 13. Disk

The box had **44.0 GiB free (96 % full)** at the registration; Segment B's remaining legs take ≈ 2 GiB each first.

- **The launcher's guard:** 17.57 GiB free REQUIRED for a round-1 launch, 16.84 GiB for a restart, 7.91 GiB for a
  probe responder. It is a floor on FREE disk, not consumption, and the launcher refuses below it.
- **What a launch leaves behind (GUESS from the banked swap dirs, 2.3–2.5 GiB each):** ≈ 2.4 GiB a lineage a
  round: 20 periodic checkpoints (1.07 GiB), a magnet sidecar beside each (≤ 54.7 MiB each, the final's is
  25.7 MiB; the guard's arithmetic does not count them), traces, logs, the compile cache in round 1. ≈ 7.2 GiB a
  round, + 0.25 GiB of archive copies.

| to launch all of | free disk needed at X72's start (GUESS: guard + what earlier launches left) |
|---|---|
| round 1 | ≈ 22.4 GiB |
| round 2 | ≈ 29.1 GiB |
| round 3 | ≈ 36.5 GiB |
| round 4 | ≈ 44.0 GiB |

**Rounds 1 and 2 fit. Round 4 does not fit on today's disk with any margin, and round 3 fits only if Segment B
leaves the box above ≈ 37 GiB.** The chain checks G4 before EVERY launch and, when it is short, **HOLDs: freeing
disk is a deletion, a deletion of run data is the owner's act, and the chain never thins, moves or deletes
anything to pass the guard.** X72's own periodic checkpoints are needed by no registered read (the finals are
archived), so thinning them after a round is the cheapest thing the owner could approve (≈ 4.7 GiB a round); it is
not registered here and is asked for in advance by the orchestrator, or not at all.

## 14. Cost

| item | GPU-h | provenance |
|---|---|---|
| training, a round (3 × +20M) | 12.75 | MEASURED: the three swap forks took 264.8 + 247.7 + 252.7 min (the proposal's 13.2 is its estimate) |
| training, four rounds | ≤ 51.0 | the cap |
| a round cross (72,000 games) | ≈ 0.40, ≤ 1.6 in all | X70's 96,000 games in 32 min on the card, scaled; plus one engine start each |
| the two reported end crosses | ≤ 0.67 | the same rate |
| the probe's two responders | 1.9–2.1 | X71 realised / registered |
| **total** | **≤ 55.4** | by §6.4's chain table, crosses and the probe included: ≈ 17 with no gain, ≈ 31 with one good round, ≈ 51 with gains that fade over four rounds, ≈ 54 with +1 pp every round (GUESS: which row is true is the question) |

| item | CPU | provenance |
|---|---|---|
| the Kakuna read, 50,400 games | 350 lane-h at 144 a lane-h; ≈ 50 h of wall at the realised rate | §8.5 |
| the probe's read, 16,000 games | ≈ 1.6 lane-h | X71's realised |
| the floor's and the diagnostics' readers | minutes | GUESS |

Wall: a round is ≈ 13.2 h of card; four rounds, crosses and the probe ≈ 2.3 days of card, then the Kakuna read.
The card lease ends 2026-12-11 15:42 PST.

## 15. Tool paths: what exists, what is NOT built (each an UNVERIFIED path until built and checked)

| piece | state |
|---|---|
| the launch commands, the pin, the inherited lever | dry-run and checkargs recorded (§4.2) |
| the stop rule and the floor's arithmetic | `x72_rule.py`, self-check OK; it reads no game and no run |
| the Kakuna spec and rule | `read_spec_kakuna.json` validates in its pending form; `b1_kakuna` is the banked rule |
| **the round cross's driver, plan and reader** | **NOT built.** Templates: X70's `cross/drive_cross.sh`, `plan_cross.py`, `read_cross.py`. The plan's guards are §11's; its files are archive copies, which X70's plan (run dirs, one-entry `pin_history`, a FORK of P\*) does not accept as written. That `main.h2h play-many` at `333a73dd` loads an archive copy is UNVERIFIED |
| **the floor's reader** (marks → three statistics → `x72_rule.py floor`) | **NOT built.** Template: CE-3d's `record/diag9_ce3d.py` |
| **the window chain** | **NOT built**; §16 is its checklist |
| **the probe's rule and reader for a fifth target** | **NOT built.** `x71_rule.py` and `read_x71.py` freeze X71's targets |
| an import path in `b1_kakuna` for same-pin banked cells | NOT built, NOT registered (§8.5) |

Each is a build unit with its own review; none may change a number, a class or a rule of this file.

## 16. What the window chain must check (for the Training session)

1. **Before anything:** G1–G6 of §4.4. Preconditions first; any failure HOLDs and launches nothing.
2. **Before EACH launch:** the dry run from a nice-0 shell with the card visible exits 0 and prints the expected
   role (round 1: `FORK of …/rb_x35_ce3dswap_s732N`; later: `RESTART` of the lineage's own run), the pin, `ent_coef
   0.0`, `magnet_kl_coef 0.05`, and the expected `--steps`; free disk ≥ its REQUIRED figure, else HOLD (§13).
3. **Within 30 min of each launch:** §4.3's banner, else stop that run and HOLD. Round 2+: record the file and step
   the restart loaded (§4.2 item 2).
4. **During each launch:** the floor at the + 5M, + 10M, + 15M marks and at the end (§7); firing ⇒ SIGTERM the
   chain's own launcher PID, launch nothing, STOP with `ENT-FLOOR`. Throughput below 1,000 steps/s in the first
   hour ⇒ PAGE, keep training.
5. **After each launch:** §11's VOID list. Any VOID ⇒ HOLD.
6. **After each round:** copy the three finals with sidecar, `model_config.json` and `metadata.json` to the
   archive and record sha256 (§3) BEFORE the next launch; play the round cross once on the card at the play pin
   (§5); `x72_rule.py decide-round <k> h.json`; act on `next.action` and on nothing else: `CONTINUE` ⇒ step 2 for
   the next round; `STOP` ⇒ step 7; `HOLD` ⇒ stop and report.
7. **After the STOP, when the read round is ≥ 1:** the two reported end crosses (§9); the probe's two responders
   (§10), each with its own dry run, disk check and X71's G6 record check. When the read round is 0: nothing more
   on the card.
8. **Then, off the card:** the amendment of §8.2, `check-spec` at the pin, the Kakuna read on the CPU lanes; the
   probe's read. The verdict is written by the read, not by the chain.
9. **Never:** a write under a banked run dir; a deletion, thinning or move to pass the disk guard; a relaunch of a
   VOID launch; a second look at a cross; a launch on a `HOLD`. A HOLD in this chain stops every chain behind it.

## 17. What this file does NOT do

It authorises no launch. It trains no control and registers no matched-dose claim, no lineage-level claim, no
second β or refresh cadence, no entropy bonus beside the magnet, no fresh-run arm, no fifth round, no Kakuna
column for X67 or X70, no class on `dose_U`, on either end cross or on goal (b)'s level, and no rule for what lies
below the entropy floor. It moves no recipe, baseline or default. It builds no driver, chain, reader or kit arm
(§15). It does not thin, delete or move run data.
