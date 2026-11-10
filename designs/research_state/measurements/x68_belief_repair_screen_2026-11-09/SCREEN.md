# X68 — can P\*'s off-pool belief be repaired after the fact?: LIGHT SCREEN (2026-11-09)

**Tier: LIGHT SCREEN. Status: REGISTERED by this file's landing commit, before the first step, game or look.** No amendments: a change to anything below is a NEW screen in a new folder that names this one.
**This screen can only PROMOTE its candidate to a full registration or DROP it. Its result is never a verdict and
is never quoted as a claim.**

## 1. Question

P\*'s belief about the opponent's unseen species is 3 to 8 nats WORSE than its own Smogon prior on teams outside
the pool (the baseline: ledger 2026-11-09; design `../../../endstate/design_offpool_belief.md`). If ONLY its three
belief residuals (212,184 parameters; trunk, policy and critic frozen) are retrained on the labels of games against
procedural teams, does that belief stop being worse than the prior on held-out teams? Worth a full registration if
it does: a finished pool-only checkpoint could then be repaired in CPU-hours.

## 2. Eligibility

| check | value |
|---|---|
| GPU cost ≤ ~15 GPU-h | 0 (CPU only, by registration) |
| CPU cost ≤ ~50 lane-h | ≈ 2: five 300-game banks ≈ 0.5 (144 games took 158–174 s); 1,728 steps × 1.9 s + ten validation passes ≈ 1.2 (timed at batch 256, one thread); one score ≈ 0.15 (455–467 s measured) |
| adopts nothing | yes |
| supports no goal claim, and no doc will quote its number as a finding | yes |

## 3. Arms and pin

| arm | what differs from the control (ONE lever) | name | start | pin |
|---|---|---|---|---|
| control | nothing: P\* (`pstar_plateau`, sha256 `814823c5…91ed`), its banked score `scores/pstar.json`, and that score's prior column | PSTAR | — | banked from the tool's tree before its commit; F re-scored byte-identical at the pin |
| candidate | the LABELS its three belief residuals are trained on: 1,200 games against `n1_proc_v1` teams (tier `ALL`), recipe `finetune.RECIPE` (lr 3e-4, batch 256, 8 epochs, Adam, clip 1.0, seed 6801), epoch picked on a 300-team validation shard | X68-FT | P\* | `67233fc3d` |

All paths under `A=~/gen3ai_archive/offpool_belief_2026-11-09`. Commands (CPU, `OMP_NUM_THREADS=1`, rule 5b):
`trainbank --corpus n1_proc_v1 --tier ALL --teams 300 --offset {0,300,600,900} --seed 6801 --out $A/x68/train_{0..3}`;
the same at `--offset 1200 --out $A/x68/val`; `finetune --train $A/x68/train_{0..3} --val $A/x68/val --out $A/x68/run`;
`score` over the four banked panels with `--mix 0.25`; `table --arm FT=… --arm PSTAR=… --versus PSTAR`.

Launch proof (`--check`, at the pin; nothing played or written):

```
trainbank … --offset {0,300,600,900,1200} --check   (five runs, the same line each)
  CHECK OK: 300 teams of n1_proc_v1, 0 held-out hits; nothing played, nothing written
finetune … --check
  CHECK OK: banks not built yet; 212184 trained / 5411995 frozen parameters; nothing trained, nothing written
```

## 4. Seeds, checked disjoint

Draw seed 6801 (one permutation of the 12,000 teams, stream 7); the five shards are slices [0, 300) … [1200, 1500)
of it, so no team is in two (`finetune.gather` refuses a repeat). Game-schedule seeds 6802, 7102, 7402, 7702, 8002
(`seed + 1 + offset`); fine-tune seed 6801. Checked: `grep -rnoE "\b(6801|6802|7102|7402|7702|8002)\b"
designs/research_state/measurements/x6* …/boundary1_scratch_2026-11-09` prints nothing, and no `models/` run is
named `s68xx`. The banked panels were played at seed 20261109. `--check` reports 0 held-out hits in every shard
(`main.heldout_panels.contamination`, exact copies and near neighbours).

## 5. Meter

**Primary: `species_logp` on `forum72` (F) and `heldout72_hpfix` (H)**, from `python -m main.offpool_belief`: the
mean log presence of the truly hidden species on the banked decisions of the seat facing each panel team (72
teams, 144 games a panel; bank rows sha256 `0caf6037fee3`, `21545e87b1fb`), as the mean over teams of team means, with
the paired team bootstrap (10,000 draws, seed 20261109, 95 %). Guards that run unchanged: the registry re-resolve
and rows sha256 of every bank, the sealed-panel refusal, the training/scoring bank schema split, the export's
bit-for-bit check that only the trained tensors changed. Reported, never deciding: `p72` (generator-adjacent to
the training corpus), `home72` (the in-pool cost), `moves_logp` and `items_logp` on all four panels, and
`finetune.json`'s validation loss by leg against the prior alone (`val_prior`).

## 6. Decision rule (ONE look)

Per panel, d_P = FT − PSTAR and d_0 = FT − prior on the primary, each with its interval from `table`.
1. `picked_epoch` = 0 (no step lowered the validation loss; the candidate IS P\*): **DROP**, nothing is scored.
2. **PROMOTE** iff on BOTH F and H: lower bound of d_P > 0 AND lower bound of d_0 ≥ −0.25 nats.
3. Anything else: **DROP**. A bound within 1e-6 of its threshold is read as not met.

(−0.25: tighter than the 0.25 prior floor reaches on P\*, F −0.30 [−0.44, −0.17]; N1-P's in-loop belief meets it.)

## 7. What each outcome does

- **PROMOTE:** a full registration is written for the repaired belief's WORTH, which this screen does not touch:
  a game read against Kakuna on unfamiliar teams (repaired P\* against banked P\*) and the belief as the searched
  player's world sampler. Until it reads, the repaired checkpoint plays nothing and seeds nothing.
- **DROP:** MF-3 step (b) closes for a frozen trunk at this size. Breadth in the loop stays the only shown repair;
  the design's trained-in floor (§7 there) is the candidate for the next fresh run. Reopens only with a new label
  source.

## 8. Cost and stop

≈ 2 CPU lane-h, 0 GPU-h. Hard stop: 2.5 lane-h in total, or the fine-tune past 5,400 s of run time
(`mem_cap.sh --max-run-s 5400`); either is reported as an overrun and the screen ends with no outcome.

## 9. Result (filled AFTER the look)

—
