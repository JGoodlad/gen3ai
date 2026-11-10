# `main.offpool_belief` — the OFF-POOL BELIEF meter

**The question it answers.** On teams no run trained on, is a checkpoint's belief about what the opponent has NOT
shown (the unseen species, the unshown moves of a revealed mon, its unstated item) better or worse than the Smogon
prior that belief is built on? The answer is offline (CPU forwards, no game against anything), per panel, with a
TEAM-clustered interval. It is the tool form of the one-off P5 probe
(`designs/research_state/measurements/arch_generalisation_2026-10-26/`, a frozen record; ledger 2026-10-26 H3).

**Why the prior is the bar.** All three belief legs are already *prior + a learned delta that starts at zero*
(`HypothesisBuilder.species_set`: scores = log prior + δ; `MoveBelief`: `move_head` + the prior's logits;
`ItemBelief`: `item_head` + the log prior). A head that learned nothing scores exactly the prior, so "worse than the
prior" means the learned delta carries information that is wrong off the pool, and the size of the gap is what a
fix has to remove.

## The three steps

```bash
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
A=~/gen3ai_archive/offpool_belief_<date>
python -m main.offpool_belief bank  --panel forum72 --pilot pstar_plateau --out $A/bank/forum72
python -m main.offpool_belief score --bank $A/bank/forum72 --bank $A/bank/p72 --ckpt <final.zip> --label c0_s3501 \
    --mix 0.25 --out $A/scores/c0_s3501.json
python -m main.offpool_belief table --arm C0=$A/scores/c0_s3501.json,$A/scores/c0_s3502.json \
    --arm PSTAR=$A/scores/pstar.json --versus C0 --md $A/table.md --json $A/table.json
```

| step | does | cost (measured where stated) |
|---|---|---|
| `bank` | the pilot plays GREEDY against itself on the Rust env core; seat 1 pilots the panel's team `g mod n`, seat 2 a `home72` team; the bank keeps seat 2's decision rows (the seat that FACES the panel team) with the core's `belief` and `item` labels | 158–174 s a panel (144 games, ≈ 6,500–6,800 rows), measured 2026-11-09 with four panels building at once on the agent cores; peak 3.5 GB |
| `score` | two CPU forwards a row: λ = 0 (the checkpoint's belief) and λ = 1 of `agents.model.belief_prior_mix` (the prior alone, bit-identical to the three deltas zeroed); per-unit metrics summed per panel team | 455–467 s a checkpoint over four banks (≈ 26,500 rows), measured with five checkpoints scoring at once on the agent cores; peak 4.5 GB |
| `table` | score files in; per panel and metric: belief, prior, belief − prior, each `--mix` blend − prior, and (`--versus`) arm − reference arm | seconds |

**The bank is a pure function of its inputs.** Game `g`'s `home72` team and battle seed come from
`default_rng([seed, g])`, never from which env ran it or when another game ended (envs that outlive the schedule
play filler games whose rows are dropped), so the env count is not a knob and every panel team gets exactly
`--games-per-team` games. The manifest names the panel's content digest, the registry's sha256, the pilot's
sha256, the commit and observation semantics the rows were encoded at, and the rows file's sha256; `score`
re-checks the rows file and re-resolves the panel against the LIVE registry before reading a row.

## What is scored

Higher is better for every column. A UNIT is what a number is the mean of.

| metric | unit | value |
|---|---|---|
| `species_logp` | a decision row with ≥ 1 hidden opponent mon | mean log presence of the true hidden species (those inside the candidate set) |
| `species_hit` | the same | share of the true hidden species among the top-k presences, k = the true hidden count |
| `moves_logp` | a (row, REVEALED opponent mon) with ≥ 1 true move the observation has not shown | mean log σ(logit) over those moves |
| `moves_ll` | the same | the Bernoulli log-likelihood of the truth over the mon's WHOLE candidate set (legal, unshown, not Hidden Power): Σ log σ(logit) over the true moves + Σ log σ(−logit) over the others, in nats a unit (a sum, not a mean) |
| `moves_prec` | the same | share of them among the top-\|unshown\| of the legal, unshown candidates |
| `items_logp` | a (row, revealed mon) holding an item the observation does not state | log probability of the true item |
| `items_top1` | the same | whether the true item is the argmax |

- **The log columns are the meter's verdict numbers**; `species_hit` / `moves_prec` / `items_top1` are descriptive
  (a top-k boundary within 1e-6 in probability is COUNTED per team as `…_near_tie`, never excluded).
- `species_logp` and `items_logp` compare beliefs of EQUAL total mass (the presences sum to the hidden count; the
  item belief is a softmax). The move belief has no such constraint (each move is its own Bernoulli), so the move
  leg has two log columns. **`moves_ll` is the proper score and the one a claim about the move belief reads**: it
  charges a move raised on a mon that does not carry it. `moves_logp` scores only the true moves, so it is blind to
  exactly that: X68's fine-tuned export is within 0.1 nats of P\* on `moves_logp` off the pool and 9 to 12 nats a
  unit below the prior on `moves_ll` (P\*: 0.6 to 2.2 below;
  `designs/research_state/measurements/move_leg_offpool_2026-11-10/README.md` §1, §5). `moves_logp` stays for
  continuity with the 2026-11-09 baseline; `moves_prec` is rank only and mass-free.
- **A score file written before a column existed does not carry it**: `table` then lists the column as ABSENT for
  that panel, naming the files to re-score, and builds the rest (the 2026-11-09 banked scores predate `moves_ll`).
- The Hidden Power family (move 237 and the typed nums) is outside the move metrics: the label carries the typed
  move, the head's table the untyped one.
- Counted, never silently dropped: a true hidden species outside the candidate set (`species_off_support`), and a
  revealed mon whose true unshown move the prior's learnset table calls unlearnable (`moves_truth_illegal`).
- A `--mix λ` column is the two forwards blended in closed form (`metrics.mix_log_prob`,
  `metrics.mix_bernoulli_logits`): the belief with a FLOOR of λ on its own prior, so its log columns can never be
  below the prior's by more than −log λ. For the species leg it IS the model run at that
  `belief_prior_mix` (pinned against a third forward in `score_test.py`). For moves and items it is NOT: those
  heads read tokens downstream of the mixed species presence, so the model run at λ computes a different delta
  (pinned too). Read a move or item blend as "this head's output, floored", never as "the model at λ".

**Statistics.** A panel's number is the mean over TEAMS of each team's mean (a team with no unit for a metric is
absent, never a zero). Every interval is a seeded bootstrap over teams (10,000 draws, one index matrix shared by
every column, so a difference is PAIRED on teams). An arm of several checkpoints averages them inside each team
first; the table also prints the per-checkpoint range, the seed spread the team interval does not contain. A Δ
column's word is BETTER / WORSE only when the whole interval is on one side of zero; an interval touching zero is
NOT DETECTED.

## What it refuses, and what a number is NOT

- **Panels come from the registry by name** (`main.heldout_panels`): a SEALED panel raises before a team is read,
  a RETIRED or unregistered one is refused, and OFF-POOL is the registry's `held_out_from: training`, never
  inferred. `home72` is the IN-POOL reference and its block is labelled so.
- **A bank is held-out material**: it is refused inside any checkout or under `models/`, never overwritten, and
  its manifest says scoring-only. Nothing the meter writes is a training input.
- A checkpoint trained under an oracle reveal is refused; a bank of another observation width is refused; a bare
  run directory is refused (name the `.zip`).
- **The registry's guard is what makes a panel off-pool**, and it covers the training POOL: the meter does not
  re-derive what each run's `--team-mix` / `--human-data` source drew. `p72` is built by the same generator
  FAMILY as `n1_proc_v1`, and `heldout72_hpfix` comes from the ladder population `hr_spec_v1` samples (with its
  near-neighbour exclusion); a run fed by either source is near-distribution on that panel, and `forum72` is the
  panel neither source resembles. Say so beside any such arm's number.
- **The rows are the PILOT's states**, not each scored checkpoint's own: the meter compares beliefs on identical
  decisions, and says nothing about the states another checkpoint would reach.
- **A checkpoint older than the bank's observation semantics is scored and badged** (`obs_gap`: the
  `OBS_SEMANTICS_HISTORY` rows between its recorded config and the bank's encoding), like every other read at head.
- It is not a strength number. No game read has yet DETECTED a win-rate effect of belief quality (ledger
  2026-10-26, the λ reads); the meter says whether a belief is right, not what a right belief is worth.

## The belief-only fine-tune (`trainbank`, `finetune`)

The cheapest test of "the off-pool failure is a fact about the LABELS" (design:
`designs/endstate/design_offpool_belief.md`): keep one checkpoint, retrain only its three belief deltas on games
against teams the pool does not hold, and read the result with the meter above.

```bash
python -m main.offpool_belief trainbank --corpus n1_proc_v1 --tier ALL --teams 300 --offset 0 --seed 6801 --out $A/ft/train_0   # --check: resolve, play nothing
python -m main.offpool_belief finetune --train $A/ft/train_0 --val $A/ft/val --out $A/ft/run [--device cuda] [--check]
python -m main.offpool_belief score --bank … --ckpt $A/ft/run/finetuned.zip --label ft --out $A/scores/ft.json
```

- **`trainbank`** is the meter's game loop with seat 1 piloting a slice of a registered `--team-mix` corpus
  (`utils.team_corpora`; its loader refuses a human-data corpus, which enters training only through
  `--human-data`). The slice is `permutation(seed)[offset : offset + teams]`, so shards of one seed are disjoint;
  every drawn team is checked against the held-out panels and one hit refuses the build. A training bank has its
  own schema: `finetune` refuses a panel's scoring bank, and `score` refuses a training bank.
- **`finetune`** trains ONLY `hypothesis_builder.delta_*`, `move_belief.{norm,move_head}` and
  `item_belief_head.{norm,item_head}` (`finetune.TRAINED`); the trunk, the policy, the critic and every other
  head are frozen and stay bit-identical. The loss is the trainer's own (`agents.training.belief_bank` rows
  `hidden_team_set` at `moves_weight` 0, `move_belief`, `item`, weight 1 each). The recipe is a constant
  (`finetune.RECIPE`), not a set of flags. The exported epoch is the one with the lowest loss on the VALIDATION
  shard (epoch 0, the untouched input, is a candidate); **no panel row is read, so the pick cannot leak a panel**.
  `finetune.json` keeps the validation loss of every epoch by leg (`val_legs`) and of the PRIOR ALONE
  (`val_prior`: the three output layers zeroed for the read), so a leg that ends above its own prior is visible.
- **The export** is the input zip with only the trained tensors of `policy.pth` replaced, re-loaded and compared
  bit for bit, beside a copy of the run's `model_config.json`.
- **What it is NOT.** A procedural corpus draws its teams from the Smogon tables the prior is built from, so its
  labels can pull a delta back to the prior and cannot be expected to beat the prior on human teams. The
  fine-tuned checkpoint's POLICY reads a changed belief through a frozen trunk that never trained on it: it is a
  belief measurement, never a checkpoint to play or adopt without its own game read.

## Files

`src/main/offpool_belief/`: `bank.py` (schedule, play, manifest, `load_bank`), `score.py` (loader, layout-derived
observation slicing, the two forwards, per-team sums), `metrics.py` (the per-unit numbers, numpy only), `stats.py`
(team means, the bootstrap), `table.py`, `panels.py` (the registry reads and the write refusals), `finetune.py`
(the training bank and the belief-only fine-tune). Tests beside
each; `bank_integration_test.py` plays the real self-check core and pins that the rows do not depend on the env
count.
