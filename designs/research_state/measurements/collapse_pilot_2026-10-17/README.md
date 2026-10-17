# The collapse pilot (2026-10-17): what kind of collapse does X26 show?

The pilot of [`../../../endstate/design_collapse_and_counterfactuals.md`](../../../endstate/design_collapse_and_counterfactuals.md)
§3 (owner request 2026-10-17: *"understand what type of collapse are we seeing? For example, equally good plays. We
need some way to give counterfactuals."*). **DESCRIPTIVE, not registered; CPU only (root `CLAUDE.md` rule 5b: nice 19,
the 3-core agent pin, one OpenMP thread, no GPU); read-only on `models/`.** The verdicts live in the design doc; this
file is how to reproduce them.

## Files

| file | what |
|---|---|
| `run.sh` | runs one step at this checkout under rule 5b (`PYTHONPATH`, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`) |
| `build_bank.py` + `bank_x26/` (committed copy; the build writes to the archive) | the TURN BANK from X26's (`rb_x26_s2601`, pin `9c033489`) own eval traces at 30M / 90M / 150M through `main.policy_spectrum.bank.build` unchanged: 7,028 decisions (6,335 free, 693 forced switches) in 210 battles, content sha `46233db2…`; gate ① PASS (7,385 recorded decisions re-encode byte-equal at the build commit) |
| `subset_x26.json` | the truth subset: 306 turns (276 free + 30 forced switches, 180 battles; `truth select --n-free 270 --n-forced 30`, the v1 draw rule) |
| `rows_X26at150M_S32.compact.jsonl.gz` | READ A's ground truth: every legal action × 32 shared dice seeds (CRN) played to the end in the TRUE world, both sides X26@150M greedy (`main.policy_spectrum.truth run`); compact form (`truth.compact`), full rows in `~/gen3ai_archive/collapse_pilot_2026-10-17/` |
| `reads.sh` | the policy reads (forward passes) of eleven X26 checkpoints (2M … 160M) on a bank |
| `starvation_readout.py` | the starvation readout: starved / TIE-COLLAPSE / MISRANK shares, dominated mass, the HELD-OUT gain of the best starved move and IMP (selection on half the seeds, scoring on the other half), greedy and sampled regret, entropy; paired battle-clustered deltas |
| `readout_x26truth.json` | READ A: X26 checkpoints on `bank_x26` / the X26@150M truth (S = 32, and the nested S = 16) |
| `readout_oldtruth.sh` + `readout_oldtruth_bankv1.json` | READ B: X26 checkpoints, N0@75M and K3 on Lane S `bank_v1` / `truth_v2` (three pre-era continuations, 64 seeds) |
| `honest_vs_oracle_starvation.py` + `.json` | READ C (post hoc, zero playouts): can an honest-world 8-world read, the oracle leaf or the critic DETECT starvation, against the value-discrimination pilot's 24 held-out true-world seeds (`rb_es_end2_s2001`, `rb_es_prod_s2001`, 15M) |
| `spectrum_x26bank.md` / `.json` | READ D: the policy spectrum / entropy of the eleven checkpoints on X26's bank, paired deltas |
| `strategic_series.py` + `.json` | READ E: per eval cycle, sentinel win rates by age, bot mean, the live Hodge scalar |
| `show_readout.py` | prints a readout JSON as compact tables |

## Commands (from the worktree root)

```bash
P=designs/research_state/measurements/collapse_pilot_2026-10-17
A=~/gen3ai_archive/collapse_pilot_2026-10-17
$P/run.sh $P/build_bank.py --out $A/bank_x26                                  # ~1 min
$P/run.sh -m main.policy_spectrum.truth select --bank $A/bank_x26 --out $P/subset_x26.json --n-free 270 --n-forced 30
$P/run.sh -m main.policy_spectrum.truth run --bank $A/bank_x26 --subset $P/subset_x26.json \
    --continuation /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_150000100_steps.zip=X26@150M \
    --out $A/rows_X26at150M_S32.jsonl --seeds 32 --threads 1 --workers 3 --chunk 6   # resumable; see the doc for its cost
$P/reads.sh $A/bank_x26 $A/reads_x26bank                                      # ~30 s per checkpoint
$P/reads.sh designs/research_state/measurements/m5_laneS/bank_v1 $A/reads_bankv1   # ~75 s per checkpoint
$P/run.sh $P/starvation_readout.py --bank $A/bank_x26 --rows X26@150M=$A/rows_X26at150M_S32.jsonl \
    --reads $A/reads_x26bank --labels X26@10M,X26@30M,X26@50M,X26@90M,X26@130M,X26@150M,X26@160M \
    --seeds 16 --pair X26@10M:X26@160M --pair X26@30M:X26@160M --out $P/readout_x26truth.json
$P/readout_oldtruth.sh                       # needs $A/reads_bankv1_combined: the bank_v1 reads + symlinks to the
                                             # ai_v14 baseline probs (~/gen3ai_archive/policy_spectrum/baseline_2026-09-29/)
$P/run.sh $P/honest_vs_oracle_starvation.py  # ~1 min
python3 $P/strategic_series.py /home/goodlad/dev/gen3ai/models/rb_x26_s2601/eval_results.jsonl $P/strategic_series.json
```

The Hodge read is `python -m main.elo <run> --out <a dir OUTSIDE models/>` (FINDING F-CC-1: without `--out` it writes
into the run directory); the plateau read is `python -m main.plateau status <run>` (reads the ledger, plays nothing).
