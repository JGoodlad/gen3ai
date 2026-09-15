# Exercises — first set (2026-09-14)

Written at the owner's request ("write down the first set of questions; one day I will have the
time"). Three tracks. Each exercise names the artifact it is read against, what *done* looks like,
and a rough time. Do them in order within a track; tracks can interleave. The point of every
exercise is the same: **be able to verify a claim from its artifact without asking anyone.**
Explicit-only doc — not auto-maintained.

Setup for the toy tracks (once, ~1 h): the `gen3ai_stable` env already has numpy, pandas and
torch. Work in `tmp/exercises/` (gitignored). A C++/Python engineer needs three idioms and
nothing else to start: numpy broadcasting (`a[:, None] * b[None, :]`), pandas `groupby` /
`merge`, and torch's `tensor → loss.backward() → optimizer.step()` loop. Exercise T1 teaches all
three on purpose.

---

## Track R — retroactive, from this project (read and judge; some hands-on)

**R1 · Read one ledger entry and find the claim that is too strong.** (1 h)
Open `designs/research_state/ledger.md`, pick any 2026-09-12 entry. Write down: the registered
bar, the number, the floor, and the tag. Then find one sentence that claims more than the number
supports. Check your answer against the next day's corrections (there is nearly always one).
*Done:* a paragraph naming the over-claim and what wording would have been safe.

**R2 · Refit a ladder and watch the newest node inflate.** (1 h, hands-on)
```
export PYTHONPATH=$PYTHONPATH:src
python - <<'PY'
from agents.training.snapshot_ladder import fit_ladder
for n in (4, 8, 12, 20):
    r = fit_ladder("models/ai_v12_02_winprob_critic", first_n=n, write=False)
    print(n, r["ratings"][-1])   # newest node
PY
```
Compare to the committed `ladder.json` and to UNDERSTANDING §3.2 rule 2 and rule 24.
*Done:* you can say in one sentence why "the newest node is inflated" and why a committed file
can be at a stale recipe.

**R3 · Read two critic reads and apply the floor yourself.** (1 h)
`measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp800/vs_ctrl10M/critic_read.md`
and the same for `vf025`. For `cond.opp_class_auc.t4_10`: point, CI, floor. Decide DETECTED /
WITHIN FLOOR / NOT DETECTED before reading the tool's label.
*Done:* your labels match the tool's on all six pairs of each, and you can explain the one
`vf025` pair that clears zero but not the floor.

**R4 · Selection inflation, by hand.** (30 min)
Read rule 21 (UNDERSTANDING §7) and the λ-0.9 replicate entries (2026-09-11). Write the two
numbers (original, replicate) and explain why "half the magnitude with CIs covering zero" is the
signature of picking a row post hoc. Then find the `strata`/`strata_b` pair and say why THAT
reversal is a different phenomenon (rule 22).
*Done:* two paragraphs, one per rule, in your own words.

**R5 · Write a registration before a number exists.** (1 h, the most important one)
Pick the next queued arm (ask, or take the paired-fork arm). Write `PREDICTION.md` yourself:
the decision row, the bar, the floor and where it comes from, the branches, what will NOT be
claimed. Commit it before the arm launches.
*Done:* when the number lands, score your own registration in the ledger paragraph.

**R6 · Reproduce one anchor cell.** (1 h, hands-on)
```
python -m main.anchors --model models/ai_v12_02_winprob_critic/final_model.zip \
  --opponent metamon:SmallRL --regime greedy --teamset away --games 50 --device cpu
```
Compare your Wilson CI to `measurements/metamon_matched_regime_2026-09-14/` (0.650 [0.553,
0.736] at n=100). *Done:* you can explain why n=50 gives a wider interval and whether your
point is inside theirs — and you have read every stamped field on one `games.jsonl` row.

**R7 · Find the confound in the flywheel pair registration.** (1 h)
`designs/research_state/flywheel_era_pair_2026-09-12.md` §4 and §5. The diff table says zero
confounds. The arm S first-cycle entry (2026-09-13) says two optimisation-block differences
against the ERA. Reconcile: which claim was about S-vs-W and which about S-vs-era, and why
`model_config.json` could not have caught it.
*Done:* one paragraph; then say what a "key-by-key" verification must read to be complete.

**R8 · Profile one training step.** (2 h, hands-on, needs the GPU free)
`train_rl_agent.py --debug --steps 5000` under `torch.profiler` (or `py-spy top` on a real
launch's child PID). Where does the wall clock go: env stepping, obs build, forward, backward?
*Done:* a four-row table with percentages and the one thing you would optimise first.

---

## Track E — evaluation toy track (the ML-adjacent job skills, on small data)

**T1 · Numpy / pandas / torch in one afternoon.** (3 h)
Load a public tabular dataset (UCI Adult or similar) with pandas; one-hot with numpy; train a
logistic regression in torch (write the loop yourself: forward, BCE, backward, step). Report
accuracy AND a calibration table (10 bins: mean predicted vs observed).
*Done:* you can explain what `loss.backward()` computed and why accuracy alone hides a badly
calibrated model.

**T2 · Calibration vs discrimination, on your own classifier.** (2 h)
Compute ECE, Brier, and the Murphy decomposition (reliability / resolution / uncertainty) for
T1's model. Then temperature-scale it and show ECE falls while AUC does not move.
*Done:* one plot, one sentence: "calibration and discrimination are different properties"
— the same sentence the step curve produced for our critic.

**T3 · A model-as-judge harness.** (4 h)
Hand-label 100 short texts on a simple rubric (e.g. "is this review about shipping?"). Use a
local or API LLM as a judge on the same 100. Compute agreement (Cohen's κ), precision/recall
vs your labels, and the judge's disagreement rate when the prompt is reworded.
*Done:* a table and the rule you would give a team: when is this judge trustworthy, and what
must be re-checked by a human.

**T4 · Pre-registered A/B with a power calculation.** (2 h)
Simulate an A/B (binary outcome, true lift 2 pp). Before generating data, write the sample size
needed for 80 % power at α 0.05. Generate, analyse, report the CI. Then run it 100 times and
count how often a "significant" result appears at half the required n.
*Done:* you have SEEN selection inflation in a simulation you wrote.

**T5 · Off-policy evaluation, the smallest version.** (3 h)
A 3-armed bandit with a logging policy; estimate a new policy's value by inverse propensity
weighting and by a direct model; compare to the truth; watch IPW's variance explode as the
policies diverge.
*Done:* you can explain why PPO wants on-policy data and what V-trace-style truncation buys.

**T6 · Distillation, tiny.** (2 h)
Train a 2-layer MLP on MNIST (teacher), then a 1-layer student on the teacher's soft labels vs
on hard labels. Compare accuracy and how much of the gap closes.
*Done:* you can say what "policy-KL distillation" means and why value distillation is a
different, harder thing.

---

## Track S — systems track (fleet efficiency; the bets are in scheduling and data movement)

**S1 · A transformer forward pass by hand.** (3 h)
Implement single-head attention + MLP block in numpy for a batch of token embeddings. Count
the flops and the bytes for one layer as functions of sequence length and width.
*Done:* the formula for attention's O(n²) memory, derived from your own code.

**S2 · A KV cache.** (2 h)
Extend S1 to autoregressive decoding with and without a cache. Plot per-token latency vs
position for both. Then implement "paged" storage (fixed blocks) and show the memory
fragmentation a naive cache has.
*Done:* you can explain what vLLM's paged attention fixes.

**S3 · Mixture-of-experts routing and the hot expert.** (3 h)
Top-k router over synthetic tokens with 8 experts. Measure expert load imbalance; add a
capacity factor and count dropped tokens; add a load-balancing loss and watch imbalance fall
and (synthetic) quality move. Then skew the tokens so one expert is hot and try replication.
*Done:* a table of imbalance vs capacity factor and one paragraph on why hot experts are the
whole problem.

**S4 · Pipeline bubbles, as a scheduler simulation.** (3 h)
Simulate 4 pipeline stages × N micro-batches under naive scheduling and 1F1B. Compute the
bubble fraction. This is a discrete-event simulator — your home ground.
*Done:* the bubble-fraction formula, recovered from the simulation.

**S5 · Ring all-reduce over processes.** (3 h)
Python multiprocessing, 4 workers, implement ring all-reduce; time it against naive gather +
broadcast for growing tensor sizes. Then reason about why MoE's all-to-all is a different
beast (latency-bound, not bandwidth-bound).
*Done:* two timing curves and one paragraph.

**S6 · Read the papers after the toys.** Megatron-LM (tensor/pipeline parallelism), Switch
Transformer (MoE routing, capacity factor), the vLLM paper (paged attention), ZeRO (memory
partitioning). Each after the toy that makes it concrete.
*Done:* for each, one sentence on what your toy got wrong.

**S7 · The SRE-defendable rollout, on paper.** (2 h)
Design the canary for a routing change in an MoE serving fleet: traffic slice, the quality
monitor, the latency percentile, the rollback trigger, the A/B that separates a latency win
from a quality loss. Use the registration format from Track R.
*Done:* one page a fleet SRE would sign.

---

*Order of attack if time is scarce:* R1, R3, T1, T2, S1, S2, R5 — then the rest.
