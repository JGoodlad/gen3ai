# X63-U s6302: the T2 trainee-load parity FATAL (2026-10-31 00:02, step 404,485,146): a selection TIE, not a miscompile

A frozen record of one diagnosis (CPU, card hidden, at pin `39eaa326`). It is not a registered read, and it changes
nothing in X63's PREREG.

## The event

- **When:** `rb_x63u_s6302` (AM-X63-BSPEC, PREREG §17; forked from `rb_x26_s2601` @380M) exited rc 3 at step
  404,485,146.
- **The error:** `ParityFailure: graph group=pool0 slot=24 bucket=256 rows=256 … legal_logprob (max|delta| 5.19e-03 >
  0.001)`.
- **The call:** `RustCollector.after_update` → `InferenceService.load(24, model.policy, "trainee:v…")`.
- **Slot 24 is the TRAINEE**: `T2 up …: 1 slot group(s) [('pool0', 31)], trainee slot(s) [24], buckets (8, 64, 256)`.
- **The gated weights** are therefore the trainee's own post-update weights, saved 0.2 s earlier as
  `final_model_exception.zip`. The archive is `~/gen3ai_archive/x63u_s6302_parity_fatal_2026-10-31/`.

## The gate

On every trainee update, and on every opponent load, `load` checks the slot in two ways:

1. A bit-exact copy check.
2. `parity.gate_slot`, run at every bucket the slot serves (8, 64 and 256 for the trainee), at `b` rows and at `b − 1`
   rows.
   - **What it compares:** the slot's compiled `decide` replayed in a CUDA graph, against the slot replica's eager
     sb3 path (`decision.policy_reference`).
   - **On which rows:** the committed 64-row real-observation fixture `compile_parity_obs.npz`, cycled up to the
     bucket size.
   - **The bars:** `max|Δ|` over legal log-probs < 1e-3 and over V < 1e-4, at fp32 `'highest'`
     (`compile_trainer._FP32_TOL`). They were measured on 2026-09-28 at ≥ 10× the healthy maximum of 2.7e-5.
   - **Other checks:** the mask contract (illegal entries are −inf), and greedy agreement outside a 2e-3 near-tie band.
   - **What it lacks:** rule 8's exclusion of near-tie rows. K9(b) has it (`tie_eps` 2e-4, about 3–4 % of its rows
     excluded in this run's `behaviour_violations.jsonl`).

## The cause, and its evidence (`cpu_tie_probe.py` → `cpu_tie_probe.out.txt`)

**The tie.** On fixture rows 54 and 55, the opponent's slot-5 mon (species 94; not addressable) has three revealed
moves and one move slot unknown.

- Two of its candidate moves, 202 and 280, have composed move logits that are **exactly equal**: −1.1628687.
- They sit at ranks 5 and 6 of the X5 move order (`hypothesis_tokens.build_op_roster`'s `stable_order`), so they
  straddle the E4 / E5-tail cut K = 6.
- At the last clean checkpoint, 404,000,148, they were 0.060 apart in logit (280 ahead). At 400M they were 0.149
  apart. The two logits were converging and **crossed at this update**.

**One-ulp noise reproduces the gate's number.** Multiply every weight by `1 + 2^-23·N(0,1)`, which is one fp32 rounding,
and recompute on CPU.

- In 2 of 6 seeded trials the pair swaps sides of the cut.
- Row 54's legal log-probs then move by **5.19e-03 and 5.20e-03**, the gate's figure to three digits. The largest move
  is on action 5, whose eager log-prob is −9.188 (p ≈ 1e-4).
- In the other four trials no row exceeds 1.5e-4. At 404,000,148, all six trials stay ≤ 7.6e-5.

**The jump is discrete.** A dispatch-level trace locates it at:

- `bench_tail_cells` (`tail = move_rank >= K`), which feeds the pointer head's E5 tail seats; and
- the pairwise op's first-K gather.

The τ solve (`fp32_alt`) is not the cause. Its decisions flip, but they move τ by about 1 ulp.

**The FATAL reproduces on CPU.** `replay_gate.py --device cpu --no-service` (`cpu_compiled_replay.out.txt`) compares
the same `decide` compiled by CPU Inductor against CPU eager on the failing weights:

- At buckets 64 and 256, max|Δlogp| is **5.21e-03**, on fixture row 54, action 5. It is the only over-bar row (all four
  copies at 256).
- The control, 404,000,148, reads 3.5e-05 and passes.

**Ranked:**

1. **(ii)/(iii), a rule-8 artefact (established on CPU).** Two valid fp32 programs, a compiled one and eager, round an
   exact tie at a discrete selection to different sides. This is TECH_DEBT P3, "piecewise-discontinuous selections".
   K9(b) tolerates the class; the T2 load gate does not. The served decision differs from eager on one fixture state
   by about 5e-7 in probability.
2. **(i), a CUDA-specific divergence on these weights (unlikely).** CPU cannot exclude it. The GPU replay below decides
   it by a prediction registered now: the over-bar rows are 54 and 55 only.

**A recorder gap that let row 54 through.** `build_op_roster` masks its per-mon gap to addressable mons:
`torch.where(ctx.opp_addressable, gap, inf)`. But `bench_tail_cells` reads every mon's order. So `near_tie_rows` does
not flag row 54, whose all-mon gap is 0.

## History

- Earlier ParityFailures in run logs: one, this one. The child logs of all 150 runs that used the T2 collector were
  grepped.
- In the ledger, no T2 load-gate FATAL is recorded. The nearest class is K9(b)'s tie tail.
- Gated updates in the surviving logs: about 147k since 10-20, about 325k in all.
- The per-update rate is therefore about 7e-6, with a 1-event 95 % CI of [1.7e-7, 3.8e-5].

## Replay (GPU, Training Oct's lease, ≤ 15 min)

`replay_gate.py` runs at the pin. It re-gates the failing weights through a T2 service, and directly compiles `decide`
against eager per row. Its control is checkpoint 404,000,148. Its verdict rule is in its docstring. The fast shape
(1 slot) comes first; `--slots 31 --lanes 8 --trainee-slot 24` gives the run's own shape.
