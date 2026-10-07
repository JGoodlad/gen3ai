# The X5 version break's weight-mapping identity proof (2026-10-07)

**Question.** The version break's part 2 (the EXACT-refactor bundle, architecture audit F1 / F6a / F7a / F16b and
the blob path's leftovers) deletes parameters and re-spells the op's maxima. The owner's bar: *the outputs are
identical except for the removed dead parameters.* Because part 2 also stops constructing modules only for their
RNG draws, the INIT bytes move, so the K9 learner golden cannot carry the proof (it is re-recorded once, at the
end of the break). The proof is a weight MAPPING instead: load the pre-break model's weights into the post-break
model, minus exactly the declared removed keys, and compare the forward and one K9 update bit for bit.

**Reference** (`capture_reference.py`, run once at `26131c0c`, the last pre-break commit; the weights are NOT
committed — 12 MB of random floats — their sha256s are in `reference_meta.json`): the K9 learner golden's
`fixed_mass` arm (= today's production surface), its seeded name-keyed-perturbed initial `policy.state_dict()`,
`evaluate_actions_functional` on its committed 64-row buffer (sha256 `4b48eaf0…`), the state after one K9
`train()`, and the pinned losses (`reference_losses.json`).

**Comparison** (`compare_mapped.py`, CPU, one thread, at the checkout it runs from): build the K9 learner; drop
EXACTLY `DECLARED_REMOVED` from the reference init (asserting the dropped set equals the declared set, nothing
missing, no shape changed) and `load_state_dict(strict=True)`; then (c) the forward on the 64 rows, `torch.equal`
on values / log_prob / entropy / masked_logp / masks_bool; (d) one K9 update seeded exactly as
`learner_golden.compute` seeds it, every SURVIVING parameter vs the reference post-state and every pinned loss,
exactly. `--readd-flat-bias` is the F16b CONTROL: it re-attaches the removed flat-pointer bias in the script
only (the reference value, in the optimizer at its pre-break position), so the remaining comparison isolates
every other removal.

```bash
/path/to/python designs/research_state/measurements/version_break_identity_2026-10-07/compare_mapped.py \
    ~/.cache/gen3ai/vb/ref [--findings F1[,F16b]] [--readd-flat-bias] [--out result.json]
```

## Result

| checkout | declared removed | forward | update (every surviving parameter) | pinned losses |
|---|---|---|---|---|
| `f001b17a` (part 1, harness check, `--baseline`: nothing removed) | — | BITWISE | BITWISE | EQUAL |
| `7340b6d9` stage A: F1 + the blob leftovers + F7a | the value tower, 10 keys, 592,129 parameters | BITWISE | BITWISE | EQUAL |
| `fd4a2de9` stage B: + F16b | + `flat_intent_head.out.bias` (1) | log_prob / masked_logp max \|Δ\| 2.4e-7, entropy 4.8e-7; values, masks EQUAL | max \|Δ\| 6.0e-8 | differ in the 8th significant digit |
| the same, `--readd-flat-bias` (CONTROL) | the tower only | BITWISE | BITWISE | EQUAL |
| `829cae1d` stage C: + F6a; and `60ddd378`, the part's final code (`result.json`) | all 11 keys, 592,130 parameters | as stage B (identical numbers) | as stage B | as stage B |
| the same, `--readd-flat-bias` (CONTROL, `result_control_readd_flat_bias.json`) | the tower only | BITWISE | BITWISE | EQUAL |

`result_stageA_f1.json` is stage A's run; `result.json` and `result_control_readd_flat_bias.json` are the part's
final commit (named inside each file as `commit`).

**What the deviation is, exactly.** F16b deletes a bias that one scorer adds to EVERY candidate of ONE softmax
(the flat opponent pointer). `softmax(s + b) = softmax(s)` exactly in real arithmetic, but not in fp32: every
logit is rounded after the shift, so the downstream log-probabilities move by ~1 ulp (2.4e-7 at |log π| ~ 1–3),
the update's gradients by rounding, and the bias itself — which receives a pure rounding-noise gradient (measured
|g| ≤ 2.3e-10 over this update, against its weight's ~1e-3) — was being moved by Adam(W) (whose `ε` turns a 1e-10
gradient into a ~3e-7 step) and shifting every later minibatch's logits again. The CONTROL removes exactly that
one parameter's effect and is bitwise, so **nothing else in the bundle moves a byte**: F1's tower (no loss read
it; it received no gradient, so it never entered the grad-norm clip or the optimizer step), item 5's deletions
(never in the state_dict, never called), F7a (the lookups were discarded) and F6a (`max_by_index`'s value is
`amax`'s exactly; its gradient differs only on an EXACT tie with a nonzero upstream gradient, and this buffer's
update has none — the control is bitwise after F6a too).

So the bar is met in the strongest form fp32 allows: bitwise for every removal but F16b, and F16b's difference is
the rounding of a mathematically exact shift invariance, isolated by a bitwise control.

**Validity.** Parts 4 / 5 of the break change the op's behaviour (its per-slot `out_gain` tied to one scalar,
the `intent_conditional` pre/post-gain), so this proof is valid at part 2's commits, not after them. The value
values (`values`) are bitwise at every stage: the critic (`sigmoid(win_head(value_pooled))`) reads nothing the
bundle changed.
