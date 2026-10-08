# The mon-tied `out_gain`'s identity proof (config v145, 2026-10-07)

**Question.** v145 (`gen3_mon_tied_gain_v1`) ties the damage operator's learned `out_gain` across OUR TEAM SLOTS
and their mons: the incoming per-mon rows 72 → 12 gains, the Choice-Band tail 12 → 2, the render matrices'
per-mon cells (production `damage_op.out_gain` 99 → 29). Tying a group by AVERAGING its per-position gains should
leave the model unchanged whenever those gains were already equal, and change only the gains' own update otherwise.

**Method** (`identity.py`, CPU, one thread, no GPU). v145 changes ONE function (`damage_op_layout.
out_gain_channel_keys`), so the v144 model is rebuilt in the same process by patching that function with its v144
form (`V144_KEYS`). Check (0): that build's K9 learner-golden init hash must equal the committed v144 golden's
(`b608d0cb…`) — the patch IS v144 byte for byte, or the script refuses. Then the v145 learner gets each tied gain =
the mean of the v144 gains it ties.

```bash
/path/to/python designs/research_state/measurements/mon_tied_gain_identity_2026-10-07/identity.py [--out result.json]
```

`result.json` was run on the v145 working tree over `25ea2cc6` (the field `commit` names that parent; the tree's only
model change is the key function) on v145's re-recorded K9 buffer (`66a14392…`), torch 2.8.0+cu126. A first run on
the v144 buffer (`75c5a772…`) read the same on every check, the trainable update's non-gain Δ aside (2.5e-7 there).

## Result

| check | result |
|---|---|
| (0) the patched build is the recorded v144 | init sha256 `b608d0cb…` = the committed golden's — YES |
| gain counts / learner parameters | 99 → 29 / 2,519,007 → 2,518,937; every non-gain parameter's init bytes EQUAL |
| (2) forward, per-slot gains already equal (set to their group mean), 64 K9 rows | values, log π, entropy, masked log π, masks: BITWISE |
| (3) one backward on the same weights | every non-gain gradient BITWISE; each tied gain's gradient = the sum of the per-slot gradients it ties, max abs Δ 1.5e-8 (max rel 1.1e-7: fp32 reduction order) |
| (4) one K9 `train()`, the gains frozen in both | every parameter BITWISE, every pinned loss EQUAL |
| (5) one K9 `train()`, the gains trainable (reported) | the gains differ by up to 1.45e-4 after the update; every other parameter by ≤ 4.7e-6 (`team_transformer`; the rest ≤ 5.7e-7); losses max abs Δ 3.6e-7 |
| (6) the golden's own PERTURBED per-slot gains (unequal, max dev from group mean 0.128), tied by averaging | log π max abs Δ 3.7e-3, values 1.5e-3 |
| (7) TRAINED divergence: per-team-slot spread of each incoming-row / CB-tail channel's gain, 22 archived X5 A/B arms (pre-break, 138 per-position gains read raw from each zip; READ-ONLY) | max/min across the 6 team slots: median 1.08, max 1.26 over 308 (run, channel) cells; the worst channel per run is a P(KO) or crit channel |

**Reading.** The tie is an exact reparameterisation at equal gains: the forward, the gradients of every other
parameter, and a whole K9 update with the gains frozen are bit-identical. With the gains trainable the update
differs ONLY through the gains (Adam normalises per element, so six per-slot gains with six different gradients take
six different steps where one tied gain takes one); every other parameter then drifts by ≤ 4.7e-6 over the update's
later minibatches, which see the slightly different gains — a consequence of the gains' own step, not an independent
change (with the gains frozen, (4), the update is bitwise). Trained per-slot gains did diverge — up to 26 % between team slots on one channel — and
since team-slot order is arbitrary that divergence is a learned positional bias the tie removes.

**What it does not cover.** CUDA / compiled numerics (deferred to a GPU lease with the other post-break GPU checks).
