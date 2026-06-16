# Design — Distributional Value Critic (`--value-dist-bins N`)

**Status:** design / not built. Authored 2026-06-16 (JGoodlad).
**Motivation:** interpretability — "review how the model is predicting" — *not* win-rate.
**Provenance:** produced by a multi-agent research workflow (7 parallel deep-dives over the
literature + this codebase, 6 adversarial verifiers, one synthesis) and hand-checked against
the live files. The load-bearing **K1** claim below was verified directly against
`designs/research_state/ledger.md`.

---

## 0. The honesty frame (read this first)

This project already **falsified the distributional critic as a strong-opponent / win-rate
lever**. From `designs/research_state/ledger.md`:

> **K1 | Distributional value critic lifts the strong-opp ceiling | ❌** — Strong-opp residuals
> SUB-Gaussian (tail-dom **0.33**, exkurt **−0.89**) — no tail to re-weight. The "fat tail" was
> outcome-conditioning + the PP-stall reward artifact.
> **Next probe:** `value-calib: V vs return-to-go residual shape`.

K1 kills the *tail-pricing / ceiling-lifting* motivation. It does **not** kill the
**interpretability / calibration** axis — and K1's own "next probe" column literally names it
(`value-calib: V vs return-to-go residual shape`). The falsifier/calibration code already defers
to the distributional critic as the gold-standard validator it hasn't built:

- `src/main/prober/falsifier.py:26` — *"the distributional-critic calibration probe … is a
  deliberate follow-up, not built here."*
- `src/main/prober/session.py:980` — *"The gold-standard per-crater resolution is re-roll →
  POLICY ROLLOUT to terminal → the return distribution → PIT … the true distributional-critic
  validator. This model-free reliability version is its cheap aggregate proxy."*

So this design is justified **on interpretability only**. Two claims must be kept strictly
separate (see §9):

- **Claim A — improves INTERPRETABILITY.** Validatable read-only, on a frozen checkpoint, risking
  nothing in the policy. This is the goal.
- **Claim B — improves VALUE LEARNING / win-rate.** Requires a fresh-run A/B and must re-argue
  Farebrother's *representation-auxiliary* mechanism **against K1** — not re-pitch the (dead) tail
  angle.

A second, equally load-bearing caveat: **a distributional head trained on a single Monte-Carlo
return is a smoothed point estimate, not a faithful aleatoric distribution** — its spread is the
imposed σ. For the interpretability goal to mean anything (bimodality, variance, PIT calibration),
the *target* must be genuinely distributional (a categorical λ-return projection, GMAC-style, or
QR), not just the loss. This is decisive, not a footnote.

---

## 1. Summary + recommendation

Build a **flag-gated `--value-dist-bins N` (default 0 = off = byte-identical scalar critic)**
distributional value head to make "how the model is predicting" a first-class, per-decision read
in the prober.

- **Method: HL-Gauss** (Farebrother et al. 2024, *"Stop Regressing: Training Value Functions via
  Classification for Scalable Deep RL"*) — `N` logits over a fixed support, cross-entropy against a
  Gaussian-CDF-projected target. Best-evidenced value loss (beats MSE **and** C51 two-hot),
  target-agnostic (GAE return drops in unchanged), and the head doubles as a readable histogram.
- **Support in PopArt-normalized units** `[−K, +K]`, K≈5, with **σ/ς = 0.75**. Keep ART, **drop
  POP** (§3).
- **Scalar `E[Z] = Σ zᵢ·softmaxᵢ`, denormalized**, at the three value sites → the entire
  GAE/advantage/rollout-buffer/policy-loss path stays **byte-identical** (verified end-to-end).
- **Versioning:** the head's final Linear goes `out_features 1→N`, a state_dict-shape toggle gated
  in `check_compatible`; OFF byte-identical → **no `ARCH_SIGNATURE` bump**; `MODEL_CONFIG_VERSION
  28 → 29`.
- **TUI:** the PROBER is the primary home (per-decision histogram + spread/bimodality/PIT); the
  LAUNCHER gets 2–3 aggregate health scalars via the zero-wiring generic metrics path.
- **Phased:** Phase A = an interpretability-only side readout (near-zero risk, `WinProbHead`
  precedent); Phase B = the train-driving critic (a separate, fresh-run A/B that engages the
  loss/PopArt/support machinery).

`32` is an **engineering choice for readability**, not a paper-measured optimum (Farebrother ran
101; the resolution-invariant knob is σ/ς, not N). It is coarse enough for a human/prober to read
the histogram without per-bin noise, fine enough to resolve a clear win/loss bimodality.

---

## 2. Two orthogonal axes + the phased plan

The decision has **two independent choices** that must be separated.

### Axis 1 — the loss / representation

| Method | Loss | Support | Mean | Verdict |
|---|---|---|---|---|
| **HL-Gauss** (Farebrother 2024) | CE to a Gaussian-CDF-projected target | fixed `[v_min,v_max]` (we place it in normalized space) | `Σ zᵢ·softmaxᵢ` | **Recommended.** Beats MSE *and* C51 *and* two-hot; target-agnostic; simplest drop-in; readable histogram. |
| **C51 two-hot** (MuZero limit) | CE to a hard two-bin target | fixed | same | Two-hot's knife-edge clamp *underperforms even MSE* (Farebrother Fig. 2). HL-Gauss's smoothing is exactly the win — no reason to ship the unsmoothed limit. |
| **QR / IQN** (Dabney 2018) | quantile-Huber (pinball) | **adaptive — no `v_min/v_max`** | `(1/N)Σθᵢ` | Fallback **only** if the return scale can't be bracketed. Forgoes the CE stability/scaling benefit (the reason to switch), is fiddlier (quantile crossing). |

**Call: HL-Gauss.** Target-agnostic — a MC/GAE scalar `R_t = GAE_t + V_old(s_t)` substitutes for
the Bellman backup with no projection-of-a-shifted-distribution. Lowest-friction swap of the MSE
value loss.

### Axis 2 — what the head drives

- **Side readout (like `WinProbHead`, v22):** a `ValueDistHead` reads `value_pooled`, emits `N`
  logits, is stashed (`last_value_dist_logits`) and **never concatenated into pi/vf** (projection
  dims unchanged → off-path byte-identical). Trained `read_only` (stop-grad) it is a **risk-free
  diagnostic** that can run on a frozen existing checkpoint. The scalar `value_net` is untouched.
- **Train-driving critic:** the distributional head **replaces** `value_net`; the value loss
  becomes cross-entropy. Deeper change (new loss, PopArt interaction, `vf_coef` retune,
  support-range choice) and the path that *could* (Farebrother's representation-auxiliary
  mechanism) improve value learning — but **K1 pre-falsifies the headline win-rate motivation**.

**Call: ship the side readout first (Phase A); make it train-driving only as a separate fresh-run
A/B (Phase B).** The stated motivation is interpretability; the side readout delivers every prober
read at near-zero risk and sidesteps the entire PopArt-POP conflict (the scalar `value_net` stays
a single Linear, so POP keeps working).

|  | **Phase A — interpretability-only** | **Phase B — train-driving critic** |
|---|---|---|
| What | `ValueDistHead` side readout off `value_pooled`; tri-state `none`/`read_only`/`shaping` | replace SB3 `value_net` with the N-atom head; CE value loss |
| Risk | **near-zero** (scalar critic untouched, no GAE/loss surgery, no PopArt conflict) | larger (loss, PopArt, support range, `vf_coef` retune) |
| Proves | **Claim A** (needs only a calibrated, distributional-target head) | **Claim B** (must re-argue vs K1) |
| Honesty | PIT/reliability diagnostic on captured traces | fresh-run ELO/win-rate A/B + `td_resid_tail`/EV |

> **Honesty gate (the most important framing):** do not let HL-Gauss's accuracy reputation, or
> the real interpretability value, smuggle the win-rate claim past the project's "learns ≠ helps"
> gate. K1 says no to the tail-pricing motivation specifically.

---

## 3. Support / normalization (the PopArt reconciliation)

*Governs Phase B. Phase A's side readout uses the same support math but never touches POP, since
the scalar `value_net` is untouched.*

### Where the support lives

A fixed support in **raw return units is wrong**: γ≈0.9999 → returns run to ±hundreds (the
documented reason PopArt exists — terminal ±30, Φ_mat ~±19.5), and that scale *drifts* as the
policy improves and reward terms anneal. A raw support re-introduces exactly the non-stationarity
PopArt solved.

**Put the support in PopArt-normalized units.** ART already tracks `(μ,σ)`; after normalization the
target is ~zero-mean unit-variance, so a fixed symmetric support `[−K,+K]` with **K≈5** covers ~5σ
at all times *as σ adapts* — the atoms never move, ART moves the data onto them. Atom centers
`zᵢ = −K + i·Δ`, `Δ = 2K/(N−1)`, bin half-width `ς/2 = Δ/2`.

### Exact target-projection math (HL-Gauss)

Per sample:

1. **Normalize the scalar return** (reuse `PopArtNormalizer.normalize`, `popart.py:61`):
   `ỹ = (R_t − μ)/σ`, where `R_t = GAE_t + V_old(s_t)` is the GAE return target.
2. **Project `N(ỹ, σ_g²)` onto the bins by integrating its CDF over each bin interval:**
   ```
   pᵢ = Φ((zᵢ + ς/2 − ỹ)/σ_g) − Φ((zᵢ − ς/2 − ỹ)/σ_g)
   ```
   with **`σ_g = 0.75·Δ`** (the σ/ς = 0.75 default — mass over ~6 bins). `Φ` is the standard-normal
   CDF (`½·erf` terms; Farebrother's reference impl is literally `erf((support−target)/(√2·σ_g))`).
3. **Edge-bin tail absorption** (out-of-support handling): the two edge bins take the full outer
   tail — `p₀ = Φ((z₀+ς/2−ỹ)/σ_g)`, `p_{N−1} = 1 − Φ((z_{N−1}−ς/2−ỹ)/σ_g)` — then **renormalize**
   so `Σpᵢ = 1`. HL-Gauss degrades gracefully here (smoothing → an out-of-range target reads as
   "near the edge", not a hard spike), strictly better than two-hot/C51 clamping. With K=5–6 the
   bias is negligible.
4. **Cross-entropy** between this soft label and `log_softmax(logits)`:
   `−Σ pᵢ·log_softmax(logits)ᵢ`.

### GAE mean extraction

1. **Extract `E[Z]` in normalized units:** `ṽ = Σ zᵢ·softmax(logits)ᵢ`.
2. **Denormalize for GAE** (reuse `PopArtNormalizer.denormalize`, `popart.py:65`): `v = ṽ·σ + μ`.
   GAE/advantages/bootstrapping see real-unit values exactly as today — they never touch the
   distribution.

### POP: keep ART, drop POP (the crux)

POP's affine weight-surgery (`popart.py:101-104`: `W' = (σ_old/σ_new)·W`,
`b' = (σ_old·b + μ_old − μ_new)/σ_new`) exists **solely** to make an ART scale update a no-op on
the de-normalized scalar output of a single Linear. For a **softmax-over-fixed-atoms head this
cancellation is unavailable and unnecessary:**

- The head emits logits over atoms fixed in normalized space; the atoms don't move when σ changes,
  so there is no output-preserving weight rewrite — POP's formula doesn't type-check against a
  categorical head (softmax is invariant to *adding* a constant to all logits, not to *scaling*
  them, and the atoms are fixed).
- There is **no catastrophic-corruption failure mode for POP to prevent here.** POP exists because
  naive running-std normalization moves a *scalar* regression target out from under a precisely
  tuned scalar output and the MSE explodes. A categorical head trained by cross-entropy over fixed
  atoms is inherently robust to target-distribution shift (a documented benefit of classification
  vs regression).

**Conclusion: keep ART, drop POP.** Mechanically: extract a `popart.update_stats_only(targets)`
from `popart.py:83-99` (the EMA + σ recompute) that **skips lines 102-104**; have the existing
`update` delegate to it then run POP. The PPO loop calls `update_stats_only` under the dist flag,
`update` otherwise.

**Accepted trade (document it):** without POP, a `(μ,σ)` step is no longer a no-op on the critic —
the same categorical shape de-normalizes to a slightly different raw value after the stats move.
Acceptable because ART's EMA `β=0.1` keeps `σ_old/σ_new ≈ 1` per call (the same approximation under
which POP already tolerates stale Adam momentum), so the per-call drift is tiny and the CE head
re-fits it within a few gradient steps. We explicitly choose "CE robustness to a slow target
drift" over "POP's exact output preservation."

### Alternative considered: symlog-categorical

DreamerV3's symlog two-hot critic (`[−20,+20]` in symlog space) **replaces** PopArt entirely and
makes out-of-range *impossible* by design (`symlog(±hundreds) ≈ ±6 ≪ 20`). **Rejected as primary**
because (a) it discards the ART plumbing we already have (bigger diff), and (b) symlog squashes
differences between large returns (+200 vs +250) — exactly the tail discrimination the
`value_tail_weight`/CVaR lever needs. Keep it as the documented fallback if we later decide ART's
running-stats non-stationarity is itself worth deleting.

---

## 4. Architecture integration

### Phase A — side readout (`ValueDistHead`, mirrors `WinProbHead`)

- **`features_extractor.py`:** add a `ValueDistHead` that reads `value_pooled` (the whole-board
  critic summary where `WinProbHead` also reads) and emits `[B, N]` logits. Declare
  `self.last_value_dist_logits: Optional[Tensor] = None` beside `last_win_prob_logits` (~`:2227`);
  set it in `forward_internal` beside the win-prob stash (~`:2376`). **Leak-safe by construction:**
  stashed only, **never** concatenated into `pi_combined`/`vf_combined` (`ProjectionAssembler` at
  ~`:2011` untouched → projection dims unchanged, off-path byte-identical). A `read_only` mode feeds
  it stop-grad `value_pooled` (risk-free diagnostic, zero trunk gradient, exactly like
  `WinProbHead.read_only`). The scalar `value_net` keeps flowing unchanged. **Stash owner is the
  extractor** here, so the player read is byte-for-byte the `_win_prob` precedent.

### Phase B — train-driving critic (replace `value_net`)

`value_net` is **SB3-built** as `nn.Linear(latent_dim_vf, 1)` inside `super().__init__`. Replace it
in `Gen3DualHeadMaskablePolicy.__init__` **after** `super().__init__()` — the exact place PopArt is
built:

```python
# policy.py __init__, after super().__init__():
self.value_dist_bins = value_dist_bins              # 0 = off
if value_dist_bins > 0:
    self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, value_dist_bins)  # logits, not →1
    self.register_buffer("value_atoms",
        th.linspace(v_min, v_max, value_dist_bins), persistent=False)  # fixed normalized support
self.popart = PopArtNormalizer() if use_popart else None
```

- `mlp_extractor` is **unchanged** — `value_net` reads `latent_dim_vf = 512` either way; only its
  `out_features` goes 1→N.
- The atom buffer is **`persistent=False`** (deterministic from N + range, like the `damage_tables`
  precedent) so it stays out of the state_dict.

**The three value sites** (`forward` `:67`, `evaluate_actions` `:89`, `predict_values` `:105`)
currently do `self._denorm(self.value_net(latent_vf))`. Wrap with a mean-extraction helper:

```python
def _value_mean(self, logits):                  # [B,N] (or [B,1] when off)
    if self.value_dist_bins == 0:
        return self._denorm(logits)             # unchanged scalar path
    mean = (th.softmax(logits, -1) * self.value_atoms).sum(-1, keepdim=True)  # [B,1]
    return self._denorm(mean)                   # normalized E[Z] → real units
```

Each site computes `logits = self.value_net(latent_vf)`, stashes
`self.features_extractor.last_value_dist_logits = logits if self.value_dist_bins else None`, and
returns `_value_mean(logits)`.

> **Stash-owner note:** for Phase B the logits are produced by `policy.value_net`, so the stash is
> set **in `policy.py`**, not in the extractor's `forward_internal`. The player's *read* pattern
> (`getattr(extractor, "last_value_dist_logits", None)`) is identical to `_win_prob` either way;
> only the *set* site differs between Phase A (extractor) and Phase B (policy). `get_distribution`
> (`policy.py:92`) is the fourth overridden method and **does not touch `value_net`** — correctly
> excluded from the value sites; do not add a value read there.

### What stays byte-identical (verified end-to-end, high confidence)

A verifier traced every consumer of the critic scalar against the real files: the rollout buffer
(`buffers.py:799` scalar column), GAE (`buffers.py:403-438`), advantages, terminal/last-value
bootstrap (`ppo_mask.py:259,278`), the async collector (`async_vec_env.py:193,221,229,268`),
`policy_loss` (`instrumented_ppo.py:719-729`), `explained_variance` (`:968`), and
`value_scale_metrics` (`:992`) — **all only ever receive or read the scalar mean.** The
distribution is consumed **only** by (a) the value loss and (b) trace capture.

### Exposing the distribution for capture

- **Trace capture** (`inference/player.py`): add `_value_dist()` mirroring `_win_prob()`
  (`:301-309`) — read the stash, `softmax`, return `None` when absent; add to
  `_last_prediction["value_dist"]` inside the `need_aux` block (`:284`) so eval's fast path skips
  it.
- **Recorder** (`battle_recorder.py`): add a `value_dist = np.full((T, N), np.nan, float32)` array
  parallel to `win_probs` (`:158`), filled from `s["value_dist"]`, added to the dict (`:171`);
  `np.savez_compressed` (`:513`) writes it for free. **The NaN-row sentinel is decisive** — it
  distinguishes "no head / uncaptured" from a real distribution.

---

## 5. PPO value-loss integration (concrete, OFF byte-identical)

*Phase B. Phase A adds only an aux CE/diagnostic loss on the side readout — like the win-prob aux
loss — with the three scalar value-loss branches untouched.*

### The guarded block

Add a class attr `self._value_dist: bool = False` set **post-construction** (the established
pattern, like `value_tail_weight` `:170` and `_async_rollout` `:112`) so OFF skips the new branch
→ byte-identical, and the upstream-drift hash check (`:76-97`, which hashes only the stock `train`)
is unaffected.

The current value loss (`instrumented_ppo.py:736-760`) has three SE branches (PopArt / unclipped /
clipped), all → `_value_loss_from_se` (`:607`). A **single early `if self._value_dist:` branch
bypasses all three** — branch C (clipping) is statically unreachable under dist (we require
`--clip-range-vf none`) and branch A (PopArt) becomes ART-only:

```python
# +DIST: categorical distributional critic. --clip-range-vf none required; OFF skips entirely.
if self._value_dist:
    dist_logits = self.policy.features_extractor.last_value_dist_logits   # [B,N], stashed by evaluate_actions
    tgt = popart.normalize(rollout_data.returns) if popart is not None else rollout_data.returns
    ce_per_sample = self._dist_ce_per_sample(dist_logits, tgt)            # [B] per-sample CE
    value_loss = self._value_loss_from_se(ce_per_sample)                  # CVaR blend (value_tail_weight)
elif popart is not None:
    ... unchanged scalar branch A ...
elif self.clip_range_vf is None:
    ... unchanged scalar branch B ...
else:
    ... unchanged scalar branch C ...
```

`_dist_ce_per_sample` (new pure method): build the HL-Gauss soft target via the Gaussian-CDF
projection of §3 against the fixed normalized support buffer, return per-sample
`−Σ pᵢ·log_softmax(logits)ᵢ` as `[B]`.

> **`evaluate_actions` must stash, not overload:** it keeps its `(values, log_prob, entropy)`
> signature (so `forward`/`get_distribution`/`predict_values` shapes are unchanged) and
> additionally stashes `[B,N]` logits on `features_extractor.last_value_dist_logits` — the
> established `last_*_logits` pattern (`:781,809,851`). The loss reads the stash +
> `popart.normalize(rollout_data.returns)`; both are already in the right space. The advantage/GAE
> path never touches the logits.

### PopArt-update reconciliation

Replace `popart.update(returns, value_net)` (`:681-685`) with the ART-only path under dist:

```python
if popart is not None:
    if self._value_dist:
        popart.update_stats_only(returns)   # ART: advance μ/σ; NO POP (no scalar Linear to rescale)
    else:
        popart.update(returns, self.policy.value_net)   # unchanged: ART + POP
```

Guard/repoint the `popart/value_weight_norm` diagnostic (`:1045`) under `not self._value_dist` —
`value_net.weight` is now `[N, latent]`, so the row norm is meaningless for a dist head.

### `value_tail_weight` (CVaR)

**Keep it, feed per-sample CE instead of per-sample SE — no new CVaR code.**
`_value_loss_from_se` (`:607-623`) takes per-sample non-negative errors and blends
`(1−w)·mean + w·CVaR(worst _VALUE_TAIL_FRAC)`; it does not require its input to be *squared*, only
non-negative — `topk` selects the worst-fit samples correctly. High CE = the true return landed
where the predicted distribution put little mass = exactly a value crater, so CVaR-over-CE sharpens
the tail bins where craters live (the `eval/td_resid_tail` lever this flag was built for). `w=0` →
`.mean()`, byte-identical default preserved.

### `clip_range_vf`

A categorical head has **no scalar prediction to clip**. Enforce `--clip-range-vf none` at parse
time, copying the PopArt enforcement verbatim (`train_rl_agent.py:1085-1093`). This makes branch C
statically unreachable under dist. The existing PopArt check already forces `none`, so
`--value-dist-bins --use-popart` composes.

### `vf_coef` + grad-balance

`grad_balance` (`grad_balance.py:83-178`) is **loss-form-agnostic** — it autograds
`vf_coef·value_loss` against the trunk regardless of CE vs MSE, so **no code change**. But the
gradient *scale* differs (CE over N bins vs normalized MSE), and `vf_coef` is **resume-immutable**
(FATAL to change on resume), so it must be **re-tuned on a fresh run** to land
`grad/value_share ≈ 0.5` (watch `grad/value_policy_logratio → 0`). Document a recommended starting
`--vf-coef` (likely **lower** than 0.5, since CE over a peaked target can pull hard early).
`value_scale_metrics` and `train/value_pred_std` read real-unit returns / the scalar
`rollout_buffer.values` — both stay valid.

---

## 6. Versioning + flags checklist

**Classification:** `value_dist_bins` is a **STATE_DICT-changing structural value-head toggle** —
same class as `use_popart` (v6) / `opp_belief_cls_k` (v9). The value head's final Linear goes
`out_features 1→N`, so a mismatch breaks the state_dict on **every** load (resume + frozen
eval/pool/distill opponents — `MaskablePPO.load` does a strict whole-policy `load_state_dict`, so
even an opponent that never runs the value forward fails to deserialize). **`MODEL_CONFIG_VERSION
28 → 29`.** OFF (N=0) reproduces the scalar head byte-for-byte → **NO `ARCH_SIGNATURE` bump** (stays
`gen3_wish_wired_v1`).

**`ModelVersion` fields (`model_version.py`):**

- `value_dist_bins: int = 0` — like `opp_belief_cls_k`: a plain int where **every distinct value
  (incl. 0↔N) is a weight-shape mismatch**, so one **unconditional** compare gates it, no on/off
  conditional.
- `value_dist_vmin: float = 0.0`, `value_dist_vmax: float = 0.0` — resume-immutable
  **value-meaning** params (the support is the meaning of the N atoms), like `value_tail_weight`.
- The dist loss coef (if any) → a **model attribute set after construction**, recorded for
  provenance in `_model_hparams`, **NOT a `ModelVersion` field, NOT version-locked, read back on
  flagless resume** (the `win_prob_coef` pattern). HL-Gauss replaces the MSE outright, so a separate
  coef is likely unnecessary — omit if so.

**`check_compatible` vs `check_*`:**

- `value_dist_bins` → **dedicated unconditional int compare in `check_compatible`** with a tailored
  message (NOT `_WEIGHT_FIELDS`, whose message is about generic shapes — same call as
  `use_popart`/`opp_belief_cls_k`). Exclude from `_WEIGHT_FIELDS`.
- `value_dist_vmin/vmax` → **new resume-only `check_value_dist`** (modeled on
  `check_value_tail_weight`), **excluded from `check_compatible`** (a frozen opponent never reads
  the value head), invoked via a new `enforce_value_dist=(vmin,vmax)` opt-in on
  `load_model_snapshot`, called only on the training-resume path.
- `check_opponent_compatible` — **no change** (gates only `arch_signature` + `total_dim`).
  **Known gotcha:** a dist-ON run and an arch-identical dist-OFF run from another run pass
  `check_opponent_compatible` (ARCH_SIGNATURE unchanged) but then **fail the strict
  `load_state_dict`** with a raw SB3 shape error rather than a clean `ModelVersionError`, for the
  cross-run-opponent path. Acceptable (load failure is the safety net, same as v27/v28
  runtime-discovered-dim cases) but worth noting.

**`_migrate_config` + version bump:**

```python
if version < 29:
    data.setdefault("value_dist_bins", 0)
    data.setdefault("value_dist_vmin", 0.0)
    data.setdefault("value_dist_vmax", 0.0)
    data["config_version"] = 29
```

`to_json` serializes the new dataclass fields automatically (`json.dumps(asdict(self))`).

**The 4 opponent-load sites** (thread `value_dist_bins` like `damage_op`/`win_prob_mode`, so a
dist-ON self-play run doesn't FATAL on its own sentinels):

1. `current_model_version` (`snapshot.py:505`) — add `value_dist_bins` (+vmin/vmax) to the
   signature, set on `policy_kwargs` **top-level** (beside `use_popart` `:574-579`, **not** under
   `features_extractor_kwargs` — the head lives in the policy).
2. `arch_toggles_from_model` (`snapshot.py:589`) —
   `"value_dist_bins": int(getattr(model.policy, "value_dist_bins", 0))` (read from the policy, like
   `use_popart` reads `model.policy.popart`). Do **not** include vmin/vmax here (arch_toggles
   carries only the structural toggles `check_compatible` gates).
3. `_run_arch_toggles` (`train_rl_agent.py:167`) — add `value_dist_bins=args.value_dist_bins`.
   **Verify the complete set:** this dict currently omits several v23–v25 toggles present in
   `current_model_version`/`arch_toggles_from_model` — confirm `_run_arch_toggles` is the set
   actually consumed at the build sites (`:1344`, `:1559`) before relying on it.
4. Sentinel/pool/distill load — no separate code; they route through `load_model_snapshot` →
   `check_compatible`, which now compares `value_dist_bins`.

**CLI (`train_rl_agent.py`):** `--value-dist-bins` (`type=int, default=None` — the None-default
`_resolve` inheritance pattern so a flagless resume inherits the saved N), `--value-dist-vmin` /
`--value-dist-vmax` (`type=float, default=None`). `_resolve` block (`:1051`) for all three.
Validation: `if args.value_dist_bins and args.clip_range_vf is not None: parser.error(...)`;
**decide the PopArt gate** — Phase B with a normalized-space support should **require
`--use-popart`** (the support needs ART's μ/σ), so
`if args.value_dist_bins and not args.use_popart: parser.error(...)`. Build sites: into
`policy_kwargs` at fresh-build (`:2145`) and resume-load (`:1911`). `_model_hparams` (`:192`) for
provenance. `enforce_value_dist` on the resume path only.

**`_run_roundtrip_test` (`train_rl_agent.py:331`):** thread the two new `value_dist_vmin/vmax`
kwargs into its `from_layout_and_policy_kwargs` call so the constructed-vs-saved version matches;
with N>0 in `policy_kwargs` it exercises the full structural round-trip (new `value_net` shape +
v28→v29 migration) at startup automatically.

**Tests** (mirror the per-toggle suites in `snapshot_test.py`):

- `test_check_compatible_value_dist_bins_mismatch_raises`
- `test_check_compatible_ignores_value_dist_vmin/vmax` (resume-only)
- `test_check_value_dist_match_and_mismatch`
- `test_migrate_v28_adds_value_dist_default`
- `test_check_opponent_compatible_ignores_value_dist`
- update `test_model_version_all_fields_present` field list
- a `value_dist_bins>0` round-trip through `to_json`/`from_json_file`
- policy/phase test: **off (N=0) value-head state_dict byte-identical to scalar**; on,
  `value_net.out_features == N` and the 4 value sites return scalar `[B,1]`
- `instrumented_ppo_test`: dist loss composes; N=0 byte-identical to current MSE.

---

## 7. TUI + interpretability plan

**Primary home: the PROBER** (per-decision forensic inspector; the histogram is inherently
per-decision). The LAUNCHER gets only aggregate health scalars via the zero-wiring generic metrics
path.

### Prober — per-decision distribution view

**(a) Engine compute (single source of truth, model-free).** Add a frozen `ValueDistView` beside
`WinProbView` (`engine.py:178-187`):

```python
@dataclass(frozen=True)
class ValueDistView:
    probs: tuple[float, ...]; support: tuple[float, ...]
    mean: float; std: float; p10: float; p50: float; p90: float
    entropy: float; bimodality: float
```

A pure `build_value_dist(npz, i, support)` follows **`_npz_win_prob` (`:321-331`)** — read
`npz["value_dist"]`, return `None` on `KeyError` (old trace) or `np.isnan(...).any()` (no
head/uncaptured). **Follow `_npz_win_prob`'s NaN+KeyError guards, NOT `_npz_value` (`:313`, which
has no NaN check).** Wire as an `InvocationAnalysis` field (`:282`) populated beside the win-prob
block (`:1212-1223`), returned at `:1254`. The `support` constant can be denormalized back to real
return units for display via the already-available `popart_stats()` (`model.py:273-285`), mirroring
how `ValueView` carries `popart_mu/sigma`.

**(b) App render (Textual histogram + scalars).** Two points in `app.py`, both reusing
`gradient_color` (`:132`):

- A `VALUE-DIST` summary line in the CRITIC block (`:831-849`): `mean` (cross-check vs
  `a.value.recorded`), `std`, `P10/P50/P90`, `entropy` — colored by `gradient_color` (wide std /
  high entropy → redder).
- A **32-bin histogram** as eighth-block bars (`▁▂▃▄▅▆▇█`) colored by magnitude with the `P50`/mean
  marked — the existing `_flow_bar` (`:1151`) / `_hp_bar` (`:1517`) idiom, in a new collapsible
  `_SECTIONS` entry (titles + `BINDINGS` auto-regenerate).

**(c) Query CLI parity (automatic).** `analyze` serializes the whole `InvocationAnalysis` via
`asdict` (`session.py:517-533`), so `value_dist` (mean/std/entropy/quantiles) appears in
`python -m main.prober.query analyze <battle_id> <inv>` with **zero extra wiring**, exactly like
`win_prob`/`belief`. Optionally extend `scan`/`decision_table` to rank by `dist_std`/`entropy` (a
few lines — a "most-uncertain decision" ranker).

**What a human learns (prober):** a **sharp unimodal** spike = the critic is confident; a
**wide/flat** distribution = genuine uncertainty; a **bimodal** shape = the critic sees a coinflip
(e.g. "I win if this 70%-accuracy move hits, else I lose") — *invisible* in scalar V, which
collapses both modes to one mean. At the loss-crater turns `scan`/`falsify` already flag, a **tight
predicted distribution immediately before a large negative realized return** is the fingerprint of
an epistemic (fixable) failure; a **wide/bimodal** shape before the same crater says "the critic
saw the risk; this was variance." This turns "the critic mis-valued this" into "the critic was
over-confident / bimodal here" — the actionable read.

### Launcher — aggregate live-training scalars (zero new wiring)

Record a `value/*` block in `instrumented_ppo.train()` (the `win_prob/*` precedent `:1034-1036`):

- `value/dist_entropy` (mean entropy — sharpening = committing)
- `value/dist_std` (the critic's own uncertainty scale)
- `value/dist_calibration` (a CRPS/PIT calibration number vs the realized return — the
  distributional analog of `win_prob/brier`)

`MetricsExporterCallback._on_rollout_end` (`metrics_exporter_callback.py:28-32`) forwards **every**
logger scalar, so any `value/*` key auto-routes to the launcher; app.py splits by section prefix →
a new `value/` section lands without app.py changes. Only `format.py` label (`_METRIC_LABELS`
`:50-108`) + order (`_METRIC_ORDER`) entries are needed (optional polish). Optionally a `📊 σ̄`
badge (the `🏅 ELO` / `⚗ distilled` precedent). **Skip a launcher sparkline** — `send_metrics`
forwards only scalars; per-shape detail belongs in the prober.

> **Capture-source caveat:** capture is `need_aux`-gated, so the npz only records on forensically
> captured eval battles. The launcher aggregate scalars must therefore be emitted from the
> **training loop** (over the minibatch's stashed logits), **not** derived from npz traces — a
> small `self.logger.record(...)` addition even though the launcher-side wiring is genuinely zero.

**What a human learns (launcher):** `dist_entropy` falling over training = the critic is
sharpening; an improving `dist_calibration` = the distribution tracks realized returns (the
tail-crater fix landing). The aggregate "is the distributional critic healthy over time" gauge,
complementing the prober's per-state detail.

---

## 8. Calibration-probe tie-in

The strongest *interpretability* justification, and it is **already written into the code** as the
named, deferred validator (`falsifier.py:26`, `session.py:980`, ledger K1's next-probe column
`value-calib: V vs return-to-go residual shape`).

**How a distributional V upgrades the existing `falsify-scan` aleatoric/epistemic split.** Today
`falsify-scan` produces a three-way bracket — `aleatoric` (LUCK) / `unattributed` (NEUTRAL
residual) / `policy_reducible` (MISTAKE) — but it is **model-free and mean-only**: it can only
*prove* `policy_reducible` (a paired alt beat the chosen action on mean margin), and the
aleatoric-vs-unattributed split is only **partially identifiable** from a mean-only re-roll (the
LUCK test asks where the realized outcome sat in an *analysis-time-constructed* dice distribution,
not the model's). `calibration` then splits the `unattributed` bucket into `critic_overvalued` vs
`lost_position` via a **population reliability curve** over scalar V vs realized G — which it
**self-diagnoses as selection-confounded** (loss-quota over-capture → `bias_on_losses>0` by
construction) and labels a "LOOSE UPPER BOUND."

A distributional V upgrades this **specifically**:

1. **Replaces the externally-constructed dice distribution with the critic's OWN predicted
   distribution.** The luck-vs-mistake question becomes a within-model **PIT** test: where does the
   realized return G(s) fall in the critic's predicted CDF at s? If G consistently lands in the
   predicted tail → the critic *knew* this was high-variance → **LUCK confirmed by the critic, not
   assumed by the analyst**. If the critic predicted a tight distribution and G craters →
   **confident-wrong → MISTAKE/epistemic**.
2. **De-confounds the calibration probe** — a per-state predicted CDF gives a PIT/reliability check
   that does not require the population reliability curve, sidestepping the loss-quota selection
   confound.
3. **Collapses the `unattributed` (NEUTRAL) residual** — the least-actionable bucket — into a
   *measured* aleatoric-vs-epistemic split.

**Two honesty caveats on the tie-in:**

- The gold standard the code defers to is re-roll → **policy-rollout-to-terminal** → return PIT. A
  distributional critic is only the **amortized stand-in** for that rollout (it's "as good as the
  critic") — it **upgrades the cheap model-free proxy to a model-based better proxy**, it does
  **not** "provide the validator" or replace the rollout-PIT ground truth.
- The PIT is only meaningful if the head is **calibrated**, which requires a **genuinely
  distributional target** (§0 / §9). HL-Gauss-on-a-single-MC-return makes the histogram a smoothed
  point estimate whose spread is the imposed σ — semantically empty for this purpose.

---

## 9. Risks / gotchas + the honesty gate

**The honesty gate — two SEPARATE claims:**

- **Claim A — improves INTERPRETABILITY.** Requires only that the predicted distribution is
  **calibrated** (PIT ≈ uniform). Confirmed by: (i) a PIT/reliability histogram on captured eval
  traces showing realized G is uniform under the predicted CDF; (ii) the `unattributed`/NEUTRAL
  bucket in `falsify-scan`/`calibration` shrinking because craters now get a *measured*
  aleatoric-vs-epistemic label, agreeing with the (expensive) rollout-PIT on a spot-check.
  **Validatable read-only, on a frozen checkpoint, risking nothing in the policy.**
- **Claim B — improves VALUE LEARNING / win-rate.** Requires that the categorical/distributional
  *target* shapes the trunk better than scalar MSE (Farebrother's representation-auxiliary
  mechanism — a *different, weaker, un-killed* claim than tail-pricing). Confirmed only by a
  **fresh-run A/B** (resume-immutable boundary) measuring `eval/elo` + `win_rate_vs_bots`
  non-regression-or-better **and** `td_resid_tail`/explained-variance improvement. **Standing
  counter-evidence: ledger K1 falsified the tail-pricing form** (residuals sub-Gaussian, tail-dom
  0.33 ≈ Gaussian 0.31, no fat tail; the apparent fat tail was outcome-conditioning + a γ=0.9999
  PP-stall reward artifact). Anyone proposing B must confront K1 and re-argue the representation
  angle, not the tail angle. The one un-killed quantitative hook (V under-spread, std V 10.9 vs R
  28.2) is flagged "small" and "a representation issue a distributional head doesn't fix."

**Do NOT let A's real value smuggle B past the gate.**

**Gotchas:**

1. **Semantic faithfulness:** HL-Gauss on a single MC return → smoothed point estimate, spread =
   imposed σ. For trustworthy bimodality/variance/PIT the **target** must be distributional — a
   categorical MC/λ-return projection (GMAC SR(λ) style) or QR — *same N-bin head, same E[Z]
   extraction, only target construction changes*. **Decisive for the interpretability goal.**
2. **PopArt POP conflict:** POP is invalid for a softmax-over-fixed-atoms head — keep ART, drop POP
   (§3), or make `--value-dist-bins` mutually exclusive with `--use-popart`. Wrong CLI gate
   silently breaks the support placement.
3. **`popart/value_weight_norm`** becomes meaningless on an `[N,latent]` matrix — guard/repoint.
4. **`vf_coef` re-tune** required on a fresh run (CE gradient magnitude ≠ MSE); resume-immutable.
5. **Fixed support must bracket the return scale** — out-of-support clips at the edge atom, biasing
   precisely the tail-craters of interest. Normalized-space support + K≈5–6 + HL-Gauss edge-tail
   absorption handles it; raw-unit support re-introduces the non-stationarity PopArt solves.
6. **Selection confound is intrinsic** (eval quota over-captures losses): the per-state PIT helps
   only if the validation sample is reweighted to the true win rate or the PIT is genuinely
   per-state selection-robust.
7. **`evaluate_actions` must stash, not overload** the `values` return (keep `(values, log_prob,
   entropy)` shape).
8. **Cross-run opponent path** surfaces an N-mismatch as a raw SB3 shape error, not a clean
   `ModelVersionError` (acceptable; the load failure is the safety net).
9. **`σ/ς = 0.75` is a default, not a guaranteed optimum** (Farebrother swept {0.25…2.0}) —
   re-tune if calibration looks off.
10. **`32` is an engineering choice**, not paper-measured — the real knob is `σ/ς`.

---

## 10. Effort estimate + phased rollout

**Phase A — interpretability-only side readout (recommended first).**

- Scope: `ValueDistHead` reading `value_pooled` (mirror `WinProbHead`); `read_only`/`shaping`/`none`
  tri-state; stash in the extractor; capture (`_value_dist`, recorder npz); prober `ValueDistView`
  + render + query parity; launcher aggregate scalars; versioning (the `win_prob_mode` precedent —
  a structural state_dict toggle in `check_compatible`, `MODEL_CONFIG_VERSION 28→29`, threaded
  through the 4 sites). **No PopArt-POP conflict** (scalar `value_net` untouched), **no GAE/loss
  surgery**.
- **Honesty target: Claim A only.** Train against a distributional target (so PIT is meaningful),
  validate via a PIT/reliability diagnostic. Can run `read_only` on an existing frozen checkpoint to
  prototype the prober reads before any retrain.
- Effort: **moderate** — almost entirely the well-precedented WinProbHead + capture + prober +
  launcher path; the genuinely new work is the histogram render and the distributional target/loss
  + PIT diagnostic.

**Phase B — train-driving distributional critic (separate, gated decision).**

- Scope: replace SB3 `value_net` (§4); the guarded CE value-loss branch (§5); ART-only PopArt
  (`update_stats_only`, §3); `--clip-range-vf none` + `--use-popart` gates; `vf_coef` re-tune; the
  support-in-normalized-space scheme (§3); resume-immutable `value_dist_vmin/vmax` +
  `check_value_dist`.
- **Honesty target: Claim B**, and **only** via a fresh-run A/B that re-argues the
  representation-auxiliary mechanism against K1. If K1's verdict holds, Phase B may legitimately
  show no win-rate movement — in which case it still keeps the Phase-A interpretability payoff if
  the critic-itself distribution is better-calibrated than the side readout.
- Effort: **larger** — the loss/PopArt/support/`vf_coef` machinery is the bulk; the
  capture/prober/launcher surfaces are shared with Phase A.

---

## 11. Open decisions

- **Phase scope:** ship Phase A (interpretability-only side readout) and stop, or commit to Phase B
  (train-driving critic) given K1 already killed the win-rate motivation?
- **Target semantics:** HL-Gauss on a single MC return (smoothed point estimate — cheaper, but the
  histogram's spread is the imposed σ, **PIT/bimodality semantically empty**) vs a genuinely
  distributional target (categorical MC/λ-return projection or QR — required for the
  calibration-probe tie-in to mean anything). The interpretability motivation *needs* the latter.
- **Bin count + smoothing:** `N=32` + `σ/ς=0.75` (recommended for readability) vs `51`/`101`
  (closer to Farebrother's runs). The real knob is `σ/ς`.
- **Support scheme:** PopArt-normalized fixed support `[−K,+K]` (recommended — reuses ART, preserves
  tail discrimination) vs symlog-categorical (replaces PopArt, squashes large-return tail) vs QR
  (adaptive, no support, but forgoes the CE stability/scaling benefit).
- **PopArt gate (Phase B):** `--value-dist-bins` **requires** `--use-popart` (normalized-space
  support) vs **mutually exclusive** with it (symlog/QR self-scale). Opposite CLI gates; the wrong
  one silently breaks support placement.
- **`read_only` vs `shaping`** for the Phase-A head: pure risk-free diagnostic (zero trunk
  gradient) vs letting the distributional objective shape the trunk (closer to Claim B).
- **Whether to retire POP entirely** for the project even off the dist path (the DreamerV3 thesis)
  — out of scope here, but the §3 analysis bears on it.

---

## Appendix — file/line index

Read-verified during the research pass (line numbers are anchors, may drift):

| Area | File | Anchors |
|---|---|---|
| Value sites / policy | `src/agents/model/policy.py` | forward `:67`, evaluate_actions `:89`, get_distribution `:92`, predict_values `:105`, `_denorm` `:53` |
| PopArt | `src/agents/model/popart.py` | normalize `:61`, denormalize `:65`, update `:69`, EMA `:83-99`, POP `:101-104` |
| Extractor heads | `src/agents/model/features_extractor.py` | WinProbHead / `last_win_prob_logits` `~:2227`, stash `~:2376`, ProjectionAssembler `~:2011` |
| Value loss | `src/agents/training/instrumented_ppo.py` | flag attrs `:112/:170`, PopArt update `:681-685`, value-loss branches `:736-760`, `_value_loss_from_se` `:607`, last_*_logits stashes `:781/:809/:851`, EV `:968`, value_scale `:992`, popart/value_weight_norm `:1045`, win_prob record `:1034` |
| Grad balance | `src/agents/training/grad_balance.py` | `:83-178` |
| Versioning | `src/agents/model/model_version.py`, `snapshot.py` | current_model_version `:505`, use_popart kwargs `:574-579`, arch_toggles_from_model `:589` |
| Training entry | `src/main/train_rl_agent.py` | `_run_arch_toggles` `:167`, `_model_hparams` `:192`, `_run_roundtrip_test` `:331`, `_resolve` `:1051`, PopArt/clip gate `:1085-1093`, build sites `:1344/:1559/:1911/:2145` |
| Capture | `src/agents/inference/player.py`, `src/agents/training/battle_recorder.py` | `_win_prob` `:301-309`, need_aux `:284`; recorder arrays `:158/:171`, savez `:513` |
| Prober | `src/main/prober/engine.py`, `app.py`, `session.py`, `model.py`, `falsifier.py` | WinProbView `:178-187`, `_npz_win_prob` `:321-331`, `_npz_value` `:313`, InvocationAnalysis `:282`, win-prob block `:1212-1223`, return `:1254`; app gradient_color `:132`, CRITIC block `:831-849`, bars `:1151/:1517`; session asdict `:517-533`, `_td` / gold-standard note `:980`; popart_stats `model.py:273-285`; falsifier deferral `:26` |
| Launcher | `src/main/launcher/format.py`, `src/agents/training/metrics_exporter_callback.py` | labels `:50-108`, exporter `:28-32` |
| Research state | `designs/research_state/ledger.md` | K1 row `:33` |

**References:**
- Bellemare, Dabney, Munos. *A Distributional Perspective on Reinforcement Learning* (C51), ICML 2017.
- Dabney et al. *Distributional RL with Quantile Regression* (QR-DQN), AAAI 2018; *Implicit Quantile Networks* (IQN), ICML 2018.
- Farebrother et al. *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL* (HL-Gauss), ICML 2024.
- Hafner et al. *Mastering Diverse Domains through World Models* (DreamerV3, symlog two-hot critic), 2023.
- van Hasselt et al. *Learning Values Across Many Orders of Magnitude* (PopArt), NeurIPS 2016.
