# Beliefs and Threats — the GPU-first observability sections

Owned by this tree. The leaf keeps what the species-clause read IS and IS NOT; this file holds the
measurements behind it and the panel-by-panel contents of both sections.

## The species clause — the two defects, and where the clean leak reading comes from


**It hangs off the RE-COMPUTED branch only, never the summary fallback** — the summary's `belief`
block carries the top-3 per slot, whose rows do not sum to 1, so running the operator on them would
answer a different question while looking identical. That is also why `battle_turns` / `/battle`
(model-free by construction) do **not** carry it.

**Two DIFFERENT defects, kept apart** (both on the view, neither folded into the other):

| field | defect | measured on gen-15 (3000 decisions, `tmp/species_exclusivity_measure.py`) |
|---|---|---|
| `max_expected_count` / `illegal_mass` | the DISTRIBUTION is jointly illegal (E[count] > 1) | 2.5% → 3.5% of decisions; peak 1.70 |
| `duplicate_top1` | the DISPLAY is illegal — two hidden slots NAME the same mon | 6.5% → 14.2% of decisions (@2M → @6M), 29.9% at 4 hidden slots |
| `revealed_leak_max` | mass on an already-revealed species — flatly wrong under any reading | **clean**: max 3.2e-4, i.e. the `SPECIES_CLAUSE_LOGIT` floor |

The duplicate-top-1 case is the common one and the whole reason this view exists: two slots can share
an expected count of 0.74 — which no clause forbids — while the panel still reads *"both hidden mons
are Salamence"*.

⚠️ **The clean leak reading is a property of `--species-prior-fusion`, not of the publication path.**
That flag floors a revealed species at `SPECIES_CLAUSE_LOGIT` (~1e-6) inside the fused prior. Nothing
downstream masks: `belief_decode.decode_species_belief` and `engine.belief_view_from_logits` both
softmax the FULL species vocab. So on a fusion-OFF run the leak is unbounded and untested — do not
carry "the belief never leaks onto revealed mons" as a general fact.

The belief itself is **re-computed from the loaded model** each analysis (`ProbeModel.belief` → the belief
head's per-slot species logits + believed mask → `engine.belief_view_from_logits`; one clean forward, since
the intervention-sweep/saliency passes clobber the extractor's stash), so it works for **any belief-on
checkpoint** — including runs whose recorder predates the summary's per-decision `belief` block. `engine.build_belief`
reads that summary block as a **model-free fallback** (available even without a captured `.npz`). Both
`belief`, `exclusive_belief` and `belief_truth` ride the `analyze` JSON output (`asdict` —
`ExclusiveBeliefView.coherent` is a stored FIELD rather than a property for exactly that reason).
`None`/absent on a belief-off run (then
only the revealed mons show). **Mon names are
blue** (`_MON_COLOR`); **disabled slots** (a fainted mon / an illegal switch / a no-PP move) render
**grey** (`_DISABLED_GREY`), NOT the red of a low value — so "dead/unavailable" reads differently
from "alive but low HP = real danger". Hidden Power shows its **type** (`hiddenpower(fire)`) — for a
revealed HP from the obs type channel, and **for OUR OWN un-revealed HP from the reconstruction
record** (`engine.build_our_hp_types` → `_retype_hp`, threaded as `analyze_invocation(our_hp_types=…)`,
loaded by `app._load_our_hp_types` / `session._our_hp_types`): Showdown's request carries only the bare
`hiddenpower` id (the type is IV-derived), so without this our own mons showed an untyped HP until they
used it. **OUR side only** — an opponent's un-revealed HP MUST stay bare (no leak), and the retype is a
no-op on websocket/older traces with no `reconstruction.json`.

### Beliefs / Threats (GPU-first observability)

The **Beliefs** section (`b`, open by default) is the model's **world-model vs ground truth** — built to
be first-class for the learned GPU obs (every datum tagged `🔷 GPU`; `_render_beliefs` + the
`_beliefs_*_text` helpers). Six self-hiding sub-panels (each blank when its belief leg is off; a fully-off
checkpoint shows one "belief heads not enabled" note):
- **species clause — the coherent reading** (`a.exclusive_belief`) — the point team hypothesis, plus the
  hidden slots where the raw top-1 and the clause-consistent read DISAGREE. Silent (one line) when the raw
  belief was already coherent. See *The SPECIES-CLAUSE reading* above for what it is not.
- **species belief vs TRUE team** — reuses `_append_belief_truth` (privileged, Hungarian-matched, `✓/≈/✗`)
  or the anonymous `_append_belief` fallback (no `reconstruction.json`).
- **move belief** (✓ revealed · ≈ believed unseen) — the revealed opp's still-unseen moves (`a.move_belief`,
  MOVED here from the old Matchups panel — it's a belief, not a threat).
- **believed SPREAD vs true** (`a.spread_belief`, `engine.build_spread_belief` → `SpreadBeliefView`) — per
  REVEALED opp mon the DamageOperator's believed DERIVED stats `[atk,def,spa,spd,spe]` (🔷) next to the
  TRUE derived stats (📋, computed from `reconstruction.team_details()` base+IV+EV+nature via the gen3 L100
  formula `_derived_stat`, which uses the mon's REAL IV — `gen3_data.priors.gen3_stat` hardcodes IV31) + the
  Smogon usage prior. Match is by **species** (exact — a revealed mon's species is known + unique), NOT
  Hungarian. A wrong spread is an otherwise-invisible damage root-cause (the op consumes these stats), so
  this surfaces e.g. "believes Metagross Atk 385 vs true 305 → over-prices its hits". `mean_abs_err` is the
  headline. Believed-only (no `true`/prior-from-truth) on a websocket trace.
- **refinement TRAJECTORY (axis B, across-battle)** — `engine.build_belief_trajectory` (model-FREE, from the
  on-disk per-decision `belief` blocks + the privileged team): a top-1 species-confidence sparkline + `✓/✗`
  correctness dots across the battle's decisions (confidence still shown without truth; the `►` marks the
  decision being viewed). Correctness mirrors `build_belief_truth`'s precision — per decision the true HIDDEN
  set is `opp_team` minus the species revealed by then (decoded model-free from the inv board), matched with
  **one-time consumption**, so guessing an already-revealed species or two slots naming the same hidden mon
  can't double-count. When the trace npz carries the captured `move_logits` / `spread_belief` arrays (new
  runs), it ALSO draws the opp-active **move-belief entropy** (`Hmv`, should decay) + believed opp-active
  **Atk** (`bAtk`) sparklines — the move/spread analog, decoded WITHOUT re-running. The "watch the belief
  sharpen as reveals accumulate" view.
- **value-dist × belief cross-read** — does critic bimodality co-occur with low belief confidence?

The **Threats** section (`6`, was *Matchups*) is reordered **GPU-first** (`_render_matchups`, still the
method name): the `🔷` DamageOperator physics (outgoing per-move · incoming worst-hit per defender · opp
the discrete per-move incoming matrix with per-pivot safe-switch) render PRIMARY into `#threats-gpu`;
the `📋` CPU obs decodes (the per-move type-multiplier table + the `their_matchups` effectiveness + the
usage-prior `incoming P(KO)`) render below into `#matchups-table`/`#matchups-threat`, **dim** when the op is
present (it subsumes them) and **full-styled** when there's no op (the demote only applies when the op is
present — the graceful-degradation contract). The **Flow** diagram flags the learned belief/physics phases
(`BeliefSlots`/`BeliefHead`/`MoveBelief`/`SpreadBelief`/`DamageOperator`) with a `🔷 GPU-computed` callout
(`_FLOW_GPU_PHASES`). New `analyze`-JSON fields (`asdict`): `spread_belief` (+ the existing
`belief`/`belief_truth`/`move_belief`/`damage_op`); `None` when the head is off.

**Switch-in OUTGOING panel (`a.switch_in_outgoing`, `engine.build_switch_in_outgoing` → `SwitchInOutgoingView`,
rendered in `_render_matchups` right after the op's "our damage (out)").** On a FORCED SWITCH the op's outgoing
block is all-zero (it prices the fainted active only), so the model picks a switch-in from INCOMING threat alone
with no estimate of what each candidate would then DO to the opp active. This **prober-only, CPU-computed (📋)**
panel fills that view: per ALIVE bench candidate → its best BP-damaging move (self-KO Explosion/Selfdestruct
excluded) vs the opp active → `low–high %HP · →KO · ×mult · P(outspeed)`, from the **privileged true spreads**
(`reconstruction.team_details(our_side)`, threaded as `analyze_invocation(our_team_details=…)` mirroring
`opp_team_details`; `SessionBackend._our_team_details` / `app._load_our_team_details`). Reuses
`observation.incoming_damage.{gen3_damage_max,p_ko,p_outspeed,type_is_physical}` + `gen3_mechanics.effective_multiplier_by_types`.
Gated to `phase == "forced_switch"`; `None` off a forced switch or without a `reconstruction.json`. NO model change —
the model still lacks switch-in outgoing damage (the op's outgoing is active-only; the symmetric "_outgoing_matrix
transpose" is a separate, un-built arch follow-up). Pair it with the per-mon INCOMING block: "what hits me on the
way in" vs "what I'd then do".

**Capture (axis B beyond species).** `RLPlayer._move_belief_active_row` (the opp-active move posterior,
`[n_moves]`) and `_spread_belief` (the opp-active believed-spread row `[5]`) stash into the trace, and
`BattleRecorder.states_arrays` writes them as `move_logits`/`spread_belief` npz arrays — **OMITTED when the
head is off**, NaN for a captured-but-headless row (parallel to `value_dist`). `build_belief_trajectory`
READS them (the `Hmv`/`bAtk` sparklines above) so move/spread trajectories decode on future runs WITHOUT
re-running the model; absent on older traces (then species-only). The capture is opp-active-row (not the
full `[6,5]`) so the trajectory needs no separate active-index array — the per-decision spread PANEL still
re-runs the model for the full `[6,5]`-vs-truth view.

