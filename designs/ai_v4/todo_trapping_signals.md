# TODO: Route trapping signals into the model (trapped / maybe_trapped / rejected switch)

**Status:** NOT STARTED. Retrain-class (obs change → `ARCH_SIGNATURE` bump).

**Assumes:** the TurnDelta full event-fold has landed and `battle_context.py` is deleted —
i.e. `TurnDelta` folds entirely from the event log + `LiveView` (lives in
`src/agents/training/turn_delta.py`), `TurnView` is the per-turn fold surface, and there is
**no `BattleContext` snapshot-diff layer**. All wiring below targets that post-migration
world. If this is picked up before that migration, stop and sequence the fold first — these
signals should be added to the event-fold path, not the retired snapshot path.

---

## Why this matters

Trapping (Arena Trap / Mean Look / Magnet Pull) is hidden-information play, and today the
model is almost blind to it:

| Signal | In obs vector? | In mask? | Model sees it? |
|---|---|---|---|
| confirmed `trapped` (this turn) | ❌ | ✅ switch bits zeroed | only via masked logits |
| `maybe_trapped` (this turn) | ❌ | ➖ switches stay on (correct) | **not at all** |
| a switch we pressed got **rejected** (`[Unavailable choice]`) | ❌ | — | **not at all** |

`LegalActions` (`src/agents/battle/live_view.py`) already carries `trapped` and
`maybe_trapped` (server-authoritative, parsed from the request) — but only `trapped` is
consumed, and only to zero the mask's switch bits. Neither is an **observation feature**.

Worse, a **rejected switch never reaches the model in any form.** When the trap is revealed,
the server sends `|error|[Unavailable choice]`; poke-env's `Player._handle_battle_message`
intercepts `|error|` and sets `_trying_again` **without** calling `battle.parse_message`
(player.py ~329-336). So `Gen3Battle` never records it, the re-prompt produces a blank
`TurnDelta` (no switch happened, no move resolved), and the "I tried to pivot and got
trapped" event is silently lost. `attempted_switch_to` was deliberately dropped from
TurnDelta on the assumption "switches always execute" — which is exactly false in the
trap-reveal case.

Net: the model can't learn the trapping read. It can only feel `trapped` as a masked logit
*after* the fact, eats `maybe_trapped` rejections blindly, and gets no history record that a
pivot was refused. These three signals fix that.

---

## The three signals

All three are retrain-class and bundle into **one** `ARCH_SIGNATURE` bump.

### (1) Current-turn `trapped` obs bit
Surface `legal.trapped` as a single scalar in the observation. Suggested home: the reactive
block alongside the existing `forced_struggle` scalar (`src/agents/observation/reactive.py`,
currently `vec[11]`). Thread the `LegalActions` snapshot into `reactive.encode` the way
`live` (LiveView) already is — `state_encoder.encode` builds the strict view per decision and
can pass `legal` too. Redundant with the mask, but gives the policy/value nets an explicit
feature rather than forcing them to infer "can't switch" from masked logits.

### (2) Current-turn `maybe_trapped` obs bit
Same location, a second scalar from `legal.maybe_trapped`. This is the one with **zero**
signal today. It lets the model learn "switching here is risky — the opponent might be
Dugtrio/Arena Trap" and weigh the pivot, instead of attempting it blind and eating a
rejection. This is the highest-value of the three.

### (3) Turn-history `attempted_switch_rejected` bit
Make the rejected pivot a first-class, learnable history event:

- **Capture the rejection as an event.** Add `EventKind.CHOICE_REJECTED` to
  `src/agents/battle/battle_event.py` (classify it in `MESSAGE_POLICY` so the conservation
  invariant stays satisfied). Because `|error|` is intercepted *before* `parse_message`, this
  needs an explicit hook in the message-handling path (`Gen3Player` / the battle-message
  handler) that appends the event to the `Gen3Battle` log — it will not flow through the
  normal parser. Carry the attempted action / switch slot if available at that point.
- **Fold it.** Surface it on `TurnView` (the fold surface) and add
  `attempted_switch_rejected: bool` to `TurnDelta` (`turn_delta.py`). Restore
  `attempted_switch_to` at the same time — the "switches always execute" assumption that
  justified dropping it is false, and the rejected slot is exactly the signal we want.
- **Encode it.** Append a bit (and the restored attempted-switch species id, if added) to the
  TurnDelta slot in `src/agents/observation/turn_delta_encoder.py`. Append-only to the layout;
  update `TURN_DELTA_DIM`, the `OFFSET_*` constants, `describe_vector`, and the embedded-id
  manifest (`TURN_DELTA_EMBEDDED_IDS`) if a species id is added. Zero on every turn with no
  rejection, so existing battles stay consistent.

---

## Files to touch
- `src/agents/battle/live_view.py` — `LegalActions` already has the flags (no change unless
  threading helpers are added).
- `src/agents/battle/battle_event.py` — new `EventKind.CHOICE_REJECTED` + policy entry.
- `src/agents/inference/player.py` (`Gen3Player` message handling) — hook to record the
  `[Unavailable choice]` rejection as an event (the out-of-band `|error|` capture).
- `src/agents/battle/turn_view.py` — fold the rejection into the per-side `TurnView`.
- `src/agents/training/turn_delta.py` — `attempted_switch_rejected` + restored
  `attempted_switch_to` on `TurnDelta` (event-fold path).
- `src/agents/observation/reactive.py` — the two current-turn obs bits (thread `legal` in).
- `src/agents/observation/turn_delta_encoder.py` — encode the history bit(s); bump
  `TURN_DELTA_DIM` + offsets + manifest + `describe_vector`.
- `src/agents/model/model_version.py` — bump `ARCH_SIGNATURE` + version doc block.
- Root `CLAUDE.md` + `README.md` — recompute the obs-dim tables from live constants.

## Verification
- Unit suite: `pytest src/ -m "not integration and not e2e" -q` (incl. `strict_api_lock_test.py`).
- `turn_history_fuzz_test.py` — must still pass: the new history bit is zero on turns with no
  rejection, so previously-validated battles stay byte-consistent under the new (bumped) layout.
- `reward_resourcing_equivalence_fuzz_test.py` — reward unaffected (0 diffs).
- **New Arena-Trap bridge fuzz** (Dugtrio + Arena Trap vs a grounded mon that always tries to
  switch — the scenario already prototyped and confirmed at `/tmp/trap_mask_bridge_test.py`):
  assert that on the trap-reveal decision the `attempted_switch_rejected` history bit fires,
  and that the current-turn `trapped` / `maybe_trapped` obs bits are set at the right
  decisions. Promote that throwaway prototype into `src/agents/action/` (or `battle/`) as a
  permanent `*_fuzz_test.py`.
- `obs_build_benchmark.py` before/after — three scalars + one history bit must not regress
  calls/encode beyond noise.

## Notes
- **Bundle as one retrain.** All three are obs-layout changes; ship them under a single
  `ARCH_SIGNATURE` so there's one retrain boundary, not three.
- **Sequence after the TurnDelta event-fold** (this doc assumes it has landed). If combined
  with that migration in one agent, land the fold first, then add these on top so TurnDelta is
  only touched once.
- Confirmed-`trapped` (signal 1) is the least critical — the mask already enforces it. If
  scope needs trimming, signals **2 (maybe_trapped obs bit)** and **3 (rejection history)**
  are the ones that add genuinely new information.
