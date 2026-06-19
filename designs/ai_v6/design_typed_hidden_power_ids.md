# Typed Hidden Power — distinct move IDs migration (overnight execution plan)

> **AS-BUILT (2026-06-16, overnight run — COMPLETE, NOT shipped).** `ARCH_SIGNATURE =
> gen3_typed_hidden_power_ids_v1`. Bare `hiddenpower`=237; typed variants=355-370 (alphabetical,
> `sync.py::_HP_TYPE_NUMS`). The per-num damage-op tables + the obs move-id channel + the turn-history
> auto-carry the distinct num & real type (loops skip only 237); the boundary holds opp HP at 237 via
> `damage_tables._belief_num` (the move-belief prior) + `gen3_env._move_num` (the belief labels). The
> `gen3_own_hp_typed_history_v1` hp_probs one-hot was reverted (own-HP hp_probs stays all-zero — the
> extractor's `is_hp_slot==237` is now opp-only, so our distinct-num HP takes the normal
> `move_embedding[distinct] + type_embedding[channel]` path). Verified: full unit suite green (2740);
> obs golden regenerated + before/after diff shows ONLY our-team + history HP dims change 237→distinct
> (ZERO opponent-block changes); obs benchmark byte-identical call count (values-only); serverless smoke
> Round-trip PASSED; obs_roundtrip 3132 decisions bit-for-bit; the new `move_id_decode_fuzz_test` over
> 172k move slots with 0 mis-associations; `hidden_power_typed_obs_fuzz_test` (per-mon + history + opp
> no-leak); belief_labels / move_alignment / hidden_power_tracker / damage_op_probe(19/19) fuzz pass;
> 4-lens adversarial review ALL CLEAN (0 blockers/majors — incl. confirming the latent-belief target's
> use of the true typed HP is training-only/never-in-pi/vf, no leak). NOT committed.



**Owner decision (confirmed via AskUserQuestion): DO THE FULL DISTINCT-ID MIGRATION.**
Scheduled to start ~22:00 PDT 2026-06-16 and run through the night. Be exhaustive and robust;
token cost is not a constraint (ultracode). **Use the Workflow tool** for parallel
investigation + adversarial review on each substantive phase. **Do NOT commit/ship** (no
`/gen3ai-ship` unless the user explicitly asks) — leave the work in this worktree, fully tested
and green, with a written summary.

Worktree: `/home/goodlad/dev/gen3ai/.claude/worktrees/bridge-cse_01MgZb1AhuspSUyiCFXqf1GM`
Python: `export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3`
(submodule + dist/node_modules symlinks already set up in this worktree.)

---

## The goal & design

All 17 Hidden Power move-ids (bare `hiddenpower` + 16 typed `hiddenpowergrass`…) currently
collide on **`num = 237`** in `data/pokemon/gen3_moves.json` (copied verbatim from Showdown).
The model therefore can't distinguish our own HP types via the move embedding — it relies on a
`hp_probs` soft-type-blend workaround (see "current state" below).

**New design — known→distinct, unknown→typeless+belief:**
- **Our own HP (type ALWAYS known):** each typed variant gets its **own distinct num** (355–370,
  free below `max_moves=400`; current max real num is 354). So `hiddenpowergrass` → e.g. 358, and
  the move embedding row 358 *is* the type. Clean — no `hp_probs` blend for our side.
- **Opponent HP (type unknown — Gen 3 never reveals it):** stays the bare **`hiddenpower` → 237**,
  with the existing **probability belief** machinery (HP tracker, the `hp_probs`/soft-type blend,
  the damage-op "expand 237 into 16 typed candidates", the move-belief prior).

**Why num 355–370 is safe & cheap:** `max_moves` is hardcoded 400 in `state_encoder.get_layout()`;
real max num is 354; 237 is HP-exclusive. So 355–370 are free, *already fit the embedding table*
(no weight-shape change), and this is a **values-only obs change** (our HP move-id channel goes
237 → distinct; previously-unused embedding rows light up). Retrain-class → bump `ARCH_SIGNATURE`.

**poke-env needs NO change.** poke-env hands us the move *string* (`hiddenpowergrass` for our team
via the `raw_id` patch already in the fork; bare `hiddenpower` for the opp). Our `agents.gen3_data`
facade maps *string → our num*; nothing reads poke-env's `Move.num` for the embedding. Confirm this
during the fuzz (assert the encoded num always == `gen3_data.moves.get(id).num`).

---

## ⚠️ THE ONE SUBTLETY THAT MUST BE FUZZED HARD (the known/unknown boundary)

The opponent's revealed HP arrives **bare (`hiddenpower` → 237)**. So **everything opponent-side
must keep aggregating to 237**, even though the *data* now gives typed HP distinct nums:
- **`damage_tables.build_move_prior_logits` (the move-belief prior over OPP moves):** when summing
  Smogon usage (which is keyed by typed HP strings like `hiddenpowerice`), the typed usage **must
  still sum into num 237** (the opp's observed bare form), NOT into the new distinct nums. If it
  leaks into 355–370, the opp's HP belief at 237 goes to ~0 and the model can't predict opp HP.
  → Add an explicit "HP → 237" aggregation in the prior builder, independent of the data num.
- **The damage-op opp-incoming "expand HP into 16 typed candidates"** stays on 237 (opp unknown).
- **The HP tracker** tracks opp HP at the bare id — unchanged.
- **OUR per-move tables** (MOVE_TYPE_IDX / MOVE_BP / MOVE_ATTR / MOVE_SECONDARY / latent), by
  contrast, **SHOULD** populate the distinct nums 355–370 with the real typed values, so our
  *outgoing* HP (the op's outgoing direction, which reads our move) is priced with the right
  type/BP. (Today num 237 is left at type-idx 0 / BP 0 — so our outgoing HP is currently
  mis-priced; distinct nums FIX that. Verify with the damage_op probe/fuzz.)

So: **two num-spaces meet** — our obs/tables use 355–370 for known HP; the opp belief/tracker/prior
use 237 for unknown HP. Fuzz the boundary exhaustively (next section).

---

## Current uncommitted state in this worktree (build on it; don't restart)

Already done + tested in this worktree (the "make our HP typed to the model" work — the distinct-ID
migration supersedes the `hp_probs` half but KEEPS the rest):

| File | What | Keep / Revert |
|---|---|---|
| `agents/battle/live_view.py` | `LegalActions.own_hp_typed_id` + `display_move_ids` + `_own_hp_typed_id` | **KEEP** (labels + the history fold source) |
| `agents/training/battle_recorder.py` | action labels use `display_move_ids` (typed) | **KEEP** |
| `agents/training/turn_delta.py` | fold types `our_move_id` via `prev_ctx.legal.own_hp_typed_id` | **KEEP** — it yields the typed STRING, which now maps to a distinct num |
| `agents/observation/turn_delta_encoder.py` | `_move_features` comment only | KEEP (revisit: distinct-num HP now takes the else branch) |
| `main/prober/engine.py` | `_display_hp` + `_matchups` typing | **KEEP** (may simplify) |
| `agents/observation/pokemon.py` | (interim `hp_probs` one-hot was **ALREADY REVERTED** to HEAD before scheduling) | nothing to revert — with distinct nums our HP isn't `is_hp_slot==237`, so it uses the normal embedding path and own-HP `hp_probs` correctly stays all-zero. `pokemon_test.py::test_hp_block_own_with_hp_is_revealed_no_probs` (own HP → no probs) STAYS VALID — do not break it. |
| `agents/model/model_version.py` | `ARCH_SIGNATURE = "gen3_own_hp_typed_history_v1"` | **RENAME** → e.g. `gen3_typed_hidden_power_ids_v1` covering the whole migration |
| tests | strict_view / battle_recorder / turn_delta_event_fold / engine new tests | KEEP, update as needed |
| `agents/training/poke_env_gaps/hidden_power_typed_obs_fuzz_test.py` | typed-obs fuzz | **KEEP + EXTEND** (see fuzz) |

Docs already touched (keep current): root `CLAUDE.md`, `model/CLAUDE.md`, `observation/CLAUDE.md`,
`battle/CLAUDE.md`, `prober/CLAUDE.md` — update them again for the distinct-ID end state.

---

## Migration steps (each a Workflow phase where it helps)

1. **Data + tool.** `tools/pokemon_data_extractor/sync.py::build_moves` (line ~254): after copying
   each move, override the num for the 16 typed HP from a STABLE `_HP_TYPE_NUMS` dict
   (`hiddenpower<type>` → 355..370 in a fixed sorted order; keep bare `hiddenpower` = 237).
   Regenerate: `python tools/pokemon_data_extractor/sync.py --datasets moves`. Then
   `extractor_parity_test.py` must pass (it re-runs the builder and compares to the committed file
   — so the override must be deterministic). Verify `data/pokemon/gen3_moves.json`: 16 distinct
   nums 355–370, bare still 237, no collision with any real move num.
   - Check no OTHER builder/consumer breaks on the new nums: `learnset` (legality), the priors
     (`smogon_stats_downloader/compute_priors.py` builds `gen3_hidden_power_priors.json` — verify it
     still keys correctly), `gen3_data` facade loaders.

2. **Damage-op tables (`agents/model/damage_tables.py`).** Audit EVERY HP/237 special-case (grep
   `237` + `hiddenpower`):
   - `build_move_type_idx`, `build_move_bp`/`MOVE_BP`, `build_move_attr` (`MOVE_ATTR`),
     `build_move_secondary` (`MOVE_SECONDARY`), the latent table, drain/recoil/priority: these are
     per-num tables read for ALL moves incl. ours → POPULATE 355–370 with the typed values (type,
     70 BP, etc.). 237 stays the unknown (skipped / type-idx 0 / BP 0).
   - `build_move_prior_logits` + `build_species_cb_prior` + any usage-aggregation: **keep summing
     typed HP usage → 237** (the opp-observed form). This is the boundary subtlety. Make it explicit
     and commented.
   - Re-derive whether the op's opp-incoming HP-candidate expansion still keys on 237 (it should).

3. **Encoders / decoders ("live viewers").**
   - `observation/moves.py`: the `else` branch already emits `md.num` (distinct) for typed HP; the
     `if move_id == "hiddenpower"` (bare) override now applies ONLY to the opp's bare HP. `describe_vector`
     (the `mid == HIDDEN_POWER_MOVE_NUM` special-case) — the reverse map now gives the typed name for
     355–370 directly, so simplify: keep the 237→recover-from-type-channel path for the OPP's bare HP
     only. `HIDDEN_POWER_MOVE_NUM = 237` stays (it now means "the typeless/opponent HP").
   - `observation/turn_delta_encoder.py`: reverse-map builder (`_num_to_name`, prefers bare for 237)
     and `_move_features` bare branch → now 237 is opp-only; our typed HP folds as the typed string →
     distinct num → real type via the else branch. Verify the history encodes the distinct num.
   - `observation/state_encoder.py`: the reverse-map builder (lines ~61-62, "all HP share 237") →
     update so 355–370 map to their typed names, 237 → bare.
   - `model/features_extractor.py`: `is_hp_slot = (all_move_ids == HIDDEN_POWER_MOVE_NUM)` now only
     matches the opp's 237 HP → our typed HP flows through the normal `move_embedding[distinct] +
     type_embedding[type]` path (NO soft-type blend). VERIFY this is what happens and that the
     hp_probs feature (lines ~655) is likewise only applied to opp HP.
   - `pokemon.py` is already at HEAD (own-HP `hp_probs` all-zero) — just CONFIRM the migration keeps
     it that way (distinct-num HP bypasses the blend), and that `test_hp_block_own_*` still passes.

4. **Prober.** `engine._display_hp` / `model.py describe_vector` HP-typing — with distinct nums the
   reverse map already yields the typed name for our side, so `_display_hp` mostly handles the
   legacy-bare case. Verify the prober renders our HP typed and the opp HP bare; keep `our_hp_types`
   for legacy traces.

5. **Versioning + golden.** Bump/rename `ARCH_SIGNATURE` → `gen3_typed_hidden_power_ids_v1` with a
   full versioning note (values-only obs change, same dim 3469, no weight-shape change). **Regenerate
   the obs golden** (`training/gen3_data_obs_parity_integration_test.py` pins the vector byte-for-byte;
   `golden_obs_capture.py` regenerates it) — this is a retrain-class value change, so the golden MUST
   be updated and the diff sanity-checked (only the HP move-id channel for our team + the reverted
   hp_probs should move).

---

## FUZZ — the owner's explicit, non-negotiable requirement

> "fuzz this extensively to ensure we can always correctly decode the incoming Showdown protocol and
> never mis-associate that move."

Write a **new** bridge fuzz (e.g. `poke_env_gaps/move_id_decode_fuzz_test.py`) + extend
`hidden_power_typed_obs_fuzz_test.py`. Over MANY real battles (≥100), at every decision, assert:
1. **Round-trip integrity, EVERY move, BOTH sides:** for each move the protocol/obs carries, the
   encoded move-id num == `gen3_data.moves.get(move.id).num`, and the reverse map decodes that num
   back to a name consistent with the move (HP-prefix-tolerant). **No move is EVER mis-associated.**
2. **Our typed HP → distinct num** (355–370) + the correct type, in the per-mon block AND the
   turn-history `our_move`.
3. **Opp HP → 237 (typeless)** everywhere (per-mon block, belief, tracker) — NO LEAK of opp HP type.
4. **The boundary:** the move-belief prior mass for opp HP lives at 237 (not 355–370); our outgoing
   damage-op prices HP with the real type.
5. Re-run ALL existing bridge fuzz: `hidden_power_tracker_fuzz_test`, `move_alignment_fuzz_test`,
   `obs_roundtrip_fuzz_test` (offline==live, bit-for-bit — critical after the num change),
   `belief_labels_fuzz_test`, `damage_op_probe_fuzz_test` + `damage_op_fuzz_test`,
   `move_outcome_fuzz_test`, `reconstruction_fuzz_test`. ALL must pass.

Use coverage counters + assert >0 (no vacuous pass), like the existing fuzz.

---

## Verification gates (ALL must pass before declaring done)

- **Obs-build benchmark** before/after, no meaningful regression (`obs_build_benchmark.py`;
  judge by calls/encode, the load-stable metric — see `observation/CLAUDE.md`).
- **Full unit suite:** `pytest src/ -m "not integration and not e2e" -q`.
- **Integration:** `pytest src/ -q` (incl. `extractor_parity_test`, `gen3_data_obs_parity`).
- **Smoke (serverless):** `python src/main/train_rl_agent.py --debug --steps 6000 --use-showdown-bridge`
  → `[ModelVersion] Round-trip smoke test PASSED`, episodes finishing, `Training complete`.
- **The new + existing fuzz tests** (above), run with a high battle count.
- **Adversarial review** (Workflow: ML + SWE + arch lenses) of the diff — especially the
  prior/belief 237-boundary and the obs golden diff.

## Robustness / process

- Use **Workflow** for: (a) a parallel CONSUMER AUDIT (find every `237` / `HIDDEN_POWER_MOVE_NUM` /
  `hiddenpower` assumption across `src/` and tests), (b) the implementation fan-out, (c) an
  adversarial multi-lens review, (d) a completeness critic ("what move-decode path did we miss?").
- Update every affected test that hardcodes 237-for-all-HP.
- Keep all docs current (the always-current CLAUDE.md rule).
- **Do not commit or ship.** End with the worktree green + a written summary of what changed, the
  new num scheme, the boundary handling, and the fuzz coverage achieved.
