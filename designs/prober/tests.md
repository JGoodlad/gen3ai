# The prober's test suite — what each file pins

Owned by this tree. The command (`python3 -m pytest src/main/prober -q`) and the tiering rules stay
in `src/main/prober/CLAUDE.md`.

**`/game`** (2026-10-08): `game_turn_events_test.py` (pure: typed events, sides from the trainee's
seat, the recoil / Leftovers sources, the board fold), `game_readout_test.py` (pure: the label →
column rule incl. OTHER_move / OTHER_species and the Hidden Power direction, calibration, seat labels,
the attention summary, the greyed move-resolution columns), `model_capture_test.py` (the recomputed
attention reproduces `BiasedEncoderLayer`'s own output; dropping the bias does not; the same for an extra pre-LN
`IdentityInitRound`, rebuilt from its captured weights, and a real `--trunk-layers 3` policy captures three rounds),
`web/game_test.py` (the degraded paths on the synthetic fixture: the story renders, a model view on an
older architecture is one sentence with the diagnosis folded, every battle surface links in; and the ACCESS
cases: a locked visitor gets the story + one unlock card, the model JSON is 403, the fragments answer the
card, a hostile `HX-Current-URL` cannot redirect the unlock, an `--open` instance is unlocked),
`web/gate_guard_test.py` (the class guard for the unlock gate, below) and
`game_integration_test.py` (`sim`: real Rust-core battles + a fresh current-architecture checkpoint —
the readout's probabilities equal the policy's, every panel present, no hook left behind, the web
views populated, and `damage_op_view` decoding on the production surface).

`engine_test.py` (pure, FakeProbeModel + offset regression, + the loss-attribution
taxonomy and the `fit_probe` stats as pure cases — decodable-vs-noise,
regression, too-few-graceful), `session_test.py` (tmp_path traces for the agent
API incl. `triage` orchestration / recoverable-ranking / no-eval-results fallback,
`probe` end-to-end with a fake feature-model — label recovery, noise→baseline,
unknown-target, CLI — the `falsify_scan` aggregation with a monkeypatched
falsifier: coverage accounting, |δ|-weighted shares, the uniform-fallback, and
concurrency-order-independence — plus the `calibration` pure helpers
(`_discounted_returns`/`_reliability_curve`/`_reliability_gap_at`/`_calibration_stats`)
and the unattributed-split integration: over-valued vs lost via the reliability gap,
+ the selection-confound diagnostics), `discovery_test.py` (tmp_path trees, checkpoint
precedence, **sharded `<outcome>_s<shard>_<idx>` parsing → distinct index**),
`core_trace_test.py` (pure: the core-trace cross-check's refusal naming the first differing line,
the live masked softmax, and `ProbeSession._meta` reading a meta-only core summary without
expanding it while `_summary` refuses its missing siblings) + `core_walk_test.py` (pure: the
`core_events` script from a reconstruction, the replay-log DISPATCH (the poke-env player's rule, the
`|init|battle` framing, one terminal line), a non-`--walk` result refused, the legal choice map, the
one-decision stall-forfeit rule) +
`core_trace_readers_test.py` (pure, `core_trace.expand` stubbed: each reader family OUTSIDE the
prober on a core trace — `mechanic_usage_baseline` counts the expanded decisions, the meta readers
(`critic_gate._trace_turns`, `quota_match`, `harvest_meter`) never expand and never report a NaN phi,
and `harvest`, `scaffolding_gauge` and the win-prob teacher REFUSE (so did `cf_audit`, deleted P6 slice 6d-1) with
`CoreTraceUnsupported`; each fails on revert) + `src/trace_summary_reader_gate_test.py` (the static
gate: no module but `core_trace.py` opens a `*_summary.json`; EMPTY allowlist; its scanner's shapes
pinned both ways) +
`core_trace_integration_test.py` (`@sim @integration`, builds the rust env cdylib, CORE-ONLY since P6 slice 6d-1: REAL core
games — two normal, one at a short `turn_limit` ending in the trainee's stall forfeit — expanded FROM THE
RUST CORE'S WALK (P5) and matched against the `states.npz` rows; the core's stream rows equal the stored rows
byte for byte and its choice maps equal the stream reading's and name exactly the stored row's legal actions;
a tampered record line and misaligned states rows REFUSED; `query summary|scan|turns` exit 0 on the core-trace
run. The poke-env ORACLE half — each game replayed LIVE through a `BattleRecorder` and the choice maps held to
the poke-env materializer — was deleted with the road it compared) +
`src/poke_env_free_entry_points_test.py::test_every_prober_command_and_the_web_app_run_with_poke_env_blocked`
(`@slow @sim @integration` — 127 s on a quiet box, so it left the routine gate 2026-10-08; run it explicitly after a prober change: every JSON-CLI command — `replay-counterfactual` included since P6, with a model
and a self-model opponent and `--narrate` — and the web app's views RUN with poke-env blocked on real core
traces + a current-arch checkpoint; the exception list `PROBER_POKE_ENV_COMMANDS` is pinned EMPTY), `forensics_test.py` (pure `move_category` + `build_decision_table`/`decision_table_digest` over a
hand-written tmp trace via a fake session — no torch, no bridge),
`falsifier_test.py` (pure: margin/percentile/paired-stats/verdict
matrix, seed determinism, δ-anchor selection incl. the forced-switch remap, and
the `anchor_deltas` δ-map) + `falsifier_integration_test.py` (`@integration`,
real bridge battle → full falsify pipeline, determinism re-run, and the run-level
`falsify_scan` over a recorded discoverable tree),
`better_line_test.py` (pure: the SEARCH backup logic — terminal sentinels, max-over-continuations,
beam pruning, principal variation) + `better_line_integration_test.py` (`@integration`, real bridge,
fake `V=obs.sum()` model: the depth-1 chosen value == sum(recorded next obs) value_crn anchor, the
depth-2 beam principal variation, determinism), `model_test.py` (the torch boundary — where each
forward stash LIVES, plus the `ArchDriftError` diagnosis and the dropped-kwarg recovery),
`awareness_test.py` (the "did it KNOW?" verdict fold over hand-built atom distributions),
`loops_test.py` (the bait-loop detector, pinned on literal Showdown protocol lines — the whole
point of the module is that it must not read the rendered timeline), `lookahead_test.py` +
`replay_test.py` (pure ORCHESTRATION of the Rust-core counterfactual — opponent resolution, regime,
per-rollout seeds, stall sides, narrate, every refusal — with the core walk and the play-out monkeypatched;
the play-out's own semantics are `src/rust_env/tests/search_playout_cf_test.rs` +
`src/utils/rust_env/cf_playout_test.py` + `successors_integration_test.py::test_the_counterfactual_keys_through_the_ffi`) +
`lookahead_integration_test.py` (`@integration @sim`, real bridge → the core's successor row →
a fake model's V), `hub_contract_test.py`, `groom_test.py` (the eval-data groomer, pure
filesystem) and `belief_obs_fuzz_test.py` (run directly — real bridge battles over the full
belief stack):

and the **web** suite under `web/` (`charts_test.py` pure Vega-Lite specs · `app_test.py`
`TestClient` over a synthetic run, each endpoint compared against a direct `ProbeSession` call ·
`runs_test.py` the run-picker / no-client-string-joined-to-a-path rule · `auth_test.py` the
fail-closed password gate · `gate_guard_test.py` every route that reaches model-loading code or starts
a job is behind the unlock gate, DERIVED from the code (AST over `session/` for the model-reaching
methods, AST over each handler, then an anonymous behavioural sweep with the model seam and the job pool
replaced by recorders) · `staleness_test.py` the template-pinning contract ·
`openapi_snapshot_test.py` the committed-contract drift gate · `render_integration_test.py`
`@integration`, headless chrome with the network blocked — see `web/CLAUDE.md`):

