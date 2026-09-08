# The prober's test suite — what each file pins

Owned by this tree. The command (`python3 -m pytest src/main/prober -q`) and the tiering rules stay
in `src/main/prober/CLAUDE.md`.

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
`forensics_test.py` (pure `move_category` + `build_decision_table`/`decision_table_digest` over a
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
`replay_test.py` (pure ORCHESTRATION with the bridge/model/players monkeypatched) +
`lookahead_integration_test.py` (`@integration @sim`, real bridge → materialized successor obs →
a fake model's V), `hub_contract_test.py`, `groom_test.py` (the eval-data groomer, pure
filesystem) and `belief_obs_fuzz_test.py` (run directly — real bridge battles over the full
belief stack):

and the **web** suite under `web/` (`charts_test.py` pure Vega-Lite specs · `app_test.py`
`TestClient` over a synthetic run, each endpoint compared against a direct `ProbeSession` call ·
`runs_test.py` the run-picker / no-client-string-joined-to-a-path rule · `auth_test.py` the
fail-closed password gate · `staleness_test.py` the template-pinning contract ·
`openapi_snapshot_test.py` the committed-contract drift gate · `render_integration_test.py`
`@integration`, headless chrome with the network blocked — see `web/CLAUDE.md`):

