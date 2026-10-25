# bridge_corpus: the frozen PER-SIDE / `|request|` regression corpus

Each `*.txt` is a self-contained single-battle bridge-fuzzer repro (a repro dir's `battle.txt`:
`SCEN` / `TEAM` / `INIT` / `CMD` / `SEED` / `CHUNK` / `END` rows), replayed by `tests/bridge_corpus_test.rs`
through the built `bridge_replay --ab`: the per-side / `|request|` byte differential against the recorded real
`getPlayerStreams`, plus the SEED ANCHOR (each decision's post-decision engine seed equals the recorded
omniscient `seedAfter`). Each fixture guards a form a fixed divergence once broke; the file name and the
`gen3_*_v1` tag in the fixing commit and `designs/rust_sim/` say which. The `_cg` / `_ou` suffix is the format
(`gen3customgame` / `gen3ou`).

## Two fixture classes

- **Untagged**: a clean battle that MUST replay `ok` (every per-side chunk and `|request|` frame byte-identical,
  the seed anchor holds). A regression re-introduces a divergence and the fixture FAILS. A name that still says
  `allowlist` (`10_allowlist_curse_target.txt`) records the class the fixture USED to sit in; untagging is strictly
  stronger than the tag was.
- **`# ALLOWLIST <reason>` tagged**: a documented-deferral battle that MUST diverge with EXACTLY the tagged
  `allowlisted` reason (a residual the classifier catalogues, not a bug). Five today: `16_*` ->
  `turn0-construction-speed-tie-order-flip`; `17_*` and `18_*` ->
  `perside-construction-speed-tie-mirror-of-flip` (the weather-`[of]` flip, and the Intimidate block permutation);
  `21_*` and `22_*` -> `turn0-construction-speed-tie-switchin-block-swap` (the two orientations). The allowlist can
  only grow by a reviewed reason-plus-fixture pair, and any un-cataloged divergence is a hard failure.

All the tagged fixtures are judged over the WHOLE battle: `tests/bridge_window_keys_whole_battle_test.rs` mutates
one `-damage` HP after `|turn|1` on fixtures 16 / 17 / 18 and requires the gate to FAIL.
`node harness/bridge_ab_fuzz.js --selftest` injects allowlist gate-integrity breakages through the real replayer
(a content change, a dropped or extra line, an internal reorder, a non-tie lead, a later divergence) and requires
each to fail.

## Adding a fixture

Run `node harness/bridge_ab_fuzz.js --mode pool --format {gen3customgame|gen3ou}` (other modes: see its header),
take a repro's `battle.txt` (or extract a clean battle from `chunks/`) and drop it here. A clean battle stays
untagged; a documented-deferral battle gets a `# ALLOWLIST <reason>` header line. The `ab_fuzz_out*` and
`bridge_ab_fuzz_out*` run dirs are gitignored: never commit raw run output.
