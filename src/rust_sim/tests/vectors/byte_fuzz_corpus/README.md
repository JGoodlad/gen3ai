# byte_fuzz_corpus: the frozen omniscient-byte regression corpus

Each `*.txt` here is a **frozen A/B `--protocol` byte-fuzzer repro**: a single, self-contained real
gen3ou / gen3customgame battle in the fuzzer's chunk golden format (`SCEN` / `TEAM` / `FMT` / `INIT` / `DEC` /
`END` / `L` rows, exactly a repro dir's `battle.txt`), replayed through the emitting engine
(`run_full_battle_logged`). Each guards a specific `|...|` emission form that the omniscient-byte fuzzer once
surfaced as a divergence and the engine now emits bit-for-bit (`gen3_omniscient_byte_fuzz_v1` in
`designs/rust_sim/fuzzers.md`). **The file name says which form it guards** (for example
`01_recover_at_full_still_fail.txt`); the fixing commit and `designs/rust_sim/` carry the story.

## Two fixture classes (the known-residual allowlist gate)

- **Emission-form fixture** (the default, no header tag): MUST replay **byte-clean** (`ok`).
- **Allowlist fixture**: carries a `# ALLOWLIST <reason>` header comment and MUST replay to a `diverged` verdict
  whose `allowlisted` reason EXACTLY equals `<reason>` (the documented residuals `ab_replay`'s
  `classify_known_residual` tags). Two today: `21_*` -> `turn0-construction-speed-tie-attribution`, `27_*` ->
  `turn0-construction-speed-tie-mirror-of-flip`. The allowlist can only grow by a reviewed reason-plus-fixture
  pair, and any un-cataloged divergence is a hard failure (stripping the tag makes the gate fail).

## How the gate works

`tests/byte_fuzz_corpus_test.rs` (plain `cargo test`) auto-discovers every `*.txt` here, runs the built
`ab_replay --protocol <file>` on each, and asserts no `kind=protocol` divergence (nor `seed` / `state` / `panic` /
`parse_error`). A per-file panic names the diverging file and line. A floor assertion on the fixture count keeps the
corpus from silently shrinking.

## Adding a fixture

1. Run the byte fuzzer to accumulate clean repros, for example:
   ```
   CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz cargo build --release --bin ab_replay
   POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/release/ab_replay \
     node src/rust_sim/harness/ab_fuzz.js --mode pool --protocol \
       --format gen3customgame --battles 80 --chunk 80 --keep-chunks \
       --out /tmp/ab_fuzz_out_cg
   ```
   (the harness's own `ab_fuzz_out*` dirs are gitignored: never commit run output).
2. Pick a CLEAN single battle whose `L` rows contain the target form (`grep` them) and drop its standalone
   `battle.txt` here, named for the form it guards. The test auto-discovers it; no code change is needed.
3. Keep fixtures SMALL (short battles preferred).
