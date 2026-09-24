# The EMISSION SELF-CHECK — zero production cost, and the milestone-scale run

<!-- A MEASUREMENT record (2026-09-23) for `gen3_core_emission_selfcheck_v1`
(designs/rust_sim/emission_selfcheck.md; program: designs/endstate/program_rust_core.md §3). -->

| # | question | verdict |
|---|---|---|
| 1 | is the check compiled OUT of the production `--release` build? | **YES** — 0 `emission_check` symbols, 0 `EMISSION SELF-CHECK` strings (the self-check build: 9 and 1) |
| 2 | does production output change? | **NO** — `sim_bridge` stdout byte-identical on the recorded 41-battle training transcript for the pre-change release build, this change's release build and the self-check build |
| 3 | does the production build pay anything? | **NO** — replay CPU B/A **0.9957×** [0.9926, 0.9997] (not slower; the CI sits just BELOW 1, a codegen-layout wobble, not a speed-up the change could cause), wall 0.9959× [0.9926, 1.0021]. The self-check build (what tests and fuzzers pay): **1.308×** [1.303, 1.313] |
| 4 | what does the whole fuzz surface find with the check ON? | **0 self-check failures** over ≈ 1.05 M omniscient emissions, ≈ 2.1 M per-viewer renders and ≈ 203 k bridge frames counted in slice E/V alone (plus every A/B fuzzer and Python fuzz script below) — **no regression, so no new pin** |

Every run: `nice -n 10`, CPU only, no port, no `models/` write, no `data/` change, the box carrying
the population-loop training arm (load1 10–24).

## 1 + 2. Compiled out, byte-identical (`byte_identity.sh` → `byte_identity.txt`)

Three builds, each in its OWN worktree target (no symlink; the baked `CARGO_MANIFEST_DIR` printed):
**A** = `origin/main` `c49ef704` (`gen3ai-wt/esc-base`, `cargo build --release`), **B** = this change,
`cargo build --release` (the production build — check compiled out), **C** = this change,
`cargo build --profile selfcheck --features emission-selfcheck` (check ON).

The transcript is the M1 record's (`../m1_transport_throughput_2026-09-23/transcript_env_seed0.txt`:
41 seeded training battles, 6,078 `CHOOSE`s). All three: stdout sha256 `4d0e37a6f80dcdd3…`,
20,216,559 bytes, 41 `__END__`, 0 `__ERR__`, 0 stderr bytes. A and B carry no `emission_check`
symbol and no marker string; C carries 9 symbols and the marker. A vs B differ in `.text` by
≈ 1 KB of codegen partitioning (a changed module set moves inlining decisions; `size` totals within
8 bytes) — no check code.

## 3. Cost (`run_pairs.py` → `replay_rows.jsonl`, `analyze.py` → `analysis.txt`)

The M1 record's REPLAY bench — ONE persistent `sim_bridge` fed the transcript, 5 replays per
sample, CPU = the reaped child's user + sys — over the three arms, pair *k* in the *k*-th of six
rotated orders. Statistic: the per-pair ratio vs A, its median and a paired bootstrap 95 % CI on
that median (20,000 resamples, fixed seed).

30 complete triples, load1 10.2–23.6 (`analysis.txt`):

| metric | B/A (production build) | C/A (self-check build) |
|---|---|---|
| replay CPU (child user + sys) | **0.9957** [0.9926, 0.9997], B slower in 10/30 — median A 182.7 ms, B 182.4 ms | **1.3078** [1.3033, 1.3131], C slower in 30/30 — C 239.3 ms |
| replay wall | 0.9959 [0.9926, 1.0021], B slower in 12/30 | 1.3055 [1.2986, 1.3082], 30/30 |

**Reading.** The production build pays nothing measurable: its CPU CI excludes any slowdown (upper
bound 0.9997). The ≈ 0.4 % on the fast side is not something the change can cause (its calls are
not compiled in). **UNVERIFIED:** it is attributed to the codegen-partitioning difference between A
and B noted in §2 — not isolated by a third build. The self-check build costs ≈ +31 % of the Rust
child's CPU — +9.3 µs per `CHOOSE` — which only tests and fuzzers pay.

## 4. The fuzz surface with the check ON

Every surface ran the self-check build (the resolver's `target/selfcheck/`, or the fuzzers' own
`--profile selfcheck --features emission-selfcheck` builds into isolated target dirs). A self-check
failure panics — it would have ended the process (`core_events`, `sim_bridge`, `search_driver`) or
been a `panic` verdict failing the green gate (`ab_replay`, `bridge_replay`); **none occurred.**

| surface | scale | result | self-check counts |
|---|---|---|---|
| `cargo test` | 850 tests (835 + 15 new) | 850 passed, 0 failed, 4 ignored | every emission of every test |
| slice E/V MILESTONE (`rust_core_parity_test.py -m slow -n 2`) | 942 battles: the whole pool (720 keys = 2 × 360 seeded-random battles), 2 × 50 `production`-policy battles, the protocol corpus × 2 + every byte-fuzz fixture (122) | 5/5 passed; slice E 0 divergences; slice V 0 divergences over 142,360 decisions (`milestone_census.txt`); verdicts recorded in `designs/ops/slow_tier_status.json` at `239ebe3e` (a re-run at the committed code, census identical) | omniscient 882,319 · per-viewer 1,771,268 · frame 170,981 |
| slice E/V on PROCEDURAL teams (`../rust_core_m2_2026-09-23/procedural_slice_v.py --battles 200 --seed 23`, `ou_random_teams.js`) | 200 battles | slice E 0 divergences; slice V: the SAME 24 view-road projection divergences the M2 record banked (`procedural_census.txt` is identical to `../rust_core_m2_2026-09-23/procedural_slice_v.txt` in its slice-V block) — the core column clean | omniscient 172,244 · per-viewer 346,932 · frame 32,058 |
| `ab_fuzz.js --protocol --mode pool --format gen3ou` | 720 battles | GREEN: 709 ok + 11 allowlisted (E1 construction speed-tie), 0 panic | (not counted) |
| `ab_fuzz.js --protocol --mode ourandom --format gen3ou` | 400 battles | GREEN: 400 ok | |
| `ab_fuzz.js --mode randbats` (state) | 300 battles | 300 ok, 0 panic | |
| `bridge_ab_fuzz.js --mode pool --format gen3ou` | 400 battles | 400 ok, 0 panic | |
| `bridge_ab_fuzz.js --mode trapping` | 200 battles | 200 ok, 0 panic | |
| `gen_sim_bridge_diff.js --mode pool --format gen3ou` | 300 battles | 300 ok (byte-identical to node's `local_sim_bridge.js`) | |
| `gen_sim_bridge_diff.js --mode randbats` | 150 battles | 150 ok | |
| Python fuzz scripts run directly (`run_py_fuzz.sh`; `_auto_selfcheck` → `target/selfcheck/`) | 10 scripts (action 60 battles, obs round-trip, event log 80, live-view memo 40, one-sided view 16, event fold 8, bridge session, search clone 6, counterfactual 4, TurnDelta fold 5 min) | 8 passed; 2 exited 1 on PRE-EXISTING oracle findings unrelated to emission (below); 0 self-check failures | |

The `|split|` check (`split_log_lines`, the replay family's `battle.log` shape) is not reached by
slice E/V (`split=0`); it runs in `cargo test` (`tests/emission_check_test.rs`, the search/replay
driver tests) and in the search-clone and counterfactual fuzz scripts, uncounted.

### Two PRE-EXISTING fuzz-script failures (not the self-check; reproduced on `origin/main`)

* **`event_log_fuzz_test.py` — a stale ORACLE.** Its raw re-derivation marks a side's turn FAILED on
  any `-fail` after that side's `|move|`; `TurnView` deliberately does NOT count a `-fail` carrying
  a real `[from]` cause (`turn_view.py`: "another effect fizzling, not this side's move failing").
  Every flagged turn carries `|-fail|pNa: Metagross|unboost|[from] ability: Clear Body|[of] …`.
  Rate: this change 5/80 and (production build) 8/240; **`origin/main` with its own production
  binary 12/240** — so it is independent of this change. Fix belongs in the script's `_rederive`
  (skip a `-fail` with a `[from]` token), not in `TurnView`.
* **`one_sided_view_parity_fuzz_test.py` — `core.successor.obs[context[k].species]` at a D10 arm
  with PE-V10 present** (protocol 0.1667, view 0.0), 2 divergences in 16 battles; **`origin/main`
  with its production binary: 1 in 16** (same class). A search-successor (D10) comparison, not an
  emission.

## Files

`byte_identity.sh` / `byte_identity.txt` · `run_pairs.py` / `replay_rows.jsonl` / `analyze.py` /
`analysis.txt` · `run_ab_fuzzers.sh` / `ab_fuzzers.txt` · `run_py_fuzz.sh` / `py_fuzz.txt` ·
`milestone_census.txt` · `procedural_census.txt`.
