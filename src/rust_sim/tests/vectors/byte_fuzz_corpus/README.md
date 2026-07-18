# byte_fuzz_corpus — the frozen omniscient-byte regression corpus

Each `*.txt` here is a **frozen A/B `--protocol` byte-fuzzer repro** — a single,
self-contained real gen3ou/gen3customgame battle in the fuzzer's chunk golden format
(`SCEN` / `TEAM` / `FMT` / `INIT` / `DEC` / `END` / `L` rows — exactly a repro dir's
`battle.txt`). Every fixture **replays byte-clean today** through the emitting engine
(`run_full_battle_logged`) and **guards a specific `|...|` emission form** that the
omniscient-byte fuzzer once surfaced as a divergence and the engine now emits
bit-for-bit (see `gen3_omniscient_byte_fuzz_v1` in `src/rust_sim/CLAUDE.md`).

The fixtures are named by the form they guard, e.g.:

| File | Emission form guarded |
|---|---|
| `01_recover_at_full_still_fail.txt` | Recover@full → `\|move\|…\|\|[still]` + `\|-fail\|…\|heal` |
| `02_toxic_into_steel_immune.txt` | status-move type-immunity → `\|-immune\|` (Toxic→Steel/Poison) |
| `03_natural_cure_switchout_curestatus.txt` | Natural Cure → `\|-curestatus\|…\|[from] ability: Natural Cure\|[silent]` |
| `04_sleep_from_move.txt` | sleep from a move → `\|-status\|…\|slp\|[from] move: <Move>` |
| `05_confusion_start.txt` | confusion onStart → `\|-start\|…\|confusion` |
| `06_beatup_activate.txt` | gen3customgame Beat Up per-strike `\|-activate\|…\|move: Beat Up\|[of]` |
| `07_protect_blocks_status_activate.txt` | Protect blocks a status move BEFORE the immune report → `\|-activate\|…\|Protect` |
| `08_shiny_details.txt` | shiny `\|switch\|` details flag (`…, shiny\|`) |
| `09_toxic_residual_from_psn.txt` | Toxic residual chip cause → `tox\|[from] psn` |
| `10_willowisp_into_fire_immune.txt` | Will-O-Wisp → Fire `\|-immune\|` |
| `11_fire_move_thaw_frz.txt` | fire-move thaw → `\|-curestatus\|…\|frz\|[msg]` |
| `12_boost_delta0_at_cap.txt` | primary self-boost at the ±6 cap → `\|-boost\|…\|<stat>\|0` |
| `13_absorb_heal_from_ability.txt` | Volt/Water Absorb heal → `\|-heal\|…\|[from] ability: <Absorb>` |
| `14_knockoff_hint_once.txt` | Knock Off `\|-hint\|` fires once per battle |
| `15_rapidspin_sideend_spikes.txt` | Rapid Spin → `\|-sideend\|…\|Spikes` |
| `16_substitute_absorb.txt` | sub absorbs a hit → `\|-activate\|…\|Substitute\|[damage]` |
| `17_still_fail_did_nothing.txt` | generic did-nothing → `\|move\|…\|\|[still]` + bare `\|-fail\|` |
| `18_freeze_clause_message_ou.txt` | gen3ou Freeze Clause → `\|-message\|Freeze Clause activated.` |
| `19_natural_cure_ou.txt` | Natural Cure `-curestatus` under the gen3ou clause-shuffle path |
| `20_protect_activate_ou.txt` | Protect-blocks-status `-activate` under gen3ou |
| `21_construction_mirror_ability_of.txt` | **ALLOWLIST fixture** — R1 turn-0 construction speed-tie `[of]` attribution (Zapdos mirror); MUST diverge with `allowlisted:turn0-construction-speed-tie-attribution` |
| `22_contact_proc_status_from_ability.txt` | contact-proc status → `\|-status\|…\|[from] ability: Effect Spore\|[of] <holder>` (Static/Poison Point/Flame Body/Effect Spore) |
| `24_endure_survive_at_one_hp.txt` | Endure survive-at-1 emits `\|-damage\|<mon>\|1/<max>` even when already at 1 HP (0-net clamp) |
| `25_natural_cure_pursuit_ko_curestatus.txt` | Natural Cure `-curestatus [silent]` on a Pursuit-KO'd switcher, BEFORE the `\|-hint\|`/`\|faint\|` |
| `26_freeze_persists_vs_hp_fire.txt` | freeze PERSISTS through a Hidden Power Fire hit (base-type Fire only thaws — the T1 state fix) |

## Two fixture classes (the KNOWN-RESIDUAL allowlist gate)

- **Emission-form fixture** (the default, no header tag) — MUST replay **byte-clean** (`ok`).
- **Allowlist fixture** — carries a `# ALLOWLIST <reason>` header comment. It MUST replay to a
  `diverged` verdict whose `allowlisted` reason EXACTLY equals `<reason>` (the documented,
  non-gen3ou-impacting residual `ab_replay`'s `classify_known_residual` tags — e.g. R1's
  `turn0-construction-speed-tie-attribution`). This makes the allowlist AUDITABLE + BOUNDED: it can
  only grow by a reviewed reason+fixture pair, and any un-cataloged divergence is a hard failure
  (proven — stripping the tag makes the gate fail).

## How the gate works

`tests/byte_fuzz_corpus_test.rs` (run by plain `cargo test`) auto-discovers every
`*.txt` here, invokes the built `ab_replay` binary in byte mode
(`ab_replay --protocol <file>`) on each, and asserts the verdict has **NO
`kind=protocol`** divergence (nor `seed`/`state`/`panic`/`parse_error`). A per-file
panic names the diverging file + line. A floor assertion (`>= 15`) keeps the corpus
from silently shrinking. This makes `cargo test` the permanent gate for the whole
corpus.

## Adding a fixture

1. Run the byte fuzzer to accumulate clean repros (both formats), e.g.:
   ```
   CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz cargo build --release --bin ab_replay
   POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/release/ab_replay \
     node src/rust_sim/harness/ab_fuzz.js --mode pool --protocol \
       --format gen3customgame --battles 80 --chunk 80 --keep-chunks \
       --out src/rust_sim/harness/ab_fuzz_out_cg
   ```
   (the `ab_fuzz_out*` dirs are gitignored — never commit run output).
2. Pick a CLEAN single battle whose `L`-rows contain the target form
   (`grep` the L-rows) and drop its standalone `battle.txt` here with a name that
   says which form it guards. The test auto-discovers it — no code change needed.
3. Keep fixtures SMALL (short battles preferred).
