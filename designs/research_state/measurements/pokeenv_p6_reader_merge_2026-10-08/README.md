# P6 slice 4 of the poke-env retirement — ONE Rust side reader (2026-10-08)

Backlog T27, plan [`../pokeenv_and_hotpath_survey_2026-10-06/README.md`](../pokeenv_and_hotpath_survey_2026-10-06/README.md)
§A4.5 (P6), finding F-P5-8 of [`../pokeenv_p5_prober_2026-10-07/`](../pokeenv_p5_prober_2026-10-07/).

**Verdict: IDENTICAL.** `core_events --obs-stream` (P5's prober reader) folded one side's text through its own
`parse_root_with` → `parse_advance_lean` loop; P4's `pokesim::side_reader::SideReader` (behind `sim_bridge`'s core
observation mode, `live_reader` and, since slice 2, `rust_env`'s `bot_reader`) folded the same chain with the
alignment rules and a sticky failure. `--obs-stream` is now a batch front end over `SideReader::advance_fold` (the
same fold without the frame; the row is encoded only at the asked decisions). Every banked stream reads byte-identical.

## The identity read

- **Corpus** (`build_corpus.py`): 240 real core-trace battles drawn (seeded shuffle, `random.Random(20261007)`) from
  the run archive's `eval_traces/**/_reconstruction.json` (read-only), each walked by `core_events --walk`; per battle
  5 streams — each side with its recorded actions (every row encoded), each side with half its actions and 3 encoded
  decisions, and one spectator read (no team). Every `--obs-stream` stdin was banked by a tee wrapper in front of the
  OLD binary (built from `e34f7db8`).
- **Compare** (`compare.py`): old vs new binary on every banked stdin, stdout byte for byte.

| streams | decisions | rows | differences | old wall | new wall |
|---|---|---|---|---|---|
| 1,200 | 43,432 | 18,839 | **0** | 5.30 s | 5.14 s |

New stdout sha256 over the corpus: `8f92869e2ec0fbaf9a2c01d033728dee17ba84738d19060fed54e01a7f012b12`.

- **Refusals** (`refusals.py`): an illegal action index — byte-identical refusal; an unknown keyword — the same
  verdict (`ok: false`), the message now carries `SideReader`'s `core_obs: parse p1:` prefix; a stream with its
  `|request|` lines dropped — byte-identical (no decision, `ok: true`).

## The standing gate

`src/rust_sim/tests/sim_bridge_core_obs_test.rs::obs_stream_reads_the_bridge_rows_from_each_sides_text`: on the
test's 8 seeded bridge battles (rejections and a forfeit included), each side's text fed to `--obs-stream` with its
recorded choices reproduces the bridge's `__OBS__` rows byte for byte up to its first `[Invalid choice]` (an
answered-again decision notes a second token the stream is never told), and every decision's mask, tokens and turn
everywhere: 112 rows + 1,112 decisions over 16 sides. TEETH: batching at `|turn|` instead of `|request|` fails it
through `SideReader`'s `[ALIGN]` refusal — a check the OLD `--obs-stream` did not have.

## FINDINGS

- **F-P6S4-1 — a THIRD fold remains, by design of this slice.** `rust_env`'s env core (`core::pool::Env::advance`)
  and its play-out (`search::game::Game::advance`) fold one side's chunks "rule for rule" with their own copy of the
  alignment rule, on `SideStream::fold_lean`. They are held equal to the bridge's rows by gate ① and
  `tests/search_game_test.rs`, and they are training's hot path; merging them onto `SideReader` is a separate,
  performance-sensitive change, not done here.
- **F-P6S4-2 — dropping `note_choice` did not move these 112 rows.** A teeth attempt that skipped the noted token kept
  the rows equal: the tokens the readers note change the row only on a refused own switch (E4's target) and the like,
  which these battles' compared prefixes do not reach. The batching teeth is the one that bites.
