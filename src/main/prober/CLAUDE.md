# CLAUDE.md — `src/main/prober/` (forensic-replay inspector)

Reads a run's `eval_traces` + a checkpoint and explains why the policy chose what it did. Battles are
read from the Rust core; no server.

```bash
python3 -m main.prober <models_dir | run_dir>   # the web app (:6008); its own leaf: web/CLAUDE.md
python3 -m main.prober.query <cmd> ...          # the JSON CLI; `--help` has a worked sequence — start with `triage`
```

Detail lives in `designs/prober/`, always current and updated with the code:

| I am about to… | Read |
|---|---|
| change `/game`, THE battle viewer | `designs/prober/battle_view_v2.md` (field map), `battle_viewer_ux_2026-10-09.md` (why) |
| read `analyze`'s panels / JSON; label order; β names | `designs/prober/analyze_panels.md`, `analyze_output.md` |
| change what the result timeline says | `designs/prober/result_timeline.md` |
| read `win` / `loss` / `draw` on a trace | `designs/prober/result_vocabulary.md` |
| touch the belief / threat views, the species-clause reading | `designs/prober/beliefs_and_threats.md` |
| call a model-free `ProbeSession` method; `critic_currency` | `designs/prober/session_reading.md` |
| call `loops` / `triage` / `probe`; the investigation recipe | `designs/prober/session_scans.md` |
| run a counterfactual probe; the trace-quota selection | `designs/prober/counterfactual_probes.md` |
| touch `ProbeModel`, discovery, core traces, the model ladder, a module's role | `designs/prober/engine_and_model.md` |
| diagnose a checkpoint that will not load; which commands need the current arch | `designs/prober/arch_drift.md` |
| the Rust search / replay engine; `--compile` | `designs/prober/sim_impl.md` |
| groom `eval_traces/` | `designs/prober/retention.md` |
| add or find a test | `designs/prober/tests.md` |
| ask what a checkpoint represents (`main.probe_battery`) | `designs/prober/probe_battery.md` |

## The engine / surface seam

- **One engine, two surfaces.** `engine/` + `session/` + `model.py` are the analysis, pure and
  framework-agnostic; `web/` renders it for a human and `query.py` prints it for an agent. Change the
  analysis once, never in a surface. Every torch call goes through the injected `model`, so the engine
  is tested with a `FakeProbeModel`.
- The battle-log vocabulary (`engine.CANT_PHRASE`, `NO_EFFECT_TEXT`, `surprise_phrase`) lives in the
  engine, so a reason one surface learns cannot go missing on another.
- `ProbeSession` methods return JSON-serializable dicts; a `battle_id` is a `*_summary.json` path or a
  short `step_<N>/<Opponent>/<outcome>_<idx>`. The CLI prints JSON, and `{"error": …}` with exit 1 on failure.

## Hazards that have cost a wrong reading

- **A turn's events land in the NEXT decision's obs**: read `describe_turn_outcome` from decision T+1.
- `core_trace` is the one `*_summary.json` reader under `src/` (`src/trace_summary_reader_gate_test.py`). The
  stored `.p1.jsonl.gz` record is the authority: a walk that differs raises `CoreTraceMismatch` and is never
  repaired.
- A model view that cannot run is one typed `ArchDriftError`, said before trying: render its `plain`
  sentence with `exc.detail` folded under it, never "analysis failed", and branch on `kind`, never on
  the message.
- **Every model load declares its use** (`ProbeModel.load(..., use=)`, `_model_for(b, use=)`): `trace_replay` for
  stored observations, `fresh` for positions encoded now, `stored_score` for a score with no faithfulness claim.
  One gap policy (`arch_status.decide`) badges every use across an `obs_semantics` gap (a trace replay's badge
  names the change list; a marker row declaring `replay: refuse` refuses it) and refuses weights that do not fit;
  a model-running output records its `obs_semantics` block (`designs/prober/arch_drift.md`).
- Per-battle model resolution is exact → nearest → most recent (the run's last snapshot, via
  `resolve_model_ref`); the `best_model` rung is last and is NAMED in `ModelChoice.detail`.
  Faithfulness is exact only on the `exact` tier.
- `classify_result` tests the turn cap before the loss, because a timeout carries a loss's flags; an
  unknown result is refused (`UnknownTraceResult`), never rendered.
- **"— no effect" is a claim** (`engine._no_effect_supported`): only a decoded fate, a sim tag
  (`|-fail|` / `|-immune|` / `|-miss|`) or a located empty window supports it; otherwise render
  `— outcome unrecorded`.
- **Find a side in the protocol by the trace's RECORDED seat** (`engine.trainee_side_of(summary)`, passed as
  `our_side=`), never by nickname: this pool ships localized (`Staross`), mojibake (`MagnÃ©ton`) and mirrored
  mons, and the nickname reader called 389 sides that moved `absent`. A located move window decides a status,
  not the recorded `events` (`designs/prober/result_timeline.md`).
- A detector (e.g. `loops.py`) reads the raw protocol, never the rendered timeline.
- The model's belief is `a.belief`, the raw marginals; `a.exclusive_belief` is a reading aid. Render
  both.
- **Read `critic_currency` before quoting any V** (`ProbeSession.critic_mode()`): it says what V means.
- A recorded β name is the species posterior, often a mon not on the team. Read time attaches
  `engine.BELIEF_NAME_CAVEAT` to every named β slot and never re-derives the name.
- **The trace quota prefers losses: read `trace_selection` first.** A tree that records none is
  SELECTION UNKNOWN, never uniform.
- Move-action labels are already in action-index order; do not re-sort them
  (`engine_test::test_recorded_actions_are_action_index_aligned`).
- A deleted obs region resolves to offset 0, and a consumer must say **absent** (`present=False`, `None`,
  omitted), never a stored `0.0` or a flat re-run: a zero reads as a measurement ("immune", "the policy
  ignores type"). The matchup, multiplier, incoming-damage and turn-history blocks are gone from every
  current run, so `threats`, `incoming` and `active_pko` are `None` and `triage` splits deaths by the mon's
  health only (`death_*_unread`). A trace whose obs length differs gets the `obs_mismatch` banner.
- A fixture shaped like an earlier era hides this class: pin the LIVE layout (`ObsOffsets.resolve()`) and a
  win-prob run (`model_config.json: critic`), as `engine_current_run_test.py` does.

## Counterfactuals

Every surface must say both: at depth ≥ 2, `better_line`'s interior opponent is the trainee standing
in for the real one (flag it); and `replay_counterfactual` with one rollout is a single realized-dice
line, not a probability.
