"""THE RUST COLLECTOR — the PPO trainer's rollout on the Rust env core (M5 Lane G).

Design of record: ``designs/endstate/program_rust_core.md`` §2 M5 (the Lane G row + paragraph, order
constraint 6 — the complete-game collector) and ``designs/training/rust_collector.md``; progress and
resume point: ``designs/research_state/measurements/m5_laneG/PROGRESS.md``.

    store.py        the ROW ARENA: every trainee row played and not yet trained on, per-game assembly,
                    complete-game GAE (sb3's arithmetic, bit for bit), the completed-game FIFO, and the
                    fill of the learner's buffer (COMPLETE-GAME = order constraint 6) — with each row's
                    behaviour log-prob and policy VERSION
    trigger.py      WHEN an update fires: ``SampleTrigger`` (the declared target sample count of
                    completed-game rows, on a quantum)
    teams.py        per-episode TEAMS and SEEDS staged into the core (the startup team table, per-env
                    seeded teambuilder draws, pinned opponent teams, the team win-rate tables)
    collector.py    the HOST LOOP: the env core (process or FFI front end) + T2 (trainee and policy
                    opponents in ONE flush) + Lane E's routing + the keyed draw
    consistency.py  the learner-side K9(b) BEHAVIOUR-POLICY CONSISTENCY check and the STALENESS probe
                    (ratio / clip fraction / KL by row age), before any optimizer step
    parity.py       THE GATE: slice N at the ROLLOUT level (the learner's buffer vs today's Python path)
                    and the learner-level check (one update on each buffer)

The trainer reaches it through ``agents.training.rust_vec_env.RustVecEnv`` (the Rust env core).
"""
