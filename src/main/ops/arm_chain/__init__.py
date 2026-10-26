"""The ARM CHAIN: one tested module that operates a sequence of production training arms end to end.

``python -m main.ops.arm_chain <verb> …`` — the design, the state machine and the hazards are
``designs/ops/arm_chain.md``. It replaces the Training session's ad hoc job-directory scripts (``arm_end_v*.sh``,
``poll_launch_v*.sh``, ``ship_record_v*.sh``, ``ship_dose_v*.sh``, ``canary_log_v*.sh``, ``live_runs.py``,
``aw_files.sh`` and the per-window ``x26r/*.sh``), each of whose version bumps found a class bug.

Layout (each module owns one concern; every external effect goes through an injectable seam so the tests run on
temp dirs, a fake ``/proc`` and fake launchers):

* ``spec``      — the chain file: arms and GPU windows in order, EACH arm with its OWN pin (a cross-pin chain);
* ``statuslog`` — the status file: anchored, offset-bounded triggers (never a self-match, never a stale line);
* ``procwatch`` — ``/proc`` identity (pid + start ticks), waits and the identity-gated signal; never ``kill -0``;
* ``live``      — live-run discovery from launcher argv (never a run-name glob);
* ``cpugate``   — the GO's CPU gate (per-CPU idle over the trainer cores; never ``load1``);
* ``launch``    — pre-launch (env guard, CPU gate, the backend dry run re-verified: FORK, pin, card visible) and GO;
* ``postspawn`` — the TEAM-MIX-style banner VOID guard and the first-hour throughput gate;
* ``armend``    — completeness, K9(b) into a SCRATCH stage, the thin, the dose record;
* ``ship``      — record shipping from a source copy through a fresh worktree (main is never dirty);
* ``window``    — a GPU window inserted between two arms;
* ``fsguard``   — the one guarded remove (a path strictly inside a root); nothing else deletes;
* ``chain``     — the persistent state machine that drives all of the above (and the async record-ship worker);
* ``cli``       — the verbs.
"""
