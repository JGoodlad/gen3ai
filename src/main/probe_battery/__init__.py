"""The REPRESENTATION PROBE BATTERY (`gen3_probe_battery_v1`) — what a checkpoint's network represents easily, and
what it does not. Detail: ``designs/prober/probe_battery.md``.

* :mod:`bank` — a fresh, deterministic DECISION BANK: games between checkpoints on the Rust eval core, replayed
  through the core with BOTH sides' views (the true hidden state), stratified by team archetype and game phase.
* :mod:`pin_worker` — the model-side worker, run BY PATH under the checkpoint's own checkout (play, replay, capture
  representations, forward edited rows).
* :mod:`facts` — the FACTS (ground-truth labels) read off the views: per-mon state, side, field, speed order, KO-ness,
  belief, the end-of-turn race, switch-in safety, aggregates, phazing.
* :mod:`probes` — cross-validated linear probes (ridge, folds BY BATTLE, fixed seeds) with a Hewitt & Liang control
  task; :mod:`report` — the per-arm summary with seed-spread CIs and the ranked catalogue.
* :mod:`behaviour` — behavioural probes for phazing on constructed obs edits.

CPU only. Every output refuses ``models/``.
"""
