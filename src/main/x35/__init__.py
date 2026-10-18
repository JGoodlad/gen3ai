"""X35, the plateau-breakers: the STAGE KIT (``TASK_BACKLOG.md`` T55; ``designs/endstate/design_compute_efficiency.md``
§5.6). Four parts, one module each:

* :mod:`main.x35.readout` -- the population-score ESTIMATOR and the deterministic decision function of §5.6.4
  (R1, R2, the guards G-F / G-E / G-C / G-K, the verdict), self-checked by :mod:`main.x35.selfcheck` on
  constructed cases with known answers BEFORE any real read;
* :mod:`main.x35.refpop` -- the node rules (A1's archive N_k, the A-block) and the reference-population FREEZE
  (the X-block's [0.20, 0.80] band, every sha256 written before the first arm);
* :mod:`main.x35.kit` -- each arm's launch argv from the frozen call record, the two-phase A3 / A6 records, and the
  pre-check helpers ``launch_x35.sh`` calls (the script lives in the staging directory);
* :mod:`main.x35.kakuna` -- the external-anchor (G-K) unit runner on the COMMITTED home72 config.

Nothing here launches, trains or plays on import. The staging record is
``designs/research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md``.
"""
