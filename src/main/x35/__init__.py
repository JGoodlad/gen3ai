"""X35, the plateau-breakers: the STAGE KIT (``TASK_BACKLOG.md`` T55; ``designs/endstate/design_compute_efficiency.md``
§5.6). One module per part:

* :mod:`main.x35.readout` -- the population-score ESTIMATOR and the deterministic decision function of §5.6.4
  (R1, R2, the guards G-F / G-E / G-C / G-K, the verdict), self-checked by :mod:`main.x35.selfcheck` on
  constructed cases with known answers BEFORE any real read;
* :mod:`main.x35.refpop` -- the node rules (A1's archive N_k, the A-block) and the reference-population FREEZE
  (the X-block's [0.20, 0.80] band, every sha256 written before the first arm);
* :mod:`main.x35.kit` -- each arm's launch argv from the frozen call record, the two-phase A3 / A6 records, and the
  pre-check helpers ``launch_x35.sh`` calls (the script lives in the staging directory);
* :mod:`main.x35.kakuna` -- the external-anchor (G-K) unit runner on the COMMITTED home72 config;
* :mod:`main.x35.d1` -- D1's (X38's) own readout (H on the unfamiliar panels, widened by the seed overdispersion
  (AM-5), and P);
* :mod:`main.x35.batch` -- batch 1's gates, the orchestrator's 2026-10-19 rulings on the review and 2026-10-20 rulings
  after the GPU battery's read: ONE pin PER SIDE (AM-1 as amended by AM-8), the production side's FP32_ALT commit (AM-8),
  one seeded pool set per side, the variance gate G1 (AM-2), the staged order (the toy block FIRST, AM-9) and its
  conditions (AM-3 / AM-4), K9(b) at each arm's end (AM-6); and batch 2's side-pinned arms (SPRT, LR-F; F-AS-8)
  with their own stage conditions; :mod:`main.x35.batch2_void` -- batch 2's registered VOID guards (SPRT V-S1 / V-S2,
  LR-F's trailing-2M KL) at arm end;
* :mod:`main.x35.ge` -- the G-E READERS (F27 closed for the readers): the fresh best responders of an arm's seed-3501
  final and C0's, their trigger (only on the arm's primary GAIN) and the ``r2`` read through ``main.best_response_gap``;
* :mod:`main.x35.x47` -- X47's readout (AM-4): s_eq at equal effective learn time, the matched-step secondary, the
  PER-STEP ONLY class, and the AM-X47 amendments (2026-10-20: ``seq`` reads only finished, K9(b)-PASS, unthinned runs
  and names the files; look 2's moved nodes; what each class does to X47b); :mod:`main.x35.x47_report` -- X47's
  REPORTED quantities (the PER-STEP ONLY gap, the per-member Δ, the new rounds' weight norms, the policy-difference
  descriptor of AM-X47b-3); :mod:`main.x35.gc` -- G-C POOLED over seed pairs of finals (AM-X47b-1, a crossed cluster
  bootstrap; each side's B2 truth, AM-X47b-2), the record every reader's ``g_c`` must be;
  :mod:`main.x35.tbrows` -- a fork's OWN TensorBoard rows (the inherited prefix excluded).

Nothing here launches, trains or plays on import. The staging record is
``designs/research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md``.
"""
