# X26 road — findings / rulings to bank at the next records ship (exit-check folder)
- 2026-10-16 01:36 — E6 scripted read OUTSIDE (fps +13.6 % vs the registered 1,631; train_s −0.2 %). ORCHESTRATOR RULING ~01:40: PASS
  on the like-for-like read — updates 10–50 (steps 1.09–5.02M) from launcher_child.full.log: rb_xs_s2701 42.00 s / 1,882 fps vs
  rb_bf_eb_s2001 42.10 / 1,865 and rb_bf_eb_s2005 42.10 / 1,841 (train −0.2 %, fps +0.9 % / +2.2 %). The registered 1,631 is EB's
  late-window figure (longer games); the readiness doc's E6 reference is to be window-matched (orchestrator's records unit).
- FINDING (mine): e6_read.py read the TRIMMED launcher_child.log (30 tables), not launcher_child.full.log.
- 2026-10-15 22:42 — rb_xs_s2701 (P_X26 71351ff5) refused at startup: UpdateWontFit (headroom 978 < 1,024 MiB); dir renamed
  rb_xs_s2701_void_updatefit; fixed by x26-memfix at 9c033489 (headroom 2,444).
- FINDING (mine): the alert watch's re-arm race (a HOLD line seen 13 min late) — fixed by alert_watch.sh (persistent per-file offsets).
- 2026-10-16 03:07 — FINDING (mine): read_w4.py used matrix_from_cells' default min_pairs (MIN_PAIRS 1,000, the closing test's)
  while §4.1 registers 500 (E7, R-T1) / 250 (R-T1b) pairs per cell ⇒ every cell refused, E7 INCONCLUSIVE, chain HOLD. Fixed per
  cross; re-read with no new games ⇒ E7 PASS (pooled −0.44 pp; s2701 −0.32, s2702 −0.56). The bugged read is kept
  (read_w4_v1_minpairs_bug.py, w4/result_v1_minpairs_bug/).
