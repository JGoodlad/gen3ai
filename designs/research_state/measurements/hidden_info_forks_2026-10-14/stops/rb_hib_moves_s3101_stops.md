# rb_hib_moves_s3101 — K9(b) stops
- 2026-10-14 15:18:46 (first live update, from the 15,047,835 fork point): TIE-ONLY stop — fp32: max 1.1e-05 < 0.0001,
  excluded_frac 0.165 NOT < 0.15 (judged 1710 of 2048; full buffer: 0 of 98,304 rows over 1e-4, max 2.24e-05, excluded 0.1563).
  Amendment 1 (closing test §3.7, carried over): resume. The launcher auto-restarted (crash 1). Dump: dumps/restart_err_20261014_151846_0ba62c.txt.
- Also in the same log, the UpdateFit DRY first update (fixture rows, before live training): K9(b) 'fp32: max 0.0376 NOT < 1e-4, excluded 0' —
  reported, not the stop; a bf16-trunk learner vs fixture behaviour log-probs (dump /tmp/update_fit_dump_gxc61sah/).
- 15:24 Amendment-2 switch (orchestrator f): SIGTERM, clean save at 15,259,653, resumed with --behaviour-check warn.
- 2026-10-14 15:23:57 (strict, the 2nd live update after auto-restart #1): TIE-ONLY stop — fp32: max 1.24e-05 < 0.0001, excluded_frac 0.151 NOT < 0.15. Crash #2; the launcher's restart was then superseded by the 15:24 Amendment-2 switch (resume from the crash save at 15,259,653). Dump: dumps/restart_err_20261014_152357_4befd3.txt.
