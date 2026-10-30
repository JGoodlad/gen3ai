# N1-P look 2: operator notes (copied into the read README at record)

## 2026-10-29 18:17 PDT: S lane PAUSED (coordinator's CPU-priority ruling)
- Why: X59-S's full Kakuna read (owner's human-data question, top priority) was starved to ~4 units/h; our S lane's
  40-min boxes held one of the 2 heavy slots. The verdict is gated by our Kakuna lane, so the pause costs it nothing.
- How: waited for the box `c0_s3505:production,x26_080m,x26_210m,x26_pstar + c0_s3505:tstar,x26_140m,x26_270m` to end
  (unit_end rc 124, the time box, t=1793323025.7); the supervisor had already started the next unit, which was still
  WAITING for a slot (no play-many child). Then `kill -TERM -- -886544` (the supervisor's own process group: it, the
  mem_cap waiter 1542003). Both gone in < 5 s. No game process was killed.
- Safe to resume (checked in the play commit's code): play-many appends ONE fsynced row per completed 100-pair batch
  and a re-run with the same request skips recorded batches (`main/h2h/play.py` "DURABLE AND RESUMABLE"); the lane
  plans from the ledger's `cell_pairs` each loop (`runner._s_pending`); an interrupted batch writes nothing. Every box
  end already is this path (rc 124).
- Ledger at the pause: c0_s3504 7 x 500 (done); c0_s3505 production/tstar/x26_080m/x26_140m 500, x26_210m 100,
  x26_270m 300, x26_pstar 0; n1p_s3504, n1p_s3505 not started. All multiples of 100.
- Anchor lane (pid 886625) left running.
- Restart: when X59-S's full supervisor (pid 3404175) exits, same play worktree (de15dfc7b), same state, `run --lane s`.

## 2026-10-30 03:19 PDT: S lane RESUMED
- X59-S's full supervisor (pid 3404175) had exited (seen at 03:18). Restarted `main.reads run --lane s --hours 48` from
  the same play worktree (de15dfc7b, clean) and state; the lane re-plans from the ledger.
