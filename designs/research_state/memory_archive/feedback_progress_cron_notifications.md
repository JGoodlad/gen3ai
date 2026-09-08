---
name: feedback_progress_cron_notifications
description: "Standing rule (owner, 2026-08-23): every long-running piece of work gets a progress watcher (cron/monitor), and the owner ALWAYS gets a push notification when it COMPLETES or is BLOCKED on his input. Applies to my agents/builds and to training-session runs (bake the requirement into relay blocks)."
metadata:
  type: feedback
---

Owner, 2026-08-23: *"I now always ask for a cron to check progress and it should always send me
a notification if it completes or is blocked and needs my input."*

**Why:** he is often away (mobile/pool) while hours-long work runs; silent completion or a
silently-blocked task wastes the window. The two events he must never discover by asking are
COMPLETE and BLOCKED-ON-ME.

**How to apply:**
- THIS session: when a background agent/build lands its final deliverable, or when I hit a
  decision only the owner can make, send a PushNotification (short, action-first). Routine
  progress does NOT notify — only completion and blocked-needs-input.
- Training-session relays: every launch order includes the standing line "arm a progress cron;
  notify on completion or when blocked on owner input" — their crons (e.g. the hourly watchers)
  already exist; the notification-on-terminal-state is the part to make standing.
- Watchers must match FAILURE states too, not just success (a silent watcher on a crashed run
  reads as "still running" — the [[feedback_waiting_on_background_work]] lesson).
