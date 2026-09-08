---
name: feedback_delta_interval_before_writing
description: "Caught TWICE in two days (2026-09-01 opponent-checkpoint 'inert'; 2026-09-03 coverage 'inside rev-3's interval'): I wrote a no-effect/equivalence sentence by comparing a POINT to an INTERVAL or to a bar. RULE: before writing any 'equivalent / indistinguishable / inert / no change' sentence, compute the DELTA's own CI (cluster-bootstrapped over the real clusters — teachers, teams, runs) and require it INSIDE the stated bar; otherwise write NOT DETECTED."
metadata:
  type: feedback
---

> **Archived 2026-09-08** — MERGED into `feedback_equivalence_needs_delta_ci` (same rule, same two catches). Read it there; this copy is the record. Preserved verbatim; nothing below is current.

Both catches came from the training session before the sentence travelled. The second was on a
cross-generation comparison where the CLUSTERS are teachers (n=3 / 6 / 20), so per-cell precision
cannot narrow it — a structurally underpowered comparison must be written as NOT DETECTED, never
as equivalent.

**How to apply (checklist before the sentence leaves my hands):**
1. Name the bar (floor, pre-registered tolerance).
2. Compute the DELTA's CI, clustered over the unit that actually varies (runs, teachers, teams).
3. CI inside the bar ⇒ "equivalent within ±X". CI straddling the bar ⇒ "NOT DETECTED at this n".
   Never "inside the other arm's interval", never "point is small".
4. State n per side; if a side has < ~8 clusters, say the comparison is structurally underpowered.
Related: [[feedback_matched_noise_control]], [[project_untaught_meter_axes]].

**Vocabulary fixed 2026-09-04 (ledger 42ed1b6f):** a delta whose CI excludes zero but whose size is
under the replicate floor is **WITHIN FLOOR** (the games are consistent; the instrument cannot tell
the arm from a re-run of the same recipe). A delta above the floor whose CI spans zero is **NOT
DETECTED**. "No effect" is retired as a label for either. Bars: no-fold floor 4.19 / fold floor POOLED 4.27 [1.23,6.92]
(the 5.94 was the endpoint alone; still ONE training draw); a loss-off arm vs the parent takes max(4.19, fold floor) like any fold arm.
