# The RESULT timeline — the protocol readers behind every claim it makes

Owned by this tree. The CLAIM RULE lives in `src/main/prober/CLAUDE.md` (`— no effect` is only ours
to make when the evidence supports it; anything else renders `— outcome unrecorded`). This file
holds the three pure protocol readers that supply that evidence and the precedence between them.

- **`protocol_action_fate`** → `moved` / `cant:<reason>` / `absent` / `None`. The `|cant|` half is
  not an edge case: a model-free trace has NO decoded TurnDelta, so `our_cant`/`opp_cant` are always
  empty and EVERY blocked move fell through.
- **`protocol_move_result`** → `immune` / `missed` / `None`, scoped to the actor's own move (the scan
  starts at its `|move|` and stops at the next, so the other side's immunity cannot be borrowed; a
  `|-miss|` names the ATTACKER first). It also turns a `missed` that used to be INFERRED from the
  move's accuracy into a recorded fact.
- **`protocol_move_effects`** → `{result, status, effects}`, its superset and the same window
  (tightened to stop at the turn's blank separator, so a residual Leftovers heal is not the move's
  doing). `result` adds the sim's own `|-fail|`; `status` is `(name, "PAR", self_targeted)`, with the
  SIDE read off the `pNa:` player tag rather than the name — this pool ships LOCALIZED nicknames
  (`Airmure` = Skarmory), so a caller maps the side to its own species spelling; `effects` is the raw
  tag list that PROVES something happened.

⚠️ **`analyze` was passing NO protocol at all** until 2026-09-07, so the deepest view in the prober
got none of these repairs while `battle_turns` got all of them. `ProbeSession.analyze` now slices the
turn once and hands it to `analyze_invocation(..., protocol=…)` as well as to the JSON's `protocol`
field.

**The RECORDED fate/effectiveness wins outright** — it comes from the TurnDelta the analysis was
built on, and `_no_effect_reason` ranks immune ABOVE miss, so feeding a protocol immunity alongside
a recorded miss would silently overrule the recorder (caught by its own test). The protocol fills
only when the recorder decoded nothing. **`absent` is claimed only when the mon FAINTED that turn**,
and only when the OTHER side was positively identified in the same slice — the guard against this
pool's LOCALIZED-species nicknames (`Triopikeur` = Dugtrio); otherwise `None` and today's behaviour.
**Measured over 120 battles of that run: 470 of 4817 move lines (9.8%) were mis-explained** — 277
never-moved, 122 blocked (61 par / 25 slp / 22 frz / 14 flinch), 71 immune. `cant_phrase` also learns
the protocol's own spellings (`move: Taunt` → taunted, `Focus Punch` → lost its focus), since the
reason now arrives straight off the log. **miss/fail is
a RECORDED fact** — the `gen3_move_outcome_v1` TurnDelta block encodes each side's `[hit, miss, fail]`,
decoded by `describe_turn_outcome` as `our_move_outcome`/`opp_move_outcome` (so it distinguishes a true
miss from a hit-that-did-nothing); only on a model-free / pre-`v1` trace does it fall back to inferring
a miss from the move's accuracy. When the OPPONENT
voluntarily switched, the recorded hp_delta can't price the hit on the switch-IN (it compares the mon
that left), so the attack shows the **resulting HP** instead (`we rockslide → celebi (now 11%)`) — the
attack is never dropped. The shared renderer is `app._append_timeline_entry` / `_append_happened`
(Summary + Review card) ·
