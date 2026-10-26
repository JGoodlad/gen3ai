# The RESULT timeline — the protocol readers behind every claim it makes

Owned by this tree. The CLAIM RULE lives in `src/main/prober/CLAUDE.md` (`— no effect` is only ours
to make when the evidence supports it; anything else renders `— outcome unrecorded`). This file
holds the pure protocol readers that supply that evidence and the precedence between them.

**A side is found by the trace's RECORDED SEAT, not by a nickname.** A core trace records which player the
trainee was (`meta.trace_source.trainee_side`; `engine.trainee_side_of(summary)`), and `_timeline_for(…,
our_side=)` hands it to every reader as `actor_side` / `our_side`. This pool ships LOCALIZED nicknames
(`Staross` = Starmie, `Electhor` = Zapdos, `Leuphorie` = Blissey), MOJIBAKE ones (`MagnÃ©ton`) and plays
mirrors, and a nickname match fails on all three: measured over 12.5k side-turns of four current runs, the
nickname reading said **`absent` (never moved) for 389 sides that did move** (which blanked their line before
any damage or status was attributed) and **nothing at all for 886** (mirrors). With a seat none of that can
happen; a trace that records none (an older one) keeps the nickname matching, mirror and all, unchanged.

- **`protocol_action_fate`** → `moved` / `cant:<reason>` / `absent` / `None`. The `|cant|` half is
  not an edge case: a model-free trace has NO decoded TurnDelta, so `our_cant`/`opp_cant` are always
  empty and EVERY blocked move fell through.
- **`protocol_move_result`** → `immune` / `missed` / `None`, scoped to the actor's own move (the scan
  starts at its `|move|` and stops at the next, so the other side's immunity cannot be borrowed; a
  `|-miss|` names the ATTACKER first). It also turns a `missed` that used to be INFERRED from the
  move's accuracy into a recorded fact.
- **`protocol_move_effects`** → `{result, status, statuses, effects}`, its superset and the same window
  (tightened to stop at the turn's blank separator, so a residual Leftovers heal is not the move's
  doing). `result` adds the sim's own `|-fail|`; `statuses` is EVERY `|-status|` in the window,
  `(name, "PAR", self_targeted, via)` (a Synchronize mon relays a second one back onto the attacker,
  `via="Synchronize"`) and `status` is the first; the SIDE is read off the `pNa:` player tag rather than
  the name, so a caller maps the side to its own species spelling; `effects` is the raw tag list that
  PROVES something happened.
- **`protocol_boost_changes`** → `{"side": {"we": "atk+1 spe+1"}, "caused": {"we": "spd-1"}}` or `None`
  (sides not told apart). The ten stat-stage messages of `deps/pokemon-showdown/sim/SIM-PROTOCOL.md` are
  its vocabulary (`BOOST_TAGS_GEN3` = what the Rust port emits; `BOOST_TAGS_LATER_GEN` are folded as notes
  and never silent; `engine_stages_test.py` pins both sets against the doc and `protocol.rs`). The text is
  the recorder's own (`atk+1 spe+2`), the NET over the turn; Belly Drum's `-setboost` is `atk=+6` (a stage
  it SET: a slice never says where a stage started); Haze, a herb and Psych Up are notes; a zero amount is
  `atk at its limit`. **`side`** is a side's own mon (its Swords Dance, an Intimidate on arrival, a berry);
  **`caused`** is what a side's MOVE put on the other mon (Psychic's Sp. Def drop), told by the log's own
  framing: inside that move's window, no `[from]` tag (a berry is the holder's), not straight after an
  Intimidate's `|-ability|`. `caused` is said on the move that did it (`we psychic did 25% (…)  +  skarmory
  spd-1`), `side` on the side's move / switch line (`we dragondance  ·  atk+1 spe+1`), else on a line of
  its own (`we stat stages  ·  atk-1`). The recorder's TurnDelta boost (older runs) wins when it exists; a
  forced replacement's decision says none of its turn's stages (the move decision before it already did).

⚠️ **`analyze` was passing NO protocol at all** until 2026-09-07, so the deepest view in the prober
got none of these repairs while `battle_turns` got all of them. `ProbeSession.analyze` now slices the
turn once and hands it to `analyze_invocation(..., protocol=…)` as well as to the JSON's `protocol`
field.

**A LOCATED WINDOW decides a status; the recorded `events` list does not.** The recorded event names a side
and a species, never a cause, and crediting it to "a move that could have caused it" billed a Rest to the
other side's Thunderbolt / Body Slam on 3 of 18 sampled status turns (`opp thunderbolt → zapdos SLP` over
`|-status|p1a: Zapdos|slp|[from] move: Rest`). With the window located, the status is the window's (the
recorder's `TOX(1)` counter spelling is kept when it recorded the same status); the recorded event is the
fallback only when no window was found. A damaging move's status is said BESIDE its damage
(`we bodyslam did 10% (skarmory …)  +  skarmory PAR`): the text used to choose the damage clause OR the
status clause and dropped 71 of 162 secondary statuses (40 battles, two runs).

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
attack is never dropped. The one renderer is `engine.timeline_entry_text` (the web page prints its `text`, the
JSON CLI carries it beside the structured fields) ·

**Known gaps (2026-10-26; the entries come from the RECORDED actions, so what the recorder did not write the
timeline does not say):** a decision whose outcome the recorder could not read (the last of a battle, a rejected
choice) records the opponent's action as `unknown` and our switch as `switch:<mon>` (not `switched_to:`), so the
killing move and the status it applied are absent and our switch reads as a move of that name; a Roar / Whirlwind
drag is recorded as OUR switch, which hides our own move that turn; and the recorded `hp_delta` is the mon's
NET change, so a Rest turn prices the other side's hit as "outcome unrecorded". The window knows the move in
each case; building the entry from it is the open fix.

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### What the timeline may CLAIM

🚨 **"— no effect" IS A CLAIM, AND IT IS ONLY OURS TO MAKE WHEN THE EVIDENCE SUPPORTS IT**
(`engine._no_effect_supported`, 2026-09-07). Exactly three things support it: the recorder DECODED
the move's fate/effectiveness, the SIM said so (`|-fail|` / `|-immune|` / `|-miss|` in the move's own
protocol window), or that window was LOCATED and is EMPTY of effect tags. Anything else renders
**`— outcome unrecorded`** — a gap in the evidence, said out loud, and *not* a synonym. And a window
that CONTRADICTS the claim (it carries effect tags) beats every recorded outcome, because the log is
the sim's own transcript while the recorded `events` list is known to have had a hole. The assertion
form of the same rule is **`verify_timeline_against_protocol`** → raises `TimelineContradiction`;
it is deliberately NOT called from `build_result_timeline` (a forensic view must still render a trace
it cannot fully explain) — it is for tests and for a surface that would rather stop than mislead.

⚠️ **"Nothing happened" had THREE causes and one sentence, so the line described the wrong thing.**
The recorded outcome says what a side CHOSE; nothing in a model-free trace says whether the choice
ever ran, so a move that never executed was explained as one that executed and achieved nothing — a
claim about the MOVE on a turn where the move never happened. Reported on gen-16 `loss_s0_004`:
Forretress CHOSE Explosion on turn 6, was outsped by a +2 Tyranitar and killed, and the timeline
read `we explosion — no effect`. Two pure readers over the turn's protocol slice fix it, siblings of
`move_order_from_protocol` and finding sides the same way (the recorded seat, else the nickname):

| the log says | now reads | before |
|---|---|---|
| no `\|move\|` for that side, and it fainted | `— never moved (fainted first)` | `— no effect` |
| `\|cant\|<mon>\|frz` (or par/slp/flinch/…) | `— couldn't move (frozen)` | `— no effect` |
| `\|-immune\|<target>` | `— no effect (immune)` | `— no effect` |

The readers that supply that evidence, the precedence between a RECORDED outcome and the log, and the
measured 9.8% of move lines they repaired, are above.

🚨 **A DETECTOR READS THE RAW PROTOCOL; ONLY A HUMAN SURFACE READS THE RENDERED TIMELINE.**
`loops.py` keys on each battle's `*_replay.html` protocol lines, never on
`engine.timeline_entry_text` — the rendering is a SENTENCE, and a detector must key on the fact
underneath it. The timeline's job is to be readable, the detector's is to be exact, and the two
must not be wired together.
