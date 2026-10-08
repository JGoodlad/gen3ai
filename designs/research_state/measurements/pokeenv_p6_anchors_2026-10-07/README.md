# P6 slice 2 of the poke-env retirement — the anchors tool fully off poke-env: the identity proof

**Date:** 2026-10-07 · **Code:** both paths present in the P6-slice-2 PART-1 commit (`feat(anchors): T27 P6 slice 2
(part 1)`: `--our-transport live` added beside the legacy `--our-transport poke-env`), built on `077197b8` and
measured there — local SHA `1b6b11c2` before this branch was rebased onto `e34f7db8` (v145); rows of the first (a)
batch and of (b) stamp `ws_frontend@077197b8` (the same tree, not yet committed), the (a) re-runs and (c)
`ws_frontend@1b6b11c2`. The legacy client is deleted in the PART-2 commit. The v144 checkpoint below is behind HEAD's
v145 floor after the rebase · **Box:** CPU only, load 3–9 on 16 threads (a live training run held the GPU) · **Servers:** local only —
the anchors harness's own front ends on 9520–9562 (explicit ports), the pinned `deps/pokemon-showdown` on
9561 / 9562 (started and stopped by PID by the tool), a master-built Node (`~/.cache/gen3ai/p4/showdown-master`,
`51ad80fa`) on 9570 started and stopped by PID by `scripts/batch_node.sh`; no public server, no 8000 / 8001, no human
contact.

## The question

After P3 (`../pokeenv_p3_anchors_2026-10-07/`) three shapes of `python -m main.anchors` still played OUR side on the
legacy poke-env client (`main.play` → `RLPlayer`, or a Python roster bot, both poke-env `Player`s): `--server node`,
`--server-uri`, and a `bot:<name>` our-side. P6 slice 2 moves all three onto the Rust stack and deletes the legacy
client:

* **a `bot:` our-side** is the RUST PORT of that roster bot (`src/rust_env/src/bots/`, gated action-equal to the
  Python bots per decision on the banked corpus) deciding on OUR side's own reading of the battle: the new
  `bot_reader` session (`src/rust_env/src/bin/bot_reader.rs` over `bot_side.rs` = `pokesim::side_reader::SideReader`
  + `bots::Bot`) under P4's live client (`main.live.client`), its RNG streams seeded as the env core's bot route
  seeds them (`--bot-seed S`: half h draws `random.Random(stream_seed(S, h, k))`);
* **`--server node` / `--server-uri`** play the checkpoint on P4's live client (`main.live`'s `live_reader` — the
  same `SideReader` chain `sim_bridge`'s core frame comes from) with the core slot's own decision arithmetic.

Do the new paths play the SAME games as the deleted one?

## Protocol

- **(a) bots, seeded, byte identity.** Every one of the NINE roster bots as our side vs `metamon:SmallRL` (ckpt 40,
  greedy), 12 games each (6 per half), on the `--server rust` front end at one `--seed-base` (each battle's sim seed)
  and `--team-seed` per cell, captures on. **new** = `--our-transport live` (the Rust bot) under
  `utils.poke_env_blocker`; **old** = `--our-transport poke-env` (the legacy client + the Python bot) with the bot's
  `_choice_rng` / `_protect_rng` seeded to the SAME streams by a comparison-harness hook
  (`scripts/legacy_seeded_bot.py`; production never seeded it). Team set alternates home / away; each bot its own
  `--bot-seed` (101 + i). Plus a **TEETH** cell: cell 00's new read against an old read seeded with the WRONG bot
  seed (102 for 101).
- **(c) the checkpoint on the live client, seeded, byte identity.** The client the Node / URI shapes now use,
  compared where a battle CAN be seeded: `--our-transport live` vs `--our-transport poke-env` on the `--server rust`
  front end (subprocess form for both), 20 games away, greedy. Checkpoint: P4's fresh v144 CPU debug checkpoint
  `~/.cache/gen3ai/p4/models/p4_shadow_v144/final_model.zip` (4,096 steps, untrained; every archived checkpoint is
  behind the v144 floor) — it plays to the turn limit, so the stall forfeit is exercised.
- **(b) Node — a battle cannot be seeded, so SHADOW + a new-path series.** The OLD path (`--our-transport poke-env
  --server node`, 20 games home vs SmallRL) ran with the NEW reader as a shadow (`scripts/legacy_shadow.py`: P4 gate
  (d)'s `main.live.gate_peer.Shadow`, fed every websocket frame the legacy client received before poke-env handled
  it; per decision: poke-env's row vs the reader's frame BYTES, the masks, the checkpoint's argmax on each row, the
  token sent vs the reader's token). Then the NEW path with the blocker: the checkpoint on Node (20 games home), a
  Rust bot on Node (`bot:heuristic2`, 10 games away), and the checkpoint against a master-built Node via
  `--server-uri` (10 games away).
- Scripts: `scripts/read.sh` (one read from the worktree: its `PYTHONPATH`, its release `sim_bridge` /
  `live_reader` / `bot_reader`, the blocker or a harness), `scripts/bot_cell.sh` + `scripts/batch_bots.sh` (a),
  `scripts/model_cell.sh` (c), `scripts/batch_node.sh` (b's new path), `scripts/analyze.py` (the identity read, P3's
  extended with each differing battle's first difference), `scripts/shadow_digest.py` + `scripts/race_kinds.py` (b's
  shadow), `scripts/read_digest.py`, `scripts/bank.sh`. Banked: each read's `games.jsonl` + `summary.json`, the
  analyses (`gate_a/analysis.json`, `gate_c/analysis.json`, `gate_b/old_shadow/shadow_digest.json`,
  `gate_b/old_shadow/race_kinds.json`, `gate_b/read_digest.json`). The captures and the raw shadow logs (tens of MB)
  were not banked; the scripts re-derive every count from them.

## Result

### (a) The nine bots — 108 / 108 battles byte-identical

| cell (bot, team set) | new (Rust bot) W/L/T | old (Python bot, seeded) W/L/T | battles byte-identical | choices (ours) |
|---|---|---|---:|---:|
| random · home | 0/12/0 | 0/12/0 | 12 / 12 | 948 (497) |
| heuristic · away | 2/10/0 | 2/10/0 | 12 / 12 | 1,021 (516) |
| heuristic2 · home | 4/8/0 | 4/8/0 | 12 / 12 | 645 (329) |
| staller · away | 0/12/0 | 0/12/0 | 12 / 12 | 1,087 (556) |
| staller_v2 · home | 1/11/0 | 1/11/0 | 12 / 12 | 740 (380) |
| aggressive · away | 1/11/0 | 1/11/0 | 12 / 12 | 877 (452) |
| aggressive_v2 · home | 3/9/0 | 3/9/0 | 12 / 12 | 675 (345) |
| setup_sweep · away | 3/9/0 | 3/9/0 | 12 / 12 | 873 (443) |
| setup_sweep_v2 · home | 2/10/0 | 2/10/0 | 12 / 12 | 703 (362) |
| **pooled** | **16/92/0** | **16/92/0** | **108 / 108** | **7,569 (3,880)** |
| TEETH: random, old seeded 102 vs new 101 | 0/12/0 | 0/12/0 | **0 / 12** | — |

- **"Byte-identical"** = each side's CHOOSE / FORCELOSE sequence to the battle child AND every per-side protocol chunk
  the front end relayed, equal in all 108 pairs. Shift new − old **0.000, Newcombe [−0.096, +0.096]** (unpaired,
  conservative; 0 discordant of 108 paired).
- **The TEETH cell differs on every battle** at our first decision (e.g. battle 1: `move meteormash` vs
  `switch Charizard`): the comparison sees a stream difference, so the 108 / 108 is not vacuous.
- **Zero failures.** `status: OK` on all 20 reads; 0 ERROR lines in the 20 front-end logs; 0 defaults, 0 re-decides;
  no game reached the 250-turn limit; Metamon's `argmax_match_rate` 1.0000 throughout.
- **Wall:** equal (new 279 s, old 274 s over the nine cells).

### (c) The checkpoint on the live client — 19 / 20 byte-identical, the 20th explained

| cell | new (live client) W/L/T | old (legacy client) W/L/T | battles byte-identical | choices (ours) |
|---|---|---|---:|---:|
| v144 checkpoint · away · greedy | 0/20/0 | 1/19/0 | **19 / 20** | 3,181 (1,635) |

- **The one difference, root-caused (finding 1).** Battle 6 (the first after battle 5, which BOTH paths ended with
  OUR identical stall forfeit at turn 250): on the old path Metamon FORFEITED at its first decision of battle 6 (`CHOOSE
  p1 switch Tyranitar`, then `FORCELOSE p2` — our first choice equal on both paths); on the new path it played the
  battle out. Metamon's own log on the old path shows the long-tail loop (`Battle is already finished, call reset` /
  `Force resetting due to long-tail error`) — hazard **H5 / H17**: after our forfeit ends a battle mid-step, the
  peer's env and agent disagree about the current battle. Whether it fires depends on WHEN the peer's last choice
  landed relative to our forfeit — the two paths interleaved the turn-250 choices in the opposite order (per-side
  sequences equal: new `… p2 switch claydol, p1 switch Skarmory, FORCELOSE p1`; old `… p1 switch Skarmory, p2 switch
  claydol, FORCELOSE p1`), a client-timing fact, not a decision. Battles 7–20 are byte-identical again.
- 0 ERRORs, 0 defaults / re-decides, 4 stall forfeits on each path, our argmax-match 1.0 on both.

### (b) Node — 0 shadow differences on 1,349 decisions; the new path 0 protocol failures

| read | server | our side | n | W/L/T | our decisions | protocol / parse failures |
|---|---|---|---:|---|---:|---|
| OLD + shadow reader | pinned Node (`showdown:e0551883f`) | legacy client, checkpoint | 20 | 2/18/0 | 1,349 shadowed | — |
| new | pinned Node | live client, checkpoint | 20 | 2/18/0 | 1,351 | 0 |
| new | pinned Node | live client, Rust `bot:heuristic2` (`--bot-seed 77`) | 10 | 0/10/0 | 342 | 0 |
| new | master Node (`51ad80fa`) via `--server-uri` (stamped `external`) | live client, checkpoint | 10 | 0/10/0 | 781 | 0 |

- **The shadow:** 1,347 compared choices + 2 stall forfeits closed, **0 row / mask / action / token differences**,
  0 reader halts. 1,328 `race` notes, every one of them an `|inactive|` battle-TIMER line (a `ROOM_SKIP` line the
  reader never folds; Metamon starts the timer — upstream `start_timer_on_battle_start`) arriving while a decision
  was open: a harness note, not a difference (`race_kinds.json`: 0 races carry battle protocol).
- **The new path:** `status: OK` on all three reads, 0 ERROR lines in the Node logs, 0 T28 halts, 0 defaults /
  re-decides, every regime verified on the peer (argmax 1.0000). After our stall forfeits at turn 250 Metamon died of
  its post-game `RecursionError` (H5) — in the ACCEPTOR half as well as the challenger half, on the old path and the
  new alike — so `peer_clean: false` with every game present (finding 7).
- The new-path Node win count equals the shadowed old one (2/18) — NOT the same battles (Node mints its own seed).

**Verdict: EQUIVALENT — the new paths play the deleted client's games, decision for decision.** Bots: 108 / 108
seeded battles byte-identical (and the teeth differ 12 / 12). Checkpoint on the live client: 19 / 20 byte-identical,
the 20th a peer-side H5 desync after an identical stall forfeit. Node: 0 differences on 1,349 shadowed decisions and
0 protocol failures on 40 new-path games across the pinned and a master Node. `our_transport` is PROVENANCE, not a
regime boundary.

## Findings

1. **H5 / H17 makes a seeded old-vs-new pair diverge after a stall forfeit, through the PEER.** (c)'s battle 6:
   Metamon's post-forfeit desync is timing-dependent (it hinges on whether its last choice landed before our
   forfeit), so two clients that play identically can still see the peer forfeit the NEXT battle on one path and
   not the other. Upstream (Metamon's `MetamonAMAGOWrapper.step` / pipelined challenger loop), costs no games of
   ours; an identity read with forfeits must expect it and root-cause each case, as here.
2. **A bot our-side never forfeits at the turn limit — on BOTH paths.** The Python bots carried no stall check
   (F-LF-5), nor does the env core's bot route, so the Rust bot client sets no forfeit; rows still stamp the plan's
   `forfeit_turn_limit` and `hit_forfeit_limit` reads `turns >= limit` (no bot game reached it here). Pre-existing,
   kept for identity; stated in the SOP.
3. **Row fields that changed meaning for a bot our-side.** `n_decisions` is now per game (the legacy bot row said
   `None` — a poke-env `Player` has no counter); `our_bot_seed` is stamped; the legacy bot drew from the
   process-wide UNSEEDED `random` in production, so a `poke_env_bot` row is the same distribution on an
   unrepeatable stream. The Rust port was gated action-equal per decision on the banked corpus; the bot-vs-bot
   ROUND ROBIN ratings the bot cells are anchored to were played by the Python bots — same policy, same distribution.
4. **The pinned Node runs a battle TIMER in an anchor read** (Metamon turns it on): 300 s per turn. Never close to
   firing here; a contended box that stalls our side for minutes would forfeit on the timer on Node only (the front
   end has no timer). Pre-existing on both paths.
5. **The checkpoint is untrained** (v144 CPU debug; the archive is behind the v144 floor), so (b)/(c) prove the
   READER and the client, not anything about strength; a trained-checkpoint repeat needs a v144+ run.
6. **Six of the first batch's 18 reads died at startup** (rc 3, `address already in use`): two cells launched at once
   raced `pick_port` for the same auto-picked 95XX port (bind-then-release). Harness-only (re-run with explicit
   ports, `PORT=` in `bot_cell.sh`); but it is a real race for anyone launching two `main.anchors` reads in the same
   second — **UNVERIFIED** how often it bites in practice; the tool's own refusal is the clean failure seen here.
7. **Metamon's post-forfeit `RecursionError` fires in the ACCEPTOR half too** (both Node checkpoint reads, old and
   new: the `ours_challenge` half's `peer_acceptor.log` carries it), but `runner.classify_peer_error` names it
   (`peer_recursion_upstream`) only for a complete `peer_challenge` half (H17's four recorded occurrences). So these
   reads say `peer_clean: false` with NO `peer_exit_note` explaining it. Pre-existing and path-independent; H17's
   "it is about WHO CHALLENGES" reading is contradicted by this acceptor-side occurrence — worth an SOP look.
