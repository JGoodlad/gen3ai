# Anchor reads come off Node: the Rust websocket front end is the DEFAULT transport

**Measured 2026-09-20 on `ai_v13_02_flywheel_winprob/final_model.zip` @ 75,005,952 steps (arm W).**
CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, `OMP_NUM_THREADS=1`), ports 9500–9599, nothing
under `models/` written, no training or launcher touched, :8000/:8001 never touched. The work is
the owner's direction, verbatim: *"I want us off node if we can for evals… it is much easier from a
conceptual view if we can just use rust for evals."*

> ## The verdict
> **`python -m main.anchors` now defaults to `--server rust`** — the in-repo websocket front end
> (`utils.bridge.ws_frontend --impl rust`) in its own subprocess, one `sim_bridge` child per
> battle, **no Node server anywhere in an anchor read**. `--server node` is the explicit opt-out.
> The two transports **agree**: 100 games each, one team seed, `Δ = +0.010 [−0.124, +0.144]`,
> **NOT DETECTED**. The owner's stated reason is the largest number here: the server tree's
> **mean RSS falls from 3,227 MB to 40 MB (80×)**, peak from 3,551 MB to 166 MB (21×), and the
> same 100 games run in **126 s instead of 203 s**.

---

## 1. Provenance

| thing | value |
|---|---|
| our checkpoint | `models/ai_v13_02_flywheel_winprob/final_model.zip` @ **75,005,952** steps (arm W; **there is no 75 M file under `checkpoints/` — the run's 75 M snapshot IS `final_model.zip`**), resolved `explicit_zip`, loaded `bare` |
| gen3ai commit | worktree `anchors_rust` at `ffc3b0ef` + this change |
| bridge binary | `POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge` (pre-built; **no cargo build in the worktree, and no rust source change was needed**) |
| `deps/pokemon-showdown` pin | `e0551883f` — recorded on BOTH transports (the port came from this tree, and the front end still validates a `/utm` team with its `validate_team.js`) |
| Metamon | `/home/goodlad/dev/metamon`, `SmallRL` ckpt 40, greedy (`sample=False`), verified per decision |
| Foul Play | `/home/goodlad/dev/foul-play`, `--search-time-ms 100`, realized 122–150 k visits/decision |
| teams | `away` = Metamon's own 20-team `competitive` set (20/20, `team_source_asymmetry` false) for the differential; `home` for the Foul Play cells |
| box | 16 cores; load 1.4–3.4 for the front-end cell, 2.6–8.6 for the Node cell, 13–30 for the Foul Play cells (recorded per cell in `cells/*/meta.txt`) |

## 2. The gate table

| # | gate | transport | result |
|---|---|---|---|
| **a** | the anchors unit suite | — | **167 passed**, 0 failed (`src/main/anchors/ -m "not integration"`) |
| **a** | the anchors END-TO-END test (2 real games, both roles), **parametrized over both transports** | rust **and** node | **4 passed, 1 skipped** in 72 s — the skip is the on-purpose skip-path guard |
| **b** | `metamon:SmallRL` greedy **away**, **100 games**, arm W | **rust** | **status OK**, 0 failed sub-cells, `argmax_match_rate` **1.0000** (both halves — `ours_challenge` 50 games, `peer_challenge` 50), **W59/L40/T1 → 0.590 [0.492, 0.681]**, **1** game at the 250-turn forfeit cap, `n_defaults` 0, `n_redecides` 0, wall **126 s** |
| **c** | the same 100 games, same `--team-seed 920001`, same arm | **node** | status OK, `argmax_match_rate` **1.0000**, **W58/L42/T0 → 0.580 [0.482, 0.672]**, 1 forfeit, wall **203 s** |
| **c** | **the differential** | rust − node | **Δ +0.010, Newcombe 95 % [−0.124, +0.144] — NOT DETECTED.** Per half: rust 0.62 / 0.56, node 0.54 / 0.62 |
| **c′** | **the byte-differential gate** (the stronger one) | both arms | `ws_frontend_byte_identity_integration_test.py` + `ws_frontend_test.py` + `ws_frontend_integration_test.py`: **40 passed** — the front end's per-side protocol text is byte-identical to the Node bridge's, node-backed *and* rust-backed |
| **d** | **memory**, server process tree, sampled every 2 s | rust | **peak 165.8 MB, mean 40.2 MB** (63 samples) |
| **d** | | node | **peak 3,551.1 MB, mean 3,226.5 MB** (97 samples) — **80× the mean, 21× the peak** |
| **e** | Foul Play, 20 games, `--search-time-ms 100` | rust | **FAILED `no_progress` after 1/10 games — the known H-1/H14 Foul Play adapter defect, NOT a transport fault** (§4) |
| **e′** | Foul Play, 2 games (1 per role), same settings | rust | **status OK**, W1/L1, realized 122 k visits/decision, 0 ERRORs in the front end's log — the front end serves Foul Play in **both** roles |

🚨 **The win-rate agreement is not byte equality, and the two 100-game cells are not replicates.**
They share our team draw (the seed) but not the battle RNG and not Metamon's hidden state, exactly
as the 2026-09-16 side-by-side recorded. A decision-level replicate needs `--seed-base`, which the
front end has and **the Node server has no counterpart for** — which is why gate (c′), the byte
differential, is the transport's real gate and (c) is the outcome-level cross-check.

## 3. What landed

* **`--server {rust,node}`, default `rust`.** `main.anchors.server` now holds a `ManagedServer`
  base (start, PID-only stop, readiness, log tail) with two subclasses: `FrontEndServer` (the
  default — `sys.executable -m utils.bridge.ws_frontend --port N --impl rust`, `src` prepended to
  the child's `PYTHONPATH`, `POKESIM_SIM_BRIDGE_BIN` inherited) and `ShowdownServer` (unchanged
  behaviour, now the opt-out). Both refuse 8000/8001 **in the base class**, so a third transport
  cannot arrive without the guard.
* **`--seed-base` / `--capture-dir`**, the front end's reproducibility pair — **REFUSED** with a
  named cause on `--server node` and on `--server-uri`, because a seed silently ignored makes an
  unrepeatable series look seeded.
* **The transport is on every row**: `server_impl` (`rust` / `node` / `external`) and
  `server_version` (`ws_frontend@<head>+rust:<bridge binary>` / `showdown:<pin>`), both added to
  `REQUIRED_ROW_FIELDS`, printed by `--dry-run` and by the result block. `--server-uri` stamps
  **`external`**: this tool cannot vouch for a transport it did not start.
* **The log is named after the server** (`ws_frontend.log` / `showdown.log`), so an output
  directory says which transport served it before anything is parsed.
* The end-to-end test is **parametrized over both transports**, and the Node-only skip reason now
  says that the default transport needs no `node` at all.

## 4. Hazards — each one is a finding

| # | hazard | evidence | what it means |
|---|---|---|---|
| **H-1** | 🚨 **A bare TCP readiness probe FABRICATES an `ERROR` in the front end's own log.** `websockets` answers a connect-then-close with `opening handshake failed` and a three-deep traceback | `cells/rust100/ws_frontend.log` line 4, at 07:18:02.655 — 3.5 s before the first client connected, and the file's only one | **"0 ERRORs in the server log" is the criterion every validation of this front end has read.** A probe that writes one poisons the instrument. **Fixed in this change**: a server that prints a readiness line (`[ws_frontend] READY`) is never TCP-probed; the Node server, which prints none we parse, still is. `cells/probefix/ws_frontend.log` is the after: **0** |
| **H-2** | 🚨 **`--opponent foulplay` still cannot run a multi-game `ours_challenge` half** — and the front end's log shows WHY | `cells/fp20_rust/`: `no_progress`, 1/10 games; the front-end log has `challenge anchorw201 -> fppeer201` twice, 0.4 s apart, the second while Foul Play was mid-battle; Foul Play's own log then sits at `Waiting for a gen3ou challenge` | poke-env PIPELINES its challenges and Foul Play drops a PM that arrives mid-battle. **Transport-independent** — reproduced 3/3 on Node on 2026-09-16 — so it is not a blocker for this promotion, but it is now diagnosed rather than inferred. A Foul Play cell is proven at **1 game per half**. Backlog |
| **H-3** | ⚠️ **The front end is not 100 % Node-free**: `/utm` team validation shells out to `deps/pokemon-showdown`'s `validate_team.js` | `utils.bridge.team_validator`, cached per distinct team | A short-lived process per distinct team (≤ 20 in an away cell), **not a resident server** — it costs none of the 3.2 GB. `--no-validate-teams` would remove it entirely at the cost of the loud refusal on a nickname-bearing team (`ws_frontend.md` H5); **not** taken, and not exposed on the anchors CLI |
| **H-4** | ⚠️ **This is a transport + SIMULATOR comparison, not a pure transport one** | unchanged from 2026-09-16 §4 | `--server rust` backs each battle with a `sim_bridge` child where Node backs it with the pinned Showdown server. The n = 100 agreement is evidence for the whole stack; gate (c′) is what isolates the protocol |
| **H-5** | ⚠️ **The RSS numbers are a process-TREE sample at 2 s, not an allocator accounting** | `cells/*/rss.txt` | The front-end tree includes its `sim_bridge` children (that is the point — counting only the parent would flatter it). The Node figure is one process. Both are `VmRSS` sums; neither is a peak-RSS guarantee between samples |
| **H-6** | ⚠️ **The two 100-game cells ran on a box whose load was falling** (3.4 → 2.7 for rust, 2.6 → 8.6 for node) | `cells/*/meta.txt` | The **wall-clock** ratio (1.61×) is therefore a weak number; the win-rate comparison is not affected, and the 2026-09-16 pair (0.575 vs 0.629 games/s under load 20–39) is the wider evidence that throughput is not the reason to switch |

## 5. What this cannot say

* **Nothing about concurrency above 1.** Every cell ran one battle at a time.
* **Nothing about `SyntheticRLV2`** (only `SmallRL` and Foul Play were played) and nothing about
  the `home` team set on the differential — the 2026-09-16 read covers `home` at n = 100.
* **Nothing about strength.** No training happened; arm W is frozen, and the absolute level here
  (0.58–0.59 away) is not a new reading of it.
* **Nothing about the deferral list.** Reconnection, a battle timer, `/search` and replays remain
  unimplemented in the front end; a read that needs one of them needs `--server node`.

## 6. What is in this directory

* `scripts/run_cell.sh` — one anchors cell with the server tree's RSS sampled beside it. The PID
  comes from the tool's own announcement, never a `pgrep` pattern.
* `cells/<label>/` — `summary.json`, `meta.txt`, `rss.txt` for every cell; `games.jsonl` for the
  two 100-game cells; the front end's log for `rust100` (the H-1 before) and `probefix` (the
  after).

---

## 7. Ready-to-append ledger paragraph

**2026-09-20 — OPS: external-anchor reads come OFF NODE. `python -m main.anchors` now starts the
in-repo websocket front end over the Rust bridge by DEFAULT (`--server rust`), and a 100-game
transport differential says the two paths agree while the Node server costs 80× the memory.**
The owner's direction was to get evals off Node ("node sucks, uses memory"), and the number that
answers it is the server process tree's resident memory over a 100-game `metamon:SmallRL`
greedy/away cell on arm W (`ai_v13_02_flywheel_winprob/final_model.zip` @ 75,005,952):
**mean 40.2 MB / peak 165.8 MB through the front end against mean 3,226.5 MB / peak 3,551.1 MB
through `deps/pokemon-showdown` — 80× on the mean, 21× on the peak** — with the same 100 games
finishing in **126 s against 203 s**. The reads themselves are the same read: **0.590 [0.492,
0.681] (W59/L40/T1) through the front end against 0.580 [0.482, 0.672] (W58/L42/T0) through Node
at one team seed, Δ +0.010 [−0.124, +0.144], NOT DETECTED**, with `argmax_match_rate` 1.0000 on
all four halves, `n_defaults` / `n_redecides` 0 / 0, one 250-turn forfeit on each path and
`team_source_asymmetry` false. 🚨 **An outcome agreement at n = 100 is not byte equality and the
two cells are not replicates** — they share only our team draw — so the transport's real gate
remains `ws_frontend_byte_identity_integration_test.py` (green on both the node-backed and the
rust-backed arm), and this cell is the outcome-level cross-check on top of the 200-battle
side-by-side of `foulplay_axes_and_frontend_validation_2026-09-16`, whose registered verdict was
"front-end MAY be the DEFAULT transport". **`--server node` stays one flag away** and is what a
differential and anything on the front end's deferral list (reconnection, a battle timer,
`/search`, replays) is taken on; `--server-uri` starts nothing and stamps rows `external`, because
a tool cannot vouch for a transport it did not start. Every row now carries `server_impl` and
`server_version` (`ws_frontend@<head>+rust:<bridge binary>` / `showdown:<pin>`) in
`REQUIRED_ROW_FIELDS` — the regime rule applied to the transport. **Two hazards, both findings.**
(1) 🚨 **A bare TCP readiness probe fabricates an `ERROR` in the front end's own log** — `websockets`
answers a connect-that-closes with `opening handshake failed` and a three-deep traceback, and "0
ERRORs in the server log" is the criterion every validation of this front end reads; fixed by
waiting for the child's `[ws_frontend] READY` line instead, with a test that the announcing server
is never dialled. (2) 🚨 **`--opponent foulplay` still cannot run a multi-game `ours_challenge`
half**, reproduced here (`no_progress`, 1/10) and now DIAGNOSED from the front end's log: poke-env
pipelines a second `/challenge` 0.4 s after the first, Foul Play drops the PM while battling and
then waits forever for a challenge already consumed — transport-independent (3/3 on Node on
2026-09-16), so a Foul Play cell is proven at 1 game per half and the multi-game half is a backlog
row. Foul Play does play **both** roles through the front end at 1 game per half (n = 2, 0 ERRORs).
⚠️ The front end is not literally Node-free: `/utm` validation still shells out to
`validate_team.js` once per distinct team — a short-lived process, not a resident server, and none
of the 3.2 GB. Tag: **OPS · ANCHOR TRANSPORT DEFAULTS TO RUST · Δ +0.010 [−0.124, +0.144] NOT
DETECTED AT n = 100 EACH WAY · SERVER RSS 3,227 MB → 40 MB (80×) · 126 s vs 203 s · BYTE-IDENTITY
GATE GREEN · FOUL PLAY MULTI-GAME `ours_challenge` STILL BROKEN (transport-independent)**.
