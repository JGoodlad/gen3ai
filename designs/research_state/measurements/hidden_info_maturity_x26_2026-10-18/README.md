# The MATURITY BATTERY on X26: the interim bank and the LEAN trajectory (2026-10-18)

**Status.**
- **The harness** is T35 (`design_hidden_info_binding.md` §12.13.7; backlog X26-6). It shipped before the
  registration.
- **The registration** is [`PREREGISTRATION.md`](PREREGISTRATION.md), shipped with the bank's ids
  (`states_mat_x26.json`) before any trajectory output existed.
- **The queue** crawls detached (§3), launched 2026-10-18 04:45 PDT from `826bee89`. The read is `read_mat.py`, committed with the harness, before any output.
- **Results:** none yet. The orchestrator dispatches the read when `QUEUE_DONE` appears.

**What it asks** (owner, 2026-10-17: "what does it take to enable search? Are we getting closer, not closer?"). It
re-runs §12.13.7's stage and dose reads at X26's 25M, 100M, 175M and 225M checkpoints on ONE frozen bank, and reads the
TRAJECTORY:
- does honest one-ply search's gap to the oracle shrink (GAP-SHRINKS)?
- does the network's effective belief quality rise toward the truth (BELIEF-RISES)?
- is the gap still moving between the last two anchors (CONVERGED)?

**Vocabulary.**
- IMP = decision improvement: the win probability gained per decision by playing a scorer's pick instead of the
  policy's top-1, judged on held-out referee seeds of the true world.
- O = search in the true world; F = search in fully honest worlds; FN = F with the network's own species and move
  belief.
- G_F = IMP_O − IMP_F. TODAY = FN's position on the dose curve, in "weight on the true world" units.
- LEAF-K8 = the mean outcome over 8 worlds played to the end.

Bulk lives in `~/gen3ai_archive/hidden_info_maturity/` (layout: `mat_common.py`'s docstring). This folder holds the
scripts, the committed ids, the anchors' sha256 (`checkpoints_x26.json`) and, later, the read.

## 1. Pipeline

| step | script | where (`run_mat.sh <where>`) | output |
|---|---|---|---|
| trigger + copies | `trigger_mat.py status / register / traces` (`maturity_series`: the first checkpoint in [m, m + 25M), the read-only sha256 copy) | any Python | `ckpt/`, `ckpts.json`, `traces/` |
| bank | `make_mat_bank.py` (`maturity_bank.draw`; no model) | `pin` | `bank/`, `census.npz`, `states_mat_x26.json` |
| roots | `roots_mat.py` (each anchor's top-1, V, tie) | `pin` | `roots/<anchor>.jsonl` |
| truth (O) | `phase1_mat.py` (the pilot's `run_playouts.one_state`) | `pin` | `rows/<anchor>/O/` |
| belief (FN) | `fn_belief_mat.py` (the pilot's `det_worlds.root_belief`) | `pin` | `fn_belief/<anchor>.jsonl` |
| worlds | `build_mat_worlds.py` (`WorldSampler`, the guard on; FN = Tier 1's `fn_worlds`; F re-sampled for CRN and checked) | `head` | `worlds/shared/<arm>/`, `worlds/<anchor>/FN/` |
| playouts | `play_mat_worlds.py` (the pilot's `run_det_playouts.one_state`) | `pin` | `rows/<anchor>/<arm>/` |
| runner | `queue_mat.py` (launched by `launch_queue.sh`) | the HEAD checkout | `logs/queue_mat.log`, `QUEUE_DONE` |
| identity | `identity_mat.py` (oracle worlds at HEAD, played at the pin = Phase 1) | pure | `identity_mat.json` |
| table | `score_mat.py` (Tier 1's `t1_data` + `score_blind.dk_family`) | pure | `score/table.pkl` (+ sha256) |
| read | `read_mat.py` (+ `read_mat_test.py`, planted effects; rules in `maturity_rules.py`) | pure | `result_mat.json` |

**CPU (root rule 5b):**
- every worker runs one thread, under `nice -n 19` and SCHED_IDLE;
- workers are pinned to `taskset -c 5-7,13-15`, each in its own `mem_cap` scope;
- `queue_mat.py --slots 2` keeps at most 2 workers.

No GPU: `CUDA_VISIBLE_DEVICES=""`. The run directory is only ever read, and only to copy.

## 2. Reproduce / operate

```bash
F=designs/research_state/measurements/hidden_info_maturity_x26_2026-10-18
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -I $F/trigger_mat.py status                               # the run's 25M anchors: registered / pending
$PY -I $F/trigger_mat.py register --pending --mode lean       # THE TRIGGER: copy + register every new anchor
bash $F/launch_queue.sh                                       # the detached runner (from the HEAD checkout)
$PY -I $F/queue_mat.py --status                               # PROGRESS: per anchor and unit, done / wanted
tail -f ~/gen3ai_archive/hidden_info_maturity/logs/queue_mat.log
$PY -I $F/identity_mat.py --out identity_mat.json             # after QUEUE_DONE: the identity per anchor
$PY -I $F/read_mat_test.py && $PY -I $F/score_mat.py && $PY -I $F/read_mat.py   # the registered read
```

## 3. The running queue

**LAUNCHED 2026-10-18 04:45 PDT** (runner pid in `~/gen3ai_archive/hidden_info_maturity/queue.pid`). It is detached
(`setsid nohup`), so it survives the launching session.

| | |
|---|---|
| runner | `queue_mat.py --slots 2`, launched by `launch_queue.sh` from the HEAD checkout |
| HEAD checkout (the world builder; §12.13.7 (f)) | `~/.cache/gen3ai/hib_mat/head`, detached at `826bee89` (the registration commit). **Keep it** until the read |
| pin checkout (bank, roots, truth, belief, playouts) | `~/.cache/gen3ai/hib_mat/pin_x26`, detached at `9c033489` (X26's pin), with its own Rust release builds. **Keep it** until the read |
| archive | `~/gen3ai_archive/hidden_info_maturity/` (checkpoint copies, copied traces, bank, every row) |
| progress (counts only, no outcome) | `python -I <this folder>/queue_mat.py --status` (done / wanted per anchor and unit); `tail ~/gen3ai_archive/hidden_info_maturity/logs/queue_mat.log` |
| done marker | `~/gen3ai_archive/hidden_info_maturity/QUEUE_DONE` (it lists any PARKED unit: a FINDING) |
| order | the shared F worlds, then 25M, 225M, 100M, 175M (roots, identity, truth, belief, F, FN worlds, FN) |
| cost | ≈ 30 CPU-h for the four LEAN anchors, at the pilot's measured 77–92 CPU-s per state (PREREGISTRATION §5) |
| **ETA** | **≈ 15–20 wall-hours on the 2 slots, i.e. ≈ 2026-10-18 20:00 to 2026-10-19 01:00 PDT.** It runs at SCHED_IDLE beside the live trainer, so it stretches whenever the trainer takes the cores (budget to 2026-10-19 midday). The first-vs-last pair (25M, 225M) completes first, about 60 % of the way in |

- **Restart after a crash or a reboot:** `bash <this folder>/launch_queue.sh` from the HEAD checkout. Every unit is
  resumable, and `--limit` caps only NEW work.
- **A new anchor** (the trigger): `trigger_mat.py register --pending --mode lean`, then relaunch if `QUEUE_DONE` has
  appeared.
- **The read** (once `QUEUE_DONE` exists): §2's last three lines. Their outputs, `identity_mat.json` and
  `result_mat.json`, land in this folder.

## 4. Notes fixed before any output

- **Not blinded.** The arms carry their labels. Every rule is fixed as code before any output exists, and the runner
  reads no outcome (E2's reasoning: Tier 1's blinding protected the frozen table and scripts, not the author's
  knowledge).
- **One pin for the whole series.** Every X26 checkpoint was saved at `9c033489`. The observation encoding, the engine
  and the critic's interface are therefore the same at every anchor, and only the weights differ.
- **F is network-free.** It is built once and played at every anchor, so the anchors' F rows differ only by the network.
  FN is built per anchor from that anchor's own belief.
- **The pilot** ran on 12 eligible states outside the bank (`<archive>/pilot/`). It is engineering only: GIGO and
  cost, never read as trajectory output.
