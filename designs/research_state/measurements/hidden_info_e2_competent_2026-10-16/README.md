# E2: the hidden-info study's COMPETENT-OPPONENT state bank (2026-10-16)

**Status.**
- **Pre-registered:** [`PREREGISTRATION.md`](PREREGISTRATION.md), shipped `5881ee83` before any E2 game was generated
  or any state drawn.
- **The harness and the bank's ids** are committed in this folder.
- **The arms** crawl detached under `queue_e2.py`; the read is `read_e2.py`, committed before any arm output existed.
- **Results:** none yet. This section is filled when the read is run.

**What it settles.** Tier 1 found that the never-seen Pokémon (class U) carry more of honest search's gap than the seen
mons' hidden moves (class M), by rule R3 on both sides. That held only on the registered mix (58 % scripted-bot
games). On the own-snapshot states it failed in both builds (`red_team.md` §2.6 (b); `third_party_comparison.md`).
E2 asks the same question, pre-registered, on 600 fresh states per checkpoint drawn ONLY from games against the
network's own snapshots.

Bulk lives in `~/gen3ai_archive/hidden_info_e2/<cp>/`: the generated games (`gen/`), the bank, `pool.npz`, the rows and
worlds per source, and the logs. This folder holds the scripts, the committed ids (`states_e2_<cp>.json`) and the read.

## 1. Pipeline

| step | script | where (`run_e2.sh <where>`) | output |
|---|---|---|---|
| games | `gen_games.sh` (the pin's `main.ops.eval_trace_gen`) | `gen_<cp>`: `~/.cache/gen3ai/hib_e2/pin_<cp>` (pin + Showdown submodule) | `gen/e2_<cp>_<step>/` |
| bank | `make_e2_bank.py` | `<cp>`: the vdp pins Tier 1 used | `bank/`, `pool.npz`, `states_e2_<cp>.json` |
| truth (O) | `phase1_e2.py` (the pilot's `run_playouts.one_state`) | `<cp>` | `rows/O/` |
| worlds | `build_e2_worlds.py` (`WorldSampler`, the guard on) | `head`: `~/.cache/gen3ai/hib_e2/head` (the E2 HEAD) | `worlds/<arm>/` |
| playouts | `play_e2_worlds.py` (the pilot's `run_det_playouts.one_state`) | `<cp>_pad`: Tier 1's pad pins | `rows/<arm>/` |
| runner | `queue_e2.py` (launched by `launch_queue.sh`) | any; it calls `run_e2.sh` | `logs/queue_e2.log`, `QUEUE_DONE` |
| read | `read_e2.py` (+ `read_e2_test.py`, planted effects) | pure | `result_e2.json` |

Arm directories use `p` / `m` for `+` / `−`: `F`, `FpM`, `FpU`, `OmM`, `OmU`, and `V_oracle` for the identity check.

**CPU** follows root rule 5b:
- each worker runs one thread, under `nice -n 19` and SCHED_IDLE;
- workers are pinned with `taskset -c 5-7,13-15`, each in its own `mem_cap` scope;
- `queue_e2.py --slots 2` keeps at most 2 workers.

## 2. Reproduce / operate

```bash
F=designs/research_state/measurements/hidden_info_e2_competent_2026-10-16
bash $F/run_e2.sh gen_end2 2 8 12h gen_games.sh           # games (then gen_prod)
bash $F/run_e2.sh end2 1 14 3h make_e2_bank.py            # the bank (then prod); exit 4 = a short quintile (§1.3)
bash $F/launch_queue.sh                                   # the detached runner, from the E2 HEAD checkout
tail -f ~/gen3ai_archive/hidden_info_e2/logs/queue_e2.log # progress (counts only)
python -I $F/read_e2.py --partial                          # COUNTS only while it crawls; the read once QUEUE_DONE exists
python -I $F/read_e2.py                                    # the pre-registered read -> result_e2.json
```

## 3. Notes fixed before any arm output

- **Not blinded.** The arms carry their labels. Every rule is fixed as code in `read_e2.py` before any arm output exists,
  and the runner reads no outcome. Tier 1's blinding protected the frozen table and scripts, not the author's knowledge
  (Tier 1 README §3), so its cost buys nothing more here.
- **The builder's HEAD differs from Tier 1's.** Since Tier 1, the world builder gained the record-slot fix and the
  throwing guard (`8702a56f`) and the packed-move canonical reader (`631db7d8`). Tier 1's worlds were unaffected by
  either: no world changed, by those commits' own audits. E2's worlds are built with both.
- **The engine is unchanged.** `git diff 67f3f7fa <E2 HEAD> -- src/rust_sim/src src/rust_env/src` touches the same 7
  files as Tier 1's README §2 (oracle reveal, the gender pad, comments).
- **The `random`-bot games are a by-product.** `eval_trace_gen` cannot generate a sentinel-only cycle. Those games are
  never drawn.

## 4. Results

(Filled by the read.)
