# `main.x72` — the tools X72's registration owes

X72 (the swap forks carried toward their plateau) is registered in
[`../research_state/measurements/x72_swap_plateau_2026-11-25/PREREG.md`](../research_state/measurements/x72_swap_plateau_2026-11-25/PREREG.md);
its §15 lists tool paths as "NOT built". This page is the map of the ones that now exist, in `src/main/x72/`. Each
module's docstring is the detail of record (what it refuses, every file it writes). **None of them changes a number,
a class or a rule of the registration.** The frozen rule files (`x72_rule.py`, X71's `x71_rule.py` and `read_x71.py`)
are never imported and never edited: each tool restates the tables it needs and its tests pin every restated value
to the frozen files.

| module | job (PREREG §) | what it hands on |
|---|---|---|
| `main.x72.finals` | the ARCHIVE COPIES of a round's three finals (§3): the copy step, and the sha256-and-record check every cross and read goes through | `~/gen3ai_archive/x72_swap_plateau/finals/r<k>/<run>/` with a `finals.json` |
| `main.x72.cross` | the round cross (§5): plan, driver, reader. Round-k finals × round-(k−1) finals, 9 cells × 4,000 mirrored pairs, X70's protocol at the play pin `333a73dd`, ONE look | `<work dir>/h.json`, the input of `x72_rule.py decide-round` |
| `main.x72.floor` | the entropy floor's reader (§7): the mean of a lineage's last 20 entropy points at a mark, against 0.2419 nats | a JSON read and an exit code |
| `main.x72.probe` | the exploitability probe at a FIFTH target (§10): the two responders' argv, their record check, the read that pools X71's twelve banked cells with three new ones | `<work dir>/x72_probe_verdict.json` |

## What the window chain calls

Set `PYTHONPATH` to the checkout's `src` first; `K` is the round (1..4).

```bash
python -m main.x72.finals archive --round K          # after a round's three launches end; then `verify --round K`
python -m main.x72.cross drive --round K --pin-dir <checkout at the play pin> --work-dir <W> --device cuda
python designs/research_state/measurements/x72_swap_plateau_2026-11-25/x72_rule.py decide-round K <W>/h.json
python -m main.x72.floor read --round K --mark 5M    # also 10M, 15M, end; --seed (repeatable) narrows it; --out F
python -m main.x72.probe argv --round K --seed 7111  # also 7112: the responder's trainer argv (K = the READ round)
python -m main.x72.probe record-check --seed 7111    # X71's G6 on the responder's record, while it trains
python -m main.x72.probe drive --round K --play-dir <checkout at X71's play commit, Rust built> --work-dir <W2>
```

| tool | exit codes |
|---|---|
| `finals` | 0 done; 3 refused (a file missing, a sha256 or a record that does not verify, an archive that already holds another file) |
| `cross drive` | X70's driver's: 0 read; 2 usage; 3 the plan or the family registration refused; 4 play failed (re-run resumes); 5 HOLD (the pin checkout does not verify, or the work dir already holds its ONE read); 6 `INCONCLUSIVE` (a cell missing or not exactly 4,000 pairs: `h.json` carries an empty matrix, which the frozen rule answers `INCONCLUSIVE` / HOLD); 7 the rows name another venue, pin or seed |
| `floor read` | 0 no lineage under the floor; **10 FIRED** (at least one statistic under it); 5 NOT-READY (the mark is not written yet); 3 refused; 2 usage |
| `probe drive` | X71's driver's: 0 a verdict; 2 usage; 3 refused (a file or record is not the registered one; the player refused); 4 a cell's play failed (re-run resumes); 5 HOLD (the play checkout, the work dir, the lock); 6 the verdict is `INCONCLUSIVE`; 8 a cell ran past its bound |

## Rules a caller must know

- **The cross needs the GPU lease** on `--device cuda` (it runs under `scripts/ops/gpu_lock.sh`); the plan and the
  read run on this checkout's code, the games on the play pin's.
- **The floor reader is read-only and safe on a live run.** A launch's last step carries two update rows
  (`designs/training/telemetry_scalars.md`): the reader keeps the LATER-written one, the registered `clean()`
  behaviour. The comparison is `x72_rule.py floor`'s: it fires only when a statistic is under the floor by more
  than 1e-9 nats; a value within that band fires nothing and is recorded. A mark that is not fully on disk yet is
  NOT-READY, never compared.
- **The probe's three cells are played at X71's play commit**, read from X71's record (`record/read/play_commit`),
  not at the cross's play pin: the two differ in game code, and the twelve banked cells and the three new ones must
  be one game. The driver HOLDs at any other commit. Its cells run on CPU (the Rust eval core), nice 19 on the agent
  cores, each under `scripts/ops/mem_cap.sh`; the work dir must be outside every checkout and `models/`.
- **A rehearsal is never a result.** `cross drive --rehearsal-pairs N` writes its matrix where the rule does not
  read it; `probe drive --rehearsal-games N` writes counts and no verdict. Both exist to check the tool path on CPU.
- **Each verdict file is written once.** A second `drive` on a finished work dir plays nothing.

## Verified by execution, and not

- **Executed on CPU:** `main.h2h play-many` at the play pin loads an archive copy; the cross end to end at two
  pairs a cell on stand-in finals; the floor reader on the three banked swap runs (it reproduces CE-3d's recorded
  end entropies exactly) and on a live run; the probe's player at X71's play commit replays X71's banked
  `swap` / `r7111` cell game for game (80 of 80 rows identical), and the probe end to end at four games a team on
  stand-in responders.
- **UNVERIFIED:** the cross on `cuda` (no lease was held); every record guard on a REAL X72 run (none exists yet: a
  false refusal is a HOLD, never a wrong read); a launcher dry run of the responder argv with an archive copy as
  `--model` (the registration's §16.7 owns it).
- **Not built:** the window chain (§16 is its checklist); §9's two reported end crosses; an import path for banked
  Kakuna cells in `b1_kakuna` (not registered: §8.5).
