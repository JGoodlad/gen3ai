# The replay corpus: how it grows, why it stops, and the guard

**What it is.** The project's own copy of the PUBLIC gen3ou replay logs, `<main checkout>/replays/showdown/gen3ou/<YYYY-MM-DD>/battle-gen3ou-<id>.log`
(gitignored; the date folder is the America/Los_Angeles date of the battle start). It has TWO populations: 376,410 spectator logs, 2026-05-18
to 2026-09-23 12:07 PDT (listed in `SPECTATOR_MANIFEST_2026-09-23.txt` in the corpus root), and the archive pull since 2026-10-20 (below). Readers: `python -m main.live.replay_scan` (the Rust reader's bulk drift scan), `main.ladder_drift_scan`, the X50 census
(`research_state/measurements/x50_replay_census_2026-10-20/`) and ai_v11's human-replay design.

🚨 **It is NOT under `data/`.** `data/replays/` is an EMPTY directory (nothing tracked, nothing written since 2026-05-12); the corpus lives under the
main checkout's top-level `replays/`. Root rule 6 (no `data/` change while a pinned run is live) is therefore not touched by any replay write.

## Status (2026-10-24; replace this section when the state changes)

**BACKFILL DONE 2026-10-21 06:47 UTC** (`BACKFILL_DONE.json` in the corpus root: all three phases, 3,652 replays written); every date folder through 2026-10-24 exists and `gen3ai-replay-daily.timer` keeps it growing. The history below is the 2026-10-20 restart.

**RESTARTED 2026-10-20 from the public replay ARCHIVE, through the owner's SOCKS5 proxy (owner: "Sure use the proxy for replay download",
2026-10-20; download only, no play, login or chat), supervised by systemd user units (§ Supervision).** The backfill runs the weeks since the
spectator stopped, then the 08-03..08-17 hole, then the 06-16..06-20 hole; the daily timer keeps it growing.

### The archive is not the spectator (measured 2026-10-20, through the proxy, 50 replays of 2026-09-22 into a scratch dir)

| check | result |
|---|---|
| log content vs the spectator's file of the same id | 50 / 50 identical from the first `\|j\|` line to `\|win\|`. The spectator file starts with `\|init\|battle` + `\|title\|`, which the archive lacks; the archive adds lines AFTER `\|win\|`: the two `\|raw\|` rating changes (50 / 50) and sometimes `\|l\|`, an empty `\|player\|pN\|` and post-game `\|c\|` chat (13 / 50). |
| the Rust reader on the archive logs | `main.live.replay_scan --corpus <scratch>`: 50 / 50 clean, 22,028 protocol lines, no refusal, PASS |
| date folder (`\|t:\|`, Los Angeles) | 50 / 50 in the same folder as the spectator's file |
| `search.json` depth | `before=` reaches 2020-09 (full pages at 2025-06 and 2020-09): no depth limit in our window |
| volume | **about 250-450 uploaded replays a day** (51-row pages span 2.7 h on 09-23, 4.8 h on 10-20), against ~3,450 a day the spectator saw: the archive holds UPLOADED replays only, roughly a tenth. 45 of 51 archive ids of 2026-05-22 are in the spectator set. **Measured 2026-10-21 (`research_state/measurements/replay_archive_bias_2026-10-21/`): the uploaded tenth is SKEWED, not random.** Against the spectator's neighbouring days, archive-only days have +35 / +101 / +128 Elo higher mean pair rating, 26-41 % more turns, 10-20 pp fewer forfeit endings and 6-11 pp fewer rated games (three windows, drift control within the bars), and a few uploaders supply 2-10 % of all sides each (top account). Never pool the two populations or treat the archive as a ladder sample; cap per player. The uploader's own result is not recoverable from a log, and the overlap contrast is impossible on disk (`sync.py` skips every id already present, so a spectator-seen battle has no archive copy). |

So the corpus after 2026-09-23 is a thinner, differently selected sample: a per-day rate or a census across the seam compares two mechanisms.
The backfill is ~10k replays (27 days) plus ~6k in the holes, ~2 requests each at the 1 s floor: hours, not the 46 h the spectator rate implied.

## Supervision (installed 2026-10-20)

Unit files live in `tools/replay_corpus_downloader/systemd/` (reference copies; installed in `~/.config/systemd/user/`); the driver is
`python -m tools.replay_corpus_downloader.service {backfill,daily}`, over `sync.py`. Both run the MAIN checkout's code (`WorkingDirectory`, `PYTHONPATH`),
write only under `<main>/replays/showdown/gen3ou/`, at `Nice=19`, `CPUAffinity=5-7 13-15`, idle I/O, `MemoryMax=2G`, one request a second.

| unit | what | stop it |
|---|---|---|
| `gen3ai-replay-backfill.service` | the phases in 4,000-replay chunks; `Restart=on-failure` every 15 min (a dead proxy, a persistent 429 / 5xx); writes `BACKFILL_DONE.json` in the corpus root when done, and `ConditionPathExists=!` keeps it from running again | `systemctl --user disable --now gen3ai-replay-backfill.service` |
| `gen3ai-replay-daily.timer` → `.service` | 05:30 Los Angeles daily, `Persistent=true`: pulls since the newest folder minus a day (≤ 6,000), then the freshness alarm | `systemctl --user disable --now gen3ai-replay-daily.timer` |

- **Every request goes through the proxy** (`--proxy socks5h://127.0.0.1:1080`, the `proxy-tunnel.service` SSH tunnel `main.play --proxy` uses; no
  credential in it). It fails CLOSED: an unreachable proxy exits 3 before any request, a proxy lost mid-run is a persistent failure (exit 1), and the
  session ignores `*_PROXY` / `NO_PROXY`, so nothing reroutes it direct (`proxy_test.py` runs it against a fake SOCKS5 server).
- **One downloader at a time:** an `flock` on `$XDG_RUNTIME_DIR/gen3ai_replay_corpus_downloader.lock`; a second one exits 75. The daily run defers to
  a running backfill (skips the pull, still checks freshness).
- **The alarm is logged** in the daily unit's journal: a STALE corpus (newest folder > 3 days old) exits 1, so the unit shows in
  `systemctl --user --failed`, and the line `[freshness] REPLAY CORPUS STALE: …` is at priority err:
  `journalctl --user -u gen3ai-replay-daily.service -p err`. Progress: `journalctl --user -u gen3ai-replay-backfill.service` (one line a chunk).
- **The units start with the user manager.** `Linger=no` for `goodlad`: after a reboot they run once the owner logs in (as every other user unit
  here); enabling linger is the owner's call.

## Why it stopped three times (the spectator, diagnosed 2026-10-20)

| fact | evidence |
|---|---|
| The corpus was produced by a LIVE SPECTATOR daemon, `src/main/collect_replays.py` (poke-env's `BattleSpectator`, a Rich dashboard, optional SOCKS5 proxy), run BY HAND in a tmux window. No cron entry, no systemd unit, no supervisor. | `git show 5d85d8a20^:src/main/collect_replays.py`; added `77ef01cac` 2026-05-18, dated subfolders `03abef85f` 2026-05-25, stuck-room reaper `5d77790d9` 2026-07-05. `crontab -l` holds only the two `tmp_sweep` lines; `systemctl --user list-timers` has no replay timer. |
| It is DELETED (T27, poke-env retired) and cannot be revived without poke-env. | `5d85d8a20` 2026-10-06 |
| Its last file is 2026-09-23 12:07:58; the box REBOOTED at 12:08:00 (a clean `systemd-reboot`, journal boot `-2` ends 12:08:00) and the tmux session was recreated by the 2026-10-04 reboot. | `ls -l --time-style=full-iso`; `journalctl --list-boots`; `journalctl -b -2` tail |
| **Cause of THIS stop: a reboot killed the unsupervised tmux window, and nobody restarted it.** | the last file and the reboot are 2 seconds apart |
| The previous stop (the census's "again"; ai_v11's D4) has the SAME cause: last file 2026-08-02 20:22:58, clean reboot at 20:23:12, restarted by hand 2026-08-18 21:20. | same sources; the gap folders 08-03..08-17 are missing |
| A THIRD hole, 2026-06-16..06-20: last file 06-15 08:07, restarted 06-21 16:30. No reboot, no OOM near 06-15 08:07 (an OOM kill hit the user slice 06-16 20:04, a day after the last file). **UNVERIFIED:** a wedge (rooms that never finish held every spectator slot); the reaper that fixes that class landed 2026-07-05. | `journalctl -b -4` around 06-15 |
| The documented "last run 2026-08-02" (GCP_INFRASTRUCTURE, the deletion commit) was only the first stop; the collector was restarted 2026-08-18 and ran to 2026-09-23. | file mtimes |


### Does a restart touch the toy runs? No, and here is the evidence

- The restart writes ONLY under `<main>/replays/showdown/gen3ou/`. It writes nothing under `data/`.
- The X35 toy runs are pinned at `4609c155` (before `3d6780ee`, so they read `data/` from the main checkout's CWD). Their trainer is
  `/tmp/launcher-4609c155-*/src/main/train_rl_agent.py`. In that tree the only non-test modules that name the corpus are `main/live/replay_scan.py`,
  `main/ladder_drift_scan.py` and `agents/training/team_completion/replay_parser.py`; none is imported from the training, eval, snapshot-ladder,
  search or launcher code (grep for the module names and for `replays` over the pin's non-test `src/`), and the pin's own `data/` tree has no `replays/`.
  (The code references are an offline scan and a CLI default.)
- Conclusion: the pinned runs never read the replay corpus, by path or by import. A replay write is safe under the data freeze **now**.
- The only readers a growing corpus changes are the OFFLINE ones, and the pull writes into old date folders too (the 09-23 folder, the two holes):
  a date range no longer names a set. **Name the population:** the spectator set is the manifest's 376,410 paths (the X50 census's set); the rest
  is the archive pull.
