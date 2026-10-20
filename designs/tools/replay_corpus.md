# The replay corpus: how it grows, why it stops, and the guard

**What it is.** The project's own copy of the PUBLIC gen3ou replay logs, `<main checkout>/replays/showdown/gen3ou/<YYYY-MM-DD>/battle-gen3ou-<id>.log`
(gitignored; 376,410 logs in 109 date folders, 2026-05-18 to 2026-09-23; ~4 GB; the date folder is the America/Los_Angeles date of the battle
start). Readers: `python -m main.live.replay_scan` (the Rust reader's bulk drift scan), `main.ladder_drift_scan`, the X50 census
(`research_state/measurements/x50_replay_census_2026-10-20/`) and ai_v11's human-replay design.

🚨 **It is NOT under `data/`.** `data/replays/` is an EMPTY directory (nothing tracked, nothing written since 2026-05-12); the corpus lives under the
main checkout's top-level `replays/`. Root rule 6 (no `data/` change while a pinned run is live) is therefore not touched by any replay write.

## Status (diagnosed 2026-10-20, read-only; replace this section when the state changes)

**STOPPED since 2026-09-23 12:07 PDT (27 days at the diagnosis). Nothing is scheduled; nothing is running.**

| fact | evidence |
|---|---|
| The corpus was produced by a LIVE SPECTATOR daemon, `src/main/collect_replays.py` (poke-env's `BattleSpectator`, a Rich dashboard, optional SOCKS5 proxy), run BY HAND in a tmux window. No cron entry, no systemd unit, no supervisor. | `git show 5d85d8a20^:src/main/collect_replays.py`; added `77ef01cac` 2026-05-18, dated subfolders `03abef85f` 2026-05-25, stuck-room reaper `5d77790d9` 2026-07-05. `crontab -l` holds only the two `tmp_sweep` lines; `systemctl --user list-timers` has no replay timer. |
| It is DELETED (T27, poke-env retired) and cannot be revived without poke-env. | `5d85d8a20` 2026-10-06 |
| Its last file is 2026-09-23 12:07:58; the box REBOOTED at 12:08:00 (a clean `systemd-reboot`, journal boot `-2` ends 12:08:00) and the tmux session was recreated by the 2026-10-04 reboot. | `ls -l --time-style=full-iso`; `journalctl --list-boots`; `journalctl -b -2` tail |
| **Cause of THIS stop: a reboot killed the unsupervised tmux window, and nobody restarted it.** | the last file and the reboot are 2 seconds apart |
| The previous stop (the census's "again"; ai_v11's D4) has the SAME cause: last file 2026-08-02 20:22:58, clean reboot at 20:23:12, restarted by hand 2026-08-18 21:20. | same sources; the gap folders 08-03..08-17 are missing |
| A THIRD hole, 2026-06-16..06-20: last file 06-15 08:07, restarted 06-21 16:30. No reboot, no OOM near 06-15 08:07 (an OOM kill hit the user slice 06-16 20:04, a day after the last file). **UNVERIFIED:** a wedge (rooms that never finish held every spectator slot); the reaper that fixes that class landed 2026-07-05. | `journalctl -b -4` around 06-15 |
| The documented "last run 2026-08-02" (GCP_INFRASTRUCTURE, the deletion commit) was only the first stop; the collector was restarted 2026-08-18 and ran to 2026-09-23. | file mtimes |

**The replacement tool is a different mechanism.** `tools/replay_corpus_downloader/sync.py` pulls the public replay ARCHIVE (`search.json` and
`<id>.log`), one request at a time. It has never been run against the public server. It is a manual command, not a service. Differences from
the collector that matter: the archive holds UPLOADED replays, the collector saw every public battle finish. **UNVERIFIED:** how deep `search.json`
paginates, that rated ladder battles are all uploaded, and that an archive log equals a spectator log byte for byte (the Rust reader's gate (b) ran
on spectator logs).

### Does a restart touch the toy runs? No, and here is the evidence

- The restart writes ONLY under `<main>/replays/showdown/gen3ou/`. It writes nothing under `data/`.
- The X35 toy runs are pinned at `4609c155` (before `3d6780ee`, so they read `data/` from the main checkout's CWD). Their trainer is
  `/tmp/launcher-4609c155-*/src/main/train_rl_agent.py`. In that tree the only non-test modules that name the corpus are `main/live/replay_scan.py`,
  `main/ladder_drift_scan.py` and `agents/training/team_completion/replay_parser.py`; none is imported from the training, eval, snapshot-ladder,
  search or launcher code (grep for the module names and for `replays` over the pin's non-test `src/`), and the pin's own `data/` tree has no `replays/`.
  (The code references are an offline scan and a CLI default.)
- Conclusion: the pinned runs never read the replay corpus, by path or by import. A replay write is safe under the data freeze **now**.
- The only readers a growing corpus changes are the OFFLINE ones. A census run while a download is in flight reads a moving set: **name the date
  range** (the X50 census is pinned to the 376,410 logs of 2026-05-18..2026-09-23).

### Restart plan (needs the owner's go: it is network load on a public server; `tools/replay_corpus_downloader/README.md`)

1. **Validate the tool on a SCRATCH dir first** (it has never met the real server): `--out <scratch> --list-only --max-replays 20`, then
   `--since 2026-09-22 --max-replays 60` into the scratch dir. Check, against the corpus' files of the same ids (the 09-22/09-23 overlap), that the
   archive log equals the spectator log (or says how it differs); that the date folder agrees; the pagination depth (does the walk reach 2026-09-23?).
   Run it from the MAIN checkout or pass an ABSOLUTE `--out`: a relative `--out` resolves against the cwd and a worktree run writes a stray `replays/`
   inside the worktree.
2. **Backfill in resumable chunks, detached, `nice -n 19`, owner-approved volume:** `--since 2026-09-23 --max-replays 4000` repeated (about 2.2 h a chunk;
   every id already on disk is skipped). Size: ~83k replays after 09-23 plus ~60k in the 06-16..06-20 and 08-03..08-17 holes (`--since 2026-06-16`)
   at ~3,000 a day, 2 requests a replay at the 1 s floor: ~80 h of wall, ~1.5 GB (437 GB free). Do the recent weeks first; the holes are optional.
3. **Keep it growing, supervised:** a systemd USER timer (the pattern of `gen3ai-prober-web-watchdog.timer`) running the pull once a day, `--since`
   the newest folder minus one day, `--max-replays 6000`, `Nice=19`, with `Persistence=true` so a reboot never loses a day. The timer is a proposal, not
   installed; installing it is the owner's / orchestrator's call.
4. **The alarm (BUILT):** `python -m tools.replay_corpus_downloader.freshness` is read-only and exits 0 fresh / 1 STALE (newest folder older than
   `--max-age-days`, default 3) / 2 no corpus. Wire it into the orchestrator's :07/:37 health check (one line when healthy) or as the timer's
   `ExecStartPost`. It would have fired on 2026-09-27; the gap went unnoticed for 27 days. `main.ops` has no corpus check today.
