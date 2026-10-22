# Is the public replay archive a fair sample of gen3ou ladder play? (2026-10-21)

## Registration (written BEFORE any statistic was computed; only log formats were inspected)

**The design the brief asked for is not possible on this disk.** The brief: take the archive battles whose id is also in
the spectator set (the overlap, = "uploaded") and compare them with the full spectator set ("all"). But
`tools/replay_corpus_downloader/sync.py` SKIPS every id already on disk, so the archive copy of an overlapping battle was
never written: the corpus holds no marker of which spectator battles were uploaded (checked 2026-10-21: of the
376,410 manifest paths exactly 0 are missing from disk and the archive-only files sit only in the days the spectator
was down, plus 09-23 afternoon and everything after). Rebuilding the overlap needs the archive's `search.json`, i.e. the
network, which the brief forbids. So the question is answered by a CONTRAST ACROSS PERIODS instead:

* **A (archive-only battles)** = the archive pull in the days the spectator was down, so no spectator copy exists:
  **W1** 2026-06-16..06-20, **W2** 2026-08-03..08-17, **W3** 2026-09-24..10-20 (date folders, Los Angeles).
* **S (spectator battles, "all")** = the comparator days next to each A window, full days only:
  W1: 06-10..06-14 + 06-22..06-26; W2: 07-19..08-01 + 08-19..09-01; W3: 08-27..09-22 (before only: the
  spectator ended 09-23).
* **Drift control (same machinery, S against S):** W1/W2 S-before vs S-after; W3 first half vs second half of the
  comparator days. It shows how far two windows of the SAME collector drift apart in time, which is the confound of
  this design.

**Not answerable from the logs:** whether the uploader won. A log names no uploader (the archive's rows carry none).
Reported instead: seat-1 win rate and "higher-rated side won" (a replay saved by the winner would show as an excess).

**Dimensions and bars (an effect is "within the bar" only if its whole bootstrap CI is inside it; the bar is a
pre-registered equivalence margin, not a significance test).** Games are public ladder gen3ou logs; "rated" = a
`|rated|` line; ratings are the `|player|` header ratings.

| dimension (A minus S) | bar | interval |
|---|---|---|
| rated share | ±3 pp | 95 % |
| rated games: mean of the pair's ratings; the higher-rated and the lower-rated player | ±20 Elo each | 95 % |
| turns: log-mean ratio | ±5 % | 95 % |
| endings (completed games): forfeit, timer ("lost due to inactivity"), normal win, tie | ±2 pp each | 95 % |
| no-result logs (no `\|win\|`/`\|tie\|`): a collection artifact, reported not judged | n/a | |
| top-30 species (share of sides on which the species is revealed) | ±1.5 pp each | 99.8 % (30 comparisons) |
| top-10 lead species (share of sides) | ±1.5 pp each | 99.5 % |
| Spikes used by either side | ±2 pp | 95 % |
| seat-1 wins; higher-rated side won (rated, unequal ratings) | ±2 pp each | 95 % |

All statistics on COMPLETED games only (a `|win|` or `|tie|`), except the no-result share. The bootstrap resamples
battles within each group (2,000 draws, fixed seed).

**Verdict rule.** FAIR only if in ALL three windows every dimension is within its bar AND the drift control is within
its bars too (otherwise drift would hide a bias). SKEWED if a dimension's CI lies wholly outside its bar, on the same
side, in at least two of the three windows and the drift control of those windows does not reproduce it (named
dimensions). Otherwise CAN'T TELL. Whatever it is, it speaks for the archive-only periods against the neighbouring
spectator days, NOT for the overlap design the brief named.

## Verdict: SKEWED (by the registered rule), for the archive-only periods against the neighbouring spectator days

The uploaded subset is NOT a fair sample of public gen3ou ladder play. It over-represents stronger players and longer
games and under-represents early forfeits, in all three independent windows, with effects about 2 to 13 times the bars and far
beyond the drift control. Numbers are A (archive-only) minus S (spectator days next to it), 95 % battle-bootstrap
intervals; the full tables (every species, lead, the drift control) are `results/results.md`.

| dimension (bar) | W1 06-16..20 (n=1,240) | W2 08-03..17 (n=3,452) | W3 09-24..10-20 (n=10,340) | drift control (S vs S) |
|---|---|---|---|---|
| rated share (±3 pp) | -11.1 [-12.9, -9.3] pp | -6.1 [-7.0, -5.2] pp | -9.0 [-9.6, -8.4] pp | within in all three |
| mean pair rating (±20 Elo) | +35 [+24, +45] | +101 [+94, +107] | +128 [+125, +132] | -11, +17, +1 Elo |
| turns, log-mean ratio (±5 %) | +28.8 [+23.5, +34.1] % | +25.6 [+22.9, +28.6] % | +41.1 [+39.1, +43.0] % | -1.7, -2.2, +1.4 % |
| forfeit share (±2 pp) | -10.5 [-13.4, -7.8] pp | -14.1 [-15.9, -12.4] pp | -19.6 [-20.6, -18.6] pp | +1.8, +1.1, +1.3 pp |
| normal-win (KO) share (±2 pp) | +13.1 [+10.3, +16.0] pp | +14.8 [+13.1, +16.5] pp | +22.1 [+21.1, +23.1] pp | -0.9, -1.6, -0.9 pp |
| timer-loss share (±2 pp) | -2.8 [-3.8, -1.8] pp (straddles) | -0.8 [-1.5, -0.1] pp (within) | -2.6 [-2.9, -2.2] pp | within |
| Spikes used (±2 pp) | +8.9 [+6.2, +11.6] pp | +3.6 [+2.0, +5.3] pp (straddles) | +20.6 [+19.7, +21.5] pp | -1.3, -2.0, +0.1 pp |
| seat-1 wins (±2 pp) | -0.5 [-3.2, +2.2] (straddles) | +1.8 [+0.1, +3.5] (straddles) | -1.3 [-2.3, -0.3] (straddles) | within |
| higher-rated side won (±2 pp) | -4.2 [-7.2, -1.2] (straddles) | -4.2 [-5.9, -2.4] OUTSIDE | -1.2 [-2.3, -0.1] (straddles) | within |

Dimensions the rule names as SKEWED (outside the bar on the same side in at least two windows, not reproduced by the
drift control): rated share, all three rating cuts, turns, forfeit share, KO share, Spikes (W1, W3), and the species
Skarmory (+3 to +18 pp), Swampert (+4 to +20 pp) and Tyranitar (W2, W3). Species and lead effects beyond those swing in
sign or size across windows and are mostly a few players' teams (below); do not read them as stable. Ties and the no-result
share are within or undefined (the corpus holds no log without a `|win|` or `|tie|`, in either collection).

**What carries it.** A is more experienced (+35, +101, +128 Elo), plays longer games (+26 to +41 %), and almost never ends
in an early forfeit (34 to 46 % of its games against 53 to 56 % of the spectator's). The band-matched table in
`results/results.md` shows the forfeit and length gaps persist inside every rating band (for example 1400-1599 in W3: 30.7 %
vs 47.5 % forfeits, 35.1 vs 29.7 turns), so rating alone does not explain them: a replay is saved after a game that was
worth saving (a played-out game), not after an early walk-over.

### Post-hoc diagnostics (not registered; explanation only)

* **A few accounts supply a large share of the archive.** The most frequent player is 2.3 % (W1), 4.6 % (W2) and 10.1 %
  (W3) of all archive sides, against 0.5 %, 0.9 %, 1.3 % in S; the top 10 are 12.7 %, 20.9 %, 30.4 % (S: 4 to 5.5 %). In
  W3 the weekly top-player share is 23 %, 13 %, 22 %, 7 %, and the Zapdos lead share falls 38 % → 41 % → 33 % → 18 % with it,
  so the W3 species and Spikes shifts are one or a few uploaders' teams. The battle-level bootstrap treats games as
  independent and so UNDERSTATES the uncertainty of every species number (a player-clustered bootstrap was not run).
* **Without those accounts the headline skew stays.** Dropping every game of the 10 most frequent archive players:
  forfeit share -11.4 / -12.3 / -11.3 pp and turns +36 / +34 / +38 % (W1/W2/W3) hold. The rating gap shrinks to
  +69 / +63 / +16 Elo and the rated-share gap widens (-15 / -11 / -25 pp: the heavy uploaders play rated games).
* **Reading the drift control.** W2's S-before vs S-after differ by 17 Elo and the W3 halves by 1 Elo, so time drift in
  the spectator alone cannot make 35 to 128 Elo. W1's S is the 06-10..06-26 span around a 6-day hole.

### What this does not say

* It is NOT the overlap contrast the brief asked for (see Registration): A and S are different days, the archive's
  coverage of the spectator's days is unmeasured, and the spectator ("all") is itself a sample of the ladder (it
  records only finished rooms; it saw ~3,500 battles a day).
* W3 compares days up to a month after the spectator ended with the 4 weeks before it; a metagame or ladder change at
  the seam would look like skew (the control cannot see across the seam). W1 and W2 sit inside the spectator's span and agree
  in direction and size order (the rating gap differs: 35 vs 101 Elo).
* **The uploader cannot be inferred**, so "did the uploader win" is unanswered. Seat-1 wins are 48 to 52 % in A and S and
  "the higher-rated side won" is 52 to 55 % in A against 56 to 57 % in S: if anything, uploaded games are LESS often
  won by the favourite, which does not point at winner-biased saving, but it is not the uploader's result.

## Findings (hazards, for the owner)

1. The brief's overlap design is impossible on this corpus: the downloader skips every id already on disk, so no archive copy
   of a spectator-seen battle exists, and the manifest cannot tell which spectator battles were uploaded. Rebuilding it
   needs the archive's `search.json` ids (the network) for 05-18..09-23, or a pull into a scratch dir that does NOT skip.
2. Any rate or census that mixes the spectator set and the archive pull mixes a skewed sample (this README) into a
   different population; the X50 census (spectator only) is unaffected.
3. A few uploaders dominate the archive (top account 10 % of W3 sides): every species/team statistic from the archive
   needs a per-player cap or a clustered interval.

## Reproduce

```
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
D=designs/research_state/measurements/replay_archive_bias_2026-10-21
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot 6 $PY -I $D/rab.py extract --cache /tmp/rab_cache --workers 3   # ~25 s, 106 date folders
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot 16 $PY -I $D/rab.py analyze --cache /tmp/rab_cache --out $D/results  # ~1 min
```

Read-only on the corpus (`<main>/replays/showdown/gen3ou`, override `$RAB_CORPUS`); the cache holds one small jsonl per
date folder. One pooled top-30 species list and top-10 lead list (over all three windows) is used in every window.
Run at corpus state 2026-10-21 (the archive backfill finished 06:47 UTC that day; later daily pulls add to W3's end).
