# BAIT-LOOP CENSUS — win-prob-critic arm vs shaped-critic comparator

## Commands (from /home/goodlad/dev/gen3ai, main checkout, read-only)
```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m main.prober.query loops models/ai_v12_02_winprob_critic --opponent 'sentinel_*'   # 2.7 s
python3 -m main.prober.query loops models/ai_v9_29_rev1_0823      --opponent 'sentinel_*'   # 4.5 s
```
Both exited 0, no stderr. Raw JSON: `bait_probe/{arm_all,rev1_all}.json`. `sentinel_*` is the
scope the tool's own baseline was measured on. Runs' `model_config.json`: arm `critic="winprob"`,
comparator has no `critic` key (= shaped default).

## Headline rates (tool output; the tool prints {n, d, rate} and NO confidence intervals)
| metric (denominator) | ARM ai_v12_02_winprob_critic | COMP ai_v9_29_rev1_0823 | gen-15 baseline the tool carries |
|---|---|---|---|
| battles matched / skipped | 576 / 0 | 834 / 0 | 843 |
| opp voluntary pivots | 3188 | 6799 | — |
| moved-into pivots | 2726 | 4567 | 4923 |
| **whiff_rate_per_pivot** | 220/2726 = **0.0807** | 677/4567 = **0.1482** | 820/4923 = **0.167** |
| whiff_rate_per_decision | 220/19032 = 0.0116 | 677/30553 = 0.0222 | — |
| **reclick_rate** | 37/220 = **0.1682** | 151/677 = **0.223** | 264 re-clicks = **0.322** |
| **loop_battle_rate** | 16/576 = **0.0278** | 84/834 = **0.1007** | 117/843 = **0.139** |
| loop_ge3_battle_rate | 7/576 = 0.0122 | 31/834 = 0.0372 | 0.062 |
| loop_steps_per_decision | 54/19032 = 0.0028 | 243/30553 = 0.0080 | — |
| worst single loop | 12 (surf→vaporeon, step_10000032/sentinel_1/win_2004) | 8 (toxic→swampert, step_20000016/sentinel_2/win_1002) | — |
| misses (never whiffs) | 62 | 180 | — |
| whiff kinds | immune 210 / fail 9 / near_zero 1 | immune 622 / fail 54 / near_zero 1 | immune 769 / fail 41 / near_zero 10 |
| median turns | 29.0 | 32.0 | — |
| median chosen-prob on loop steps | 0.9065 | 0.915 | 0.963 |
| mirror (CONTROL — they whiff into us) | 351/2913 = 0.1205; loop-battle 37/576 = 0.0642 | 561/4701 = 0.1193; loop-battle 70/834 = 0.0839 | 0.145 |

Win-arm to win-arm (the tool's registered confound 2): ARM wins whiff 66/1080 = 0.0611,
loop-battle 2/240 = 0.0083; COMP wins whiff 317/2123 = 0.1493, loop-battle 44/360 = 0.1222.

## α/β readout on the same pivots
| | ARM | COMP | gen-15 baseline |
|---|---|---|---|
| β slot acc first-time / repeat / loop-step | 0.4966 (220/443) / 0.5430 (518/954) / **0.7200 (36/50)** | 0.4775 (371/777) / 0.6584 (2184/3317) / **0.7363 (148/201)** | 0.52 / 0.659 / 0.821 |
| β species correct | 949/3188 = 0.2977 | 2926/6799 = 0.4304 | — |
| α switch top-1, all pivots / loop steps | 0.8504 / **0.8148 (44/54)** | 0.6163 / **0.6749 (164/243)** | — / 0.762 |
| α switch p (median) on loop steps | 0.7372 | 0.5390 | 0.60 |

Critic Δ on loop steps (median): ARM ΔV −0.0264 and ΔP(win) −0.0264 — **identical, because under
`--critic winprob` V is P(win)**; other_bait −0.0039, other +0.0001. COMP (shaped units) ΔV −1.955
/ ΔP(win) −0.0240; other_bait −1.503 / −0.0271; other −0.053 / −0.0012.

## Arm per-step trend (sentinel_*; 2M and 4M have NO sentinel_* opponents — pool empty, so the series starts at 6M)
| step | battles | pivots | whiff_rate | reclick | loop-battle | loop≥3 | med turns |
|---|---|---|---|---|---|---|---|
| 6.0M | 19 | 52 | 0.0577 | 0.000 | 0/19 | 0/19 | 26.0 |
| 8.0M | 39 | 112 | 0.0446 | 0.000 | 0/39 | 0/39 | 25.0 |
| 10.0M | 60 | 293 | 0.0956 | 0.393 | 1/60 = 0.0167 | 1/60 | 30.0 |
| 12.0M | 78 | 353 | 0.0595 | 0.048 | 1/78 = 0.0128 | 0/78 | 30.0 |
| 14.0M | 94 | 493 | 0.0669 | 0.091 | 2/94 = 0.0213 | 1/94 | 28.5 |
| 16.0M | 95 | 455 | 0.0989 | 0.222 | 6/95 = 0.0632 | 2/95 | 30.0 |
| 18.0M | 95 | 475 | 0.0926 | 0.159 | 5/95 = 0.0526 | 2/95 | 29.0 |
| 20.0M | 96 | 493 | 0.0832 | 0.122 | 1/96 = 0.0104 | 1/96 | 29.0 |

Comparator over its 11 steps (4M–24M) for reference: whiff_rate 0.087 → 0.105 → 0.155 → 0.119 →
0.110 → 0.132 → 0.173 → 0.141 → 0.187 → 0.157 → 0.159; loop-battle rate peaks 17/92 = 0.185 at 20M.

## Refusals / flags the tool printed
- **Nothing refused.** `coverage` = 576 matched / **0 skipped** (arm) and 834 / **0 skipped** (comp) —
  no undecidable sides, no missing `*_replay.html`. No SELECTION-UNKNOWN or reconstruction-sibling
  warning applies (`loops` is model-free and does not use `*_reconstruction.json`).
- Six standing caveats ride the result: the gen-15 baseline is a REFERENCE POINT NOT A TARGET and
  must be read at matched scope; confound 1 = loop_battle_rate rises with game LENGTH (read the
  per-pivot / per-decision rates first); confound 2 = loops concentrate in WINNING positions, so
  compare win-arm to win-arm; a MISS is never a whiff; β slot accuracy is decidable only for
  already-revealed arrivals so its denominator skews toward REPEAT pivots; the mirror block is a
  CONTROL measuring the opponent (a frozen self in a sentinel matchup).
- Baseline carried internally: gen-15, `ai_v9_18_gen15_v8rewards_0818`, measured 2026-08-19, scope
  `sentinel_*` every step, 843 battles / 4923 moved-into pivots. Source cited by the tool:
  `designs/research_state/ledger.md (2026-08-19)`; `designs/research_state/bait_loop_hunt.md`.
- The arm has fewer sentinel battles (576 vs 834) and a shorter step range (6M–20M vs 4M–24M): it is
  a live run and its self-play pool is empty before 6M.

All numbers above are as printed by `main.prober.query loops`; I computed nothing from raw files.
