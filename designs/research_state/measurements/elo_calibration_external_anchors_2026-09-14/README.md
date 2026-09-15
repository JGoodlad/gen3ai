# A LARGE ELO CALIBRATION — two EXTERNAL anchors for the bot-anchored scale

**2026-09-14 · pre-registered in [`PREDICTION.md`](PREDICTION.md), committed before the first game
(`d655843f`).** 4,000 games, greedy-vs-greedy on the HOME team set (our 719-team gen3ou pool, both
sides), our pinned Showdown `e0551883f`, Metamon @ `0a00a759`, CPU only, role-balanced inside every
cell, regime VERIFIED per decision on both sides. **39/39 cells `status: OK`**, every one with
`their_argmax_match_rates == [1.0]` and `team_source_asymmetry == false`; 0 ties, 15 games at the
250-turn cap out of 4,000.

---

## 0. The answer, in one paragraph

**The two anchors land where the campaign needed them to — near the frontier, and informative
there.** `metamon:SmallRL` fits at **1940.0 ± 9.5** and `metamon:SyntheticRLV2` at
**1983.4 ± 9.7** on our bot-anchored scale, straddling `ai_v12_02_winprob_critic`'s 74M node
(1984.6) and sitting just under `ai_v13_01_flywheel_shaped`'s 72M node (2034.6). Against our ten
snapshots the median cell is **0.070 away from even**; against the nine bots the same statistic is
**0.340**. That is the whole point: *the bots are saturated where the anchors are not.* But the
anchors did **not** buy what the campaign predicted they would buy. Frontier standard errors fell
only **8.6%** (registered: ≥ 15%) and the newest-node inflation did not move at all
(+11.1 → +10.3 and +9.8 → +10.0 Elo). **Precision was never the problem — accuracy was**, and the
evidence for that is the one number the registration did not ask for: the nine bots' individual
edges to `SmallRL` imply ratings spanning **1775 to 2049, a 274-Elo disagreement**, while the fit's
standard error on that node is 9.5. A frame can be precise and wrong at the same time, and this one
is.

| pre-registered | verdict |
|---|---|
| **P1** both anchors within ±100 of their registered value, se ≤ 25 | **PASS** — 1940.0 ± 9.5 / 1983.4 ± 9.7 |
| **P2** the anchors are NOT saturated (median \|wr − 0.5\| < 0.25 over snapshot cells) | **PASS** — 0.070, against 0.340 for the bot cells |
| **P3** frontier se falls ≥ 15% | **FALSIFIED** — it falls 8.6% |
| **P4** mean Δ negative and \|mean Δ\| < 15 Elo | **PASS** — −3.0 over 56 nodes, −8.2 over the ten played |
| **P5** newest-node inflation shrinks ≥ 25% | **FALSIFIED** — −7.2% and +2.0% |
| **P6** bot-edge mean\|err\| does not rise > 0.01 | **PASS** — +0.0000 |
| **P7** every pinned bot moves < 40 Elo when re-fit jointly | **FALSIFIED as registered** (the arm was mis-specified — see §7); **PASS** on the corrected arm, max 15.4 Elo |
| **P8** `SyntheticRLV2` beats `SmallRL` at 0.55–0.75 and the fit agrees | **PASS** — 0.635 [0.566, 0.699], fit agrees |

**Recommendation: adopt `metamon:SmallRL` as a standing anchor edge; do NOT adopt
`SyntheticRLV2` per promotion.** The reasoning, the cost and the exact diff are in §10.

---

## 1. THE FINDING the registration did not ask for — the bot frame is precise and wrong

Each of the nine bots is pinned, so each bot's own 100-game edge to an anchor implies a rating for
that anchor on its own. Nine estimates of one quantity:

| bot | pinned | bot's win rate vs `SmallRL` | ⇒ implied `SmallRL` | vs `SyntheticRLV2` | ⇒ implied `SyntheticRLV2` |
|---|---:|---:|---:|---:|---:|
| `heuristic2` | 1639 | 0.16 | 1927 | 0.16 | 1927 |
| `aggressive_v2` | 1630 | 0.17 | 1906 | 0.13 | 1960 |
| `setup_sweep_v2` | 1619 | 0.17 | 1894 | 0.22 | 1839 |
| `setup_sweep` | 1598 | 0.16 | 1886 | 0.19 | 1850 |
| `heuristic` | 1578 | 0.16 | 1866 | 0.10 | 1959 |
| `staller` | 1571 | 0.06 | **2049** | 0.17 | 1846 |
| `staller_v2` | 1555 | 0.10 | 1937 | 0.08 | 1979 |
| `aggressive` | 1512 | 0.18 | **1775** | 0.17 | **1787** |
| `random` | 1000 | 0.00 | (perfect score — prior-limited) | 0.00 | (perfect score) |

**The eight informative bots disagree by 274 Elo about `SmallRL` and 192 about `SyntheticRLV2`.**
The fitted values (1940.0 / 1983.4) are averages over a frame whose members do not agree, and the
se the fit reports (9.5 / 9.7) is the precision of that average, not the accuracy of the frame.

This is what "saturated" costs, made concrete. A Bradley–Terry edge at p = 0.9 carries Fisher
information ∝ p(1−p) = 0.09 against 0.25 at p = 0.5 — under 40% of the information per game — and,
worse, the logistic is flat there, so a ±2 pp wobble in a bot edge moves the implied rating by tens
of Elo. `staller` and `aggressive` differ by 12 pp against `SmallRL` and that is worth 274 Elo.

It also means the three-way ordering is **not transitive through the bots**: the bot frame alone
places `SmallRL` (mean 1905) *above* `SyntheticRLV2` (mean 1893), while 200 direct games say
`SyntheticRLV2` wins **0.635 [0.566, 0.699]** — a +96 Elo gap in the other direction. With the
direct edge in the fit the order comes out right. Without it, the saturated frame had it backwards.
## 2. How it was measured

| | |
|---|---|
| protocol | **greedy-vs-greedy**, verified per decision on BOTH sides (`their_argmax_match_rates == [1.0]` is a fit precondition, not a footnote) |
| team set | **home** — our 719-team gen3ou pool on both sides; `team_source_asymmetry` false on every included cell |
| games | 100 per cell, role-balanced across two half-series (50 where we challenge, 50 where the peer does) |
| server | ours, pinned `e0551883f`; one Showdown per lane on a 9500–9527 port, started and stopped by PID |
| opponents | Metamon @ `0a00a759`, `SmallRL` ckpt 40 (13.9M params), `SyntheticRLV2` ckpt 48 (200.9M) |
| compute | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, 3 lanes; **one torch thread per process** |
| tooling | `python -m main.anchors` at `1c680877`, run from this worktree so the campaign's imports are pinned |

Everything is reproducible from this directory:

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 run_campaign.py --out <dir> --lanes 3 --games 100          # the 38 cells
python3 run_campaign.py --out <dir> --only h2h --lanes 1 --base-port 9560
python3 fit_joint.py   --cells <dir> --out .                       # the three-arm fit + games.jsonl
python3 report.py      --fit joint_fit.json --h2h <dir>/cells/h2h_SyntheticRLV2__SmallRL/summary.json
```

`fit_joint.py` **refuses to run** unless its single-run fit reproduces
`snapshot_ladder.fit_ladder` to 0.2 Elo — the joint fit is only readable as a delta against the
production one, and that precondition has already caught a real defect (`elo.BOT_ANCHORS_PATH` is
relative, so from the wrong directory the fit silently lost its pins, came out ~700 Elo low, and
reported the newest-node inflation with the **wrong sign**).

## 3. The three arms, and why the middle one is the answer

| arm | bot edges | dense frozen edges | external edges | bots |
|---|---|---|---|---|
| **JOINT-NOEXT** | ✅ | ✅ (within a run) | ❌ | PINNED |
| **JOINT-EXT** | ✅ | ✅ | ✅ | PINNED |
| **JOINT-EXT-FREEBOT** | ✅ | ✅ | ✅ | only `random` pinned |
| **JOINT-EXT-FREEBOT-RR** | ✅ | ✅ | ✅ + the 36-pair bot ROUND ROBIN | only `random` pinned |

A cross-run joint fit moves every rating **by itself**, before any anchor is added — four runs that
have never played each other are joined only through the pinned bots, and pooling them is already a
different object from four separate `ladder.json`s. `JOINT-EXT − JOINT-NOEXT` holds the node set and
the internal edges FIXED and changes only the external edges, so the difference is the anchors and
nothing else.

The fourth arm was added AFTER the data, because the third one's own result showed it was asking
the wrong question — the bots' 86,000-game round robin is what the pins ARE, and a fit that frees
them without it holds them only by `random`. §7's P7 row states the correction and both numbers.

**Nodes** are keyed the ladder's own way, with one change forced by fitting four runs at once: a
snapshot key is RUN-QUALIFIED (`snap:<run>@<step>`), because two of these runs both have a
74,000,016 and a bare step would weld them into one player.

**Every committed `ladder.json` was REFIT at the current recipe first** (UNDERSTANDING rule 24).
Two of the four carry no `eval_sentinel_edges_dropped` stamp; `ai_v12_02_winprob_critic`'s 74M node
reads **2057.3 committed and 1984.2 refit** — the +73 stale-recipe trap. Nothing in this artifact
quotes a committed file.

---

## 4. Where the anchors landed, and what moved

**`ext:metamon:SmallRL` = 1940.0 ± 9.5 · `ext:metamon:SyntheticRLV2` = 1983.4 ± 9.7** on the
bot-anchored scale, against our own frontier at 1984.6 (`ai_v12_02` @ 74M) and 2034.6
(`ai_v13_01` @ 72M). Both inside the registered ±100 bands, both far inside the se ≤ 25 bar.

**The scale barely moved, and that is a real answer rather than a null.** Mean Δ over all 56
snapshot nodes is **−3.0 Elo**; over the ten nodes that actually played an anchor it is **−8.2**.
The registered direction (down) is right and the registered magnitude bound (< 15) holds.

**One node moved a lot, and it is the one the mechanism predicts.**
`ai_v12_11_ladder_ctrl10M` @ 10M fell **−42.4 Elo**, six times the median move. It is the
least-connected node in the campaign — a 4-node ladder with 6 dense pairs against the 190 that
`ai_v13_01` and `ai_v12_02` each carry — so it was the node whose absolute level rested most
heavily on saturated bot edges, and it is the node an unsaturated edge corrects most. Its se also
falls furthest (16.7 → 13.4). **A short control run is exactly where the bot-only scale is least
trustworthy**, which matters because the ladder of 10M control arms is a comparator this
project uses constantly.

## 5. P3 and P5 — what the anchors did NOT buy, and why that is the interesting half

**P3 (precision) FALSIFIED: mean se over the ten played nodes falls 10.04 → 9.18, −8.6%, against a
registered bar of ≥ 15%.** **P5 (newest-node inflation) FALSIFIED: it does not move** — +11.1 →
+10.3 Elo on `ai_v12_02` and +9.8 → **+10.0** on `ai_v13_01`.

Both failures have the same cause and it is arithmetic, not measurement. A frontier node in this
fit already carries roughly **25 bot edges at ~500 games each plus ~19 dense frozen edges at 100**;
this campaign adds **two edges at 100 games**. Two edges against forty-four cannot move a standard
error by 15% however informative they are — and the newest-node inflation is a property of **BT
re-solving as nodes are ADDED**, which is a structural feature of the dense internal matrix that an
external edge does not touch.

**So the registered theory of the problem was wrong in an instructive way.** The campaign was
designed on "saturated anchors ⇒ imprecise frontier ⇒ inflation", and the data say the frontier is
*precise* (se ≈ 9 Elo) and *possibly mis-levelled* (274 Elo of disagreement among the nine bot
edges about one external opponent). Two external edges fix neither the precision (they are
outnumbered) nor the inflation (wrong mechanism) — what they fix is the thing nothing else could
check: **whether the frame is pointing at the right place at all.** That is a validity instrument,
not a variance instrument, and it should be used as one.

## 6. P6 — fit quality, and the residual that was pre-registered to be allowed to rise

| edge family | JOINT-NOEXT mean\|err\| | max | JOINT-EXT mean\|err\| | max |
|---|---:|---:|---:|---:|
| bot (each snapshot's historical edges) | 0.0239 | 0.1394 | **0.0239** | 0.1493 |
| dense (frozen snapshot pairs) | 0.0346 | 0.1682 | 0.0352 | 0.1716 |
| **external (this campaign)** | — | — | **0.0542** | 0.1521 |
| ALL | 0.0290 | 0.1682 | 0.0302 | 0.1716 |

The bot-edge error is **unchanged to four decimals** — the anchors are not distorting the existing
frame. But the external edges are the **worst-fitting family in the whole model** (0.0542, more
than double the bot edges' 0.0239), which is exactly the shape §1 predicts: a scalar rating cannot
simultaneously satisfy nine bot edges that disagree by 274 Elo and ten snapshot edges that are all
near even. **The non-transitivity was always there; the external anchor is the first instrument
that makes it visible**, because a saturated edge can absorb a large rating error inside a small
win-rate error and a near-even one cannot.


---

## 7. Every pre-registered row, evaluated in code

The verdicts below are computed by `report.py` against the thresholds in `PREDICTION.md`, so a prediction cannot quietly become a description of whatever happened.

### Every cell

| our side | anchor | n | W/L/T | win rate (ours) | Wilson 95% | implied Elo gap | mean turns | caps | in fit |
|---|---|---:|---|---:|---|---:|---:|---:|---|
| `bot:random` | SmallRL | 100 | 0/100/0 | 0.000 | [0.000, 0.037] | -2400 | 32.9 | 1 | yes |
| `bot:heuristic` | SmallRL | 100 | 16/84/0 | 0.160 | [0.101, 0.244] | -288 | 27.3 | 0 | yes |
| `bot:heuristic2` | SmallRL | 100 | 16/84/0 | 0.160 | [0.101, 0.244] | -288 | 34.3 | 1 | yes |
| `bot:staller` | SmallRL | 100 | 6/93/1 | 0.060 | [0.028, 0.125] | -478 | 27.6 | 0 | yes |
| `bot:staller_v2` | SmallRL | 100 | 10/90/0 | 0.100 | [0.055, 0.174] | -382 | 39.5 | 2 | yes |
| `bot:aggressive` | SmallRL | 100 | 18/81/1 | 0.180 | [0.117, 0.267] | -263 | 25.7 | 1 | yes |
| `bot:aggressive_v2` | SmallRL | 100 | 17/83/0 | 0.170 | [0.109, 0.255] | -275 | 23.6 | 0 | yes |
| `bot:setup_sweep` | SmallRL | 100 | 16/83/1 | 0.160 | [0.101, 0.244] | -288 | 26.1 | 0 | yes |
| `bot:setup_sweep_v2` | SmallRL | 100 | 17/82/1 | 0.170 | [0.109, 0.255] | -275 | 25.9 | 0 | yes |
| `bot:random` | SyntheticRLV2 | 100 | 0/100/0 | 0.000 | [0.000, 0.037] | -2400 | 30.9 | 0 | yes |
| `bot:heuristic` | SyntheticRLV2 | 100 | 10/89/1 | 0.100 | [0.055, 0.174] | -382 | 26.6 | 0 | yes |
| `bot:heuristic2` | SyntheticRLV2 | 100 | 16/84/0 | 0.160 | [0.101, 0.244] | -288 | 35.4 | 1 | yes |
| `bot:staller` | SyntheticRLV2 | 100 | 17/82/1 | 0.170 | [0.109, 0.255] | -275 | 36.6 | 1 | yes |
| `bot:staller_v2` | SyntheticRLV2 | 100 | 8/91/1 | 0.080 | [0.041, 0.150] | -424 | 46.1 | 2 | yes |
| `bot:aggressive` | SyntheticRLV2 | 100 | 17/79/4 | 0.170 | [0.109, 0.255] | -275 | 31.5 | 0 | yes |
| `bot:aggressive_v2` | SyntheticRLV2 | 100 | 13/87/0 | 0.130 | [0.078, 0.210] | -330 | 30.3 | 0 | yes |
| `bot:setup_sweep` | SyntheticRLV2 | 100 | 19/80/1 | 0.190 | [0.125, 0.278] | -252 | 31.8 | 0 | yes |
| `bot:setup_sweep_v2` | SyntheticRLV2 | 100 | 22/77/1 | 0.220 | [0.150, 0.311] | -220 | 26.6 | 0 | yes |
| `snap:ai_v9_29_rev1_0823@2000016` | SmallRL | 100 | 17/83/0 | 0.170 | [0.109, 0.255] | -275 | 39.2 | 1 | yes |
| `snap:ai_v9_29_rev1_0823@8000016` | SmallRL | 100 | 48/52/0 | 0.480 | [0.385, 0.577] | -14 | 45.6 | 1 | yes |
| `snap:ai_v9_29_rev1_0823@24000000` | SmallRL | 100 | 57/43/0 | 0.570 | [0.472, 0.663] | +49 | 43.5 | 1 | yes |
| `snap:ai_v12_11_ladder_ctrl10M@10000032` | SmallRL | 100 | 40/60/0 | 0.400 | [0.309, 0.498] | -70 | 44.5 | 0 | yes |
| `snap:ai_v12_02_winprob_critic@36000000` | SmallRL | 100 | 53/47/0 | 0.530 | [0.433, 0.625] | +21 | 47.0 | 3 | yes |
| `snap:ai_v12_02_winprob_critic@56000016` | SmallRL | 100 | 61/39/0 | 0.610 | [0.512, 0.700] | +78 | 52.6 | 1 | yes |
| `snap:ai_v12_02_winprob_critic@74000016` | SmallRL | 100 | 48/52/0 | 0.480 | [0.385, 0.577] | -14 | 47.0 | 0 | yes |
| `snap:ai_v13_01_flywheel_shaped@22000032` | SmallRL | 100 | 56/43/1 | 0.560 | [0.462, 0.653] | +42 | 44.3 | 0 | yes |
| `snap:ai_v13_01_flywheel_shaped@48000000` | SmallRL | 100 | 54/46/0 | 0.540 | [0.443, 0.634] | +28 | 43.4 | 1 | yes |
| `snap:ai_v13_01_flywheel_shaped@72000000` | SmallRL | 100 | 64/36/0 | 0.640 | [0.542, 0.727] | +100 | 44.1 | 1 | yes |
| `snap:ai_v9_29_rev1_0823@2000016` | SyntheticRLV2 | 100 | 16/83/1 | 0.160 | [0.101, 0.244] | -288 | 35.3 | 0 | yes |
| `snap:ai_v9_29_rev1_0823@8000016` | SyntheticRLV2 | 100 | 30/70/0 | 0.300 | [0.219, 0.396] | -147 | 43.6 | 0 | yes |
| `snap:ai_v9_29_rev1_0823@24000000` | SyntheticRLV2 | 100 | 45/55/0 | 0.450 | [0.356, 0.548] | -35 | 47.6 | 2 | yes |
| `snap:ai_v12_11_ladder_ctrl10M@10000032` | SyntheticRLV2 | 100 | 36/64/0 | 0.360 | [0.273, 0.458] | -100 | 47.6 | 2 | yes |
| `snap:ai_v12_02_winprob_critic@36000000` | SyntheticRLV2 | 100 | 42/58/0 | 0.420 | [0.328, 0.518] | -56 | 47.7 | 1 | yes |
| `snap:ai_v12_02_winprob_critic@56000016` | SyntheticRLV2 | 100 | 58/42/0 | 0.580 | [0.482, 0.672] | +56 | 51.8 | 2 | yes |
| `snap:ai_v12_02_winprob_critic@74000016` | SyntheticRLV2 | 100 | 56/44/0 | 0.560 | [0.462, 0.653] | +42 | 46.1 | 0 | yes |
| `snap:ai_v13_01_flywheel_shaped@22000032` | SyntheticRLV2 | 100 | 55/45/0 | 0.550 | [0.452, 0.644] | +35 | 49.6 | 0 | yes |
| `snap:ai_v13_01_flywheel_shaped@48000000` | SyntheticRLV2 | 100 | 51/49/0 | 0.510 | [0.413, 0.606] | +7 | 46.2 | 1 | yes |
| `snap:ai_v13_01_flywheel_shaped@72000000` | SyntheticRLV2 | 100 | 54/46/0 | 0.540 | [0.443, 0.634] | +28 | 44.4 | 1 | yes |
| `ext:metamon:SyntheticRLV2` | SmallRL | 200 | 127/73/0 | 0.635 | [0.566, 0.699] | +96 | 51.2 | 0 | yes |

39/39 cells entered the fit; 4000 games, 15 ties, 27 at the 250-turn cap.

### P1 — anchor placement

| anchor | registered | fitted (JOINT-EXT) | se | verdict |
|---|---|---:|---:|---|
| `ext:metamon:SmallRL` | 1970 ± 100, se ≤ 25 | **1940.0** | 9.5 | PASS |
| `ext:metamon:SyntheticRLV2` | 1984 ± 100, se ≤ 25 | **1983.4** | 9.7 | PASS |

**P1: PASS**

### P2 — are the anchors saturated too?

median |win rate − 0.5| over the 20 snapshot cells: **0.070** (registered: < 0.15 expected, FALSIFIED at ≥ 0.25)

for comparison, the same statistic over the 16 anchor-vs-bot cells: **0.340**, and our own frontier nodes beat the eight bots at 0.88–0.92 (|wr − 0.5| = 0.38–0.42).

**P2: PASS**

### P3 — frontier precision

| node set | mean se, JOINT-NOEXT | mean se, JOINT-EXT | change |
|---|---:|---:|---:|
| the 10 PLAYED nodes | 10.04 | 9.18 | -8.6% |
| all 56 snapshot nodes | 9.56 | 9.32 | -2.5% |

**P3: **FALSIFIED**** (registered: ≥ 15% reduction on the played nodes; REFUTED if se rises)

### P4 — how far the scale moved

| node | JOINT-NOEXT | JOINT-EXT | Δ | se before | se after |
|---|---:|---:|---:|---:|---:|
| `snap:ai_v9_29_rev1_0823@2000016` | 1723.0 | 1715.6 | **-7.4** | 10.1 | 9.5 |
| `snap:ai_v9_29_rev1_0823@8000016` | 1961.5 | 1946.2 | **-15.3** | 10.6 | 9.7 |
| `snap:ai_v9_29_rev1_0823@24000000` | 2053.3 | 2035.9 | **-17.4** | 11.3 | 10.2 |
| `snap:ai_v12_11_ladder_ctrl10M@10000032` | 2018.7 | 1976.3 | **-42.4** | 16.7 | 13.4 |
| `snap:ai_v12_02_winprob_critic@36000000` | 1954.5 | 1954.7 | **+0.2** | 8.4 | 8.0 |
| `snap:ai_v12_02_winprob_critic@56000016` | 1965.4 | 1971.4 | **+6.0** | 8.4 | 8.0 |
| `snap:ai_v12_02_winprob_critic@74000016` | 1984.2 | 1984.6 | **+0.4** | 8.5 | 8.1 |
| `snap:ai_v13_01_flywheel_shaped@22000032` | 1982.7 | 1983.0 | **+0.3** | 8.7 | 8.2 |
| `snap:ai_v13_01_flywheel_shaped@48000000` | 2022.4 | 2017.8 | **-4.6** | 8.8 | 8.3 |
| `snap:ai_v13_01_flywheel_shaped@72000000` | 2036.6 | 2034.6 | **-2.0** | 8.9 | 8.4 |

mean Δ over all 56 snapshot nodes: **-3.0 Elo** (over the 10 played nodes: -8.2; min -42.4, max +6.0)

**P4: PASS** (registered: mean Δ NEGATIVE and |mean Δ| < 15 Elo)

### P5 — newest-node inflation

| run | mean infl, JOINT-NOEXT | mean infl, JOINT-EXT | change | max before | max after |
|---|---:|---:|---:|---:|---:|
| `ai_v12_02_winprob_critic` | +11.1 | +10.3 | -7.2% | +24.2 | +22.2 |
| `ai_v13_01_flywheel_shaped` | +9.8 | +10.0 | +2.0% | +37.7 | +38.3 |

**P5: **FALSIFIED**** (registered: ≥ 25% reduction in mean inflation)

### P6 — fit quality

| edge family | JOINT-NOEXT mean\|err\| | max | JOINT-EXT mean\|err\| | max |
|---|---:|---:|---:|---:|
| bot (anchor edges to our snapshots) | 0.0239 | 0.1394 | 0.0239 | 0.1493 |
| dense (frozen snapshot pairs) | 0.0346 | 0.1682 | 0.0352 | 0.1716 |
| external (this campaign) | None | None | 0.0542 | 0.1521 |
| ALL | 0.029 | 0.1682 | 0.0302 | 0.1716 |

bot-edge mean|err| moved by **+0.0000** (registered: must not rise by > 0.01; `max_abs_err` over ALL edges is ALLOWED to rise — an informative node exposes residuals the saturated bot edges could not)

**P6: PASS**

### P7 — would the PINNED bots move if re-fit jointly?

🚨 **The arm P7 was REGISTERED on is mis-specified, and its own result is what shows it.** `JOINT-EXT-FREEBOT` frees the eight bots but leaves their OWN round robin — the 36 pairs and 86,000 games that ARE the pins — out of the edge set. The bots are then held only by their edges to our snapshots and by `random`'s pin, and `random` loses ~100% of everything, so its edges are near-perfect scores the Gaussian prior resolves rather than the data. The cluster slides toward the prior and every bot reads low, uniformly. `JOINT-EXT-FREEBOT-RR` adds the round robin back and is the arm that answers the question.

| bot | pinned (2026-06-06) | FREEBOT (registered arm) | Δ | **FREEBOT-RR** (correct arm) | **Δ** | se |
|---|---:|---:|---:|---:|---:|---:|
| `heuristic2` | 1638.8 | 1413.6 | -225.2 | 1623.4 | **-15.4** | 6.6 |
| `aggressive_v2` | 1630.1 | 1445.1 | -185.0 | 1618.8 | **-11.3** | 6.7 |
| `setup_sweep_v2` | 1618.7 | 1415.6 | -203.1 | 1605.3 | **-13.4** | 6.7 |
| `setup_sweep` | 1597.8 | 1409.3 | -188.5 | 1586.1 | **-11.7** | 6.6 |
| `heuristic` | 1577.5 | 1401.1 | -176.4 | 1566.9 | **-10.6** | 6.6 |
| `staller` | 1570.9 | 1375.9 | -195.0 | 1558.6 | **-12.3** | 6.6 |
| `staller_v2` | 1554.9 | 1341.9 | -213.0 | 1541.2 | **-13.7** | 6.6 |
| `aggressive` | 1511.7 | 1378.6 | -133.1 | 1505.0 | **-6.7** | 6.6 |
| `random` | 1000.0 | 1000.0 | +0.0 | 1000.0 | **+0.0** | 0.0 |

largest move: **225.2 Elo** in the registered arm (the artifact), **15.4 Elo** in the arm with the round robin (registered bar: < 40 Elo)

**P7 as registered: **FALSIFIED**. P7 on the corrected arm: PASS.** Reported only — `data/gen3_bot_elo_anchors.json` and the calibration store are NOT written by this campaign.

### P8 — the head-to-head

`SyntheticRLV2` vs `SmallRL`, greedy-vs-greedy, home teams, n=200: **0.635** [0.566, 0.699] (W127/L73/T0), implied gap +96 Elo

the fit orders them the SAME way (SyntheticRLV2 1983.4 vs SmallRL 1940.0)

**P8: PASS** (registered: 0.55-0.75 AND the fit agrees)


## 8. The regime caveat, and what it does NOT excuse

`data/gen3_bot_elo_anchors.json` — the nine pins this whole scale hangs on — was computed
**2026-06-06 at git `1081a8e0`**: a dense bot-vs-bot round robin, 2,000–2,700 games on each of the
36 pairs, `random` pinned at 1000, `converged: true`, `max_abs_err` **0.0381**, `min_pair_games`
2000 against a `target_games` of 5000 (so `complete: false` — the anchor is a valid partial, which
is how that tool is designed to be used).

**That is three months before the 2026-09-07 eval-sentinel regime boundary, and the boundary does
not reach it.** The boundary moved the eval *sentinels* — snapshot-vs-snapshot opponents — to greedy
and to the trainee's own teams, worth **+8.9 pp** to the trainee. A bot has no sampling knob to flip
and never drew sentinel teams, so **every bot edge is unaffected**, which is exactly what the SOP
and `snapshot_ladder`'s own recipe comment already say. The bot-vs-bot round robin is
bot-policy-vs-bot-policy on the shared pool, the same operation in September as in June.

**The exposure that IS real, named rather than buried:** the bots' *code* could have changed since
`1081a8e0`. `bot_elo_store.load` warns on a git-hash mismatch and **keeps the counts** — it does not
re-measure. This campaign does not replay the 36-pair matrix, so it cannot settle that; what it can
do is expose it, which is what the FREEBOT arm is for.

**One asymmetry this campaign accepts and records rather than corrects:** the bot-anchor calibration
and `snapshot_ladder._play_pair` both draw teams with `bias_prob = 0.1` toward the sample teams;
`main.anchors`'s home source is the UNBIASED 719-team pool on both sides. Every cell here is
internally symmetric so no cell is confounded, but the external edges are not drawn from exactly the
same team distribution as the existing bot edges.

## 9. Hazards met, and what each cost

| # | what happened | how it presented | fix |
|---|---|---|---|
| **A** | the per-battle observer assumed `battle.strict_view()` — our fork's `Gen3Battle`. A roster bot is a plain poke-env `Player` and gets a plain `Battle` | **`0/4 games completed`** while the server log read `finished=2`: games that happened, recorded as games that did not | read the strict view when it exists, the battle otherwise; both pairings regression-tested |
| **B** | a cross-run frozen snapshot FAILS a bare `MaskablePPO.load` (`unexpected keyword argument 'threat_prob_outspeed'`) — but only for `ai_v9_29`, not for `ai_v12`/`ai_v13` | a campaign that spans eras breaks on SOME cells only | `--model-load auto` falls back to `load_foreign_opponent` and **stamps which loader ran on every row** |
| **C** | `ai_v8` snapshots are below `MIGRATION_FLOOR` | a refusal, correctly | the v8 leg was **dropped before the campaign started**, and the registration says so |
| **D** | each Metamon peer burned **210% CPU** on B=1 CPU inference | a 4,000-game campaign projecting at **18 hours** at 33 s/game | `OMP_NUM_THREADS=1`; the same three lanes then ran a 100-game cell in **~150 s** |
| **E** | the peer-pair watchdog measured its progress deadline from the START of the half (`max(last_progress, started)` never advances) | the 200-game head-to-head was FAILED as `no_progress` at game 41, with both peers alive and mid-turn | the clock resets on each new row, and each row prints a per-game line; a regression test feeds games while a shorter budget elapses |
| **F** | Metamon APPENDS to its battle CSV across runs of the same `--results-dir` | a retry silently inherits the previous attempt's games | the driver RETIRES a failed cell directory (`.failed.<n>`) rather than reusing it — which also keeps the evidence of the failure |

Hazards H1–H12 in [`EXTERNAL_ANCHORS_SOP.md`](../../../ops/EXTERNAL_ANCHORS_SOP.md) were all
already handled by the tool and none of them fired. D is now H13 there.

## 10. The PROPOSAL — what adopting the anchors would actually cost

🚨 **Nothing in this campaign was applied.** `data/gen3_bot_elo_anchors.json`,
`designs/ai_v5/elo_calibration/gen3_bot_elo_games.json` and every run's committed `ladder.json` are
untouched. `EXTERNAL_ANCHORS_SOP.md` licenses an external anchor as a **READ** (tiers B and C); it
does not license changing the ladder's anchor set, so that stays a proposal for the owner.

### The mechanism, in three commits

**(1) a pinned external-anchor file, beside the bot one.** The bots are pinned from a dense round
robin; the two Metamon nodes would be pinned from this campaign the same way.

```diff
+ data/gen3_external_elo_anchors.json
+ {
+   "version": 1,
+   "computed_at": "2026-09-14T…",
+   "protocol": {"regime": "greedy", "teamset": "home", "games_per_cell": 100,
+                "showdown_pin": "e0551883f", "metamon_commit": "0a00a759"},
+   "base": 1000.0,
+   "ratings": {"metamon:SmallRL": …, "metamon:SyntheticRLV2": …},
+   "se":      {"metamon:SmallRL": …, "metamon:SyntheticRLV2": …},
+   "source": "designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/"
+ }
```

**(2) `elo.py` learns the third key family, and `snapshot_ladder.fit_ladder` folds in the pins.**

```diff
  # src/agents/training/elo.py
+ EXT = "ext:"
+ def ext_key(name: str) -> str: return f"{EXT}{name}"
+ def load_external_anchors(path: str = EXTERNAL_ANCHORS_PATH) -> dict | None: ...

  # src/agents/training/snapshot_ladder.py, fit_ladder()
- pinned = {elo_mod.bot_key(n): float(e) for n, e in pins.items()} if pins else None
+ pinned = {elo_mod.bot_key(n): float(e) for n, e in pins.items()} if pins else {}
+ pinned.update({elo_mod.ext_key(n): float(e)
+                for n, e in (elo_mod.load_external_anchors() or {}).get("ratings", {}).items()})
+ # (3) each snapshot's EXTERNAL edges, from <run>/external_anchors/games.jsonl
+ results.extend(external_edges(run_dir, keep_keys))
+ ladder["anchored_to_external"] = bool(ext_pins)
+ ladder["external_edges"] = len(ext_results)
```

**(3) the RECIPE STAMP moves with it.** Rule 24 exists because a committed `ladder.json` that does
not name its recipe is unreadable later. `anchored_to_external` and `external_edges` are that stamp
for this change, and **every `ladder.json` written before it must be refit, not quoted** — exactly
as the sentinel-edge change already requires.

### The tax, priced

An external edge is not free the way a bot edge is. Bot edges are a **by-product** of eval cycles
the run already pays for; an external edge is 100 games against a third-party process that must be
played on purpose.

| per promotion | games | wall clock (3 lanes, one torch thread, measured here) |
|---|---:|---|
| `metamon:SmallRL` only | 100 | **~2.5 min** |
| both anchors | 200 | **~25 min** (SyntheticRLV2 dominates: 200M params, a multi-minute build per half) |

A 75M-step run promotes ~20 snapshots, so both anchors cost **~8 h of CPU per run** and `SmallRL`
alone costs **~50 min**. That is the decision, and it is a cost question rather than a correctness
one.


---

## 11. THE RECOMMENDATION

**Adopt `metamon:SmallRL` as a standing anchor edge. Do NOT adopt `SyntheticRLV2` per promotion.
Do NOT change `data/gen3_bot_elo_anchors.json`.**

Three reasons, in order of weight.

1. **The bots cannot be trusted about anything far above them, and only an external edge can say
   so.** 274 Elo of disagreement among nine pinned edges about one opponent (§1) is not a small
   effect, and no internal instrument can detect it — `ladder.json`'s se, `fit_quality` and the
   dense matrix are all computed *inside* the frame. The external edge is the only check on the
   frame's validity we have.
2. **The cost is asymmetric and `SmallRL` is the cheap half.** 100 games against `SmallRL` costs
   **~2.5 min** at three lanes with one torch thread; the same against `SyntheticRLV2` costs
   **~21 min**, because a 200M-parameter policy pays a multi-minute build per half-series. Over a
   20-snapshot run that is **~50 min** versus **~8 h**. `SmallRL` is at 1940, close enough to the
   frontier to be unsaturated (its snapshot cells median 0.070 from even) and cheap enough to pay
   every promotion.
3. **Do not expect it to change any rating.** The scale moved −3.0 Elo on average and the
   inflation did not move at all. Adopting the anchor buys a **falsification instrument**, not a
   better number. Sold as a precision improvement it will disappoint; sold as the thing that would
   have caught a 274-Elo frame error, it is cheap.

**What to do with `SyntheticRLV2`:** keep it at the SOP's tier C (era gates only), where it already
sits. It is the anchor that is *above* our frontier and that is worth having, but not 8 h per run.

**What NOT to do, and why:** the registered P7 arm reads every bot 133–225 Elo low, and that is an
artifact of dropping the round robin, not a finding (§7). `data/gen3_bot_elo_anchors.json` must not
be touched on the strength of it. The honest arm moves the bots by at most **15.4 Elo**, inside the
registered 40-Elo bar, so **the 2026-06-06 calibration is corroborated, not impeached** — a real
result and the opposite of what the naive arm appeared to say.

**One backlog row falls out of this**, for `designs/ops/TECH_DEBT_BACKLOG.md`, unrun and
undispatched: *the bots' code HAS moved since the calibration's git hash `1081a8e0` — `8a46d394`
("the global-random coupling GENRE — four more seams, and the staller was the smallest") changed
staller behaviour after it — and `bot_elo_store.load` warns-and-keeps rather than re-measuring.
`staller` is also the bot with the most extreme anchor edge in §1 (0.06 vs `SmallRL`, implying
2049, the top of the 274-Elo spread), so it is the one worth re-measuring first. The round robin is 36 pairs at 2,000
games, ~86,000 games; at the measured bridge rate it is a few hours. Nothing in this campaign says
it is wrong — the FREEBOT-RR arm says it is right to within 15 Elo — but nothing says it is current
either.*

---

## 12. A READY-TO-APPEND LEDGER PARAGRAPH

> Not appended by this artifact. `designs/research_state/ledger.md` and `UNDERSTANDING.md` are
> untouched; paste this under a `## 2026-09-14 · MEASUREMENT (MAJOR)` heading and regenerate
> `ledger_index.md` with `python -m main.ledger_index --write`.

**MEASUREMENT (MAJOR) · TWO EXTERNAL ANCHORS ON THE BOT-ANCHORED SCALE — the bot frame is PRECISE
AND WRONG, and the scale itself barely moves.** 4,000 games, pre-registered before the first one
(`d655843f`), greedy-vs-greedy on the HOME 719-team pool, both regimes verified per decision on
both sides, 39/39 cells `status: OK` on our pinned Showdown `e0551883f` against Metamon
`0a00a759`: each of the NINE PINNED eval bots and TEN frozen snapshots spanning
`ai_v9_29_rev1_0823` → `ai_v13_01_flywheel_shaped` played 100 games against both
`metamon:SmallRL` (ckpt 40) and `metamon:SyntheticRLV2` (ckpt 48), plus a 200-game head-to-head.
In a joint Bradley–Terry fit over 995 edges the anchors land at **`SmallRL` 1940.0 ± 9.5** and
**`SyntheticRLV2` 1983.4 ± 9.7**, straddling `ai_v12_02_winprob_critic`'s 74M node (1984.6) and
below `ai_v13_01`'s 72M (2034.6). 🚨 **THE FINDING: the nine pinned bots' individual edges to
`SmallRL` imply ratings from 1775 to 2049 — 274 Elo of disagreement about one opponent — against a
fitted se of 9.5**, and the bot frame alone orders `SmallRL` ABOVE `SyntheticRLV2` while 200 direct
games say `SyntheticRLV2` wins **0.635 [0.566, 0.699]** (+96 Elo the other way). The anchors ARE
unsaturated where the bots are not: median |win rate − 0.5| is **0.070** over the 20 snapshot cells
against **0.340** over the bot cells. But the pre-registered benefits did NOT land: frontier se
fell only **8.6%** (bar ≥ 15%, FALSIFIED) and the newest-node inflation did not move (+11.1→+10.3,
+9.8→+10.0, FALSIFIED) — a frontier node already carries ~25 bot edges at 500 games plus ~19 dense
edges, so two 100-game edges cannot move its variance, and the inflation is a property of BT
re-solving as nodes are ADDED, which no external edge touches. The scale moved **−3.0 Elo** on
average (−8.2 over the ten played nodes), with one exception that confirms the mechanism:
`ai_v12_11_ladder_ctrl10M` @ 10M, the least-connected node in the campaign (6 dense pairs against
190), fell **−42.4** — **a short control run is where the bot-only scale is least trustworthy.**
The external edges are the WORST-fitting family in the model (mean|err| **0.0542** against the bot
edges' 0.0239, unchanged by their addition), which is the same non-transitivity seen from the other
side. **METHOD ERROR, self-caught:** the registered "would the pinned bots move" arm freed the eight
bots but left their OWN 36-pair, 86,000-game round robin out of the edge set, so they were held
only by `random`'s pin — whose edges are perfect scores the prior resolves — and every bot read
133–225 Elo low; with the round robin restored the largest move is **15.4 Elo**, inside the
registered 40-Elo bar, so **the 2026-06-06 bot calibration is CORROBORATED**. RECOMMENDATION:
adopt `metamon:SmallRL` as a standing per-promotion anchor (100 games, **~2.5 min**) as a
FALSIFICATION instrument rather than a precision one; keep `SyntheticRLV2` at era gates only
(**~21 min** per cell, ~8 h per run); change nothing in `data/`. Also landed: `main.anchors` can now
play an external anchor against a pinned BOT (`--our-side bot:<name>`), against a cross-run
snapshot (`--model-load auto|foreign` — every `ai_v9_29` node FAILS a bare `MaskablePPO.load`), and
against another anchor (`--our-side metamon:<Agent>`); and **one torch thread per peer is worth
~20×** — each Metamon peer was burning 210% CPU on B=1 inference and the campaign projected at 18 h
before `OMP_NUM_THREADS=1` brought a 100-game cell to ~150 s. Artifact:
`designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/`.
