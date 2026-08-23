# Post-paydown idle-box baselines — 2026-08-23

The first re-baseline of all four performance benchmarks on a genuinely idle box since the tree
absorbed the frame-deletion era, v96–v100, the cf plumbing and the entry-point decomposition.

**The claim under test was that none of that moved a hot path. It held on all four.** No code
regression was found. Two **documentation** findings were, and both are the kind that would make
the next reader draw a wrong conclusion from a correct measurement.

Box: 16 cores, **no training run live** (load 0.75 at start). Data: `post_paydown_baselines_2026-08-23.json`.

---

## The table

| benchmark | measured | documented baseline | verdict |
|---|---|---|---|
| **obs_build** | 0.373 ms/decision · **5431 calls/encode** · obs 2501 | ~3.46k calls/encode (**not comparable** — see below) | ✅ **no regression** (settled by A/B: **+2.4%** calls vs bcdd868) |
| **trainer_turn** | our-CPU **0.899 ms** · obs 63% / reward 27% / parse 9% | obs ~88% / reward ~4% / parse ~7% | ✅ **no regression** (A/B: 0.921 → 0.899 ms, shares flat) — ⚠️ **the documented shares are stale** |
| **bridge throughput @8** | node 2172 / rust 2594 = **1.195×** | node 1852 / rust 2182 = 1.18× | ✅ **no regression** (ratio reproduces, both arms faster) |
| **bridge throughput @48** | node 2160 / rust 3132 = **1.450×** | node 1942 / rust 2729 = 1.41× | ✅ **no regression**; child RSS 223/9 MB matches exactly |
| **bridge heap growth** | 189.8 → 229.3 MB · **+0.16 MB/1k battles** | 189 → 229 MB, ~0 growth | ✅ **no regression** — textbook match |

Every figure carries its load average in the JSON. The two bridge-throughput arms saturate the box
by construction, so their *start* load is recorded rather than treated as contention; no benchmark
printed the "THE BOX IS BUSY" banner.

---

## Why obs_build needed an A/B instead of the documented threshold

The observation leaf names **calls/encode** the primary regression detector and sets the bar at
"~3.46k, >10% above is a regression". Measured today: **5431**. Applied mechanically that reads as a
**+57% regression** — and it is not one.

That 3.46k is a **naked-encode** figure. Since 2026-08-16 the benchmark threads the full env
protocol (`update_progress_clock`, recency, the H-A pair loop, the H-B event-window fold), and the
leaf's own re-baseline note says so — but the *threshold* was never restated in the new units. The
profile confirms it: `episode_tracker._pair_sat_norm` alone is 164 calls/encode, and the whole
tracker family accounts for the bulk of the gap.

So the documented rule could not answer the question, and the box was idle, so it was settled
directly — a same-session A/B against **bcdd868** (the frame deletion; everything after it is the
change under test):

| | obs dim | ms/decision | calls/encode |
|---|---|---|---|
| bcdd868 (before) | 2437 | 0.360 | 5303 |
| **ba5d63f (now)** | **2501** | **0.373** | **5431** |
| delta | +2.6% | +3.6% | **+2.4%** |

**+2.4% against a 10% bar, and it tracks the +2.6% obs-dim growth almost exactly** — the same
algorithms writing more dimensions, not a new hot loop. The cProfile top-of-list structure is
unchanged.

---

## The trainer-turn finding — reward is not 4% and has not been for a long time

Root `CLAUDE.md` documents the trainer-turn split as *"obs build ≈ 88% of our CPU
(`state_encoder.encode` ≈ 80%), parse ≈ 7%, reward ≈ 4%, everything else <1%"*.

That matches **none** of the three arms measured today:

| arm | our-CPU | obs build | reward | parse | `process_turn_reward` |
|---|---|---|---|---|---|
| e80db47 (pre-frame-deletion) | 1.001 ms | 65% | **23%** | 11% | 0.207 ms |
| bcdd868 (post-frame-deletion) | 0.921 ms | 62% | **25%** | 11% | 0.199 ms |
| **ba5d63f (now)** | **0.899 ms** | **63%** | **27%** | 9% | **0.210 ms** |

Two things follow, and they are different:

1. **No regression.** 0.921 → 0.899 ms with flat shares across the change under test. The frame
   deletion itself bought **−8%** of our controllable CPU (1.001 → 0.921 ms) and removed the
   0.060 ms turn-history line — the first direct measurement of what that deletion was worth on CPU.
2. **The documented figure is stale by generations, not by the frame deletion.** Reward was already
   23% *before* the deletion, so obs shrinking cannot explain the gap: at reward's measured 0.23 ms
   absolute, a 4% share would require our-CPU near 5.75 ms — six times anything observed. The
   documented line describes a long-gone tree.

The practical cost of leaving it: `process_turn_reward` is **0.21 ms, the second-largest
per-decision CPU consumer**, and the root doc says everything but obs is noise.

---

## Notes on method

- The heap benchmark's **first attempt carried the busy banner** (load 33 — residual from the
  preceding 48-worker throughput arm, whose 1-minute average had not decayed). It was **discarded
  and re-run** after waiting for load < 3, rather than recorded with a caveat: the docs' own rule is
  to re-run idle before recording, and a caveated number in a file called *baselines* is exactly the
  measurement a later reader quotes without the caveat.
- The clean heap run played **2521 battles**, more than the ~2137 a child sees inside a 3h launcher
  window — so "no recycle needed under the launcher" is re-confirmed on the production horizon
  rather than extrapolated to it.
- Benchmarks were run strictly one at a time, never against each other.
