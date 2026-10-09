# Where is static worse? A CPU diagnostic of the static-token screen (2026-10-09)

**DESCRIPTIVE / exploratory. This is NOT part of the registered screen** (`designs/endstate/design_static_tokens.md`
§8.1–§8.2) and cannot change its outcome. The owner ruled that look 3 adopts on the registered non-inferiority rule.
Every number here describes the 11 finished screen seeds; none of them is a registered test.

Owner's question (2026-10-09): *"can we have a little CPU time spent to see where it [static] is worse or less
expressive? Could it be that attention is slow to learn and now we rely on that more?"*

## In plain language

- **Attention is NOT slow to learn, at least not in a way that explains the gap.** Static's deficit against legacy is
  about the same at 5M, 10M and 15M steps (−2.6, −3.2 and −2.1 pp, paired-seed games). It does not shrink with
  training. Static's attention does not start diffuse either: early on its first layer is *sharper* than legacy's,
  and it gets broader later. One thing static does learn more slowly: keeping a mon's own HP readable through the
  trunk. At 2M it trails legacy by a lot. By 15M it has nearly caught up.
- **Static IS less expressive for a few board facts.** The clearest is **Spikes on our own side**. After the whole
  trunk a linear probe reads it from our mon tokens with R² 0.31–0.33 under static, against 0.46–0.50 under legacy.
  It is the only fact whose gap *grows* with training: legacy learns to keep it and static does not. A second, smaller
  gap is **our active's HP in its move tokens** (R² 0.59 vs 0.80). Static's move tokens carry no HP; legacy's move
  network mixed it in. Weather, the turn clock and the faint counts reach static's mon tokens almost fully by the last
  layer (within ~0.04).
- **Depth matters (H3).** Under static, board facts arrive through attention, and mostly only in the SECOND layer.
  After layer 1 the gaps are 2–7× larger than after layer 2. Static's active mon puts 3× legacy's attention on the
  board tokens at layer 2. With two layers, board context arrives at the last round, leaving no round to *combine*
  it with anything.
- **Where static loses games: slow, hazard-heavy teams.** In the 50,000 look-1/2 cross games, compare static's win
  rate with each team archetype against legacy's win rate with the same teams. Static is **10.7 pp worse with stall
  teams** (24 of 25 cells negative) and 8.2 pp worse with semi-stall. With hyper-offense it is only 1.2 pp worse.
  Spikes, spin, phazing and Wish teams show the same pattern. This is the game the Spikes probe gap would hurt: long
  games, many switches, entry damage.
- **Per decision on the shared bank, the policies hardly differ.** Static is slightly more uncertain everywhere
  (+0.04 nats of entropy, t 1.45). It agrees with the legacy consensus about as often as one legacy seed agrees with
  the others (−0.7 pp). The win-prob critic's calibration is the same (Brier 0.193 vs 0.197). The deficit lives in
  long-horizon play, not in single-turn choices the bank can see.
- **Next lever:** the evidence points at **depth for board context**. The candidate is a third trunk round under
  static, which the owner's architecture direction allows ("add a trunk round if needed"). The cheaper, narrower
  alternative is to put **our side's Spikes layers into D** (the per-mon dynamic part). Either is a separate,
  one-lever screen after this one, never a change to the running screen.

## Hypotheses and their predictions (as given in the brief, declared before reading)

| | prediction | read | verdict |
|---|---|---|---|
| **H1 slow to learn** (a) | static's deficit larger EARLY, shrinking | paired-seed diagonal Δ: 5M **−2.60**, 10M **−3.21**, 15M **−2.13** pp; paired change 5M→15M **+0.47 pp** (se 2.45), 10M→15M +1.08 (se 0.94) | **NOT SUPPORTED** (flat within noise) |
| H1 (b) | static's late slope ≥ legacy's | bots-8 win rate 10→14M: static +0.74 vs legacy +0.34 pp/M (Welch t 2.0), but the bots sit at the 89–90 % ceiling, and the 14M bots gap (−0.3 pp) does not show the −2.7 pp the cross shows | weakly consistent, CONFOUNDED by the ceiling |
| H1 (c) | static's attention more diffuse early, sharpening later | layer-1 entropy of our active's query: 2M static **1.94** vs legacy 2.76 nats (t −4.3); 15M 2.52 vs 2.77. Static starts SHARPER and broadens | **NOT SUPPORTED** (opposite) |
| **H2 less expressive** | a dynamic fact is less decodable from static's mon token after the trunk | our side's Spikes after layer 2: R² 0.33 vs 0.46 (our active), 0.31 vs 0.50 (bench mon); our active's HP in its move token 0.59 vs 0.80; per-mon facts in D (own HP, status, boosts) within 0.05 | **SUPPORTED for board facts (Spikes most), NOT for per-mon facts** |
| **H3 depth-limited** | H2's gap large after layer 1, smaller after the last; errors concentrate where reasoning is multi-step | board-fact gaps after layer 1 vs after layer 2: fainted (theirs) −0.20 → −0.04, clock −0.14 → −0.03, weather −0.07 → −0.03, Spikes −0.25 → −0.13; game deficit concentrated in slow teams (stall −10.7 pp) | **SUPPORTED** (partly: Spikes stays open at the last layer; the bank strata show no per-turn concentration) |

## What was read

- **Seeds:** the 11 FINISHED seeds at P_st `6c6d2e09`: legacy s1001–s1006 and static s1001–s1005. Static s1006 was
  still training (5M) and is left out. Each run is read at the checkpoint nearest 2M, 5M and 10M (static s1003's
  "10M" is 9.6M, its resume gap), plus its `final_model.zip` (15M) — 44 checkpoints (`checkpoints_read.txt`).
- **Code:** every model load ran on the pin's code (`PYTHONPATH` = the read-only pin checkout
  `.claude/worktrees/st-look1-pin-6c6d2e09`, HEAD `6c6d2e09` read from its git admin file). CPU only (CUDA hidden),
  `nice 19`, ≤ 4 threads, one heavy job at a time, each under `scripts/ops/mem_cap.sh`.
- **States:** the M5 Lane S bank (`m5_laneS/bank_v1`, 20,712 decisions, 580 battles) re-encoded ONCE by the pin's
  encoder (41 s). Both arms read the same obs rows. 19,332 / 19,964 checkable rows are byte-equal to the recorded
  ones, which is expected: the bank was built at `7013c288`, and the pin's encoder has moved since (`encode_gate.json`).
- **Per checkpoint** (`capture.py`, ~20 s each): one eager forward over all 20,712 rows, with read-only hooks on the
  two trunk layers. It records the masked policy, the win-prob, the attention entropy per layer / head / query group,
  the attention mass from each active's query onto six key groups, and four probe seats' tokens at three depths. The
  depths are the trunk input, after layer 1 and after layer 2. The seats are our active, their active, our first
  alive bench mon, and our active's first move seat (E3).
- **Probes** (`probe.py`): ridge (RidgeCV α ∈ {1, 10, 100}) on standardised tokens, GroupKFold(5) by BATTLE in bank
  order (deterministic, never by row). The out-of-fold R² is reported for a continuous fact and the ROC-AUC for a
  binary one. Facts come from the obs row at the pin's layout. Tokens were deleted after probing (disk).
- **Learning curves** (`curves.py`): TensorBoard (value loss, explained variance, entropy, approx-KL, clip fraction,
  win-prob Brier) in 1M bins, and the in-loop bots eval (8 bots, `random` excluded, every 2M).
- **Early paired-seed diagonal** (`early_cells.py`, `run_h2h_early.sh`, `early_read.py`): static seed i vs legacy
  seed i at ~5M and ~10M, 400 mirrored pairs per cell. Played by `main.h2h play-many` at the pin on CPU (eager, 64
  envs), in the screen's own regime `adfdbee824c9eb0a`, purpose `audit`, into THIS study's own ledger root
  (`~/gen3ai_archive/static_diag_2026-10-09/ledger`, never `models/`), 8,000 games. The 15M diagonal is the
  registered cross's own diagonal (1,000 pairs, CUDA graph). Seeds do not pair runs, so the diagonal is only a
  fixed, balanced subset of the cross.
- **Cross games by team** (`h2h_teams.py`): the look-1 + look-2 rows (requests `st_look1_steps` +
  `st_look2_steps`), read ONLY through the eval ledger's declared family read (`main.h2h.cross.FAMILY_READ_OFF`).
  Every row carries per-team [games, wins] for both sides. All 717 teams are mapped to the pool classifier
  (`agents.training.team_archetypes.classify_team`). Game LENGTH is not on a row (only inside the outcome digest), so
  no length breakdown was possible.
- **Added after seeing the probes (exploratory):** the archetype breakdown, and `opc_norms.py` (does the zero-init
  per-mon op-content route carry Spikes' entry chip?).

## H1 — slow to learn?

**The paired-seed diagonal** (`early_h2h.json`; score = static seed i vs legacy seed i, pp):

| stage | s1001 | s1002 | s1003 | s1004 | s1005 | mean (Δ vs 50) | sd | mean without s1005 |
|---|---|---|---|---|---|---|---|---|
| ~5M | 53.62 | 49.94 | 48.81 | 49.19 | **35.44** | 47.40 (**−2.60**) | 6.95 | 50.39 (+0.39) |
| ~10M | 49.19 | 46.44 | 46.81 | 47.88 | 43.62 | 46.79 (**−3.21**) | 2.07 | 47.58 (−2.42) |
| 15M (the cross) | 47.33 | 50.52 | 48.15 | 48.95 | 44.40 | 47.87 (**−2.13**) | 2.27 | 48.74 (−1.26) |

Without the outlier seed (static s1005, very weak at 5M) the gap *opens* from 5M to 10M and then narrows a little.
With it, the gap is flat. Neither reading is "a larger deficit early that training closes".

**Bots eval** (`curves.json`, the 8 training bots, arm mean, static − legacy): 2M −1.9, 4M −4.2, 6M −1.9, 8M −1.5,
10M −1.9, 12M −0.2, 14M −0.3 pp. The apparent closing at 12–14M is the 89–90 % ceiling. At 14M the cross still reads
−2.7 pp. TensorBoard: value loss, explained variance (15M: 0.803 vs 0.792), entropy, approx-KL and the win-prob Brier
track each other from 4M on. Static's explained variance trails by ~0.01 throughout, and no TB slope over the last 5M
differs by more than its seed spread.

**Attention entropy** (head mean, nats; `analysis.json` → `attention`):

| query, layer | 2M L / S | 5M L / S | 10M L / S | 15M L / S |
|---|---|---|---|---|
| our active, L1 | 2.76 / **1.94** | 2.72 / 2.23 | 2.74 / 2.48 | 2.77 / 2.52 |
| our bench, L1 | 2.61 / **1.84** | 2.57 / 2.16 | 2.58 / 2.38 | 2.58 / 2.41 |
| board seat(s), L2 | 3.60 / 3.59 | 3.53 / 3.40 | 3.56 / 3.21 | 3.53 / **3.17** |
| our active, L2 | 3.25 / 3.38 | 3.00 / 3.09 | 3.02 / 2.91 | 2.96 / 2.95 |

Static's first layer starts SHARP and broadens. Its board tokens sharpen over training (they learn whom to read).

**The one place static learns slower:** a mon's OWN HP (in D under static) after the trunk. At 2M, after layer 2,
R² is 0.37 vs 0.68 (our active), 0.22 vs 0.58 (bench), 0.33 vs 0.65 (their active). At 15M it is 0.79 vs 0.83, 0.72
vs 0.77 and 0.69 vs 0.71. So "slow to learn" holds for carrying the D signal through the trunk, and it has mostly
caught up by 15M while the game deficit has not moved.

## H2 / H3 — what is less decodable, and at which depth

Linear probe, arm mean over seeds, legacy / static (Δ, Welch t over seeds), at 15M (`analysis.json` → `probes`;
2M / 5M / 10M there too). R² for continuous facts, AUC for binary (b) ones. Seats: OA our active, OB our bench mon,
TA their active, M0 our active's first move seat.

| seat : fact | trunk input | after layer 1 | after layer 2 |
|---|---|---|---|
| OA : Spikes, our side | 0.64 / 0.19 (−0.46) | 0.48 / 0.23 (−0.25) | **0.46 / 0.33 (−0.13, t −5.2)** |
| OB : Spikes, our side | 0.74 / 0.19 (−0.56) | 0.56 / 0.28 (−0.28) | **0.50 / 0.31 (−0.19, t −5.6)** |
| TA : Spikes, their side | 0.48 / 0.38 (−0.10) | 0.42 / 0.39 (−0.03) | 0.41 / 0.39 (−0.02, t −1.1) |
| OA : weather (b) | 0.98 / 0.82 (−0.16) | 0.95 / 0.87 (−0.07) | 0.92 / 0.89 (−0.03, t −2.2) |
| M0 : weather (b) | 0.89 / 0.67 (−0.21) | 0.96 / 0.88 (−0.08) | 0.92 / 0.89 (−0.03, t −2.9) |
| OA : turn clock | 0.55 / 0.31 (−0.24) | 0.64 / 0.50 (−0.14) | 0.74 / 0.70 (−0.03, t −1.5) |
| OA : our faint count | 0.38 / 0.15 (−0.23) | 0.51 / 0.42 (−0.09) | 0.69 / 0.67 (−0.01, t −0.5) |
| OA : their faint count | 0.40 / 0.08 (−0.32) | 0.56 / 0.36 (−0.20) | 0.73 / 0.69 (−0.04, t −1.8) |
| **M0 : our active's HP** | 0.83 / 0.34 (−0.49) | 0.86 / 0.65 (−0.21) | **0.80 / 0.59 (−0.21, t −7.3)** |
| OA : own HP | 0.91 / 0.91 (−0.01) | 0.87 / 0.83 (−0.04) | 0.83 / 0.79 (−0.05, t −4.2) |
| OB : own HP | 0.92 / 0.92 (+0.01) | 0.83 / 0.80 (−0.04) | 0.77 / 0.72 (−0.05, t −3.1) |
| OA : own status (b) | 0.97 / 0.98 | 0.93 / 0.92 | 0.90 / 0.88 (−0.02) |
| OA : our active boosted (b) | 0.89 / 0.90 | 0.89 / 0.89 | 0.89 / 0.89 (0.00) |
| OB : our active boosted (b) | 0.63 / 0.63 | 0.73 / 0.72 | 0.78 / 0.80 (+0.03) |
| OA : their active's HP | 0.10 / 0.11 | 0.19 / 0.14 | 0.32 / 0.38 (+0.07, t 2.4) |

- **Spikes on our side is the gap that GROWS with training.** After layer 2 the gap is −0.04 at 2M (both arms ~0.30)
  and −0.13 at 15M: legacy rises to 0.46, static stays at 0.33. Their side's Spikes is nearly closed (−0.02).
- The per-mon op content (`op_content.amount_proj`, zero-init) DOES carry Spikes' entry chip into static's mon tokens:
  its entry-chip column grew to norm 2.7–3.6 across the five static seeds, vs ~1.0 for the weather chip
  (`opc_norms.json`). So the AMOUNT has a route. What legacy has and static lacks is the plain "Spikes are on my side"
  fact in every token, its move tokens included.
- **Attention mass** (15M, head mean, from our active's query): layer 1 static puts 0.30 on the event seats (legacy
  0.18, t 2.7) and 0.10 on their mons (legacy 0.22, t −4.4). Layer 2 static puts **0.099 on the board tokens vs 0.035**
  (t 4.4). Board context is read at the last round.

## Where static is worse

**By team archetype** (`h2h_teams.json`; 50,000 cross games, 717 teams; per cell, static's win rate holding the
archetype minus legacy's win rate holding it in the same games, so the teams' own strength cancels; mean over the 25
cells, sd, cells negative):

| archetype (teams) | static − legacy, holding | sd over cells | cells < 0 |
|---|---|---|---|
| stall (90) | **−10.7 pp** | 6.5 | 24 / 25 |
| semi-stall (129) | **−8.2 pp** | 6.0 | 23 / 25 |
| balance (205) | −5.6 pp | 5.1 | 20 / 25 |
| offense (154) | −3.2 pp | 4.4 | 18 / 25 |
| hyper-offense (139) | **−1.2 pp** | 5.0 | 17 / 25 |

Static's own holding rate: stall 43.4 %, semi-stall 45.5 %, balance 46.9 %, offense 47.8 %, hyper-offense 50.2 %.
Slow minus fast per cell: −4.3 pp (23 / 25 cells). By style tag, pooled: static holds Wish / spin / spinblock / phaze /
Spikes teams at 45.0–46.1 % and choice / setup-heavy teams at 49.4 %. Cells share seeds, so the cell spread describes
the data; it is not a test.

**Per decision on the bank** (15M, `analysis.json` → `policy`; arm mean of seed means, legacy / static):

| stratum | agreement with the legacy consensus (leave-one-out) | entropy (nats) | critic Brier |
|---|---|---|---|
| all (20,712) | 0.723 / 0.715 (t −0.7) | 1.125 / 1.165 (t 1.5) | 0.197 / 0.193 |
| midgame | 0.715 / 0.704 | 1.133 / 1.187 (t 1.8) | 0.205 / 0.198 |
| endgame | 0.741 / 0.744 | 1.078 / 1.087 | 0.153 / 0.155 |
| free, setup legal | 0.769 / 0.741 (t −1.8) | 1.005 / 1.041 | 0.211 / 0.209 |
| free, hazard legal | 0.562 / 0.579 | 1.480 / 1.571 (t 2.1) | 0.176 / 0.170 |
| free, recovery legal | 0.692 / 0.698 | 1.247 / 1.311 (t 2.2) | 0.179 / 0.181 |
| vs pool snapshots | 0.717 / 0.701 | 1.057 / 1.107 (t 1.6) | 0.194 / 0.192 |

Category mass on free decisions is the same within the seed spread (attack 0.516 / 0.509, switch 0.267 / 0.270,
status 0.116 / 0.119, setup 0.032 / 0.037, hazard 0.019 / 0.020, recovery 0.051 / 0.046). Legacy seeds agree with
each other only ~72 % of the time, so per-turn agreement is a weak instrument here. The bank's states come from
older-era policies' games.

## Recommended next lever (a proposal; the orchestrator owns it)

1. **A third trunk round under `static`** (one lever, its own screen after this one). All board facts reach static's
   tokens only at layer 2, so two layers leave no round to use them. Spikes and the move-token HP stay short even
   there. The owner's direction allows it. Cost: one more `BiasedEncoderLayer` (~+50 % trunk FLOPs at 2 → 3; the trunk
   is part of the forward, not all of it; UNVERIFIED in ms).
2. Cheaper and narrower: **our side's Spikes layers as a D input** (one column, the one fact whose gap grows), and/or
   the mon's HP onto its move tokens (`m_k + Linear([pp_k; legal_k; hp])`). These move a dynamic fact back into the
   token, which the static design otherwise avoids.
3. Not indicated: an LR / warmup change for attention. Nothing here says attention trains slowly; the gap is flat
   from 5M to 15M.

## FINDINGS (standing rule 7)

1. **Disk was at 100 % when this study started** (2.2 GB free on `/`, ~07:00 PDT). It was reported to the
   orchestrator at once and later cleared by someone else (584 GB free by 08:10). This study keeps bulk tokens off
   disk (probed in place, then deleted). The archive folder holds ~300 MB.
2. **Not read:** static s1006 (in training), game length (not on a ledger row), the frozen-pool and SmallRL panels.
   The per-decision bank strata come from older-era policies' games. A linear probe measures LINEAR decodability, not
   use: a lower R² can mean a non-linear code or a fact the policy does not need.
3. **The early diagonal mixes engines:** 5M / 10M on CPU eager (400 pairs), 15M from the registered cross on a CUDA
   graph (1,000 pairs). Same regime id. A device change can flip only near-tie decisions (rule 8's listing).
4. **The bots curve is ceiling-bound (89–90 %)** and does not see the −2.7 pp the cross sees. Do not read H1's slope
   from it.
5. **CPU contention:** the capture pass (~35 min, ≤ 4 threads, nice 19) and the early h2h (~40 min, 2 core + 3 torch
   threads, nice 19) ran beside the live screen and four nice-5 python jobs that were not this study's. The screen's speed read
   discards contended cycles, so it may have lost quiet cycles in 07:05–08:10. Its strength read is unaffected.
6. **The h2h archetype breakdown's FACING rate counts a draw for static** and its HOLDING rate counts it against
   static (a row keeps per-team wins only; draws 0.65 %). The controlled skill gap uses holding rates for both arms,
   so draws are non-wins on both sides there.

## Files

| file | what |
|---|---|
| `run_at_pin.sh` | runs a script here against the pin's code: CPU only, nice 19, 4 threads, 24 GB cap |
| `encode_bank.py` | re-encodes the Lane S bank once at the pin → archive `bank_rows.npz`, `bank_meta.json`, `obs_layout.json` |
| `plan.py` → `checkpoints_read.txt` | the 44 checkpoints read |
| `capture.py`, `probe.py`, `run_all.sh` | per checkpoint: hooks + forward, then the probes (archive `cap/`, `probe/`) |
| `analyze.py` → `analysis.json` | probes, attention, policy and critic per stratum, arm-level |
| `curves.py` → `curves.json`; `tb_tags.py` | TB + in-loop eval learning curves |
| `early_cells.py`, `run_h2h_early.sh`, `early_read.py` → `early_h2h.json` | the early paired-seed diagonal |
| `h2h_teams.py` → `h2h_teams.json` | the cross games by team archetype / tag |
| `opc_norms.py` → `opc_norms.json` | the op-content route's per-input weight norms |
| `encode_gate.json` | the re-encode's byte-equality check vs the recorded obs |

Bulk, logs and this study's ledger root: `~/gen3ai_archive/static_diag_2026-10-09/`.

## Commands (as run)

```bash
D=designs/research_state/measurements/static_diag_2026-10-09
python $D/curves.py
$D/run_at_pin.sh encode_bank.py
python -I $D/plan.py > ~/gen3ai_archive/static_diag_2026-10-09/plan.txt
$D/run_all.sh                       # 44 × (capture + probe), resumable
python -I $D/analyze.py
python -I $D/early_cells.py both > ~/gen3ai_archive/static_diag_2026-10-09/cells_both.json
$D/run_h2h_early.sh ~/gen3ai_archive/static_diag_2026-10-09/cells_both.json 400
python -I $D/early_read.py
$D/run_at_pin.sh h2h_teams.py
$D/run_at_pin.sh opc_norms.py
```
