# P\* against Metamon's HUMAN-ONLY models (FR-1 `SmallRL`, FR-3 `LargeRL` / `LargeIL`): the PREREGISTRATION, 2026-10-25

**The owner's questions (2026-10-25):** "does a model using only human data outperform ours?" and, in part, "is human
data better than our data?". These are the first reads FR-1 and FR-3 of
[`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md) §2.1 / §2.3 (at `431b1fdf`), Q-H10
of [`human_data_architecture.md`](../../../endstate/human_data_architecture.md) §7.1. This file is committed **before
any counted game** of either read. It turns the design's §2.1 into a standalone registration and adds FR-3's.

Abbreviations: P\* = `rb_x26_s2601` @280,000,170 (X26's plateau); H = `heldout72_hpfix`; F = `forum72`; P72 = the
procedural panel; U = H, F and P72 at equal weight; home72 = our 72 sample teams (in the pool); CI = 95 % interval;
LB / UB = its bounds; pp = percentage points; IL = imitation learning (behaviour cloning); RL = offline actor-critic.

## 1. The three opponents (all human-only, all upstream's, none trained by us)

Read from `/home/goodlad/dev/metamon` @ `0a00a759` (`metamon/rl/pretrained.py`, `README.md`). Each was trained only on
human replays (upstream: "1M human battles"); none saw self-play. Their weights are at Hugging Face hub
`jakegrigsby/metamon` revision **`ac8e88e7da8dc2a2b85176d88f81fd8854b1445b`** (the hub's `main` on 2026-10-25,
checked by `HfApi().model_info`), in Metamon's own cache, never in `models/` or `data/`.

| agent | upstream's description | params (checkpoint check) | file at `ac8e88e7` | bytes | sha256 |
|---|---|---:|---|---:|---|
| `SmallRL` | small actor-critic, human replays | 13,938,567 (pinned before) | `small-rl/ckpts/policy_weights/policy_epoch_40.pt` | 55,821,762 | `f4258211777480643fbfe573ff906a2ef1a86b0bc2ad9cce117efdc797e40369` |
| `LargeRL` | "200M actor-critic model trained on 1M human battles" | 194,792,947 | `large-rl/ckpts/policy_weights/policy_epoch_40.pt` | 779,316,162 | `04ed6d5179a0fbc1254384a8199cbf09eed56982c41adf0e22465bfe95044d59` |
| `LargeIL` | "200M imitation learning model trained on 1M human battles" | 194,405,089 | `large-il/ckpts/policy_weights/policy_epoch_40.pt` | 777,764,866 | `8a43fcc7798c41df404a72486002a2d6a7c3ef9f988b948ae062af6a2bb1e89b` |

`LargeRL` and `LargeIL` were downloaded today (the orchestrator's approval, plain `hf_hub_download` at the pinned
revision; the hub's blob names equal the sha256 we computed). `designs/ops/anchors.json` pins both at checkpoint 40
(upstream's default). The machine-readable copy is [`scripts/WEIGHTS.json`](scripts/WEIGHTS.json); the queue re-hashes
the weights before every unit and refuses on a mismatch.

## 2. Feasibility (measured, no counted game)

Two 2-game smokes, `--our-side bot:heuristic` vs each Large model, `away` set, local port 9590 (claimed by flock),
`mem_cap.sh`, CPU, rule 5b cores, the play tree below. Both `status OK`, `regime_verified_decisions = true`,
argmax match **1.0** on every decision (89 for LargeRL, 87 for LargeIL), attention `VanillaAttention` with no window
(the H20 rebind: `large_agent.gin` binds no window). Per decision: LargeRL median 0.08 s (p90 0.08–0.10), LargeIL
0.10–0.16 s (p90 0.11–0.20), first decision a 1.5–12 s warm-up; peak 2.4–2.7 GB a scope. So a 20-game P\* unit costs
about a Kakuna unit (X51: ≈ 12 s a game): ≈ 4–5 min, inside the short lane's 600 s.

## 3. The harness, and why every game twins an X51 game

- **Our side:** P\* = X51's sha-checked byte copy of `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`,
  sha256 `814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed`, `bare` load, greedy, the in-process Rust
  core slot, one torch thread.
- **The play tree:** a detached checkout of **`d32ffcee`** (`~/.cache/gen3ai/hr_reads/pin`), the commit the
  metamon_strongest read plays at. Its identity unit replayed X51's `pstar_home72_00` (P\* vs Kakuna, harness
  `63f97a57`) **20 / 20, winner and turns** (`../metamon_strongest_2026-10-25/`), and nothing under `src/main/anchors`
  or `scripts/ops` changed between `d32ffcee` and `431b1fdf`. This is the one deviation from the design's "the harness
  X51 used": the same play, a later commit; no identity unit is re-run here.
- **Cells:** H, F, P72 and home72, each at X51's committed symmetric config
  (`../x51_plateau_vs_kakuna_2026-10-20/configs/anchors_x51_sym_<panel>.json`; FR-3 adds the two agents,
  `scripts/make_configs.py`; every config's sha256 in `scripts/CONFIG_SHA.json`). **400 games a cell = 20 units × 20
  games**, role-balanced halves, `--team-seed = --seed-base =` the unit seed **0–19**. Both sides pilot the panel's
  teams; every game therefore twins X51's P\* vs Kakuna game of the same panel, seed, half and index (the same two
  team files), and only the opponent differs. 1,600 counted games per opponent; 4,800 in all.
- **Greedy vs greedy** (SOP rule 2); Showdown pin `e0551883f`, server `rust`, the 250-turn forfeit,
  `HF_HUB_OFFLINE=1` (the peers resolve `ac8e88e7` from the cache and never ask the hub).
- **The queue** ([`scripts/hrrq.py`](scripts/hrrq.py)): two lanes; lane A plays FR-1 first, lane B FR-3 only (LargeRL
  and LargeIL interleaved by seed); each unit its own `mem_cap.sh --max-s 600` job on a 95xx port claimed by flock,
  `nice 19`, cores `5-7,13-15`, `CUDA_VISIBLE_DEVICES=""`. A short-lane overrun is re-run once as a long capped job.

**Counting:** X51's rule. `main.anchors.results.unit_counts` (undeclared, symmetric), status OK, n = 20, both sides'
content digests equal (recomputed from the sha-checked files), the capture check (our packed team is the row's team;
the peer's is a team of the same set), `regime_verified_decisions` with argmax match 1.0 and `sample == [False]` on
both halves. A failed unit is re-run with the same seed (at most twice), never replaced by a new seed; a cell with
> 1 uncounted unit is reported as such and read on its counted units.

## 4. The reads and their rules (registered; ties are non-wins)

For each opponent X ∈ {SmallRL (FR-1), LargeRL, LargeIL (FR-3)}, computed by [`scripts/aggregate.py`](scripts/aggregate.py):

| quantity | statistic |
|---|---|
| **W-U(X) (primary)** | P\*'s win rate vs X on U: `main.x35.d1.strat_score` (equal-weight panel mean, Kish n_eff; at 400 games a panel it equals the pooled rate), its interval |
| W-panel(X) | each panel's Wilson 95 %, vs 0.50 (BETTER / WORSE / NOT DETECTED) |
| Δ-K(X) (secondary) | W(X) − P\*'s X51 rate vs Kakuna ckpt 34 on the same twins: U `newcombe_strat`, panels Newcombe; X51's numbers are **U 0.487 [0.459, 0.515]**, home72 **0.598 [0.549, 0.644]** |
| paired descriptor | per twin (seed, half, index): mean difference, normal and seed-bootstrap 95 %, discordant counts |
| FR-3 only: RL − IL | W(LargeIL) − W(LargeRL) on U (`newcombe_strat`) and home72: which human-only recipe is the stronger opponent |

**The rule (the design's §2.1, per opponent):**
- **HUMAN-ONLY-BEATS-OURS** iff W-U's UB < 0.50;
- **OURS-BEATS** iff W-U's LB > 0.50;
- else **UNDECIDED**. A bound within 1e-9 of 0.50 takes UNDECIDED (root rule 8).

Δ-K's classes: **X WEAKER THAN KAKUNA** (LB > 0: P\* wins more vs X), **X STRONGER THAN KAKUNA** (UB < 0), else NOT
DETECTED; the same 1e-9 rule. FR-3 is two tests at 95 % each, no correction (descriptive across models; each verdict
stands alone). FR-1's verdict is sent and recorded first; FR-3's when both of its opponents finish.

**Power:** a panel's Wilson half-width at 400 is ≈ 4.5–4.9 pp; W-U's ≈ 2.6–2.8 pp. A human-only model within ≈ 3 pp of
an even match with P\* on U reads UNDECIDED.

**What follows:**
- OURS-BEATS on all three ⇒ Q-H10's answer is "no, at Metamon's human-only scales up to 200M, in our regime"; X59-S
  stays the read in OUR architecture.
- HUMAN-ONLY-BEATS-OURS on any ⇒ that model is a stronger teacher than P\* on U; it moves H-INIT / H-BC up the lever
  ranking (the orchestrator decides) and becomes a candidate U meter beside Kakuna.
- UNDECIDED ⇒ reported with its interval; no lever moves on it.

**Predictions (GUESS, registered, read by no rule):** OURS-BEATS on all three (Metamon's own gen3 table ranks its
human-only models 34–48 % against 55–67 % for its self-play ones). SmallRL: W-U **0.72** (0.62–0.82), home72 **0.80**.
LargeRL: W-U **0.64** (0.54–0.74), home72 **0.72**. LargeIL: W-U **0.68** (0.58–0.78). Every Δ-K positive (each
weaker than Kakuna). On each panel the same order as vs Kakuna (F ≥ H > P72).

## 5. The regime and its id

`agents.training.eval_ledger.schema.regime_id` (v1 rule) over X51's fields with the opponent changed (it reproduces
X51's `e30dd0b859e45753` for `metamon:Kakuna`, `Kakuna@ckpt34`): `battle_format gen3ou, forfeit_turn_limit 250,
mirrored_pairs false, model_loader bare, model_step 280000170, model_sha256 814823c5…91ed, opponent_commit 0a00a759,
our_regime greedy, their_regime greedy, our_side model, our_transport rust_core_slot, regime_matched true, server_impl
rust, showdown_pin e0551883f, units "20 x 20 games, role-balanced, --team-seed = --seed-base = unit seed 0-19"`:

| opponent | `opponent_version` | regime id |
|---|---|---|
| `metamon:SmallRL` | `SmallRL@ckpt40` | `f0828cc22aa53b41` |
| `metamon:LargeRL` | `LargeRL@ckpt40` | `1664a6b4a5c16de4` |
| `metamon:LargeIL` | `LargeIL@ckpt40` | `541710e5b509e8b7` |

Every row also carries `main.anchors`' own per-row stamp (SOP rule 1).

## 6. What these reads cannot say (registered limits)

1. **"Human-only" is upstream's recipe, not human data in OUR architecture.** A Metamon model differs from P\* in
   architecture, observation, action space, team handling and training scale as well as data. A loss to it would
   not show that human data beats ours inside our network; X59-S (the matched-student contest) is that read.
2. **Unfamiliar is unfamiliar to P\*.** H is Metamon's `hl_05_26` (teams extracted from ladder replays); the
   human-only models trained on the same ladder population, so H may be FAMILIAR to them. F and P72 are not from
   their replays. Per-panel numbers are reported for that reason.
3. **H18 (Metamon misreads a Baton Pass) favours P\*** on Baton Pass teams; its size is unmeasured.
4. **One checkpoint per model** (upstream's default, ckpt 40); greedy only. A sampling peer is a different regime.
5. Metamon's peers use their `poke-env` backend (upstream's choice for these three), as every earlier SmallRL read.

## 7. FINDINGS at registration

1. The play tree is `d32ffcee`, not X51's `63f97a57` (§3; identity shown by the metamon_strongest read, not re-run).
2. `main.anchors` resolves a Metamon checkpoint through `hf_hub_download` without a revision and without
   `HF_HUB_OFFLINE`, so an earlier read could have fetched a newer hub revision had upstream's `main` moved. It had
   not (`main` = `ac8e88e7` today); this read sets `HF_HUB_OFFLINE=1`. The fix in `main.anchors` itself (pin the
   revision or force offline) is a separate harness unit.
3. The design's FR-1 text calls SmallRL "human-only offline RL"; upstream's table also lists `SmallIL` and Medium
   models. Only SmallRL, LargeRL and LargeIL are read; no other model was downloaded.
