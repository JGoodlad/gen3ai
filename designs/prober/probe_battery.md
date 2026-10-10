# The representation PROBE BATTERY (`python -m main.probe_battery`)

**Always-current topic doc** for `src/main/probe_battery/` (`gen3_probe_battery_v1`, built 2026-10-09 at the owner's
request: *"I really want to understand what the model represents easily and what it doesn't, so we can go hunt this
if needed."*). The first read is `designs/research_state/measurements/probe_battery_2026-10-09/`.

**What it answers.** For a checkpoint: which battle FACTS a linear read can take off its representations, at which
SITE (token, depth, head), how that compares with the trunk's INPUT, with a RANDOM network of the same architecture
and with a species-memorising CONTROL — and, for phazing, whether the policy USES what it represents. A linear probe
measures LINEAR decodability, not use: a low score can be a non-linear code or a fact the policy does not need, and a
high score at a token says nothing about whether the head reads it.

## 1. The pipeline

| step | command | runs at | output (outside `models/`; every command refuses `models/`) |
|---|---|---|---|
| bank | `bank --checkout <ckpt's checkout> --expect-commit <sha> --legacy L1=…zip … --static S1=…zip … --pairs 16 --target 25000 --out <dir>` | the checkpoints' commit (worker) | `games.jsonl.gz` (input logs), `replay_all/`, the selected `rows.npy` / `masks.npy` / `decisions.jsonl.gz` (both sides' views per decision), `archetypes.json`, `manifest.json` |
| capture | `capture --checkout … --bank <dir> --ckpt L1=…zip … Lr1=…zip@rand1 --out <caps>` | worker | one `<label>.npz` per checkpoint: probs, win-prob, float16 tokens at 10 seats × every depth, the policy state, the critic pool |
| probe | `probe --bank <dir> --caps <caps> --out <probes>` | HEAD, no model | `facts.json` / `facts.npz` (the labels, cached) + one `<label>.json` per capture |
| report | `report --probes <probes> --arm legacy=L1,… --arm static=S1,… --random legacy=Lr1,… --random static=Sr1,… --out <dir>` | HEAD | `summary.json`, `catalogue.md` |
| behaviour | `behaviour --checkout … --bank <dir> --ckpt L1=…zip … --arm legacy=… --arm static=… --out <dir>` | HEAD + worker forwards | `behaviour.json` |
| depth | `depth --checkout … --bank <dir> --ckpt … --out <dir> [--n-rows 4000]`, then `depth-report --depth <dir> --arm … --random … --out <dir>` | worker, then HEAD | one JSON per checkpoint; `depth_summary.json` with the verdict |

**The worker** (`pin_worker.py`) is run BY PATH with `PYTHONPATH=<checkout>/src`, the checkout as its working
directory, CUDA hidden, `nice 19`, inside `scripts/ops/mem_cap.sh`. It imports nothing from `main.probe_battery`, so a
checkpoint is always read by the code it trained at (its obs layout, encoder and model); `--expect-commit` refuses a
checkout at any other HEAD. It uses only the checkout's long-standing APIs (`main.h2h.play.H2HEngine`, the
`core_events` replay's `run_core`, `main.policy_spectrum.reader.load_checkpoint`).

## 2. The bank

Games are mirrored pairs between checkpoints on the checkout's Rust eval core (greedy vs greedy, the h2h regime), ONE
CPU engine, one cycle seed per cell (`seed + 1000·k`): the cells are each arm-A seed vs the same-index arm-B seed, and
a ring within each arm. Every game's INPUT LOG is replayed through the checkout's `core_events --views --trackers
--obs`, which yields, per side and per answered decision, the obs row, the mask, the legal tokens AND both sides'
`present()` views at that board: the viewer's own (`V`) and the opponent's own (`W`, whose `ours` is the opponent's
TRUE state: exact HP, stats, item, ability, moves, boosts, volatiles). The opponent's choice at the same board is its
TRUE next action. Selection keeps decisions with >= 2 legal actions, `target / 5` per viewer-team ARCHETYPE
(`agents.training.team_archetypes`, the pool table built by the worker), each archetype's quota spread over the
phases (opening = turn <= 3, endgame = either side <= 2 alive, else midgame) in proportion, ranked by
`sha256(seed:decision id)`. Re-running any step reproduces it byte for byte (the manifest's content hashes).

## 3. The facts

`facts.py` reads ~115 facts off `V` / `W` (per-mon state, side, field, speed order, KO-ness, belief, the opponent's
next action, the end-of-turn race, switch-in safety, aggregates, phazing). Each names its family, kind (continuous →
R², binary → AUC), the mon it is ABOUT (the control task's key) and its DECISION SITES (where a decision would need it).
**Rule 8:** a speed TIE is excluded from "who moves first"; a best-move damage ratio within ±3 % of the target's HP is
excluded from "can KO". The damage / speed / residual physics is a small gen-3 calculator on the TRUE stats (STAB, the
type chart, the immunity abilities, Thick Fat, Huge / Pure Power, Choice Band, the +10 % type items, burn, screens,
weather, Explosion's Defense halving; no crit, multi-hit at a fixed count) — an approximate LABEL, checked against
what happened (`label_check.py` in the measurement folder). Opponent-hidden facts (`TA_hidden_*`, `opp_hidden_has_*`)
are scored only on rows where the fact is NOT revealed.

## 4. The probes and their baselines

Ridge on standardised features (the standardisation fitted on the training folds only), α per target chosen inside
each training fold by the closed-form leave-one-out error, five folds BY BATTLE (`sha256(battle id) mod 5`: never by
row, never by a seed). Sites: `OA` / `TA` / `OB` (our active, their active, our first alive bench mon), `M0` (our
active's first move seat) and `Mcat` (the four move seats concatenated), the three board sites (`B_OURS`,
`B_THEIRS`, `B_FIELD` — the static arm's three board tokens; legacy's ONE global token for all three), each at every
depth (`in` = the trunk input, `L1`, `L2`), plus `PI` (the actor latent the pointer head reads) and `VF` (the critic
pool). A fact is read only with >= 300 rows and >= 40 of each class. A fact's DECISION SITES are the tokens the
pointer head scores (our mon tokens for a switch, the move seats `Mcat` for a move), `PI` and, for the value-relevant
aggregates, `VF`; the board tokens are where a side / field fact is STORED, so they are reported per slot but are
never a decision site (otherwise "Spikes on our side" would read high at its own storage token and hide that it never
reaches the tokens that decide).

| baseline | what it says | how |
|---|---|---|
| the INPUT layer | what the trunk adds or loses | the same probe at depth `in` |
| a RANDOM network | what the architecture alone gives | `@rand<SEED>`: a fresh build under `torch.manual_seed(SEED)` (the policy class with the checkpoint's own kwargs; refused if any key or shape differs or nothing moved) |
| the CONTROL task (Hewitt & Liang 2019) | how much of a read a species-memorising probe also gets | a label with the fact's marginal, assigned as a random function of the species of the mon the fact is ABOUT, seeded by the fact's name; selectivity = real − control |
| the SPECIES LOOKUP | how much of the fact the species alone fixes | out-of-fold (by battle) per-species mean of the label; a property of the bank |

**Why selectivity is not the tier gate.** A mon token carries its species almost perfectly, and there are ~150
species, so a species-keyed control is itself decodable at a token (AUC ~0.99 at our active's input token): the
control measures memorisation capacity, which is high everywhere. A fact is therefore only CAUTIONED as "not
selective" when its selectivity is low AND the species lookup says the fact is species-predictable; the pool's 717
teams nearly fix a species' set, so most belief facts are species-determined (lookup >= 0.8) — their decodability is
not evidence of state tracking.

## 5. The report

Binary facts are put on R²'s scale as `2·AUC − 1` wherever facts are ranked. Per fact the catalogue reads `final` =
the best decision site at the trunk's LAST layer (tokens) or the heads, `input` / `L1` = the best decision token site
at those depths, each arm's mean over its seeds with a 95 % t-interval, and the second arm minus the first (Welch).
**The tier** is what the reference arm represents at the decision sites: `poor` = `final < 0.5` or the other arm
significantly worse (gap interval below −0.03) — a HUNT candidate either way; `easy` = `final >= 0.8` with
`max(input, L1) >= 0.7`; else `ok`. **The notes are diagnostics of CAUSE, never a tier**: the trunk LOSES it
(`final − input < −0.05`, interval below 0: present at the input token, diluted by the trunk); a random network reads
it as well (`final − random < 0.10`: the architecture alone delivers what is linearly there); species-determined
(lookup >= 0.8); not selective over the control (selectivity < 0.10 with a species lookup >= 0.5).

## 6. Behaviour (phazing) and depth / capacity use

`behaviour.py`: (a) states where our active has a legal Roar / Whirlwind — the mass on the phaze when their active is
boosted vs not, and the paired change when their boosts are ZEROED / their Spikes REMOVED / +2 Atk +2 SpA INJECTED;
(b) states where our active is boosted with a legal setup move — mass on setup / attack / switch by what the viewer
knows of the opponent's phazers, and the change when OUR boosts are zeroed; (c) the win-prob's paired change. An edit
writes only the encoder's columns for that fact (the active-context boost pair, the global env's Spikes scalar); the
event window still remembers the boost / Spikes EVENTS, so an edit's effect is a LOWER bound. The layout is
self-checked against the views before any edit (a mismatch refuses the run).

`depth.py` (the owner's scope addition, 2026-10-09): the LOGIT LENS (`lens_L2`, the heads read the representation
after layer 1; for this post-LN block it equals skipping the layer), `zero_update` (both residual updates zeroed,
LayerNorms kept), `no_attn` / `no_ffn`, each head ablated, per-token-type `‖Δx‖/‖x‖`, the participation ratio / n90 /
n99 of the token representations, the singular-value spectra of the trunk matrices and the input projections, a
low-rank truncation of the trunk (and of the input projections) to r ∈ {16, 32, 48, 64, 96}, and dead / always-on FFN
units. Every trunk round is read in execution order: the post-LN `BiasedEncoderLayer`s and, under `--trunk-layers
3/4`, the pre-LN `IdentityInitRound`s after them (for a pre-LN round zeroing both updates IS the identity, so
`zero_update` equals `lens`; the verdict reads the LAST round, whichever it is). A round replica computes every variant
and is refused unless the full replica equals the real trunk (max |Δ| ≤ 1e-4 on the policy and the win-prob) and the
rounds ran in module order; `probe_battery_test.py` pins the replica against both round classes. `main.capacity` computes a pooled effective rank on eval-trace states; this
reads it per token type and per depth on the bank instead (the same participation-ratio definition).

## 7. Limits

- Linear decodability ≠ use; the behaviour probes are the only USE reads, and only for phazing.
- The bank is the screen arms' own greedy play at 15M: other policies visit other states.
- The damage / speed / residual labels are approximate physics (§3); rare facts (screens, rain, Wish, Substitute)
  are too rare to read in a 25k bank.
- The policy readout of the read checkpoints is `tower`, so there is no state-query site; `PI` is the actor latent.
