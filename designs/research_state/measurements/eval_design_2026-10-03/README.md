# Eval-system design inputs (2026-10-03, CPU-only, read-only on `models/`)

Small analyses on banked data behind the numbers in [`designs/endstate/design_evaluation.md`](../../../endstate/design_evaluation.md).
Nothing under `models/` was written and no game was played. Run from `scripts/` with the `gen3ai_torch28` interpreter:
`simchk.py` and `nashdemo.py` import `cyc.py` from the same directory.

| script | output | what it measures |
|---|---|---|
| `cyc.py` | `out/cyc.txt` | For every run with `snapshot_ladder/games.jsonl`: a Bradley–Terry fit on the frozen-vs-frozen pair counts (eval-cycle copies dropped, the nodes in `ladder.json`), the residual deviance and Pearson χ² over df = edges − (K − 1), and the implied excess (cyclic) SD on the pp scale |
| `simchk.py` | `out/simchk.txt` | The same fitter on BINOMIAL data simulated from each fitted θ on the same graph (30 replicates): checks that the fitter itself gives χ²/df ≈ 1 |
| `repchk.py` | `out/repchk.txt` | Pairs played TWICE on the ladder (both rows ladder protocol): var of the binomial z of the difference |
| `nashdemo.py` | `out/nashdemo_n0.txt` | N0 (`ai_v14_01_base`): max-entropy Nash of the archive (all nodes but the newest) on 200 posterior draws of the matrix (Beta(w+1, l+1) per measured edge; BT-imputed with a small logit jitter for the 49 unmeasured), the frequency of each node's weight > 0.01, and the newest node's score vs the mixture |
| `mirror_var.py` | `out/mirror_var.txt` | The P0 h2h rows: the mirrored pair's score variance vs that of two independent games at the same mean |

## Results (MEASURED here)

- **No within-run snapshot pool shows a cyclic component above noise.** Over 70 ladders, Pearson χ²/df has a median of
  **0.76** (q10–q90 0.54–1.14); the same fitter on simulated binomial data on the same graphs gives **1.01–1.03 (SD 0.12–0.14)**.
  N0: 0.79 on 122 df (141 measured pairs × 100 games, 20 nodes 24M–72M).
- **The ladder's per-edge noise is BELOW binomial (cause UNKNOWN).** χ²/df < 1 on most ladders, and the 133 pairs played
  twice give var(z) = **0.82** (expected 1; SE of a variance at n = 133 ≈ 0.12). The ladder plays unseeded, unmirrored,
  greedy vs greedy with independent team draws (`snapshot_ladder._play_pair`), so iid binomial games are expected. A
  binomial SE is therefore CONSERVATIVE for these rows; the cause is an open FINDING (`design_evaluation.md` §11).
- **Nash averaging on a near-transitive archive concentrates on the frontier.** N0's BT-smoothed matrix gives a PURE
  max-entropy Nash on the 70M node (Balduzzi et al. 2018, Thm 1 P3: a transitive game's Nash is uniform on the top). On
  posterior draws at 100 games per edge, P(weight > 0.01) is 0.88 (70M), 0.64 (68M), 0.68 (58M), 0.34 (62M), 0.34 (52M),
  0.26 (54M), 0.28 (46M), 0.15 (44M), and ≤ 0.04 for every node at or below 42M except 34M (0.20). The newest node's score against the posterior mixture has SD
  **3.6 pp** from the mixture's uncertainty alone, larger than the 2 pp a plateau check must resolve. That is why
  `design_evaluation.md` §8 freezes the reference mixture per check.
- **Mirroring saves ~12 % of games on a cross edge.** P0's nine cross edges (near-equal runs): the pair-score variance is
  **0.866–0.907** of two independent games' (pair SD 0.328–0.336 vs 0.352–0.354). Self-play edges: 0.09–0.11. The antithetic
  correlation of the two games of a pair is about −0.12 for different players.
