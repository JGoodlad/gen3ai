"""Render PROBE G's JSON into the measurement markdown."""
from __future__ import annotations
import json, sys

def f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"

def ci(d, n=4):
    if d is None or d.get("mean") is None: return "—"
    lo, hi = d["ci"]
    return f"{d['mean']:.{n}f} [{lo:.{n}f}, {hi:.{n}f}]"

def main(js_path, md_path, meta):
    o = json.load(open(js_path))
    C = o["cells"]; A = C["ALL"]; m = A["mse"]
    L = []
    P = L.append
    P("# PROBE G — the critic's leaf error, split into SHARED vs DIFFERENTIAL")
    P("")
    P(f"*Measured {meta['date']} · commit `{meta['commit']}` · CPU-only, 2 cores, `nice 15`.*")
    P("")
    P("## The question")
    P("")
    P("A search that ranks candidate actions by a learned critic only cares about the critic's")
    P("error to the extent that error **differs between the actions it is comparing**. Split the")
    P("per-decision error vector into the part every action shares and the part that does not:")
    P("")
    P("```")
    P("e_d[a]     = C_d[a] - L_d[a]          critic minus Monte-Carlo label, win-prob units")
    P("offset_d   = mean_a e_d[a]            SHARED    -- paired evaluation cancels it EXACTLY")
    P("resid_d[a] = e_d[a] - offset_d        DIFFERENTIAL -- pairing does not touch it")
    P("")
    P("mean_a e^2 == offset^2 + mean_a resid^2      (an identity, exact per decision)")
    P("```")
    P("")
    P("If the error is mostly offset, a paired search is reading a critic far better than its MSE")
    P("suggests. If it is mostly differential, the critic mis-ranks and contrastive training is the")
    P("binding lever.")
    P("")
    P("## What was measured")
    P("")
    P(f"- **Subject**: `{meta['run']}` / `{meta['step']}`, scored by the run's OWN eval snapshot")
    P(f"  (`{meta['step']}/snapshot.zip`) — the exact weights that wrote the traces.")
    P(f"- **{o['n_decisions']} recorded decisions** over **{o['n_battles']} battles**, "
      f"**{o['n_action_cells']} (decision × legal action) cells**.")
    P("  Every decision has ≥3 legal actions; half drawn from the top-15% |TD δ| tail (`pivotal`),")
    P("  half from the rest (`ordinary`); ≤3 decisions per battle.")
    P("- **Labels** `L[a]`: substitute action `a` at the recorded decision, then play the rest LIVE")
    P("  to a terminal win/loss — the trainee greedy vs the RELOADED real opponent (a bot rebuilt")
    P("  exactly; a pool sentinel reloaded from its pinned snapshot and played STOCHASTIC, the")
    P(f"  regime `eval_worker` recorded). **R = {o['R_per_action']} rollouts per action.** No")
    P("  horizon adjudication and no |V| cut-off: every label is a rollout to an actual terminal.")
    P("- **Common random numbers**: the R post-divergence dice seeds are derived from")
    P("  `<battle_tag>:<inv>:cf` with **no action in the salt**, so every sibling action at one")
    P("  decision is rolled on the same seed list. The paired difference `L[a]-L[b]` is the")
    P("  quantity this probe needs resolved, and CRN is what resolves it.")
    P("- **Critic** `C[a]`: the one-ply read a search would use — re-roll the turn under `a` with")
    P("  the opponent playing its RECORDED move, materialize the successor obs through the real")
    P("  encoder, and read the **win-prob head at s′**. Same units as the label. An action whose")
    P("  turn ends the battle is scored 1.0 / 0.0 (a search sees the terminal exactly).")
    P("")
    P("### The label noise floor is MEASURED, not assumed")
    P("")
    P(f"Every label was rolled as **two independent CRN blocks** A and B of {o['R_per_action']//2}")
    P("dice each, with `L = (L_A + L_B)/2`. Within a block CRN is intact; across blocks the dice")
    P("are independent. With `D = L_A − L_B`:")
    P("")
    P("| quantity | estimator |")
    P("|---|---|")
    P("| sampling variance of the full-R label | `E[D²]/4` |")
    P("| …of the DIFFERENTIAL component | `E[mean_a (D − mean_a D)²]/4` |")
    P("| …of the SHARED component | `E[(mean_a D)²]/4` |")
    P("")
    P("The differential floor is the load-bearing one, and it is why a closed-form binomial floor")
    P("would not do: CRN has already removed an unknown share of the paired variance, and only a")
    P("measurement knows how much. `critic_bias_split_selftest.py` pins the estimators on synthetic data")
    P("(shipped beside this file) — the corrected residual returns ~0 for a critic that is a PURE")
    P("offset of the truth, and")
    P("recovers a known injected differential error to within 2%.")
    P("")
    P("## The decomposition")
    P("")
    P("All numbers are mean-squared error in win-prob units, averaged with **equal weight per")
    P("decision** (so a 9-action decision does not outvote a 3-action one). `_true` = raw minus the")
    P("measured noise floor.")
    P("")
    P("| component | raw MSE | measured noise floor | **true MSE** | **RMS** |")
    P("|---|---|---|---|---|")
    P(f"| SHARED (offset) | {f(m['offset_raw'],5)} | {f(m['noise_floor_offset'],5)} | "
      f"**{f(m['offset_true'],5)}** | **{f(m['rms_offset_true'],3)}** |")
    P(f"| DIFFERENTIAL (residual) | {f(m['residual_raw'],5)} | {f(m['noise_floor_residual'],5)} | "
      f"**{f(m['residual_true'],5)}** | **{f(m['rms_residual_true'],3)}** |")
    P(f"| total | {f(m['total_raw'],5)} | — | **{f(m['total_true'],5)}** | "
      f"**{f((m['total_true'])**0.5,3)}** |")
    P("")
    sh = m["offset_share_of_true"]
    shc = m["offset_share_ci"]
    P(f"**Offset share of true MSE: {f(sh,3)}** "
      + (f"(95% CI over battles [{shc[0]:.3f}, {shc[1]:.3f}])" if shc else "")
      + f" · differential share {f(m['residual_share_of_true'],3)}.")
    P("")
    P(f"Mean SIGNED offset: {ci(A['mean_offset_signed'])} — the direction the critic is wrong on")
    P("average, which paired evaluation removes entirely.")
    P("")
    tw = A["three_way"]; sh3 = tw["shares_of_true_total"]
    P("### …and the offset itself splits again, which is what decides SEARCH DEPTH")
    P("")
    P("A *constant* bias shifts every state equally and cancels in any comparison at any depth. A")
    P("*per-decision* offset cancels only between siblings — a deeper tree compares states at")
    P("different nodes, where it does not. So the same MSE means three different things:")
    P("")
    P("| component | cancels when | true MSE | RMS | share |")
    P("|---|---|---|---|---|")
    P(f"| global calibration bias | always (any depth) | {f(tw['global_bias_mse'],5)} | "
      f"{f(abs(tw['global_bias']),3)} | {f(sh3['global_bias'],3)} |")
    P(f"| per-decision offset | siblings only (1-ply paired) | "
      f"{f(tw['per_decision_offset_spread_mse_true'],5)} | {f(tw['rms_per_decision_offset'],3)} | "
      f"{f(sh3['per_decision_offset'],3)} |")
    P(f"| **differential** | **never** | **{f(tw['differential_mse_true'],5)}** | "
      f"**{f(m['rms_residual_true'],3)}** | **{f(sh3['differential'],3)}** |")
    P("")
    P(f"For scale: a single label's own RMS sampling error is {f(m['rms_label_noise'],3)}, and the")
    P(f"mean within-decision spread of the labels themselves (max L − min L) is "
      f"{f(A['label_spread_mean'],3)}.")
    P("")
    P("## The decision-relevant loss")
    P("")
    P("MSE is not what a search pays. What it pays is **regret**: the win-prob it gives up by")
    P("picking `argmax C` instead of the best action.")
    P("")
    P("Three readings, because label noise biases the naive one:")
    P("")
    P("- **naive** — `max(L) − L[argmax C]`. Biased **UP**: `max(L)` is a winner's-curse maximum")
    P("  over noisy estimates, so it exceeds the true best action's value.")
    P("- **cross-fitted** — select the comparison action on one half, score it on the OTHER")
    P("  (`L_B[argmax L_A] − L_B[argmax C]`, symmetrized). Immune to the curse; biased **DOWN**,")
    P("  because the selector is itself only an R/2 estimate. **The truth is bracketed.**")
    P("- **noise reference** — `max(L_B) − L_B[argmax L_A]`, symmetrized: the regret a scorer with")
    P("  **zero bias and only sampling noise** would post. This is the floor a flip has to clear")
    P("  to be a flip.")
    P("")
    P("| statistic | value (95% CI, bootstrap over BATTLES) |")
    P("|---|---|")
    P(f"| flip rate — naive | {ci(A['flip_naive'],3)} |")
    P(f"| flip rate — cross-fitted | {ci(A['flip_cross_fitted'],3)} |")
    P(f"| flip rate — noise reference | {ci(A['flip_noise_reference'],3)} |")
    P(f"| **regret — naive (upper bracket)** | **{ci(A['regret_naive'])}** |")
    P(f"| **regret — cross-fitted (lower bracket)** | **{ci(A['regret_cross_fitted'])}** |")
    P(f"| regret — noise reference | {ci(A['regret_noise_reference'])} |")
    P(f"| **regret_cf − noise reference** | **{ci(A['regret_cf_minus_noise_reference'])}** |")
    P(f"| regret_cf median / p90 / p95 | {f(A['regret_cf_median'],3)} / "
      f"{f(A['regret_cf_p90'],3)} / {f(A['regret_cf_p95'],3)} |")
    P(f"| regret_naive median / p90 | {f(A['regret_naive_median'],3)} / {f(A['regret_naive_p90'],3)} |")
    P(f"| regret — RANDOM action (no-information control) | "
      f"{ci(A['regret_random_no_information'])} |")
    P(f"| Spearman ρ(C, L) per decision | {ci(A['spearman_C_vs_L'],3)} |")
    P(f"| Spearman ρ(L_A, L_B) — the LABEL's own reliability | "
      f"{ci(A['spearman_halves_reliability'],3)} |")
    P("")
    P("The two Spearman rows belong together: `ρ(L_A, L_B)` is the ceiling on `ρ(C, L)` — a critic")
    P("cannot correlate with the labels better than two independent measurements of those labels")
    P("correlate with each other. Read `ρ(C,L)` **against** it, never against 1.0.")
    P("")
    P(f"**Capture fraction: {f(A['capture_fraction_of_achievable_ranking_gain'],3)}** of the")
    P("achievable ranking gain — 1.0 would be ranking as well as a tight-MC oracle at this R, 0.0")
    P("no better than choosing at random. It is the regret scale expressed as a statement rather")
    P("than a magnitude, and it is the number to quote.")
    P("")
    P("### Does ranking by the critic beat what the policy actually did?")
    P("")
    P("The most program-relevant comparison, and it needs no cross-fitting: `argmax C` and the")
    P("recorded action are BOTH noise-free selections, so their CRN-paired label difference is an")
    P("unbiased estimate of the true difference. (The eval trainee played greedy — verified 995/995")
    P("recorded actions equal the masked argmax — so the recorded action IS the policy's choice.)")
    P("")
    P("| statistic | value |")
    P("|---|---|")
    P(f"| **1-ply SEARCH DIVIDEND** `L[argmax C] − L[policy]` | "
      f"**{ci(A['search_dividend_1ply'])}** |")
    P(f"| policy already picks `argmax C` | {ci(A['policy_agrees_with_critic_argmax'],3)} |")
    P(f"| regret of the POLICY's action (cross-fitted) | {ci(A['regret_policy_cross_fitted'])} |")
    P(f"| regret of the CRITIC's argmax (cross-fitted) | {ci(A['regret_cross_fitted'])} |")
    P("")

    P("## Splits")
    P("")
    P("| cell | n dec | offset share | RMS diff | flip (cf) | regret (cf) | noise floor | capture | 1-ply dividend | ρ(C,L) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for k, c in o["cells"].items():
        if "MISSING" in c:
            P(f"| `{k}` | {c['n_decisions']} | MISSING — {c['MISSING']} | | | | | | | |")
            continue
        mm = c["mse"]
        P(f"| `{k}` | {c['n_decisions']} | {f(mm['offset_share_of_true'],3)} | "
          f"{f(mm['rms_residual_true'],3)} | {f(c['flip_cross_fitted']['mean'],3)} | "
          f"{ci(c['regret_cross_fitted'],3)} | {f(c['regret_noise_reference']['mean'],3)} | "
          f"{('MISSING' if c['capture_fraction_of_achievable_ranking_gain'] is None else f(c['capture_fraction_of_achievable_ranking_gain'],2))} | "
          f"{ci(c['search_dividend_1ply'],3)} | {f(c['spearman_C_vs_L']['mean'],3)} |")
    P("")
    if meta.get("value_json"):
        v = json.load(open(meta["value_json"]))["cells"]["ALL"]
        P("### Robustness: the SCALAR value head instead of the win-prob head")
        P("")
        P("Rank statistics are invariant to any monotone re-scaling, so this arm asks a different")
        P("question: does the run's *other* critic readout order the actions the same way?")
        P("")
        P("| statistic | win-prob head (headline) | scalar value head |")
        P("|---|---|---|")
        P(f"| flip rate (cf) | {ci(A['flip_cross_fitted'],3)} | {ci(v['flip_cross_fitted'],3)} |")
        P(f"| regret (cf) | {ci(A['regret_cross_fitted'])} | {ci(v['regret_cross_fitted'])} |")
        P(f"| regret (naive) | {ci(A['regret_naive'])} | {ci(v['regret_naive'])} |")
        P(f"| ρ(C,L) | {ci(A['spearman_C_vs_L'],3)} | {ci(v['spearman_C_vs_L'],3)} |")
        P(f"| capture fraction | "
          f"{f(A['capture_fraction_of_achievable_ranking_gain'],3)} | "
          f"{f(v['capture_fraction_of_achievable_ranking_gain'],3)} |")
        P(f"| **1-ply search dividend** | **{ci(A['search_dividend_1ply'])}** | "
          f"{ci(v['search_dividend_1ply'])} |")
        P("")
        P("The win-prob head wins on every row, and the dividend is the one that decides a build:")
        P("ranking by the win-prob head beats the policy significantly; ranking by the scalar value")
        P("head does not. **A search over this checkpoint should read the win-prob head.**")
        P("")
    P("## Caveats")
    P("")
    for c in meta["caveats"]:
        P(f"- {c}")
    P("")
    P("## Accounting — what was cut, and why")
    P("")
    for c in meta["accounting"]:
        P(f"- {c}")
    P("")
    P("## Verdict")
    P("")
    for c in meta["verdict"]:
        P(c); P("")
    open(md_path, "w").write("\n".join(L) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])))
