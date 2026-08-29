# Learning note — the GLU family, the clock's journey, and the survival reframe (2026-08-29)

## 1. GLU family
LSTM gates (1997) → GLU (Dauphin 2016): `(W₁x+b) ⊗ σ(W₂x+c)` — content × learned relevance,
elementwise. Shazeer 2020: SwiGLU/GeGLU as transformer FFN beat plain ReLU generically; now the
LLaMA/PaLM/Mistral default. Why: (expressive) a bilinear AND/veto in ONE layer instead of
assembled-across-layers with optimization-resisted large weights; (gradient) closed gates stop
credit from irrelevant contexts — features stop being averaged across situations; (philosophy)
nobody picks what gates what — capability shipped, semantics learned. Costs: 3 matrices (offset
by ⅔ width), arch-version bump, retrain-class, torch.compile re-verification.

## 2. Placement — trace the clock
Current path: 3 clock scalars → global-env encoder → trunk/attention → CLS pool →
`value_pooled` (~10² dims summarizing the whole game) → linear+sigmoid head. A cliff needs the
LOGIT to carry −huge×(t>240); on an additive path that means trunk units with very large
weights at clock thresholds — resisted by decay/smoothness, bought only by data mass. Two
tiers: **systemic** = SwiGLU across trunk towers + FFN sublayers + heads (upstream is where
representation is built; SB3's pi/vf nets are tiny); **surgical first arm** = a GATED READOUT
on the win-prob head (GLU over the head input: content(value_pooled) ⊗ gate(value_pooled)) —
late fusion, small, targeted at the measured defect, predicts whether the systemic swap is
worth its cost.

## 3. The survival reframe (owner insight, formalized)
The head should compute P(win | position, STILL GOING at t) — survival conditioning. Non-
conversion is evidence: convertible positions end games; conditioning on "still going" filters
toward unbreakable-wall / impotent-win-con games. "If it's dragging, drop it" = the posterior
computed by experience; THE DRAG IS THE DIAGNOSIS. Consequences: (1) the cliff decomposes into
a smooth Bayesian slide + small terminal step — Tanh-learnable, softening the architectural
demand; (2) the fix re-weights to the DATA leg (the harvest over-represents dragged games —
where "still going" has bitten) — reinforcing the registered sequencing (data → gated readout →
systemic swap); (3) humans RESIGN — human data embodies the conditioning; our self-play grinds
to the cap because neither side computes "pointless". CAP_TRADE's 0.86-at-245 = failure to
condition on survival, not failure to represent a step. The 0.999 head was never taught that
time-without-conversion is information; the obs has the ingredients (clock + progress), the
label mass for their JOINT is the missing piece.
