# Quality-diversity and open-endedness — the archive view of the flywheel

*MAP-Elites, POET, and stepping stones: the literature for when you want a good one of EACH kind
rather than the best one overall — and why the slice registry is already a hand-built instance.*

## 1. Intuitive: illumination, not optimization

Ordinary optimization asks "what is the best policy?" Quality-diversity (QD) asks "for each KIND
of situation, what is the best policy of that kind?" — it wants a well-lit MAP of the possibility
space, not a single peak. The motivating discovery (Lehman & Stanley's novelty search): for hard
problems, directly chasing the objective often fails, because the stepping stones to the best
solutions do not themselves score well — a system rewarding only the objective discards them.
Diversity is not a nicety; it is how search escapes deception.

The flywheel's owner-decided structure is already QD-shaped: the goal was never "one exploiter
that wins the most" but "coverage of the meta's kinds" — ~50 teams across curated similarity
slices, each slice getting its own specialist, folded into one generalist.

## 2. The formal objects

**MAP-Elites.** Define BEHAVIOR DESCRIPTORS (axes that say what kind of solution this is, not how
good); grid the descriptor space into cells; keep the best-so-far solution PER CELL (the elite);
mutate elites and file offspring wherever they land. Output: an archive — quality everywhere, one
per niche. **The slice registry is a hand-built MAP-Elites archive**: the descriptors are team
composition (pace class + style tags in `gen3_team_archetypes.json`), the cells are the curated
slices, the elites are the slice teachers, and the tock is the mutation operator. Two QD metrics
translate directly: COVERAGE (how many cells have an elite — the ~50-team target) and QD-SCORE
(sum of elite qualities — the sum of per-slice headroom captured). The archetype-novelty
regression (T1.5: does marginal gain track slice novelty?) is the QD question "are new cells
still being illuminated, or are we re-filling old ones?"

**Descriptor choice is the load-bearing decision.** MAP-Elites is only as good as its axes: a
descriptor that doesn't separate genuinely different strategies collapses distinct niches into
one cell. Today's descriptors are COMPOSITION (what the team is), not STRATEGY (how it is
played) — the known limitation the deferred fingerprint aux would address (a learned behavioral
descriptor: "two teams are similar if the policy plays them similarly"). The pace-class probe
result — decodable from raw obs at 0.456 but discarded by the trunk — says the composition axes
are real; whether they are the RIGHT axes is what revolution-1 evidence should say.

**POET (paired open-ended trailblazer).** The step beyond a fixed archive: co-evolve the NICHES
themselves alongside the agents — generate new environments at the frontier of solvability,
transfer agents between niches, let the curriculum invent itself. The flywheel's analog question
arrives at revolution 3+: today the owner curates slices from a fixed 719-team pool; the POET
move is letting the system PROPOSE slices — teams (or team perturbations) where the current
generalist's exploitability bound is fattest. Team-building as niche generation is the natural
end state and is deliberately out of scope until the fixed-pool loop is proven.

**Stepping stones and goal-switching.** QD's deepest claim: solutions to hard niches are often
REACHED VIA other niches (an agent transferred from an easier environment solves one the direct
optimizer never cracks). The flywheel already banks on a weak form — the warm-fork result (fork
0.84@2M vs scratch ~0.65@20M) is "the generalist is a stepping stone to every exploiter." The
strong form worth watching for: an EXPLOITER as the stepping stone to another exploiter (a
trap-core specialist warm-starting a spin-denial specialist), which the task-arithmetic preview
could detect as unusually compatible deltas.

## 3. What this buys operationally

- Report the flywheel in QD terms alongside ELO: coverage (slices with a banked teacher),
  QD-score (Σ headroom captured), and illumination rate (new-cell gains vs re-fill gains, T1.5).
  A flat ELO with rising QD-score is a real and reportable state — the map is getting better
  before the single deployable point does.
- Curation reviews are DESCRIPTOR reviews: when two slices' teachers turn out interchangeable
  (task-arithmetic deltas near-parallel), the cells were one niche — merge them; when one slice's
  teacher bifurcates behaviorally, the cell was two — split it. The archive vocabulary makes
  those calls principled instead of aesthetic.
- The kill condition gains a third reading: flat ELO + flat piloting + flat COVERAGE is true
  convergence; flat ELO with coverage still growing is an archive still filling — different
  verdicts, one extra number.

**The question you can answer after this note:** *the flywheel's revolution 2 shows +40 ELO but
T1.5 says gains no longer track novelty — healthy or not?* Healthy for the spine, warning for the
map: the folds are compressing known niches better rather than illuminating new ones — expect
diminishing returns unless the descriptor set (or the pool itself) grows; that is the moment the
fingerprint aux or POET-style slice proposal earns its re-visit.

Related: [`population_game_theory.md`](population_game_theory.md) (the spinning-top width these
niches live in); the flywheel design (`../ai_v10/design_flywheel_tick_tock.md`) §2-§3; the
descriptor artifacts are `data/teams/gen3_team_archetypes.json` + the owner's slice worksheet.
