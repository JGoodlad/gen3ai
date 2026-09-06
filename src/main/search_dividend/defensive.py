"""DEFENSIVE PAIRED SEARCH — the composite the G/H/I probe trio each supplied one part of.

Every previous search arm in this battery lost, and it lost in one shape: it **overruled the
policy on noise**. The mirror cells put plain depth-1 search at 0.292 against a null of 0.500
that is a construction rather than an estimate, and the only arm that did not lose (``playoff``,
0.450) is the one that searched 14.5% of decisions and changed 7.4% of them. So this strategy is
not "search, but better allocated". It is **the policy, with a search that must earn the right to
interrupt it**, and every one of its four components is a measurement rather than a preference:

1. **The GATE (probe H).** ``n_legal <= 1 or |P(win) - 0.5| >= 0.15`` plays the policy instantly.
   H swept the whole frontier: raw action-flip rate is *unseparable* (flat at ~0.69 no matter how
   the forced class is cut, and it RISES to 0.78 as the class shrinks), but flip COST is highly
   separable and `|P(win) - 0.5|` is the only cheap feature that finds it — every
   policy-confidence feature (logit gap, entropy, top-1 probability) sits at or BELOW the random
   null, and a drop-one ablation says removing the logit gap *improves* the classifier. At 0.15
   the rule forces 82.5% of decisions and still retains 31.0% of the claimed dividend against a
   random triage's 16.5%. *The policy does not know when search will overrule it; the critic knows
   when being overruled would not matter.*

2. **The LEAF (probe G).** The per-decision score is the **win-prob head's** one-ply read, and
   that is a measured choice with a measured cost to getting it wrong. On 317 decisions with
   142,208 terminal Monte-Carlo rollouts, ranking by the win-prob head beat the action the policy
   actually played by **+0.0219 [+0.0089, +0.0364]** win probability; the same measurement on the
   SCALAR value head gives **+0.0135 [-0.0007, +0.0280]**, which does not clear zero. Capture
   fraction 0.712 vs 0.569. So :func:`check_leaf` treats a silent fall-back to the value head as
   an ERROR rather than a degradation — see its docstring, and the test that fails if anyone
   reverts it.

3. **The RACE (probe G x I).** Candidates are compared on CRN-PAIRED rounds. G decomposed the
   critic's leaf error and found **72.8% of it is a per-decision OFFSET shared by every action at
   that decision** [0.674, 0.780] — which a paired comparison cancels *exactly* and for free. Only
   27.2% is differential. That offset is per-DECISION and not global (the global calibration bias
   is 0.26% of the total), so it cancels between SIBLINGS and not between NODES: the pairing
   argument holds at depth 1 and collapses at depth >= 2, which is why a defensive round is depth
   1 and why that is a finding rather than a shortcut.

4. **The FUTILITY STOP (probe I).** I measured the separation distribution and it is U-SHAPED
   with an empty middle: **52.2% of root decisions never separate at all** within 32 paired
   samples, and of those that do, the MEDIAN separates at the minimum-samples floor. There is
   almost no "try harder and it will resolve". So a race that does not separate is not an
   unfinished race — it is a decision the search cannot tell apart, and the honest verdict there
   is the policy's own action plus the clock handed back.

**The rule the whole module exists to enforce: an OVERRULE requires SEPARATION.** Not a higher
mean, not a wider margin — separation under the `seq` rule's anytime-valid union bound, which is a
family-wise error guarantee over every look and every comparison in the race rather than a
per-look p-value. Everything else plays what the policy would have played anyway.

**Two counters that must never be read as one.** ``forced`` (the gate declined to search) and
``futility`` (the search ran and declined to overrule) are opposite findings: the first says the
position was decided, the second says the actions were indistinguishable. A cell where one is
large and the other is zero is a different instrument from a cell where they are balanced, and
folding them together would hide exactly the thing this design is a bet about.

This module is PURE — no sim, no model, no clock. The engine's side of the contract lives in
:mod:`search`; :func:`fold_defensive` is the additive results-row fold, the same shape
``playoff.fold_playoff`` and ``racing.fold_racing`` already use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from agents.model.critic_mode import is_winprob

#: The leaf readouts a defensive search may score on. ``winprob`` is the DEFAULT and it is
#: measured, not preferred (probe G, above). ``value`` exists so the losing arm of that
#: measurement can be re-run as a control rather than argued about.
LEAF_WINPROB = "winprob"
LEAVES = (LEAF_WINPROB, "value")

#: probe H's chosen operating point. A decision is FORCED (play the policy, spend nothing) when
#: ``|P(win) - 0.5| >= this``. 0.15 is where the 150 s ladder budget affords ~16-20 s on the ~15%
#: of decisions that remain — the same searched-fraction regime as the only historical arm that
#: did not lose. Raising it searches more and retains more claimed dividend; lowering it is
#: strictly safer and strictly less useful.
DEFAULT_WP_MARGIN = 0.15

#: A decision with this many legal actions or fewer cannot exhibit a ranking error, so there is
#: nothing for a search to find. It is a separate clause from the win-prob one because it is a
#: fact about the position's branching, not about its decidedness.
FORCED_N_LEGAL = 1

# -- the gate's reasons. Recorded per decision so a cell's forced mass can be attributed. --------
GATE_SEARCH = ""                       # not forced: the search runs
GATE_N_LEGAL = "n_legal"               # <= FORCED_N_LEGAL legal actions
GATE_WP_EXTREME = "wp_extreme"         # the position is decided; being overruled would not matter

#: What the defensive verdict was, per decision. These are DECISIONS the strategy made, not
#: failures — ``kept`` and ``futility`` both play the policy's action, and they say different
#: things about why.
VERDICT_FORCED = "forced"              # the gate declined to search at all
VERDICT_FUTILITY = "futility"          # the race ran and never separated -> keep the policy action
VERDICT_KEPT = "kept"                  # the race separated ON the policy's own action
VERDICT_OVERRULED = "overruled"        # the race separated on a DIFFERENT action -> play it
VERDICT_NONE = ""                      # the decision never reached the defensive rule


class DefensiveLeafError(RuntimeError):
    """The scorer returned a different leaf than the strategy requires.

    Its own type because the failure it names is specific and silent by nature: a checkpoint with
    no win-prob head makes ``batch_scores`` return the SCALAR VALUE readout with no error at all,
    and the search would then run to completion, report a full set of healthy-looking counters,
    and be ranking on the one estimator probe G measured as NOT beating the played action. This
    class is what turns that into a counted ``search_error`` instead of a wrong result.
    """


@dataclass(frozen=True)
class DefensiveConfig:
    """The four knobs, in the order the strategy applies them.

    ``wp_margin`` is the gate (H); ``leaf`` is the scorer (G); the RACE's own parameters live in
    :class:`~main.search_dividend.racing.RacingConfig` and are not duplicated here, because a
    second copy of an elimination threshold is how two allocators end up disagreeing about what
    they measured. ``confirm_rollouts`` is the OPTIONAL fourth stage: 0 (the default) means an
    overrule acts on the race alone, and any positive value settles the proposed overrule with
    that many PAIRED terminal rollouts through the playoff mechanism first.

    ``confirm_deadline_s`` is the clock the CONFIRM stage runs on, and it exists because the
    built confirm had no reachable one. The confirm was handed the race's OWN
    :class:`~main.search_dividend.budget.Deadline`, i.e. whatever the race left over — and
    iteration 2 measured that residual at roughly a second on a separated race against a
    ``playoff``-measured **1.5 s per PAIR**. The playoff runner always runs its first pair and
    then requires ``deadline.fits(2 * rollout_cost_s)``, so a shared clock buys exactly ONE pair,
    ``paired_stats`` returns ``se = inf`` at ``n = 1``, and :func:`~playoff.is_conclusive` refuses
    below ``MIN_PAIRS = 4``. Every confirm would have declined, for want of a clock rather than
    for want of evidence — a null that would have read as a finding. ``None`` (the default) keeps
    that shared-clock behaviour exactly, so the built code is unchanged until asked; a positive
    value gives the confirm its OWN fresh deadline, which is simultaneously the adjudication CAP
    that keeps a nested rollout family from out-accruing the live battle's idle watchdog.

    ``contested_deadline_s`` is the TIME MANAGER, and it is the one change iteration 2 makes.
    ``None`` (the default) means a contested decision gets the same ``--budget`` every decision
    would have got, which is exactly what the first cell measured. A positive value spends the
    bank instead: the gate already hands back the whole budget on the ~74% of decisions it
    forces, so the notional clock a uniform search would have burned there is real and unspent
    (measured: **0.77 s of every 1 s, 28.8 s per game**), and this is what lets a contested
    decision draw on it. The first cell's diagnosis is the entire reason the knob exists — its
    mean race ran **4.61 rounds against the ``seq`` rule's elimination FLOOR of 5**, and every
    one of its 3,301 futility stops was also ``deadline_truncated`` (an exact identity), so the
    strategy was BUDGET-limited at the floor rather than evidence-limited. It buys ROUNDS and
    nothing else: :func:`~main.search_dividend.budget.allocate`'s only live output on the racing
    path is ``m_opp``, which the first cell already ran at a mean of 5.77 against its cap of 6.
    """

    wp_margin: float = DEFAULT_WP_MARGIN
    leaf: str = "winprob"
    confirm_rollouts: int = 0
    confirm_deadline_s: Optional[float] = None
    contested_deadline_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.leaf not in LEAVES:
            raise ValueError(f"unknown defensive leaf {self.leaf!r} (want one of {LEAVES})")
        if not (0.0 <= float(self.wp_margin) <= 0.5):
            # 0.5 forces nothing on the win-prob clause (no probability is 0.5 away from 0.5);
            # 0.0 forces everything. Outside that range the rule is not a rule.
            raise ValueError("wp_margin must lie in [0.0, 0.5]")
        if int(self.confirm_rollouts) < 0:
            raise ValueError("confirm_rollouts must be >= 0")
        if self.confirm_deadline_s is not None and float(self.confirm_deadline_s) <= 0.0:
            # Same reasoning as the contested deadline below: 0 is not "off" (``None`` is), and a
            # zero confirm clock would decline every overrule for lack of a second rather than for
            # lack of evidence — the exact failure this field exists to make impossible.
            raise ValueError("confirm_deadline_s must be > 0 (None = share the race's clock)")
        if self.contested_deadline_s is not None and float(self.contested_deadline_s) <= 0.0:
            # 0 is NOT "off" — off is `None`. A zero deadline would expire before the first round
            # and turn every contested decision into a fallback, which reads in the counters as a
            # strategy that refused rather than a clock that was never granted.
            raise ValueError("contested_deadline_s must be > 0 (None = use --budget)")

    def deadline_for(self, budget_s: float) -> float:
        """The wall-clock a CONTESTED decision actually gets.

        One function, so the ``Deadline`` and the width ``allocate`` can never be handed two
        different numbers — a plan sized to a budget the clock does not honour over-runs, and a
        plan sized below the clock silently leaves the extra seconds unreachable, which is the
        precise failure the first cell measured on the *forced* side of the gate.
        """
        return (float(budget_s) if self.contested_deadline_s is None
                else float(self.contested_deadline_s))

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_score_mode(leaf: str) -> str:
    """The ``batch_scores`` mode this leaf demands — EXPLICIT, never ``auto``.

    ``auto`` is the battery's default and it prefers the win-prob head *and silently falls back to
    the value head* when the run trained none. For a defensive search that fall-back is precisely
    the failure probe G measured, so the mode is named outright and the result is checked (see
    :func:`check_leaf`). A default that can quietly become the losing arm is not a default.
    """
    if leaf not in LEAVES:
        raise ValueError(f"unknown defensive leaf {leaf!r} (want one of {LEAVES})")
    return "win_prob" if leaf == "winprob" else "value"


# ---------------------------------------------------------------------------
# the CRITIC MODE narrows both legal sets (gen3_winprob_critic_mode_v1, design gap B10)
# ---------------------------------------------------------------------------

#: `--score`'s legal set on a `shaped` policy. Unchanged, and `auto` is still the CLI default.
SCORES = ("auto", "value", "win_prob")


def leaves_for_critic(critic_mode: object) -> tuple:
    """The `--defensive-leaf` values a checkpoint trained under `critic_mode` can honestly take.

    Under `--critic winprob` there is exactly ONE readout — `predict_values` IS
    `sigmoid(win-prob logit)` — so `value` names a critic that is in no loss graph and would be
    the SAME number wearing a different label. It is refused rather than aliased: probe G's whole
    finding is that the two leaves are different arms with different measured worth, and a run
    that silently made them one would report a control it never ran.
    """
    return (LEAF_WINPROB,) if is_winprob(critic_mode) else LEAVES


def scores_for_critic(critic_mode: object) -> tuple:
    """The `--score` values that mean something on `critic_mode`. See :func:`resolve_for_critic`."""
    return ("win_prob",) if is_winprob(critic_mode) else SCORES


def resolve_for_critic(critic_mode: object, score: str, leaf: Optional[str] = None):
    """``(score, leaf, notes)`` — the CLI's request, narrowed by the critic the model actually has.

    **`auto` does not survive contact with a winprob policy, and it does not survive as a
    SILENT default either.** The design's §3.10 says `--score auto` dies; the shape that ships is
    a RESOLUTION rather than a refusal, and the distinction is worth stating because both readings
    of "dies" are defensible:

    * A refusal would make the FLAGLESS invocation — `python -m main.search_dividend <ckpt>` — a
      usage error on every winprob checkpoint, since `auto` is the CLI default. The battery is
      required to run end to end on this critic; an arm that cannot be launched without a flag is
      not a working battery.
    * What made `auto` dangerous was never the spelling — it was the **fall-back**: it prefers the
      win-prob head *and silently drops to the value head* when the run trained none, which is
      exactly the substitution :func:`check_leaf` exists to catch. On a winprob policy that
      fall-back is unreachable by construction: there is one readout, so `auto` can only ever
      resolve to it. The hazard is gone, not merely tolerated.

    So `auto` RESOLVES to `win_prob` and the resolution is ANNOUNCED and recorded, never inferred
    by a reader from the absence of a message. An EXPLICIT `value` on either flag RAISES, because
    that is a request for the arm this checkpoint does not have — and a default that quietly
    became the losing arm is the class this module was written against.

    `shaped` is returned untouched, notes empty: not "narrowed to the same values", but never
    consulted at all.
    """
    if not is_winprob(critic_mode):
        return score, leaf, []

    notes = []
    if str(score) == "value":
        raise ValueError(
            "--score value is refused on a --critic winprob model: this checkpoint has ONE value "
            "readout (predict_values IS the win-prob head's sigmoid), so 'value' names a critic "
            "that is in no loss graph. Pass --score win_prob.")
    if str(score) == "auto":
        score = "win_prob"
        notes.append("--score auto -> win_prob (a winprob critic has ONE readout, so `auto` has "
                     "nothing to fall back to; resolved rather than left implicit)")
    if leaf is not None and leaf not in leaves_for_critic(critic_mode):
        raise ValueError(
            f"--defensive-leaf {leaf!r} is refused on a --critic winprob model: the only leaf is "
            f"{LEAF_WINPROB!r} (probe G's `value` control arm does not exist on this critic — "
            "predict_values and the win-prob head are the same number). Pass "
            f"--defensive-leaf {LEAF_WINPROB}.")
    return score, leaf, notes


def check_leaf(mode_used: str, cfg: Optional[DefensiveConfig]) -> None:
    """Raise unless the scorer actually delivered the leaf the strategy asked for.

    Called at the ONE seam where a leaf can silently change — the return of ``batch_scores``,
    which reports the mode it *used* rather than the mode it was *asked for*. ``cfg`` of ``None``
    means this is not a defensive search and the check is a no-op, so the grid and racing paths
    are untouched.

    This is the assertion probe G's headline is worth: the win-prob head beats the played action
    by +0.0219 [+0.0089, +0.0364] and the scalar value head by +0.0135 [-0.0007, +0.0280] — one
    clears zero and one does not, on the same decisions, with the same labels. A run that degraded
    from the first to the second would look identical in every counter this battery records.
    """
    if cfg is None:
        return
    want = resolve_score_mode(cfg.leaf)
    if str(mode_used) != want:
        raise DefensiveLeafError(
            f"defensive leaf is {cfg.leaf!r} (score mode {want!r}) but the scorer returned "
            f"{mode_used!r}. On this checkpoint the requested head is unavailable, so the search "
            "would silently rank on the readout that does NOT beat the played action (probe G: "
            "win-prob +0.0219 [+0.0089, +0.0364] vs value +0.0135 [-0.0007, +0.0280]). Train the "
            "win-prob head, or ask for --defensive-leaf value explicitly.")


def gate(n_legal: int, win_prob: Optional[float],
         cfg: DefensiveConfig = DefensiveConfig()) -> str:
    """The triage gate. Returns a reason string; :data:`GATE_SEARCH` (``""``) means SEARCH.

    Two clauses, ORed, and they are separate because they mean different things:

    * ``n_legal <= 1`` — there is no choice to make. This is checked FIRST and without reference
      to the win probability, because it holds whatever the critic thinks.
    * ``|P(win) - 0.5| >= wp_margin`` — the position is decided, so the win-probability at stake
      in being overruled is small. This is H's whole finding: the flip RATE is flat and
      unseparable, the flip COST is not.

    ``win_prob`` of ``None`` is a MISSING MEASUREMENT, not a mid-range one. The caller must have
    already refused the decision (the engine counts ``defensive_no_win_prob``); imputing 0.5 here
    would silently convert every decision on a headless checkpoint into a searched one, which is
    the most expensive possible reading of an absent number.
    """
    if int(n_legal) <= FORCED_N_LEGAL:
        return GATE_N_LEGAL
    if win_prob is None:
        raise ValueError("gate() needs a win probability — see the module docstring; the caller "
                         "must count `defensive_no_win_prob` rather than impute 0.5")
    if abs(float(win_prob) - 0.5) >= float(cfg.wp_margin):
        return GATE_WP_EXTREME
    return GATE_SEARCH


def verdict(separated: bool, race_action: int, policy_action: int) -> str:
    """The post-race verdict: :data:`VERDICT_OVERRULED` requires SEPARATION, and nothing else does.

    This function is the entire "defensive" claim in three lines, so it is worth stating what it
    REFUSES: a race that ended with several actions still live has a leader (the racer always has
    one — it is the empirical best mean), and acting on that leader is exactly what every losing
    arm in this battery did. Probe I measured that 52.2% of decisions never separate, so the
    refused mass is the majority of decisions, not an edge case.
    """
    if not separated:
        return VERDICT_FUTILITY
    return VERDICT_KEPT if int(race_action) == int(policy_action) else VERDICT_OVERRULED


def resolve_action(verdict_name: str, race_action: int, policy_action: int) -> int:
    """Which action a verdict actually plays. Only an OVERRULE leaves the policy's own choice."""
    return int(race_action) if verdict_name == VERDICT_OVERRULED else int(policy_action)


def fold_defensive(decisions: Sequence[Mapping]) -> dict:
    """Fold the per-decision defensive counters into the ones a results row carries.

    ADDITIVE by construction, the same rule ``playoff.fold_playoff`` and ``racing.fold_racing``
    follow (ladder requirement 3, 87a3f91): a GRID or RACING row folds to zeros and reads exactly
    as it always did, so one schema still covers the whole battery.

    SUMS, never means of per-game means — a cell pools games whose decision counts differ by 2-3x,
    and ``eval_sharding``'s exact Sigma-won/Sigma-finished rule applies here for the same reason.

    ``defensive_banked_s`` is the clock the strategy did NOT spend: the whole per-decision budget
    on a gated decision, and whatever the race left on a futility stop. It is the quantity a time
    manager would redistribute, so it is carried as a total rather than inferred from a rate.

    ``n_defensive_confirmed`` counts an overrule the CONFIRM STAGE UPHELD, and "upheld" is read
    off the VERDICT rather than off the playoff's stage — because ``stage == "played"`` does not
    mean the overrule survived. :func:`~playoff.decide` returns ``STAGE_PLAYED`` whenever the
    paired difference clears ``2·SE`` **in either direction**, so a confirm whose rollouts
    conclusively preferred the POLICY's action is also ``played``, and counting it as a
    confirmation would have inverted the one number this stage exists to report. The rejections
    are therefore split four ways — ``reversed`` (conclusive, and against the overrule),
    ``inconclusive`` (the honest refusal), ``no_budget``, ``error`` — because only the first two
    are findings about the leaf and the last two are findings about the clock.

    ``defensive_confirm_events`` carries ONE compact record per attempted confirm (turn, legal
    count, root P(win), the race's own leaf margin, the paired mean/SE and the verdict). It is a
    LIST rather than more counters because the registered diagnostic is "what distinguishes a
    confirmed overrule from a rejected one", which is a per-decision question and cannot be
    answered from sums. At ~2 confirms per game it costs the row a few hundred bytes.

    ``n_defensive_futility_deadline`` SPLITS the futility mass, and the split is the whole reason
    iteration 2 can be scored. A futility stop means "the race did not separate", but that sentence
    has two readings which the first cell could not tell apart because they coincided in 100% of
    its 3,301 stops: the race **ran out of clock** (``deadline_truncated`` — a budget finding, and
    the one the time manager is meant to remove), or it ran its supply out and the actions were
    GENUINELY indistinguishable (probe I's U-shape — a fact about the game). Reporting the total
    alone would let a strategy that merely bought more rounds look identical to one that learned
    something, so the two are counted apart at the fold rather than inferred afterwards.
    """
    out = {"n_defensive": 0, "n_defensive_forced": 0, "n_defensive_forced_n_legal": 0,
           "n_defensive_forced_wp": 0, "n_defensive_raced": 0, "n_defensive_separated": 0,
           "n_defensive_overruled": 0, "n_defensive_futility": 0,
           "n_defensive_futility_deadline": 0, "n_defensive_kept": 0,
           "n_defensive_no_win_prob": 0, "n_defensive_confirm_attempted": 0,
           "n_defensive_confirmed": 0, "n_defensive_confirm_declined": 0,
           "n_defensive_confirm_reversed": 0, "n_defensive_confirm_inconclusive": 0,
           "n_defensive_confirm_no_budget": 0, "n_defensive_confirm_error": 0,
           "defensive_confirm_s": 0.0, "defensive_banked_s": 0.0}
    events: list = []
    for d in decisions:
        w = d.get("widths") or {}
        v = str(w.get("defensive_verdict") or "")
        if not v and not w.get("defensive_no_win_prob"):
            continue
        out["n_defensive"] += 1
        if w.get("defensive_no_win_prob"):
            out["n_defensive_no_win_prob"] += 1
        if v == VERDICT_FORCED:
            out["n_defensive_forced"] += 1
            reason = str(w.get("defensive_gate_reason") or "")
            if reason == GATE_N_LEGAL:
                out["n_defensive_forced_n_legal"] += 1
            elif reason == GATE_WP_EXTREME:
                out["n_defensive_forced_wp"] += 1
        elif v in (VERDICT_FUTILITY, VERDICT_KEPT, VERDICT_OVERRULED):
            out["n_defensive_raced"] += 1
            if v == VERDICT_FUTILITY:
                out["n_defensive_futility"] += 1
                if w.get("deadline_truncated"):
                    out["n_defensive_futility_deadline"] += 1
            else:
                out["n_defensive_separated"] += 1
                if v == VERDICT_KEPT:
                    out["n_defensive_kept"] += 1
                else:
                    out["n_defensive_overruled"] += 1
        stage = str(w.get("defensive_confirm_stage") or "")
        if stage:
            out["n_defensive_confirm_attempted"] += 1
            out["defensive_confirm_s"] += float(w.get("defensive_confirm_s", 0.0) or 0.0)
            upheld = (v == VERDICT_OVERRULED)
            if upheld:
                out["n_defensive_confirmed"] += 1
            else:
                out["n_defensive_confirm_declined"] += 1
                if stage == "played":
                    # CONCLUSIVE, and against the overrule: the rollouts cleared 2·SE preferring
                    # the policy's own action. The sharpest single reading of leaf bias this
                    # instrument produces — it is not "we could not tell", it is "the leaf was
                    # wrong" — so it never shares a counter with the refusals below it.
                    out["n_defensive_confirm_reversed"] += 1
                elif stage == "inconclusive":
                    out["n_defensive_confirm_inconclusive"] += 1
                elif stage == "error":
                    out["n_defensive_confirm_error"] += 1
                else:
                    out["n_defensive_confirm_no_budget"] += 1
            po = d.get("playoff") or {}
            events.append({
                "turn": int(d.get("turn", 0) or 0),
                "n_legal": int(w.get("n_our_actions", 0) or 0),
                "wp": round(float(w.get("defensive_root_win_prob", -1.0) or -1.0), 4),
                "leaf_margin": round(float(w.get("defensive_leaf_margin", 0.0) or 0.0), 5),
                "stage": stage,
                "upheld": bool(upheld),
                "r": int(po.get("r", 0) or 0),
                "mean": float(po.get("mean", 0.0) or 0.0),
                "se": float(po.get("se", 0.0) or 0.0),
                "wall_s": round(float(w.get("defensive_confirm_s", 0.0) or 0.0), 3),
            })
        out["defensive_banked_s"] += float(w.get("defensive_banked_s", 0.0) or 0.0)
    out["defensive_banked_s"] = round(out["defensive_banked_s"], 3)
    out["defensive_confirm_s"] = round(out["defensive_confirm_s"], 3)
    out["defensive_confirm_events"] = events
    return out


def defensive_block(a: Mapping) -> Optional[dict]:
    """The report block for one aggregated cell, or ``None`` if the cell never ran defensively.

    ``None`` rather than a block of zeros, for the reason ``playoff._playoff_block`` gives: "this
    arm has no defensive stage" and "the defensive stage ran and never overruled anything" are
    opposite findings, and the second one is the whole bet.

    Every rate is quoted against ``n_defensive`` — the decisions the strategy actually handled —
    so ``forced + raced`` accounts for all of them and no rate can be flattered by a denominator
    that excludes the branch it lost on.
    """
    n = int(a.get("n_defensive", 0) or 0)
    if not n:
        return None
    raced = int(a.get("n_defensive_raced", 0) or 0)
    sep = int(a.get("n_defensive_separated", 0) or 0)
    over = int(a.get("n_defensive_overruled", 0) or 0)
    fut = int(a.get("n_defensive_futility", 0) or 0)
    fut_dl = int(a.get("n_defensive_futility_deadline", 0) or 0)
    att = int(a.get("n_defensive_confirm_attempted", 0) or 0)
    conf = int(a.get("n_defensive_confirmed", 0) or 0)

    def rate(k: int, d: int) -> Optional[float]:
        return round(k / d, 4) if d else None

    return {
        "decisions": n,
        "forced": int(a.get("n_defensive_forced", 0) or 0),
        "forced_n_legal": int(a.get("n_defensive_forced_n_legal", 0) or 0),
        "forced_wp": int(a.get("n_defensive_forced_wp", 0) or 0),
        "raced": raced,
        "separated": sep,
        "futility": fut,
        # The two readings of a futility stop, kept apart (see `fold_defensive`): a race the CLOCK
        # ended, versus a race that ran and found the actions genuinely indistinguishable.
        "futility_deadline": fut_dl,
        "futility_genuine": fut - fut_dl,
        "kept": int(a.get("n_defensive_kept", 0) or 0),
        "overruled": over,
        "no_win_prob": int(a.get("n_defensive_no_win_prob", 0) or 0),
        "forced_rate": rate(int(a.get("n_defensive_forced", 0) or 0), n),
        "race_rate": rate(raced, n),
        # THE headline: how often the strategy actually interrupted the policy, over every
        # decision it saw. The registered prediction for the first cell is 0.08-0.17.
        "overrule_rate": rate(over, n),
        # ...and the same number among decisions that were raced at all, which is what a time
        # manager buying more contested decisions would move.
        "overrule_rate_raced": rate(over, raced),
        "separation_rate": rate(sep, raced),
        "futility_rate": rate(fut, raced),
        # Of the futility mass, how much the time manager could still buy back. 1.0 is the first
        # cell's reading — every stop was the clock — and driving it down is what a larger
        # contested deadline is FOR, so it is the number that says whether the knob worked.
        "futility_deadline_frac": rate(fut_dl, fut),
        "banked_s": round(float(a.get("defensive_banked_s", 0.0) or 0.0), 2),
        "banked_s_per_decision": (round(float(a.get("defensive_banked_s", 0.0) or 0.0) / n, 3)
                                  if n else None),
        "confirm_attempted": att,
        "confirmed": conf,
        "confirm_declined": int(a.get("n_defensive_confirm_declined", 0) or 0),
        "confirm_reversed": int(a.get("n_defensive_confirm_reversed", 0) or 0),
        "confirm_inconclusive": int(a.get("n_defensive_confirm_inconclusive", 0) or 0),
        "confirm_no_budget": int(a.get("n_defensive_confirm_no_budget", 0) or 0),
        "confirm_error": int(a.get("n_defensive_confirm_error", 0) or 0),
        # THE LEAF-BIAS METER IN VIVO. An attempted confirm is a decision the race certified as an
        # overrule on the one-ply leaf; the rejection rate is the fraction of those certifications
        # that terminal paired rollouts — which contain the opponent's response, and the leaf does
        # not — would not stand behind.
        "confirm_reject_rate": rate(int(a.get("n_defensive_confirm_declined", 0) or 0), att),
        "confirm_s": round(float(a.get("defensive_confirm_s", 0.0) or 0.0), 2),
        "confirm_s_per_attempt": (round(float(a.get("defensive_confirm_s", 0.0) or 0.0) / att, 3)
                                  if att else None),
    }
