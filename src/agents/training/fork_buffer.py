"""BRANCH ROWS INTO THE PPO BUFFER (`--fork-fraction`, `gen3_fork_v1`).

Two jobs, one module, because neither is meaningful without the other: turning a branch's captured
decisions into ordinary PPO rows (:func:`build_branch_rows`), and getting those rows in front of
``train()`` (:class:`ForkRolloutBuffer`).

WHY THE BUFFER IS SUBCLASSED AND NOT RESIZED
--------------------------------------------
``MaskableDictRolloutBuffer.get`` iterates ``buffer_size * n_envs`` and ``reset()`` allocates from
``buffer_size``, so RAISING ``buffer_size`` to make room would resize the NEXT rollout's arrays and
quietly change how much the run collects. Padding the injected rows out to a whole multiple of
``n_envs`` is the other tempting move and is worse: a pad row is a fabricated transition, and the
one thing this subsystem must never do is put a fabricated number into the objective.

So the rows live beside the collected ones in their own FLAT arrays and ``get()`` yields
minibatches over the concatenation. Three properties follow, and they are the ones worth testing:
``reset()`` drops them (they are as per-rollout as the buffer itself), an empty fork set makes
``get()`` byte-identical to upstream's, and a shape the concatenation cannot take is a REFUSAL at
injection time rather than a silent truncation inside a generator.

THE MASK RULE — ONE RULE, AND IT IS UNIFORM
-------------------------------------------
🚨 **The FORK STEP is excluded from the POLICY term for EVERY branch, the top-2 included.** The
design offered a choice — mask only the random branch (whose action the policy did not choose) and
let the clip handle the top-2 (whose true log-prob is recorded). This takes the uniform rule, for
one reason: **an exclusion criterion that depends on WHICH branch a row came from re-weights the
policy gradient by the branch mix.** Masking only ``rand`` would leave ``top1`` and ``top2`` as the
only fork-step rows in the policy term, i.e. it would silently up-weight the policy's own two
candidates at exactly the contested states the arm selects for — an on-policy re-weighting shipped
under a flag whose declared job is to buy VALUE data. The uniform rule buys nothing and costs
nothing: at 3 branches it removes 3 rows per fork from a term that keeps every one of the ~25
post-fork rows the same fork contributes.

The fork step stays FULLY IN the value terms — its return is its own branch's — which is the whole
point of playing the branch at all.

The carrier is the ``fork_pg_m`` obs key: a per-row multiplier that survives
``RolloutBuffer.get()``'s shuffle aligned to its own row, the same mechanism ``win_row_w`` uses.
Collected rows carry the env's 1.0 placeholder, so the term is unchanged wherever the arm did not
reach.

THE PREFIX IS COUNTED ONCE
--------------------------
A branch's rows begin AT the fork step. The turns BEFORE it are the parent episode's, they are
already in the buffer as the parent's own rows, and they are identical across the branches by
construction (that is what "common random numbers" means). Injecting them per branch would put the
same transition in the objective ``branches`` times and multiply the prefix's weight by the fork
rate — a re-weighting of the training distribution by a flag that is supposed to add states.

The fork STATE itself does appear once per branch, with a DIFFERENT action each time. That is the
exploring start, not a duplicate: those rows differ in the action, the return and the successor.

THE FILL TABLE, AND WHY AN UNKNOWN KEY REFUSES
----------------------------------------------
A branch is played by two `RLPlayer`s, not by a `Gen3Env`, so the capture yields ``observation``
and ``action_mask`` and nothing else — while the buffer's obs Dict may carry a dozen flag-gated
LABEL keys. :data:`FILL` declares, per key, what an injected row honestly holds; a key that is not
in the table makes the arm REFUSE AT SETUP, naming the key and the flag. That is deliberate: the
alternative is a heuristic ("anything ending in ``_mask`` is zero"), and a heuristic would keep
working, silently and wrongly, the first time someone adds a key it happens to match.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.observation.true_team import TRUE_TEAM_KEY
from agents.training.fork_arm import PG_MASK_KEY
from agents.training.opp_intent_labels import KIND_UNKNOWN, SWITCH_SLOT_NONE

#: Injected rows are played against a SELF-LIKE opponent (the current snapshot), which is the
#: `MaskableAgentWrapper.OPP_CLASS_POOL` population. Naming it POOL rather than the parent
#: episode's real class is the honest label: the class tags WHO THE ROW WAS PLAYED AGAINST, and
#: these rows were not played against the parent's opponent.
_OPP_CLASS_POOL = 1


def _const(value, dtype):
    return lambda n, ctx, _v=value, _d=dtype: np.full((n,), _v, dtype=_d)


def _const_block(value, shape, dtype):
    return lambda n, ctx, _v=value, _s=shape, _d=dtype: np.full((n,) + tuple(_s), _v, dtype=_d)


#: What an INJECTED row holds for every obs key other than ``observation`` / ``action_mask``.
#: Each entry is ``(builder, why)``; the builder takes ``(n_rows, ctx)`` where ``ctx`` carries
#: ``outcome`` (the branch's win bit) and ``pg_mask`` (the per-row policy-term multiplier) and
#: returns an array whose leading axis is the row axis.
#:
#: 🚨 **THE RULE FOR A LABEL THE BRANCH CANNOT SUPPLY IS "NOT SCORED", NEVER A GUESS.** Every
#: privileged belief label is a fact about the OPPONENT'S TEAM read from ``battle2`` inside the
#: env; a branch has no env, so those rows are masked out of their losses instead of being filled
#: with a plausible number. The cost is supervision on ~1 row in 3; the alternative cost is a
#: belief head trained on fiction.
FILL: Dict[str, Tuple[Any, str]] = {
    # ── the two the arm actually MEANS ──────────────────────────────────────────────────────
    "win_target": (lambda n, ctx: np.full((n, 1), float(ctx["outcome"]), dtype=np.float32),
                   "the BRANCH's own terminal outcome — the label the whole arm exists to add"),
    "win_mask": (_const_block(1.0, (1,), np.float32),
                 "scored: a branch always ran to a real terminal (a CAPPED one is dropped before "
                 "it gets here)"),
    PG_MASK_KEY: (lambda n, ctx: np.asarray(ctx["pg_mask"], dtype=np.float32).reshape(n, 1),
                  "0.0 on the FORK STEP, 1.0 after — see THE MASK RULE"),
    # ── honest present-state values ─────────────────────────────────────────────────────────
    "opp_class": (_const_block(_OPP_CLASS_POOL, (1,), np.int64),
                  "the branch WAS played against a self-like (pool) opponent — the ecology "
                  "approximation, labelled rather than hidden"),
    "win_row_w": (_const_block(1.0, (1,), np.float32),
                  "a MULTIPLIER: an injected row is weighed normally. `--win-prob-rollout-weight` "
                  "doses ROLLOUT ANCHORS, which an injected row is not"),
    # ── not reconstructible outside the env ⇒ NOT SCORED ────────────────────────────────────
    "win_margin": (_const_block(0.0, (1,), np.float32),
                   "Phi_mat lives in the env's reward manager and a branch has none. 0.0 is the "
                   "CONTESTED stratum, which is where a contested fork's rows belong anyway — but "
                   "it is a FILL, so `--win-prob-strata-weight` is REFUSED alongside this arm"),
    "belief_species": (_const_block(-1, (6,), np.int64), "PAD/not-scored sentinel"),
    "belief_moves": (_const_block(-1, (6, 4), np.int64), "PAD/not-scored sentinel"),
    "known_moves": (_const_block(-1, (6, 4), np.int64), "PAD/not-scored sentinel"),
    "belief_spread": (_const_block(0.0, (6, 5), np.float32), "masked off below"),
    "belief_spread_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "belief_nature": (_const_block(0, (6,), np.int64), "masked off below"),
    "belief_nature_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "belief_ev": (_const_block(0.0, (6, 5), np.float32), "masked off below"),
    "belief_ev_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "hp_type_label": (_const_block(-1, (6,), np.int64), "PAD/not-scored sentinel"),
    "hp_type_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "item_label": (_const_block(-1, (6,), np.int64), "PAD/not-scored sentinel"),
    "item_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "opp_action_kind": (_const_block(KIND_UNKNOWN, (1,), np.int64),
                        "UNKNOWN is the intent labels' own MASK value — 'there was no choice we "
                        "can read', which is exactly true of a row whose delta we never folded"),
    "opp_action_num": (_const_block(0, (1,), np.int64), "unread under KIND_UNKNOWN"),
    "opp_switch_slot": (_const_block(SWITCH_SLOT_NONE, (1,), np.int64),
                        "the labels' own 'cannot be tied to a slot' sentinel"),
    "opp_switch_species": (_const_block(0, (1,), np.int64), "unread under KIND_UNKNOWN"),
}

#: Keys whose presence REFUSES the arm rather than being filled, each with the reason. Separated
#: from "not in the table at all" so the refusal can say *why* instead of only *that*.
REFUSE_KEYS: Dict[str, str] = {
    TRUE_TEAM_KEY:
        "--value-true-team declares `opp_true_team`, which the extractor's value route READS and "
        "RAISES on when missing. A branch is played by two RLPlayers with no env, so there is no "
        "`battle2` to build the opponent's TRUE party from, and a zero block would be a fabricated "
        "privileged input rather than an absent one. --value-true-team is a CEILING PROBE, not a "
        "shippable channel; run it or the fork arm, not both.",
    "defensive_opportunity":
        "--defensive-entropy-boost declares `defensive_opportunity`, a REAL present-state fact "
        "(is a recovery/cure legal right now) computed inside Gen3Env. A branch has no env, and a "
        "0.0 fill would tell the state-conditioned entropy boost that every injected row is a "
        "non-defensive state — a systematic, state-dependent lie to the exploration schedule.",
    "bait_opportunity":
        "--bait-* declares `bait_opportunity`, a REAL present-state fact computed inside Gen3Env; "
        "see `defensive_opportunity`. (The bait hunt is CLOSED, so this should not be on.)",
    "distill_mask":
        "distillation declares `distill_mask`, which gates WHICH rows the teacher KL scores. An "
        "injected row has no teacher decision behind it, and both fills are wrong: 1.0 scores a "
        "row the teacher never saw, 0.0 silently shrinks the distillation dose by the fork rate.",
    "aux_target":
        "--win-prob-dense-aux declares `aux_target`/`aux_mask`/`aux_turn`. The targets are the "
        "END-OF-BATTLE per-slot facts of the episode, back-filled by DenseAuxLabelCallback from "
        "`battle1` at the terminal — which for a branch is a battle this process never held. "
        "Masking every injected row out would make the dense head's dose a function of the fork "
        "rate; supplying one would need the branch's own terminal facts threaded back from the "
        "worker. Neither is built, so the combination REFUSES rather than reading half a dose.",
}


def unfillable_keys(obs_keys: Sequence[str]) -> List[str]:
    """Obs keys this module cannot honestly fill for an injected row, in declaration order.

    The RETURN of this function is the arm's setup gate: a non-empty list is a refusal, and the
    caller prints :func:`refusal_text`. ``observation`` and ``action_mask`` come from the capture;
    everything else must be in :data:`FILL`.
    """
    bad = []
    for k in obs_keys:
        if k in ("observation", "action_mask") or k in FILL:
            continue
        bad.append(str(k))
    return bad


def refusal_text(keys: Sequence[str]) -> str:
    """One message per unfillable key — the declared reason when there is one, else the generic."""
    lines = []
    for k in keys:
        why = REFUSE_KEYS.get(k)
        if why is None:
            why = (f"the obs Dict carries `{k}`, which `agents.training.fork_buffer.FILL` does not "
                   f"declare a value for. An injected row is not played inside a Gen3Env, so every "
                   f"env-computed key needs an explicit decision: a real value, a NOT-SCORED "
                   f"sentinel, or a refusal. Add the row (with its reason) or turn the flag off.")
        lines.append(f"  - {k}: {why}")
    return ("🚨 --fork-fraction REFUSED: an injected branch row cannot be built for "
            f"{len(keys)} obs key(s).\n" + "\n".join(lines))


def gae(rewards, values, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(advantages, returns)`` for ONE complete branch that ends at a TERMINAL.

    `RolloutBuffer.compute_returns_and_advantage`'s recursion, specialised to a trajectory that is
    a whole episode: there is no bootstrap, because there is no state after the last one —
    ``next_non_terminal`` is 0 at the end and 1 everywhere else. That specialisation is what makes
    the branch's value target the ORDINARY GAE/lambda-return of its own branch, which the design
    registers as the value target and the reason there is NO ranking term (CLOSED as a lever by
    `paired_refit_discrimination_2026-09-14`: -0.0107 [-0.0249, +0.0028], NOT DETECTED, negative on
    points at every coefficient).
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = r.shape[0]
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        non_terminal = 0.0 if t == n - 1 else 1.0
        next_v = 0.0 if t == n - 1 else v[t + 1]
        delta = r[t] + gamma * next_v * non_terminal - v[t]
        last = delta + gamma * gae_lambda * non_terminal * last
        adv[t] = last
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def branch_rewards(n_rows: int, outcome: float) -> np.ndarray:
    """The branch's whole reward sequence: zeros, then the terminal WIN INDICATOR.

    🚨 **This is why the arm REFUSES any critic but ``winprob``.** Under ``--critic winprob`` the
    reward stream IS the terminal indicator (`combination_checks
    .winprob_critic_needs_the_indicator_terminal`, plus ``--victory-value 1.0`` and
    the terminal-only reward), so a branch's rewards are reconstructible from its outcome bit alone.
    Under ``shaped`` the terminal is the SIGNED one (±V, ``--draw-penalty`` at the cap), which
    this builder does not reproduce. A flag that silently injected zero-reward rows into a
    shaped objective would be teaching the critic that a third of the buffer is inert.
    """
    r = np.zeros(int(n_rows), dtype=np.float32)
    if int(n_rows) > 0:
        r[-1] = float(outcome)
    return r


def build_branch_rows(*, obs: np.ndarray, masks: np.ndarray, actions: np.ndarray,
                      values: np.ndarray, log_probs: np.ndarray, outcome: float,
                      gamma: float, gae_lambda: float, obs_keys: Sequence[str],
                      mask_dims: int) -> Dict[str, Any]:
    """One branch's captured decisions as a block of flat PPO rows.

    ``obs``/``masks``/``actions`` are the worker's capture IN TIME ORDER — row 0 the FORK STEP,
    row 1 the successor. ``values``/``log_probs`` come from the PARENT's live policy, which is the
    policy that collected the buffer, so ``old_log_prob`` is exactly the behaviour policy's and the
    PPO ratio is 1.0 at the first epoch.
    """
    n = int(obs.shape[0])
    rewards = branch_rewards(n, outcome)
    adv, ret = gae(rewards, values, float(gamma), float(gae_lambda))
    # THE MASK RULE: row 0 (the fork step) out of the policy term, every later row in.
    pg = np.ones(n, dtype=np.float32)
    if n:
        pg[0] = 0.0
    ctx = {"outcome": float(outcome), "pg_mask": pg}
    observations: Dict[str, np.ndarray] = {
        "observation": np.asarray(obs, dtype=np.float32),
        "action_mask": np.asarray(masks, dtype=np.float32).reshape(n, int(mask_dims)),
    }
    for key in obs_keys:
        if key in observations:
            continue
        builder, _why = FILL[key]
        observations[key] = builder(n, ctx)
    return {
        "observations": observations,
        "actions": np.asarray(actions, dtype=np.int64).reshape(n, 1),
        "values": np.asarray(values, dtype=np.float32).reshape(n, 1),
        "log_probs": np.asarray(log_probs, dtype=np.float32).reshape(n, 1),
        "advantages": adv.reshape(n, 1),
        "returns": ret.reshape(n, 1),
        "action_masks": np.asarray(masks, dtype=np.float32).reshape(n, int(mask_dims)),
        "n_rows": n,
        "n_masked": 1 if n else 0,
    }


def concat_blocks(blocks: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Concatenate per-branch blocks into ONE flat block, or None when there are none."""
    blocks = [b for b in blocks if int(b.get("n_rows", 0)) > 0]
    if not blocks:
        return None
    keys = list(blocks[0]["observations"])
    out: Dict[str, Any] = {
        "observations": {k: np.concatenate([b["observations"][k] for b in blocks], axis=0)
                         for k in keys},
    }
    for name in ("actions", "values", "log_probs", "advantages", "returns", "action_masks"):
        out[name] = np.concatenate([b[name] for b in blocks], axis=0)
    out["n_rows"] = int(sum(int(b["n_rows"]) for b in blocks))
    out["n_masked"] = int(sum(int(b["n_masked"]) for b in blocks))
    return out


class ForkRolloutBuffer:
    """Mixin over `MaskableDictRolloutBuffer` that lets EXTRA flat rows ride into ``train()``.

    Defined as a mixin so the concrete class can be built against whichever buffer the algorithm
    already uses (:func:`fork_buffer_class`), rather than pinning one import path here.
    """

    def reset(self) -> None:                                          # type: ignore[override]
        # Fork rows are as per-rollout as the buffer itself: a row left over from the PREVIOUS
        # rollout was measured under weights this one has already left behind.
        self._fork_rows = None
        super().reset()                                               # type: ignore[misc]

    #: The per-row arrays upstream's `get()` flattens beside the obs dict. ``action_masks`` is the
    #: maskable buffer's and is filtered out on a base that does not carry one — the mixin is
    #: advertised as composing with whichever buffer the algorithm selected, and a hard-coded list
    #: would make that a lie the first time it composed with anything else.
    _FLAT_TENSORS = ("actions", "values", "log_probs", "advantages", "returns", "action_masks")

    def _flat_tensor_names(self) -> Tuple[str, ...]:
        return tuple(t for t in self._FLAT_TENSORS if t in self.__dict__)

    @property
    def n_fork_rows(self) -> int:
        rows = getattr(self, "_fork_rows", None)
        return 0 if rows is None else int(rows["n_rows"])

    def add_fork_rows(self, block: Optional[Dict[str, Any]]) -> int:
        """Attach ``block`` (a :func:`concat_blocks` result) for THIS rollout. Returns the count.

        REFUSES a block whose obs keys or widths do not match the collected arrays: a shape the
        concatenation cannot take must fail here, where the message can name the key, rather than
        inside ``get()``'s generator where it would surface as a torch stack error three frames
        into the loss.
        """
        if block is None or int(block.get("n_rows", 0)) <= 0:
            self._fork_rows = None
            return 0
        own = set(self.observations)                                  # type: ignore[attr-defined]
        got = set(block["observations"])
        if own != got:
            raise ValueError(
                f"fork rows carry obs keys {sorted(got)} but the buffer holds {sorted(own)} "
                f"(missing {sorted(own - got)}, extra {sorted(got - own)})")
        for k, arr in block["observations"].items():
            want = self.observations[k].shape[2:]                     # type: ignore[attr-defined]
            if tuple(np.asarray(arr).shape[1:]) != tuple(want):
                raise ValueError(f"fork rows' `{k}` has shape {np.asarray(arr).shape[1:]}, "
                                 f"buffer wants {tuple(want)}")
        self._fork_rows = block
        return int(block["n_rows"])

    def get(self, batch_size=None):                                   # type: ignore[override]
        rows = getattr(self, "_fork_rows", None)
        if rows is None:
            # BYTE-IDENTICAL to upstream when the arm injected nothing — including the order the
            # permutation is drawn in, because upstream's generator is simply called.
            yield from super().get(batch_size)                        # type: ignore[misc]
            return
        n_own = int(self.buffer_size) * int(self.n_envs)               # type: ignore[attr-defined]
        n_extra = int(rows["n_rows"])
        total = n_own + n_extra
        if not self.generator_ready:                                   # type: ignore[attr-defined]
            for key, obs in self.observations.items():                 # type: ignore[attr-defined]
                self.observations[key] = self.swap_and_flatten(obs)    # type: ignore[attr-defined]
            for tensor in self._flat_tensor_names():
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            # Concatenated ONCE, at the same point upstream flattens, so every later call sees one
            # contiguous array per field and `_get_samples`' fancy indexing is unchanged.
            for key, arr in rows["observations"].items():
                self.observations[key] = np.concatenate(                # type: ignore[attr-defined]
                    [self.observations[key], np.asarray(arr, dtype=self.observations[key].dtype)],
                    axis=0)
            for tensor in self._flat_tensor_names():
                cur = self.__dict__[tensor]
                add = np.asarray(rows[tensor], dtype=cur.dtype).reshape((n_extra,) + cur.shape[1:])
                self.__dict__[tensor] = np.concatenate([cur, add], axis=0)
            self.generator_ready = True                                # type: ignore[attr-defined]
        indices = np.random.permutation(total)
        if batch_size is None:
            batch_size = total
        start = 0
        while start < total:
            yield self._get_samples(indices[start:start + batch_size])  # type: ignore[attr-defined]
            start += batch_size


def fork_buffer_class(base):
    """``type(base.__name__ + 'WithForks', (ForkRolloutBuffer, base), {})`` — memoised.

    A factory rather than a literal subclass so the arm composes with whatever buffer the algorithm
    already selected (today `MaskableDictRolloutBuffer`), and so the test can build one over a
    stand-in without importing sb3_contrib.
    """
    cache = fork_buffer_class.__dict__.setdefault("_cache", {})
    cls = cache.get(base)
    if cls is None:
        cls = type(base.__name__ + "WithForks", (ForkRolloutBuffer, base), {})
        cache[base] = cls
    return cls


def install_fork_buffer(model) -> None:
    """Replace ``model.rollout_buffer`` with a fork-capable one, in place, on BOTH build paths.

    Called from `main.train.model_build.apply_training_hparams`, which is the one function both the
    fresh build and the resume run through — and AFTER ``_setup_model``, because that is what
    created the buffer this one replaces. A fresh allocation rather than a class re-tag: the arrays
    are untouched at this point in the launch, so building the right object costs one allocation
    and leaves no half-initialised instance behind.

    ⚠️ **`MaskablePPO._setup_model` hard-codes its buffer class** — it does not read
    ``rollout_buffer_class`` — so anything that re-runs ``_setup_model`` after this point (a
    ``set_env`` on a live model) would silently put the plain buffer back and the arm would then
    disable itself with `ForkArmCallback`'s "not a ForkRolloutBuffer" message rather than fail
    quietly. Nothing in the launch does that today; the callback's check is what keeps it honest if
    something ever does.
    """
    buf = model.rollout_buffer
    if isinstance(buf, ForkRolloutBuffer):
        return
    cls = fork_buffer_class(type(buf))
    model.rollout_buffer = cls(
        buf.buffer_size, buf.observation_space, buf.action_space, device=model.device,
        gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=buf.n_envs)
