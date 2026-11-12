"""THE PROBES' OWN FORWARD (gen3_probe_own_graph_v1): each gradient a diagnostic probe reads is taken from a
re-run of the micro-step it describes, never from the training step's retained graph.

WHY. The grad-balance probe and the per-term noise probe read one gradient per loss term. Reading them off
the micro-batch's live graph (`autograd.grad(retain_graph=True)` before the real `backward()`) costs a whole
backward's transient ON TOP of the retained graph: the learner's peak, by a wide margin (d_model 256 at the
production shape: 9,106 MiB allocated in the noise probe and 9,053 in grad-balance, against 7,139 for the
forward and 6,881 for the real backward — `designs/training/learner_lifecycle.md`, "The update fit check").
That peak is what the startup fit check must admit, so the diagnostics, not training, decided whether an arm
could start; and the noise probe, which ran out of memory first, was caught and dropped per call, so its
scalars existed at one width and not at the other.

HOW. After the micro-batch's real `backward()` (its graph is gone) and BEFORE the optimizer step (the
parameters are the ones the real forward read), a probe calls `ProbeForward` once per gradient: R1 again, on
the same rows, through the same installed region (`TrainSetup._micro_region`, the un-graphed compiled route —
the signature the update already compiled, so no new one). The gradient is taken with `retain_graph=False`,
which frees the graph as it is walked. A probe's peak is then a training forward's plus one gradient.

WHAT IT MAY NOT CHANGE, and why it does not:

  * the update's numbers — `autograd.grad` writes no `.grad`, nothing here touches the optimizer, and the
    re-run sits under `torch.random.fork_rng` (R1 draws nothing today; the guard makes that a property of
    this module, not of the model). Pinned on the learner golden: `probe_own_graph_test.py` and
    `diagnostics_cadence_test.py` (parameters, optimizer state, losses and all three RNG streams, probes on
    versus skipped);
  * the probes' values — the same function of the same inputs. On one thread on the CPU the two routes agree
    to the bit (`probe_own_graph_test.py`); on CUDA they agree to the kernels' run-to-run determinism;
  * what the eager tail reads — R1 re-stashes the extractor's `last_*` tensors with the values the real
    forward left (same rows, same parameters). Every stash READER of a micro-batch (the rank probe, the
    ride-along heads, the calibration read, the capacity snapshot) still runs before the first re-run, and
    `policy._last_pi_distribution` is set outside R1 and is not touched.

WHAT IT COSTS: one R1 forward per probe gradient — per diagnostic update, `2 + n_aux` for grad-balance and
`groups × accum` for the per-term noise probe (`train/noise_per_term_ms` meters the latter live). The
compiled route's dispatch counter counts them (`lifecycle/compiled_region_calls` is higher on a diagnostic
update).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import torch as th

from agents.training.belief_bank import ROWS as _BELIEF_ROWS
from agents.training.grad_balance import BalanceTerms
from agents.training.instrumented_ppo.noise_scale_terms import grouped_terms

# R1's probe terms in the inline fold's REGISTRATION order (the grad-balance probe's and K9(c)'s names):
# the belief bank's rows' `probe` names, then the win-prob head. DERIVED from `belief_bank.ROWS`, never retyped —
# a row registered under a name this tuple does not hold is never looked up, so its `grad/<name>_*` family is
# never produced. (That is how `hidden_team_set`, registered `species_belief_set`, lost `grad/species_belief_*`
# from the X5 version break to 2026-10-15: `gen3_metric_producer_guard_v1`.)
R1_PROBE_TERMS = tuple(_row.probe for _row in _BELIEF_ROWS) + ("win_prob",)


def select_probe_terms(terms: Mapping[str, Any], present: Mapping[str, bool]) -> Dict[str, Any]:
    """The R1 terms the grad-balance probe and K9(c)'s finiteness verdict see: every `R1_PROBE_TERMS` name the
    micro-batch REGISTERED (`terms`) and SCORED (`present`), in registration order."""
    return {name: terms[name] for name in R1_PROBE_TERMS if name in terms and present.get(name, False)}


def probe_aux_terms(terms: Mapping[str, Any], present: Mapping[str, bool]) -> Dict[str, Any]:
    """Every auxiliary term the grad-balance probe breaks out on the shared trunk, for one R1 forward:
    `select_probe_terms`, then the opponent-intent term when the micro-batch scored it.

    THE FIGHT DETECTOR. Registering the intent term is what produces `grad/opp_intent_policy_cosine` — the
    angle between the intent objective's pull on the shared trunk and the policy's. Under
    `--opp-intent-grad-mode detached` the intent gradient cannot reach the trunk at all and this reads ~0 BY
    CONSTRUCTION, which is the correct and expected value, not a bug. It only becomes informative under
    `shaping`, which is precisely when you need to know."""
    out = select_probe_terms(terms, present)
    if "opp_intent" in terms and present.get("opp_intent", False):
        out["opp_intent"] = terms["opp_intent"]
    return out


def balance_terms(mo: Any, present: Mapping[str, bool], *, ent_coef: float, vf_coef: float,
                  critic_winprob: bool) -> BalanceTerms:
    """One R1 forward's ``(policy_term, value_term, aux_terms)`` for `grad_balance`, each AS FOLDED.

    +PG-COEF: the policy term is `terms["policy"]` (the folded `policy_grad_coef * policy_loss`; at the 1.0
    default the raw `policy_loss` tensor) plus the entropy bonus. Under the win-prob critic the REAL critic
    term is the head's BCE — the scalar `vf_coef * value_loss` is dropped from the loss — so the BCE is the
    value term, and (being the same tensor object as `aux["win_prob"]`) is reported once, as the value."""
    aux = probe_aux_terms(mo.terms, present)
    win_prob_term = aux.get("win_prob")
    return (mo.terms["policy"] + ent_coef * mo.entropy_loss,
            (win_prob_term if (critic_winprob and win_prob_term is not None) else vf_coef * mo.value_loss),
            aux or None)


def _fork_devices(policy: Any) -> List[int]:
    """The CUDA devices `fork_rng` must cover: the policy's own, read off a parameter (asking `torch.cuda`
    which device is current would open a CUDA context in a CPU process)."""
    dev = next((p.device for p in policy.parameters()), th.device("cpu"))
    if dev.type != "cuda":
        return []
    return [dev.index if dev.index is not None else th.cuda.current_device()]


class ProbeForward:
    """ONE micro-batch's R1, re-runnable: ``region(*args)`` is exactly the call `train()` made for it.

    ``present`` is the real forward's presence read (which terms these rows scored); the re-run registers
    the same terms, and a probe that finds otherwise raises `grad_balance.ProbeGraphError`. ``calls`` counts
    the re-runs (the cost, and a test's proof that the probes went through here)."""

    def __init__(self, region: Callable[..., Any], args: Sequence[Any], present: Mapping[str, bool], *,
                 ent_coef: float, vf_coef: float, critic_winprob: bool) -> None:
        self._region = region
        self._args: Tuple[Any, ...] = tuple(args)
        self._present = dict(present)
        self._coefs = {"ent_coef": ent_coef, "vf_coef": vf_coef, "critic_winprob": bool(critic_winprob)}
        self._devices = _fork_devices(self._args[0])
        self.calls = 0

    def __call__(self) -> Any:
        """R1 on the micro-batch's rows at the current parameters: a fresh `MicroOut` on its own graph."""
        self.calls += 1
        with th.random.fork_rng(devices=self._devices):
            return self._region(*self._args)

    def balance_terms(self) -> BalanceTerms:
        """`grad_balance.grad_balance_metrics_fresh`'s ``terms_fn``: a fresh forward's balance terms."""
        return balance_terms(self(), self._present, **self._coefs)

    def grouped_terms(self) -> Dict[str, List[th.Tensor]]:
        """`PerTermNoiseSampler.flush_micro`'s ``reforward``: a fresh forward's live terms by noise group."""
        mo = self()
        return grouped_terms(mo.terms, mo.term_groups, self._present)
