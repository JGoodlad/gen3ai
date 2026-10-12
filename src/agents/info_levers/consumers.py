"""THE THREE CONSUMERS of the information-lever spec and what each honours (``agents.info_levers.spec.Consumer``).

A lever a consumer cannot honour is REFUSED by name before anything is built (``Consumer.validate``); a fact a consumer
can only ever give TRUE is its named RESIDUAL, reported beside every honest spec it runs (``Consumer.residual_of``).
The matrix is the design doc's §3 table (``designs/endstate/design_information_levers.md``), and a test pins that
each consumer's code path exists for every lever declared here.
"""
from __future__ import annotations

from agents.info_levers.spec import Consumer

_T = ("truth",)
_TPM = ("truth", "placebo", "map")

#: (a) The TRAINING OBSERVATION (``--oracle-reveal <spec>``): the Rust encoder states the told facts in the opponent
#: block (``pokesim::encoder::oracle``) — a never-seen mon's slot exists only where its species is told, so its set
#: facts need ``species:unseen``. The observation can only state a fact or not: a mass p is a per-(episode, side,
#: mon, class) Bernoulli. It has no cell for an exact HP or a hidden duration (a layout change), and no prior sampler
#: for a placebo / map fact.
OBS = Consumer(
    name="the training observation (--oracle-reveal)",
    tells={("species", "unseen"): _T,
           ("moves", "seen"): _T, ("moves", "unseen"): _T,
           ("item", "seen"): _T, ("item", "unseen"): _T,
           ("ability", "seen"): _T, ("ability", "unseen"): _T,
           ("spread", "seen"): _T, ("spread", "unseen"): _T},
    needs={("moves", "unseen"): ("species", "unseen"), ("item", "unseen"): ("species", "unseen"),
           ("ability", "unseen"): ("species", "unseen"), ("spread", "unseen"): ("species", "unseen")},
    why={("hp", "seen"): "the opponent's HP cell is the DISPLAYED fraction; an exact-HP cell would be a layout change",
         ("counters", "seen"): "the observation has no cell for a hidden duration (the elapsed sleep / toxic counters "
                               "it has are public)"},
)

#: (b) The SEARCH WORLD BUILDER (``main.value_search.worlds``): every fact is drawn per world from the mixture, then
#: the world must replay to the searched side's exact protocol bytes and rows. ``counters`` is its RESIDUAL: the
#: world replays the prefix on the TRUE battle seed, so a duration rolled BEFORE the root (sleep, confusion, Encore,
#: a lock) is the truth's in every world (the hidden-info study's A5 caveat); no honest world can hide it without a
#: root-state resample in the playout core.
SEARCH = Consumer(
    name="the search world builder",
    tells={("species", "unseen"): _TPM,
           ("moves", "seen"): _TPM, ("moves", "unseen"): _TPM,
           ("item", "seen"): _TPM, ("item", "unseen"): _TPM,
           ("ability", "seen"): _TPM, ("ability", "unseen"): _TPM,
           ("spread", "seen"): _TPM, ("spread", "unseen"): _TPM,
           ("hp", "seen"): _T,
           ("counters", "seen"): _T},
    residual=(("counters", "seen"),),
    why={},
)

#: (c) The MODEL-INPUT injection (``agents.info_levers.inject``): the network's own belief inputs are pushed toward
#: the target with mass p — the typed move posterior (every consumer of ``last_move_belief_logits``), the item
#: posterior (the op's P(Choice Band) / P(Quick Claw)) and the believed stats (the op's spread). An ``unseen`` move
#: lever applies on an X5 HYPOTHESIS slot whose hypothesised species is one of the opponent's true never-seen species.
#: There is no ability or species belief input to push, and no exact-HP / duration input.
INJECT = Consumer(
    name="the model-input injection",
    tells={("moves", "seen"): _TPM, ("moves", "unseen"): _T,
           ("item", "seen"): _TPM,
           ("spread", "seen"): _T},
    why={("species", "unseen"): "the X5 hypothesis species are the network's own guesses; forcing them is a different "
                                "intervention (not built)",
         ("ability", "seen"): "the network has no ability belief input to push",
         ("hp", "seen"): "the network reads the displayed HP; it has no exact-HP belief input",
         ("counters", "seen"): "the network has no hidden-duration belief input",
         ("item", "unseen"): "the item head reads seen slots only",
         ("spread", "unseen"): "the spread head's hypothesis-seat outputs are unsupervised; not pushed"},
)

ALL = {"obs": OBS, "search": SEARCH, "inject": INJECT}
