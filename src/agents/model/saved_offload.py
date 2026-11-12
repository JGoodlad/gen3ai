"""THE fp32 REFERENCE ARM'S SAVED ACTIVATIONS, PAST A FIXED BUDGET, WAIT ON THE HOST (gen3_reference_offload_v1).

WHY. Under the production bf16 trunk the R1 gate and the mid-run canary judge the compiled and eager arms
against R1's EAGER body at fp32 (`compile_regions._r1_reference`). That arm saves every activation at twice
the bytes of the function the run trains, so it — a parity check's yardstick, not training — set the card's
second-highest peak: at d_model 256 its transient is 6,838 MiB against 5,369 for the eager bf16 arm and 4,738
for the compiled arm, and the canary measured inside the startup fit check peaked at 8,703 MiB allocated
(`designs/training/learner_lifecycle.md`, "The update fit check"). With the probes on their own graphs
(`agents/training/instrumented_ppo/probe_forward.py`) the canary would have been the next thing to refuse a
256-wide arm.

HOW. `fp32_reference_scope` is `trunk_precision.fp32_scope` plus `torch.autograd.graph.saved_tensors_hooks`
(the hook K9(b)'s probe and the clip relocation already use to keep NOTHING). Autograd hands every tensor it
is about to save to `SavedOffload.pack`, in forward order:

  * a tensor that is not on a CUDA device, is smaller than `OFFLOAD_MIN_BYTES`, is a parameter (the module
    keeps it on the card anyway) or does not own all of its elements (an expanded view) stays where it is,
    uncounted;
  * the rest are decided per STORAGE (a view saved by one node and its base saved by another are one block
    of device memory): the first `DEVICE_BUDGET_BYTES` of storage stay on the device;
  * every tensor of a later storage is copied to host memory and the copy is what the graph holds, so the
    device block is freed the moment the forward drops it. The backward copies it back when (and only when)
    its node runs, and frees it when the node is done.

The same tensor saved by several nodes is copied once (keyed on the tensor object and its version counter,
held weakly). The rule is one rule at every width, in bytes: a reference arm that saves less than the budget
is not touched at all. Measured on the CPU at the production micro-batch (2,048 rows; distinct saved storages
of 1 MiB or more, 2026-11-11): 4,774 MiB at d_model 128 and 7,214 MiB at 256, against 4,294 and 6,254 for the
eager bf16 arm. Those are UPPER bounds for the card (the CPU's attention saves its weight matrices; on the
card at 128 the reference arm's whole transient, saved tensors and working set, is 4,474 MiB), so how many
bytes each width offloads there is the GPU validation's reading (`LAST`), not a number this module knows.

WHAT IT MAY NOT CHANGE, and why it does not: the reference arm's NUMBERS. A host round trip is a byte copy;
a copy whose strides differ from the original's (a view that is not dense in its storage) is refused and the
original kept, so every backward kernel sees the operand layout it saw before. The arm's loss, log-probs
and gradient are therefore bit-identical with and without the hooks (`saved_offload_test.py`: CPU, with the
offload forced; on the card, where the plain arm's own gradient is not run-to-run bit-stable, the bounded arm
is judged against the measured spread of four plain arms under deterministic algorithms), the bf16 rule's drift bars and envelopes keep their basis, and
the parameter-noise null (`canary_param_null`) reads the same reference. What does change: the arm's wall
time (two transfers of the offloaded bytes) and a host-memory transient of the same size.

ONE CHECK IS LOST INSIDE THE SCOPE: autograd cannot version-check a hooked saved tensor, so an in-place
write to a tensor after it was saved would be read silently here. R1's eager bf16 arm runs the same body
without hooks in the same check and still raises on one.

THE LEDGER COUNTS HANDLES, and a handle is one host copy (`_Hosted`). Autograd promises neither that a saved
tensor is unpacked nor that it is unpacked once, so "copies back >= copies out" is not an invariant (on the
card at B = 2,048, 2026-11-11: 451 out / 407 back with nothing kept, 99 out / 108 back at the default budget):

  * NEVER unpacked: a node saves its tensors in the forward whether or not a backward will ever reach it,
    and a node the loss does not reach never runs — everything computed with a graph and then read only
    through a `.detach()`, or returned beside the loss (WHICH of the model's nodes these are was not
    traced; the toy is `tanh(h).detach() * h`). Its copy just waits and is dropped with its node;
  * unpacked MORE THAN ONCE: one tensor saved by several nodes is one handle (one copy out), and every node
    that runs copies it back for itself (`h * h.sigmoid()`: `h` is saved by three nodes); a retained graph
    backpropagated twice unpacks everything twice.

So each handle is in exactly one state, and `offloaded == returned + never_unpacked + host_live`:
`returned` = handles unpacked at least once, `never_unpacked` = handles dropped without an unpack,
`host_live` = handles whose host copy still exists; `unpacks` (>= `returned`) is the copies back, the
number the transfer seconds scale with. The graph is the ONLY owner of a handle (`_seen` holds it weakly), so
a node's host copy is freed when the backward releases the node and all of them when the graph dies:
`unsettled(ledger)` is None exactly when every handle is accounted for and nothing waits on the host.

WHAT OUTLIVES THE SCOPE: only what the model itself keeps of the graph. The extractor stashes a few of its
forward's tensors WITH their graph (`damage_op.stash`, `value_entity_pool.last_att`) until its next forward
replaces them, and the nodes behind those that the loss never reached still hold their saved tensors — host
copies here, device tensors in any other forward. `LAST["host_live"]` is that remainder at scope exit (CPU,
64 rows, everything forced: 4 handles, 0.03 MiB; at 2,048 rows about 1 MiB), so `LAST` is judged with
`graph_freed=False` and the handles' owner again after the next forward. It is one forward's remainder,
replaced, never added to: nothing accumulates across canary calls or `canary_param_null`'s arms.

`LAST` holds the most recent scope's ledger (bytes kept / offloaded, the handle states, seconds in the
transfers), for the tests and the GPU validation; nothing reads it to decide anything.
"""
from __future__ import annotations

import contextlib
import time
import weakref
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

MiB = 1 << 20
#: a saved tensor smaller than this never leaves the device (the transfer costs more than it frees)
OFFLOAD_MIN_BYTES = 1 * MiB
#: the saved bytes the reference arm keeps on the device — the FIRST ones saved; the rest wait on the host.
#: Sized so the bounded arm stays under the eager bf16 arm it is compared with (d_model 256: 5,369 MiB of
#: transient = this budget + the unbudgeted small tensors + up to ~3 GiB of backward working set), which is
#: then the canary's binding arm. The cost of a smaller budget is transfer seconds only.
DEVICE_BUDGET_BYTES = 2048 * MiB

#: the most recent `fp32_reference_scope`'s ledger (`SavedOffload.ledger`); {} before the first
LAST: Dict[str, float] = {}


class _Hosted:
    """One saved tensor waiting in host memory, the device it returns to, and how often it has. Owned by the
    graph alone; its death is booked on the `SavedOffload` that made it."""

    __slots__ = ("host", "device", "nbytes", "unpacks", "_owner", "__weakref__")

    def __init__(self, host: torch.Tensor, device: torch.device, nbytes: int, owner: "SavedOffload") -> None:
        self.host, self.device, self.nbytes, self.unpacks, self._owner = host, device, nbytes, 0, owner

    def __del__(self) -> None:
        self._owner._dropped(self)


class SavedOffload:
    """The pack / unpack pair of one reference arm (module docs). ``force`` is the CPU test seam: treat every
    tensor as if it were on a card (the "host copy" is then a real copy on the same device)."""

    def __init__(self, *, budget_bytes: int = DEVICE_BUDGET_BYTES, min_bytes: int = OFFLOAD_MIN_BYTES,
                 force: bool = False) -> None:
        self.budget_bytes, self.min_bytes, self.force = int(budget_bytes), int(min_bytes), bool(force)
        self.kept_bytes = 0            # storage bytes counted against the budget, on the device
        self.offloaded_bytes = 0       # copied to the host (each distinct tensor once)
        self.offloaded = 0             # handles made: one per distinct tensor and version
        self.returned = 0              # handles unpacked at least once
        self.unpacks = 0               # copies back to the device (a handle may be unpacked by several nodes)
        self.never_unpacked = 0        # handles dropped without one unpack (their node never read them)
        self.never_unpacked_bytes = 0
        self.host_live = 0             # handles whose host copy still exists
        self.host_live_bytes = 0
        self.refused_layout = 0        # copies dropped because their strides differed
        self.refused_bytes = 0
        self.seconds = 0.0
        #: id(tensor) -> (weak ref to it, its version, weak ref to what the graph holds for it)
        self._seen: Dict[int, Tuple[Any, int, Any]] = {}
        #: storage address -> (weak ref to the storage, True = it stays on the device)
        self._storages: Dict[int, Tuple[Any, bool]] = {}

    # ------------------------------------------------------------------ the hooks
    def pack(self, t: torch.Tensor) -> Any:
        nbytes = t.numel() * t.element_size()
        if (not (t.is_cuda or self.force) or nbytes < self.min_bytes or (t.is_leaf and t.requires_grad)
                or t.layout != torch.strided or nbytes > t.untyped_storage().nbytes()):
            return t.detach()
        hit = self._seen.get(id(t))
        if hit is not None and hit[0]() is t and hit[1] == t._version:
            again = hit[2]()
            if again is not None:      # else every node that saved it is gone: a new save is a new handle
                return again
        held: Any
        if self._stays(t.untyped_storage()):
            held = t.detach()
        else:
            t0 = time.perf_counter()
            host = t.detach().to("cpu", copy=True)
            self.seconds += time.perf_counter() - t0
            if host.stride() != t.stride():
                self.refused_layout += 1
                self.refused_bytes += nbytes
                held = t.detach()
            else:
                self.offloaded_bytes += nbytes
                self.offloaded += 1
                self.host_live += 1
                self.host_live_bytes += nbytes
                held = _Hosted(host, t.device, nbytes, self)
        self._seen[id(t)] = (weakref.ref(t), t._version, weakref.ref(held))
        return held

    def _dropped(self, held: _Hosted) -> None:
        """A handle died (its last node was released): its host copy is gone."""
        self.host_live -= 1
        self.host_live_bytes -= held.nbytes
        if not held.unpacks:
            self.never_unpacked += 1
            self.never_unpacked_bytes += held.nbytes

    def _stays(self, storage: Any) -> bool:
        """Whether this storage's tensors stay on the device: decided once per storage, the first that fit."""
        key = storage._cdata
        rec = self._storages.get(key)
        if rec is not None and rec[0]() is storage:
            return rec[1]
        nbytes = int(storage.nbytes())
        stays = self.kept_bytes + nbytes <= self.budget_bytes
        if stays:
            self.kept_bytes += nbytes
        self._storages[key] = (weakref.ref(storage), stays)
        return stays

    def unpack(self, held: Any) -> torch.Tensor:
        if not isinstance(held, _Hosted):
            kept: torch.Tensor = held
            return kept
        t0 = time.perf_counter()
        out = held.host.to(held.device, copy=True)
        self.seconds += time.perf_counter() - t0
        self.unpacks += 1
        if not held.unpacks:
            self.returned += 1
        held.unpacks += 1
        return out

    # ------------------------------------------------------------------ the ledger
    def ledger(self) -> Dict[str, float]:
        return {"kept_mib": self.kept_bytes / MiB, "offloaded_mib": self.offloaded_bytes / MiB,
                "offloaded": float(self.offloaded), "returned": float(self.returned),
                "unpacks": float(self.unpacks), "never_unpacked": float(self.never_unpacked),
                "never_unpacked_mib": self.never_unpacked_bytes / MiB,
                "host_live": float(self.host_live), "host_live_mib": self.host_live_bytes / MiB,
                "refused_layout": float(self.refused_layout), "refused_mib": self.refused_bytes / MiB,
                "seconds": self.seconds, "budget_mib": self.budget_bytes / MiB}


def unsettled(ledger: Dict[str, float], graph_freed: bool = True) -> Optional[str]:
    """None when a ledger is settled (module docs), else what is wrong: every handle is returned, never
    unpacked or still live, with no fewer copies back than handles returned — and, once the arm's graph is
    freed (``graph_freed``; False for a ledger read while something may still hold a piece of it), no host
    copy is left."""
    bad = []
    if graph_freed and (ledger["host_live"] != 0 or ledger["host_live_mib"] != 0):
        bad.append(f"{ledger['host_live']:.0f} host copies ({ledger['host_live_mib']:.1f} MiB) outlive the graph")
    if ledger["returned"] + ledger["never_unpacked"] + ledger["host_live"] != ledger["offloaded"]:
        bad.append(f"offloaded {ledger['offloaded']:.0f} != returned {ledger['returned']:.0f} + never unpacked "
                   f"{ledger['never_unpacked']:.0f} + live {ledger['host_live']:.0f}")
    if ledger["unpacks"] < ledger["returned"]:
        bad.append(f"{ledger['unpacks']:.0f} unpacks for {ledger['returned']:.0f} returned handles")
    return "; ".join(bad) or None


@contextlib.contextmanager
def fp32_reference_scope(policy: Any, offload: Optional[SavedOffload] = None) -> Iterator[SavedOffload]:
    """`trunk_precision.fp32_scope(policy)` with the reference arm's saved activations bounded on the device
    (module docs). Yields the `SavedOffload`; its ledger is published to `LAST` on exit."""
    from agents.model.trunk_precision import fp32_scope
    off = offload if offload is not None else SavedOffload()
    try:
        with fp32_scope(policy), torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack):
            yield off
    finally:
        off._seen.clear()              # weak entries only; the graph owned the tensors and is gone
        LAST.clear()
        LAST.update(off.ledger())
