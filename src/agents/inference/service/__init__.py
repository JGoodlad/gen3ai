"""M5 Lane T2 — the unified inference service (fixed weight slots x fixed buckets, a declared
lifecycle, every slot x bucket parity-gated). Design: ``designs/endstate/program_rust_core.md``
§2, the "T2 DESIGN" paragraph; resume point: ``designs/research_state/measurements/m5_t2/PROGRESS.md``.
"""
from agents.inference.service.decision import DecisionModule, UnservableArchitecture, policy_reference
from agents.inference.service.service import Decision, InferenceService, Ticket
from agents.inference.service.spec import (
    BACKENDS, CallerError, LifecycleViolation, ParityFailure, Priority, ServiceError, ServiceSpec,
    SlotArchMismatch, SlotGroupSpec,
)

__all__ = [
    "BACKENDS", "CallerError", "Decision", "DecisionModule", "InferenceService",
    "LifecycleViolation", "ParityFailure", "Priority", "ServiceError", "ServiceSpec",
    "SlotArchMismatch", "SlotGroupSpec", "Ticket", "UnservableArchitecture", "policy_reference",
]
