"""M5 Lane T2 — the unified inference service (fixed weight slots x fixed buckets, a declared
lifecycle, every slot x bucket parity-gated). Design: ``designs/endstate/program_rust_core.md``
§2, the "T2 DESIGN" paragraph; resume point: ``designs/research_state/measurements/m5_t2/PROGRESS.md``.
"""
from agents.inference.service.decision import DecisionModule, UnservableArchitecture, policy_reference
from agents.inference.service.service import Decision, InferenceService, Ticket
from agents.inference.service.spec import (
    BACKENDS, CallerError, CopyParityFailure, LifecycleViolation, NonFiniteWeights, ParityFailure,
    Priority, ServiceError, ServiceSpec, SlotArchMismatch, SlotGroupSpec, VacuousParity,
)

__all__ = [
    "BACKENDS", "CallerError", "CopyParityFailure", "Decision", "DecisionModule", "InferenceService",
    "LifecycleViolation", "NonFiniteWeights", "ParityFailure", "Priority", "ServiceError",
    "ServiceSpec", "SlotArchMismatch", "SlotGroupSpec", "Ticket", "UnservableArchitecture",
    "VacuousParity", "policy_reference",
]
