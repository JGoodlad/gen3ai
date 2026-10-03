"""The `--compile-trainer` DEFAULT resolvers, kept pure (and therefore unit-testable without a GPU).

(The `--compile-opponents` family and its preload resolver were deleted with the Python env core —
deletion pass U3 / R6.)

`main.compile_defaults_test` imports these through the `train_rl_agent` hub.
"""


def resolve_compile_trainer_default(device, debug: bool, cuda_available=None) -> bool:
    """The AUTO default for `--compile-trainer`: ON for cuda, OFF for cpu, OFF under `--debug`.

    `--compile-trainer` REFUSES a non-cuda device by design (the CPU backward provably does not
    lower). A default must therefore never be a flat `True`, or every CPU invocation that works
    today — a smoke, a laptop, a CI run — would start failing with a FATAL_CONFIG. So the default
    is conditioned on the device rather than softened: `auto`/`cuda*` with a card ⇒ ON, anything
    else ⇒ OFF, and an EXPLICIT `--compile-trainer` on cpu still refuses exactly as before.

    `--debug` is excluded outright even with an explicit `--device cuda`: a smoke exists to prove
    the pipeline in ~1 minute, and a multi-minute Inductor compile (plus a CUDA context a live GPU
    run does not want a smoke taking) defeats the point of it.

    Pure and injectable (`cuda_available`) so the matrix is unit-testable without a card.
    """
    if debug:
        return False
    dev = str(device or "auto").strip().lower()
    if dev.startswith("cuda"):
        return True
    if dev != "auto":
        return False                                   # cpu / mps / anything explicit and non-cuda
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available
    try:
        return bool(cuda_available())
    except Exception:                                  # noqa: BLE001 — a probe must never crash a launch
        return False


def update_rows_for(*, rollout_target_samples, n_steps: int, n_envs: int) -> int:
    """The REAL update size of a run: the rows one update trains on (`rust_rollout.trigger.update_rows`).
    An untyped / zero target is ``n_steps * n_envs``.
    This — not ``n_steps * n_envs`` alone — is what `check_shape_stability` judges (P10-E, F9)."""
    from agents.training.rust_rollout.trigger import update_rows
    return update_rows(n_envs=int(n_envs or 0), n_steps=int(n_steps or 0),
                       target=int(rollout_target_samples or 0))


def resolve_compile_trainer_auto(*, device, debug: bool, n_steps: int, n_envs: int,
                                 batch_size: int, rollout_target_samples=0, cuda_available=None):
    """The full AUTO decision for `--compile-trainer`. Returns `(enabled, downgrade_reason)`.

    Two gates, and the SECOND one is the non-obvious half. `check_shape_stability` REFUSES an update
    that does not divide by `--batch-size` (a second micro-batch shape the one compiled learner graph
    does not declare) — because for someone who ASKED for the compile, silently getting eager is the
    whole failure this flag exists to prevent. The update size is the real one
    (`--rollout-target-samples`, else ``n_steps * n_envs``; `update_rows_for`).

    But a DEFAULT is not an ask. Applying that refusal to the default would convert a class of
    command that works today into a `FATAL_CONFIG` exit, which is exactly the failure the cpu
    conditioning avoids, one flag over. So the auto default YIELDS to the config that was actually
    typed and returns the reason for the caller to announce; an EXPLICIT `--compile-trainer` never
    reaches here and still hits the refusal.
    """
    if not resolve_compile_trainer_default(device, debug, cuda_available):
        return False, None
    from agents.model.compile_trainer import CompileTrainerError, check_shape_stability
    try:
        check_shape_stability(
            update_rows=update_rows_for(rollout_target_samples=rollout_target_samples,
                                        n_steps=n_steps, n_envs=n_envs),
            batch_size=int(batch_size or 0))
    except CompileTrainerError as exc:
        return False, str(exc)
    return True, None
