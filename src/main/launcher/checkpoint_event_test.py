"""The launcher's events panel reads a child `[CHECKPOINT]` line as a SAVE only when it names its file
(P10-A2: SIGUSR1's deferred request line was announced as "Checkpoint saved" ~40 s before the save)."""
from main.launcher.child import checkpoint_event


def test_only_a_line_naming_its_file_is_a_save():
    saved = "💾 [CHECKPOINT] Forced save → /m/run/checkpoints/checkpoint_forced_0000890624_044903.zip"
    assert checkpoint_event(saved) == "💾 Checkpoint saved → checkpoint_forced_0000890624_044903.zip"
    req = "[CHECKPOINT] SIGUSR1 received — saving at the next safe point (rollout / step boundary)"
    assert "saved" not in checkpoint_event(req) and "requested" in checkpoint_event(req)
    failed = "💾 [CHECKPOINT] Forced save FAILED (OSError: disk full) — training continues"
    assert "FAILED" in checkpoint_event(failed) and "saved" not in checkpoint_event(failed)
