"""The `# --- Reward config ---` section. Resume-immutable, value-checked.

The shaped reward's flags (`--bias-additivity`, `--mat-alive-weight`, `--no-progress-penalty`,
`--bias-redesign`, `--switch-bias-weight`, `--self-ko-hp-penalty`, `--drop-redundant-bias`,
`--drop-switch-bias`, `--all-shaping-pbrs`, `--stall-pbrs`) were DELETED with the shaped reward path
(gen3_shaped_reward_deletion_v1, 2026-09-26; `designs/deleted_flags.md`). The terminal's own
`--draw-penalty` is what remains here.

Lifted VERBATIM out of the old single-file `parser.py` (lines 260-328); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse



def add_reward_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    parser.add_argument("--draw-penalty", "--draw_penalty", dest="draw_penalty", type=float,
                        default=0.0, help="Terminal reward for a DRAW / 250-turn timeout (no "
                        "winner). DEFAULT 0.0 = what the default --terminal-indicator REQUIRES (the "
                        "win-prob critic's terminal pays 0 on everything but a win; the bare-argv "
                        "default since 2026-10-02). It applies to the SIGNED terminal only "
                        "(--no-terminal-indicator), where the validated ai_v8 value was -35.0 with "
                        "--victory-value 30 (stalling to the turn cap strictly worse than losing "
                        "cleanly) -- still what a config that never recorded the field means. "
                        "Resume-immutable (recorded + value-checked in model_config.json).")
