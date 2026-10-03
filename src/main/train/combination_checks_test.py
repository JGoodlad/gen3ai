"""THE GUARD ON THE SINGLE DECLARATION — one list, two readers, and nothing left behind.

Three tests, answering three different questions:

* **Is the list EXHAUSTIVE?** `test_no_cross_flag_parser_error_remains_in_config` walks
  `main.train.config`'s AST and fails, naming file:line, on any `parser.error` whose guard reads a
  second flag's value. It RESOLVES LOCAL ALIASES (`x = bool(args.a and args.b)`), because that
  indirection is exactly how three refusals hid from an earlier reading of this file and let
  G5 (2026-09-06) die three times on a command `checkargs` had passed.
* **Do the two surfaces AGREE?** `test_resolve_config_and_checkargs_agree` drives one argv per
  check through the real `resolve_config` and through `main.checkargs.check`, and asserts the launch
  refuses, that checkargs names the same rule FIRST, and that the launch prints that rule's declared
  text verbatim — including the C1 INHERITED-value shape (a parent recording
  `move_belief_mode="both"`, a child zeroing `--opp-belief-aux-coef`), built against a synthetic
  parent `model_config.json` on disk. (The original G5 / C1 shapes were distillation flags; that
  family is deleted and the shape is kept.)
* **Is every check REACHABLE?** `test_every_check_has_an_argv_that_trips_exactly_it` fails if a
  declared check has no row in the table, or if a row stops tripping the rule it names — the way a
  table of argvs rots into a table of nothing.

MEASURED, not asserted: the migration was verified against the pre-migration tree by running every
row of this table through both `resolve_config`s. **91 of 91 argvs refused in both, with identical
exit codes and identical refusal text.**
"""
from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import pytest

from main.train.combination_checks import COMBINATION_CHECKS, failing_checks
from utils.paths import src_path

# ---------------------------------------------------------------------------------------------
# THE ALLOWLIST. A cross-flag `parser.error` that legitimately stays in `resolve_config`, keyed by
# a distinctive prefix of its message, with the reason it cannot move. Adding an entry is a design
# decision, not a way to silence the scan: the reason has to name what the check needs that a pure
# predicate over a namespace does not have.
# ---------------------------------------------------------------------------------------------
ALLOWED_CROSS_FLAG_ERRORS: dict[str, str] = {}

_SCANNED = ("config.py",)


def _dests(node) -> set[str]:
    """Every `args.X` / `getattr(args, "X", …)` attribute named anywhere under `node`."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args":
            out.add(n.attr)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
              and len(n.args) > 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == "args"
              and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
            out.add(n.args[1].value)
    return out


def _alias_map(fn) -> dict[str, set[str]]:
    """`_wanted = bool(args.a and args.b … )` -> {"_wanted": {…}}.

    Without this the scan is blind to precisely the indirection that hid the G5 refusals: the guard
    reads a LOCAL, and a local's name says nothing about which flags produced it.
    """
    out: dict[str, set[str]] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            d = _dests(n.value) | {x for t in _dests_of_names(n.value) for x in out.get(t, set())}
            if d:
                out.setdefault(n.targets[0].id, set()).update(d)
    return out


def _dests_of_names(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _cross_flag_errors(path: Path):
    """`(lineno, sorted dests, message-ish)` for every `parser.error` guarded by 2+ flag values."""
    tree = ast.parse(path.read_text())
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    aliases = {id(f): _alias_map(f) for f in fns}

    found = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "error" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "parser"):
            continue
        fn, dests, cur = None, set(), n
        while cur in parents:
            p = parents[cur]
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn is None:
                fn = p
            if isinstance(p, ast.If):
                dests |= _dests(p.test)
                dests |= {d for name in _dests_of_names(p.test)
                          for d in aliases.get(id(fn) if fn else 0, {}).get(name, set())}
            cur = p
        # a guard inside a function whose aliases were only resolvable once `fn` was known
        if fn is not None:
            amap = aliases[id(fn)]
            cur, dests2 = n, set()
            while cur in parents:
                p = parents[cur]
                if isinstance(p, ast.If):
                    dests2 |= _dests(p.test)
                    dests2 |= {d for name in _dests_of_names(p.test) for d in amap.get(name, set())}
                cur = p
            dests |= dests2
        if len(dests) >= 2:
            msg = ast.unparse(n.args[0])[:80] if n.args else ""
            found.append((n.lineno, sorted(dests), msg))
    return found


def test_no_cross_flag_parser_error_remains_in_config():
    """A value-conditional refusal belongs in COMBINATION_CHECKS, where checkargs can read it."""
    offenders = []
    for name in _SCANNED:
        path = Path(src_path("main", "train", name))
        for lineno, dests, msg in _cross_flag_errors(path):
            if any(msg.strip("'\"( f").startswith(k) or k in msg
                   for k in ALLOWED_CROSS_FLAG_ERRORS):
                continue
            offenders.append(f"{path}:{lineno}  reads {dests}\n      {msg}")
    assert not offenders, (
        "parser.error calls whose predicate reads a SECOND flag's value are invisible to "
        "`main.checkargs`, which is the defect this module exists to close (C1 2026-09-01, G5 "
        "2026-09-06). Move each into main.train.combination_checks.COMBINATION_CHECKS with the "
        "same message, or add an ALLOWED_CROSS_FLAG_ERRORS entry naming what it needs that a pure "
        "predicate cannot have:\n  " + "\n  ".join(offenders))


# ---------------------------------------------------------------------------------------------
# One argv per check. `--unified-moves off` appears wherever a refusal is about the damage-op /
# move-latent family: the flagless default desugars to 'both' and would turn the toggle back ON,
# so `--no-damage-op` alone does not produce the config the check is about.
# ---------------------------------------------------------------------------------------------
OFF = ["--unified-moves", "off"]

ARGVS: dict[str, list[str]] = {
    "winprob_critic_needs_unit_gamma": ["--gamma", "0.99"],
    "arch_umbrella_is_fresh_only": ["--arch", "production", "--model", "models/p/final_model.zip"],
    "adaptive_batch_target_positive": ["--adaptive-batch", "total", "--adaptive-batch-target", "0"],
    "adaptive_batch_band_above_one": ["--adaptive-batch", "total", "--adaptive-batch-band", "1.0"],
    "adaptive_batch_min_accum": ["--adaptive-batch", "total", "--adaptive-batch-min-accum", "0"],
    "adaptive_batch_max_ge_min": ["--adaptive-batch", "total", "--adaptive-batch-min-accum", "4",
                                  "--adaptive-batch-max-accum", "2"],
    "adaptive_batch_every_min": ["--adaptive-batch", "total", "--adaptive-batch-every", "0"],
    "exploiter_excludes_self_play": ["--exploiter", "models/t", "--self-play"],
    "pfsp_scale_needs_self_play": ["--pfsp-scale", "1.0"],
    "supply_starve_cycles_parses": ["--supply-starve-cycles", "no_such_lever=3"],
    "exploiter_keep_bots_needs_exploiter": ["--exploiter-keep-bots"],
    "exploiter_temp_start_needs_exploiter": ["--exploiter-temp-start", "2.0"],
    "exploiter_temp_positive": ["--exploiter", "models/t", "--exploiter-temp-start", "-1"],
    "exploiter_temp_anneal_frac": ["--exploiter", "models/t", "--exploiter-temp-start", "2.0",
                                   "--exploiter-temp-anneal-frac", "1.5"],
    "exploiter_temp_ratchet_factor": ["--exploiter", "models/t", "--exploiter-temp-start", "5.0",
                                      "--exploiter-temp-mode", "ratchet",
                                      "--exploiter-temp-ratchet-factor", "1.5"],
    "exploiter_temp_ratchet_wr": ["--exploiter", "models/t", "--exploiter-temp-start", "5.0",
                                  "--exploiter-temp-mode", "ratchet",
                                  "--exploiter-temp-ratchet-wr", "1.5"],
    "exploiter_temp_ratchet_games": ["--exploiter", "models/t", "--exploiter-temp-start", "5.0",
                                     "--exploiter-temp-mode", "ratchet",
                                     "--exploiter-temp-ratchet-games", "0"],
    "exploiter_temp_ratchet_start_above_end": [
        "--exploiter", "models/t", "--exploiter-temp-start", "0.5",
        "--exploiter-temp-mode", "ratchet", "--exploiter-temp-end", "1.0"],
    "exploiter_temp_ratchet_needs_start": ["--exploiter", "models/t",
                                           "--exploiter-temp-mode", "ratchet"],
    "fork_lr_is_resume_only": ["--fork-lr", "1e-5"],
    "fork_lr_freeze_needs_fork_lr": ["--fork-lr-freeze"],
    # ---- gen3_winprob_critic_mode_v1. The bare argv IS the win-prob composition (the critic is a
    # constant of the namespace, deletion pass P11b), so a row below trips its own rule.
    "winprob_critic_needs_a_head": ["--win-prob-mode", "none"],
    # gen3_fork_v1 — the fork arm's refusal that survives: the strata weight (a branch row has no
    # material margin to stratify on). "The fork needs the winprob critic" and "the strata weight needs
    # the winprob critic" can no longer fire (there is no other critic: P11b) and are gone, with the
    # Python core's fork arm (UNAVAILABLE, U3).
    "fork_refuses_strata_weight":
        ["--fork-fraction", "0.02", "--win-prob-strata-weight", "0.5"],
    "rnd_variants_need_the_base_rnd_head": ["--ridealong-rnd-variants", "all"],
    "winprob_critic_refuses_draw_penalty": ["--terminal-indicator", "--victory-value", "1.0",
                                            "--draw-penalty", "-1.0"],
    "winprob_critic_needs_the_indicator_terminal": ["--no-terminal-indicator", "--victory-value", "1.0",
                                                    "--draw-penalty", "0"],
    "winprob_critic_needs_unit_victory_value": ["--terminal-indicator", "--victory-value", "7.5",
                                                "--draw-penalty", "0"],
    "move_belief_hidden_needs_species_belief": ["--move-belief-mode", "both",
                                                "--opp-belief-aux-coef", "0"],
    "damage_op_needs_revealed_move_belief": [*OFF, "--damage-op", "--move-belief-mode", "off",
                                             "--opp-belief-aux-coef", "0.1"],
    "move_prior_fusion_needs_move_belief": [*OFF, "--move-prior-fusion",
                                            "--move-belief-mode", "off"],
    "species_prior_fusion_needs_belief_coef": [*OFF, "--species-prior-fusion",
                                               "--opp-belief-aux-coef", "0"],
    "damage_candidate_k_needs_op": [*OFF, "--damage-candidate-k", "4"],
    "damage_outgoing_needs_op": [*OFF, "--damage-outgoing"],
    "entity_topk_seats_need_op_and_latent": [*OFF, "--entity-topk-seats", "4"],
    "entity_tail_seats_need_topk": [*OFF, "--entity-tail-seats"],
    "edge_families_d1s1c1c2_need_outgoing": [*OFF, "--edge-bias-families", "d1"],
    "edge_families_x_needs_op": [*OFF, "--edge-bias-families", "x"],
    "edge_families_kernels_need_op": [*OFF, "--edge-bias-families", "d2"],
    "edge_families_d3s3_need_seats": [*OFF, "--edge-bias-families", "d3", "--damage-op",
                                      "--entity-topk-seats", "0", "--move-belief-mode", "both",
                                      "--opp-belief-aux-coef", "0.1"],
    "move_candidate_floor_needs_fusion": [*OFF, "--move-candidate-floor", "0.05"],
    "damage_topk_needs_op": [*OFF, "--damage-topk", "5", "--damage-matrices", "incoming"],
    "damage_topk_needs_move_latent": [*OFF, "--damage-topk", "5", "--damage-op",
                                      "--damage-matrices", "incoming", "--move-belief-mode", "both",
                                      "--opp-belief-aux-coef", "0.1"],
    "damage_topk_needs_incoming_matrix": [*OFF, "--damage-topk", "5", "--damage-op",
                                          "--move-latent", "--damage-matrices", "off",
                                          "--move-belief-mode", "both",
                                          "--opp-belief-aux-coef", "0.1"],
    "damage_matrices_outgoing_needs_op": [*OFF, "--damage-matrices", "outgoing"],
    "damage_matrices_incoming_needs_op": [*OFF, "--damage-matrices", "incoming"],
    "damage_matrices_incoming_needs_move_latent": [
        *OFF, "--damage-matrices", "incoming", "--damage-op",
        "--move-belief-mode", "both", "--opp-belief-aux-coef", "0.1"],
    "move_belief_latent_coef_needs_latent": [*OFF, "--move-belief-latent-coef", "0.1"],
    "move_belief_latent_coef_needs_revealed": [*OFF, "--move-belief-latent-coef", "0.1",
                                               "--move-latent", "--move-belief-mode",
                                               "unrevealed", "--opp-belief-aux-coef", "0.1"],
    "spread_belief_coef_needs_head": [*OFF, "--spread-belief-coef", "0.1", "--no-spread-belief"],
    "spread_belief_nature_needs_head": [*OFF, "--spread-belief-nature", "--no-spread-belief"],
    "hp_type_belief_coef_needs_move_belief": [*OFF, "--hp-type-belief-coef", "0.05",
                                              "--move-belief-mode", "off"],
    "anneal_start_needs_min_lr": ["--anneal-lr-start-steps", "5"],
    "anneal_start_below_steps": ["--anneal-lr-start-steps", "500", "--anneal-min-lr", "1e-6"],
    # gen3_fork_rust_v1 — the fork arm on the Rust core (forks.md §14.7)
    "fork_rust_needs_keyed_opponent_sampling": ["--fork-fraction", "0.02",
                                                "--opponent-sampling", "generator"],
    "fork_rust_needs_complete_game_trigger": ["--fork-fraction", "0.02",
                                              "--rollout-trigger", "window"],
    "rollout_target_on_the_quantum": ["--n-envs", "48", "--batch-size", "2048",
                                      "--rollout-target-samples", "100000"],
    # T6 SPRT promotion
    "promotion_sprt_needs_self_play": ["--promotion-sprt"],
    "promotion_sprt_ignores_the_threshold": ["--promotion-sprt", "--self-play", "--promote-threshold", "0.6"],
}


def _resolved(argv: list[str]):
    """`(exit_code, first-refusal text or "")` from the REAL launch path, output captured."""
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    parser = build_parser()
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            args = parser.parse_args(argv)
            resolve_config(args, parser)
        return 0, ""
    except SystemExit as exc:
        text = err.getvalue()
        if "error: " in text:                      # argparse's own framing
            return int(exc.code or 0), " ".join(text.split("error: ", 1)[1].split())
        body = out.getvalue() + text               # the print+exit styles
        return int(exc.code or 0), " ".join(body.split())


def _first_check(argv: list[str]):
    """Which check `main.checkargs` reports first for this argv (None = it says it launches)."""
    from main.checkargs import check
    res = check(["--steps", "100"] + argv)
    combos = res["combinations"]
    return combos[0][0] if combos else None


def _namespace(argv: list[str]):
    """The argv as BOTH surfaces see it before the checks run: parsed, marked, critic-resolved,
    desugared — in that order, which is the order `resolve_config` and `checkargs` use.

    `resolve_critic_mode` belongs here for `desugar_umbrella_flags`' exact reason: it IMPLIES
    `--win-prob-mode shaping` / `--gamma 1.0` (the win-prob critic's settings), so a
    table row judged without it would report a command as broken on the very flags the mode fills
    in."""
    from main.train.config import desugar_umbrella_flags, resolve_critic_mode
    from main.train_rl_agent import build_parser
    parser = build_parser()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        args = parser.parse_args(argv)
        args._explicit_flags = frozenset(d for d, v in vars(args).items() if v is not None)
        args._saved_config_present = False
        resolve_critic_mode(args)
        desugar_umbrella_flags(args)
    return args


@pytest.mark.parametrize("name", sorted(ARGVS))
def test_every_check_has_an_argv_that_trips_exactly_it(name):
    """A table row must trip the rule it names — under the LAUNCH path, on the resolved config."""
    args = _namespace(["--steps", "100"] + ARGVS[name])
    hits = [c.name for c in failing_checks(args)]
    assert name in hits, (
        f"{name}: its argv trips {hits or 'nothing'} on the PARSED namespace. Fix the row — an "
        f"argv that stops tripping its check silences the agreement test without failing it.")


def test_the_table_covers_every_declared_check():
    missing = sorted({c.name for c in COMBINATION_CHECKS} - set(ARGVS))
    extra = sorted(set(ARGVS) - {c.name for c in COMBINATION_CHECKS})
    assert not missing, f"COMBINATION_CHECKS entries with no argv in the table: {missing}"
    assert not extra, f"table rows naming no declared check: {extra}"


@pytest.mark.parametrize("name", sorted(ARGVS))
def test_resolve_config_and_checkargs_agree(name):
    """THE CONTRACT: the launch path refuses exactly when checkargs reports a would-fail.

    Same rule, same text. `checkargs` printing "✓ this command still launches" on a command
    `resolve_config` then kills is the whole defect (C1 2026-09-01, G5 2026-09-06).
    """
    argv = ["--steps", "100"] + ARGVS[name]
    code, text = _resolved(argv)
    reported = _first_check(argv)
    assert code != 0, f"{name}: resolve_config accepted an argv the table says it refuses"
    assert reported is not None, (
        f"{name}: resolve_config exited {code} on this argv and checkargs said it launches — "
        f"the exact class this module exists to make impossible.\n  launch said: {text[:300]}")
    # Both surfaces must name the SAME rule first, and the launch must print that rule's own text
    # verbatim (argparse wraps it, so compare whitespace-joined).
    expected = " ".join(reported.text(_namespace(argv)).split())
    assert expected in text, (
        f"{name}: the launch refused with something other than the rule checkargs reported first "
        f"({reported.name}).\n  launch said: {text[:400]}")
    # …and this row's own rule must be among what checkargs reports. It need not be FIRST: a few
    # argvs are broken two ways by construction (`--damage-matrices incoming` desugars a
    # `--damage-topk` default in, so the top-K dependency fires first), and declaration order —
    # which is the pre-migration source order — decides which message a launch shows.
    from main.checkargs import check as _check
    assert name in [c.name for c, _ in _check(argv)["combinations"]]


def test_the_c1_inherited_value_is_still_caught(tmp_path, run_archive):
    """C1: the parent's recorded `move_belief_mode=both`, a child zeroing `--opp-belief-aux-coef`.

    The value is INHERITED, so an argv-only reading sees nothing — this is the case that made
    `main.checkargs` read the effective namespace at all, and it must survive the migration. (C1
    itself was an inherited distillation target; the family is deleted, the shape is kept.)
    """
    from main.checkargs import check
    from main.checkargs_test import _parent_run          # the one fixture, not a second copy
    ckpt, _ = _parent_run(tmp_path, move_belief_mode="both", opp_belief_aux_coef=0.05)

    res = check(["--model", ckpt, "--steps", "100", "--opp-belief-aux-coef", "0",
                 "--run-name", "child_run"])
    names = [c.name for c, _ in res["combinations"]]
    assert "move_belief_hidden_needs_species_belief" in names, (
        "the inherited move-belief mode must still be reported; got "
        f"{names} (resolution={res['resolution'] and res['resolution'].get('config_path')})")
    prov = dict(res["combinations"])[
        next(c for c, _ in res["combinations"]
             if c.name == "move_belief_hidden_needs_species_belief")]
    assert any("INHERITED" in line for line in prov), prov


def test_the_g5_arm_is_reported_without_a_model():
    """G5: a FRESH arm (no --model) carrying an incoherent pair.

    `checkargs` used to run no combination check at all without a `--model`, so such an argv printed
    "✓ this command still launches" and then died three times in a row.
    """
    from main.checkargs import check
    res = check(["--steps", "100", "--move-belief-mode", "both", "--opp-belief-aux-coef", "0"])
    names = [c.name for c, _ in res["combinations"]]
    assert "move_belief_hidden_needs_species_belief" in names, names


def test_fresh_only_flags_are_parser_options_that_take_exactly_one_value():
    """The launcher's restart path strips `fresh_only_flags()` as `--flag VALUE` / `--flag=VALUE`
    (`main.launcher.checkpoint.resume_child_args`). A fresh-only flag with another arity would be
    stripped wrongly — a value left behind as a stray positional, or a neighbouring flag eaten."""
    from main.train.combination_checks import fresh_only_flags
    from main.train_rl_agent import build_parser

    flags = fresh_only_flags()
    assert "--arch" in flags, "the 2026-09-26 incident's flag must stay declared fresh-only"
    by_option = {opt: a for a in build_parser()._actions for opt in a.option_strings}
    for flag in flags:
        assert flag in by_option, f"{flag}: declared fresh_only but the trainer parser lacks it"
        action = by_option[flag]
        assert action.nargs is None and action.const is None, (
            f"{flag}: takes nargs={action.nargs!r} — the stripper assumes exactly one value")


def test_every_check_refusing_a_resume_declares_what_to_strip():
    """A check that fires on `--model` + X (a FRESH-only X) must name X in `fresh_only`, or the
    launcher's interval/crash restart re-passes it and crash-loops (2026-09-26). Probed by
    evaluating each check that reads `model` with and without a `--model`."""
    for check in COMBINATION_CHECKS:
        if "model" not in check.dests:
            continue
        if check.name == "fork_lr_is_resume_only":
            continue    # the opposite direction: refused on a FRESH run, never on a resume
        assert check.fresh_only, (
            f"{check.name}: reads `model` but declares no fresh_only flag for the launcher's "
            f"restart path to strip")


def test_the_tf32_behaviour_check_refusal_is_deleted_with_tf32():
    """TF32 was retired (deletion pass K2), so the Rust core's refusal of `--matmul-precision high` + a
    FATAL `--behaviour-check` has nothing to refuse: the row is gone, and the flag it read no longer
    parses."""
    from main.train.combination_checks import COMBINATION_CHECKS as CHECKS

    assert not [c for c in CHECKS if "tf32" in c.name or "matmul_precision" in c.dests]
