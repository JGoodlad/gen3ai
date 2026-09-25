"""Revert ONE Rust fix at a time in the fix tree, run the tracker pins, restore. A fix whose revert
leaves every pin green is an unpinned fix."""
import subprocess, sys, pathlib
R = pathlib.Path("/home/goodlad/dev/gen3ai-wt/gigo-inputs/src/rust_sim")
MUT = [
  ("W1 boost sign", "src/trackers/history.rs", "r.hp_delta = amt;", "r.hp_delta = if e.kind == K::Boost { amt } else { -amt };", "a_stat_drop_is_a_negative_boost_row"),
  ("W4 row block", "src/trackers/history.rs", "                            om.failed = true;\n                        }\n                    }\n                }\n                K::Immune", "                        }\n                    }\n                }\n                K::Immune", "a_protect_block_fails_the_blocked_move_and_freezes_the_clock"),
  ("W4/T2 outcome", "src/trackers/turnview.rs", "} else if self.failed || self.blocked {", "} else if self.failed {", "a_protect_block_fails_the_blocked_move_and_freezes_the_clock"),
  ("W5 hazard sign", "src/trackers/history.rs", 'r.hp_delta = if ev::s(e, "op") == Some("sideend") { -1.0 } else { 1.0 };', "r.hp_delta = 0.0;", "a_rapid_spin_clear_is_a_negative_hazard_row"),
  ("W3 swapped", "src/trackers/history.rs", '''    if ["trick", "thief", "covet", "switcheroo"].iter().any(|w| fc.contains(w)) {
        item_tr::SWAPPED
    } else if kind == K::Item {''', '''    if kind == K::Item {
        item_tr::REVEALED
    } else if ["trick", "thief", "covet", "switcheroo"].iter().any(|w| fc.contains(w)) {''', "trick_and_thief_item_lines_are_swapped_on_both_mons"),
  ("W2 lethal", "src/trackers/history.rs", "self.last_dmg_lethal[ri(side)].unwrap_or(false),", "true,", "a_faint_no_damage_line_caused_is_not_an_attack"),
  ("L1/L2 drag", "src/trackers/mod.rs", "|| d.opp_dragged ", "", "a_dragged_mon_is_never_labelled_a_chosen_switch"),
  ("L3 replacement", "src/trackers/mod.rs", " || d.opp_switch_is_replacement", "", "a_replacement_in_the_window_after_its_faint_is_masked"),
  ("L4 caller", "src/trackers/mod.rs", '.opp_called_via\n            .as_deref()', '.opp_move_id\n            .as_deref().filter(|_| false)', "a_called_move_is_labelled_as_its_caller"),
  ("L5 encore", "src/trackers/mod.rs", "d.phase_is_forced_switch || d.opp_choice_overridden", "d.phase_is_forced_switch", "an_encore_override_is_masked"),
  ("T1 own hit", "src/trackers/clock.rs", " && d.our_move_hit_delta <= -PROGRESS_DMG_EPS", "", "a_status_move_in_sand_is_not_progress"),
]
for name, f, old, new, test in MUT:
    p = R / f; src = p.read_text()
    assert src.count(old) == 1, (name, src.count(old))
    p.write_text(src.replace(old, new))
    try:
        out = subprocess.run(["nice", "-n", "10", "cargo", "test", "-j", "4", "--profile", "selfcheck", "--features", "emission-selfcheck",
                              "--test", "tracker_semantics_test"], cwd=R, capture_output=True, text=True).stdout
    finally:
        p.write_text(src)
    failed = sorted(l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED"))
    ok = test in failed
    print(f"{'PINNED ' if ok else 'UNPINNED'} {name:16s} failing: {failed}", flush=True)
