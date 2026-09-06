"""REGISTERED ACID GATE against content_locality. Pure post-processing; runs in either tree.

  python acid_vs_content_locality.py <zswap_nN.json> <../content_locality/v8_era_nN.json> [mode]

TWO different claims depending on which teacher checkpoint this run used.

* The STATE BATCH is a function of the PARENT and the REFERENCE OPPONENT only, and neither
  changes between modes -- so `n_states`, the per-team counts and the team order MUST match
  exactly in BOTH modes, and the two FLOOR models (checkpoints of the parent run) must reproduce
  content_locality's per-team KL BIT-FOR-BIT in both. That is the gate on the state batch and on
  the shim.

* The TEACHER rows are a claim about which file was loaded.
    mode=final_interrupted -> the same files content_locality used ⇒ must be BIT-IDENTICAL.
    mode=best_model        -> a DIFFERENT checkpoint (the one the fold's own resolver picks) ⇒
                              must DIFFER. Agreement there would mean the two files are the same
                              model, which would falsify the whole reason for the re-run, so it is
                              reported as a FAILURE rather than quietly accepted.
"""
import json, sys
import numpy as np

TOL = 1e-9


def main(zpath, cpath, mode="best_model"):
    Z, C = json.load(open(zpath)), json.load(open(cpath))
    zm, cm = Z["_meta"], C["_meta"]
    ok = True

    print(f"=== ACID: {zpath}\n         vs {cpath}   (teacher-file mode = {mode}) ===")
    print(f"  recorded teacher rungs: {zm.get('teacher_rung')}")

    same = zm["n_states"] == cm["n_states"]
    print(f"  states   zswap {zm['n_states']}   content_locality {cm['n_states']}   "
          f"{'MATCH' if same else 'MISMATCH'}")
    ok &= same
    same = zm["states_per_team"] == cm["states_per_team"]
    print(f"  per-team state counts identical: {same}")
    ok &= same
    same = [t["sha10"] for t in zm["teams"]] == [t["sha10"] for t in cm["teams"]]
    print(f"  team order identical: {same}")
    ok &= same

    print(f"\n  {'model':16s} {'role':10s} {'max|zswap.a - CL|':>20s}  {'mean KL':>9s}  verdict")
    for tag in [k for k in Z["per_team_kl"] if k in C["per_team_kl_fwd"]]:
        a = np.array(Z["per_team_kl"][tag]["a"])
        b = np.array(C["per_team_kl_fwd"][tag])
        d = float(np.abs(a - b).max())
        is_floor = tag.startswith("FLOOR_")
        must_match = is_floor or mode == "final_interrupted"
        good = (d <= TOL) if must_match else (d > TOL)
        role = "floor" if is_floor else "teacher"
        want = "must match" if must_match else "must DIFFER"
        print(f"  {tag:16s} {role:10s} {d:20.3e}  {a.mean():9.6f}  "
              f"{'OK' if good else 'FAIL'} ({want})")
        ok &= good

    print(f"\n  {'PASS' if ok else 'FAIL -- RUN IS VOID'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "best_model"))
