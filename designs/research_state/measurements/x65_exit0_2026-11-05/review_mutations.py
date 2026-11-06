"""X65 ExIt-0 REVIEW (REVIEW.md, item 6): mutate one guard at a time in a SCRATCH worktree at c9d86ef53, run the
guard's own test file, revert. KILLED = some test fails on the mutation; SURVIVED = no test notices it.

    python review_mutations.py <scratch worktree> [M1 M4 ...]   > review_mutations.txt

Run beside a live training run: nice 19 on the agent cores, CUDA_VISIBLE_DEVICES="" (root rules 5, 5b). The bank
build test is deselected (its E4 engine check needs a cargo-built worktree: env/Cargo.lock is in the env stamp).
"""
import subprocess, sys, os, json
WT = os.path.abspath(sys.argv[1])          # a scratch worktree at c9d86ef53 (each mutation is reverted after its run)
PY = sys.executable
B, S, D, R = "src/main/x65/bank.py", "src/main/x65/student.py", "src/main/x65/delivery.py", "src/main/reads/rules_distill_play.py"
BT, ST, DT, RT = "src/main/x65/bank_test.py", "src/main/x65/student_test.py", "src/main/x65/delivery_test.py", "src/main/reads/rules_distill_play_test.py"
M = [
 ("M1 frame_sha256 guard off", B, 'if frame_sha256(r["b64"]) != rec["frame_sha256"]:', 'if False:', [BT]),
 ("M2 honest_cut arrival order", B, 'rows.sort(key=lambda r: (r[1], 0 if r[2] == "p1" else 1, r[0]))', 'rows.sort(key=lambda r: r[0])', [BT]),
 ("M2b honest_cut keeps opp same-update", B, 'if upd[i] < u or (upd[i] == u and _token(cmds[i])[0] == side)]', 'if upd[i] <= u]', [BT]),
 ("M3 pi_prime Q*lam", B, 'float(q[str(i)]) / lam', 'float(q[str(i)]) * lam', [BT]),
 ("M4 legal_mask guard off", B, 'if legal != [i for i, m in enumerate(r["mask"]) if m]:', 'if False:', [BT]),
 ("M5 played guard off", B, 'if int(rec["played"]) != played:', 'if False:', [BT]),
 ("M7 key/deal guard off", B, 'if rec.get("key") != key or rec.get("deal") != deal:', 'if False:', [BT]),
 ("M18 inner_val residue", B, '% INNER_VAL_MOD == 0', '% INNER_VAL_MOD == 1', [BT]),
 ("M20 pstar tie exclusion off", B, 'if top[0] - top[1] <= TIE_EPS:', 'if False:', [BT]),
 ("M21 turn guard off", B, 'if int(rec["turn"]) != r["turn"]:', 'if False:', [BT]),
 ("M22 unavailable re-request counted", B, '                if err[s]:\n                    err[s] = False\n                else:\n                    count[s] += 1', '                err[s] = False\n                count[s] += 1', [BT]),
 ("M23 no-digest label kept", B, 'if not row["frame_checked"]:\n                    drop("no_frame_sha256", True)', 'if False:\n                    drop("no_frame_sha256", True)', [BT]),
 ("M24 pstar mismatch kept", B, '        if c == 0:\n            n_mis', '        if False:\n            n_mis', [BT]),
 ("M25 void threshold off", B, 'if p["label_drop_frac"] > MAX_DROP_FRAC:', 'if False:', [BT]),
 ("M8 split read-deal check off", S, 'if leaked:', 'if False:', [ST]),
 ("M8b split read-panel row check off", S, 'if (z["panel"] == read_ix).any():', 'if False:', [ST]),
 ("M9 train_label includes inner-val", S, '"train_label": np.flatnonzero(lab & ~iv)', '"train_label": np.flatnonzero(lab)', [ST]),
 ("M9b train_anchor includes inner-val", S, '"train_anchor": np.flatnonzero(~iv)', '"train_anchor": np.arange(len(iv))', [ST]),
 ("M26 beta 0", S, 'return kl_lab + BETA * kl_anc, kl_lab, kl_anc', 'return kl_lab, kl_lab, kl_anc', [ST]),
 ("M27 pick ties to higher lr", S, 'key=lambda r: (r["L"], r["lr"], r["checkpoint"])', 'key=lambda r: (r["L"], -r["lr"], r["checkpoint"])', [ST]),
 ("M14 delivery eps 0", D, 'cmp(lb, 0, RULE8_EPS, above=True)', 'cmp(lb, 0, 0, above=True)', [DT]),
 ("M15 delivery a_pstar vs student", D, '"a_pstar": (pia[lab_k] == played[lab_k]).astype(np.int8)', '"a_pstar": (s_arg[lab_k] == pia[lab_k]).astype(np.int8)', [DT]),
 ("M28 delivery row bootstrap", D, '    sums = {p: _deal_sums(per_panel[p]["deal"], per_panel[p]["diff"]) for p in panels}', '    sums = {p: (per_panel[p]["diff"].astype(float), np.ones(len(per_panel[p]["diff"]))) for p in panels}', [DT]),
 ("M10 rule boundary off", R, '        if abs(d) <= eps:\n            return False, True\n        return (d > 0 if above else d < 0), False', '        return (d > 0 if above else d < 0), False', [RT]),
 ("M12 imputation swapped", R, '("go_imputed", (0, 2)), ("null_imputed", (2, 0))', '("go_imputed", (2, 0)), ("null_imputed", (0, 2))', [RT]),
 ("M13 z look1 always", R, '    z = p["z_go"][str(look)]', '    z = p["z_go"]["1"]', [RT]),
 ("M16 drop cap >=", R, 'Fraction(drops, int(x["deals"])) > Fraction(str(p["max_drop_frac"]))', 'Fraction(drops, int(x["deals"])) >= Fraction(str(p["max_drop_frac"]))', [RT]),
 ("M17 gbank >", R, 'ident >= int(gb["min_identical"])', 'ident > int(gb["min_identical"])', [RT]),
 ("M29 null uses go-imputed", R, 'ub = _bound(est["go_imputed"], z, -1), _bound(est["null_imputed"], p["z_null"], +1)', 'ub = _bound(est["go_imputed"], z, -1), _bound(est["go_imputed"], p["z_null"], +1)', [RT]),
 ("M30 null bar <=", R, '"UB < null_bar": test(ub, bar, False)', '"UB < null_bar": test(ub, bar + Fraction(1, 1000), False)', [RT]),
 ("M31 SE divisor B", R, '/ (B - 1)', '/ B', [RT]),
 ("M32 engine void off", R, 'off = [u["unit"] for u in counted if u.get("engine") != reg]', 'off = []', [RT]),
 ("M33 sha void off", R, 'bad = [u["unit"] for u in counted if u["label"] == lab and u.get("model_sha256") != want]', 'bad = []', [RT]),
 ("M34 regime-consistency void off", R, '            if len(vals) > 1 or None in vals:', '            if False:', [RT]),
 ("M35 delivery consistency off", R, '        if (doc["delivered"], doc["boundary"]) != (crosses, near):', '        if False:', [RT]),
]
only = sys.argv[2:]
env = dict(os.environ, PYTHONPATH=WT + "/src", CUDA_VISIBLE_DEVICES="", GEN3AI_SKIP_DEPS_GUARD="1")
res = []
for name, f, old, new, tests in M:
    if only and not any(name.startswith(o + " ") for o in only):
        continue
    p = os.path.join(WT, f)
    src = open(p).read()
    n = src.count(old)
    if n != 1:
        res.append((name, f"PATTERN x{n}")); print(name, "PATTERN", n, flush=True); continue
    open(p, "w").write(src.replace(old, new))
    try:
        r = subprocess.run(["timeout", "600", "nice", "-n", "19", "taskset", "-c", "5-7,13-15", PY, "-m", "pytest", *tests, "-q", "-x", "-p", "no:cacheprovider",
                            "--deselect", "src/main/x65/bank_test.py::test_the_bank_builds_checks_and_never_opens_a_result_file"],
                           cwd=WT, env=env, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if l.startswith("FAILED") or " passed" in l or " failed" in l]
        verdict = "KILLED" if r.returncode == 1 else ("SURVIVED" if r.returncode == 0 else f"rc{r.returncode}")
        res.append((name, verdict, tail[-2:]))
        print(name, verdict, tail[-2:], flush=True)
    finally:
        open(p, "w").write(src)
