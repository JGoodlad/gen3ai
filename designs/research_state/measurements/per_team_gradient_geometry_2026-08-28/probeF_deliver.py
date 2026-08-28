"""PROBE F step 4 — assemble the deliverable JSON (every number, both epochs)."""
import json
import os

OUT = "/tmp/probeF"
DEST = ("designs/research_state/measurements/"
        "per_team_gradient_geometry_2026-08-28.json")


def main():
    geo = json.load(open(f"{OUT}/geometry.json"))
    geo_i = json.load(open(f"{OUT}/geometry_init.json"))
    idx = json.load(open(f"{OUT}/grads_index.json"))
    idx_i = json.load(open(f"{OUT}/grads_index_init.json"))
    st = json.load(open(f"{OUT}/states_per_team_meta.json"))
    acid = json.load(open(f"{OUT}/acid_test.json"))

    def strip(d):
        """drop the per-row fold index (large, reproducible from the seed)."""
        return {k: v for k, v in d.items() if k != "fold_index"}

    out = {
        "probe": "F — per-team gradient geometry (PCGrad substrate test)",
        "date": "2026-08-28",
        "acid_test": acid,
        "teams": st["teams"],
        "state_construction": {
            "source": "the five R2 fork runs' own eval_traces (both recorded "
                      "steps); each fork's --trainee-teams pins exactly the two "
                      "slice teams its traces contain",
            "per_team_states": st["per_team_target"],
            "batches_per_team": idx["n_folds"],
            "states_per_batch_target": idx["batch"],
            "batch_split": "BATTLE-disjoint (whole games assigned to folds)",
            "seed": st["seed"]},
        "losses": {
            "distill": {
                "role": "PRIMARY",
                "definition": "masked mean of -log pi_S(a_teacher) over the "
                              "legal set; a_teacher = the team's slice "
                              "teacher's argmax",
                "matches_production_flags": "--distill-target action "
                                            "--distill-topk 1 --distill-gate none",
                "deviations": [
                    "the AWR row weight w = clamp(exp(|A_hat|/beta), 20) is "
                    "dropped (A_hat is not recoverable offline) -> every row "
                    "carries w = 1",
                    "eccfe630ec08de27 is pinned by BOTH F5a and F5e and the "
                    "live fold averages the two teachers' KLs; this probe uses "
                    "F5a only, per the mission"]},
            "bc": {
                "role": "SECONDARY (well-conditioned)",
                "definition": "masked mean of -log pi_S(a_recorded) — the "
                              "policy gradient with every advantage set to +1"},
            "pgmc": {
                "role": "SECONDARY (directional only)",
                "definition": "-(A_hat * log pi_S(a_recorded)).mean() with "
                              "A_hat = per-batch-standardized (R - V_rec(s)), "
                              "R = +1 win / -1 loss / 0 tie",
                "status": "PARTIALLY RESOLVED — 2 of 9 teams fail the "
                          "cross-half positivity test"},
            "pg": {
                "role": "SECONDARY (directional only)",
                "definition": "-(A_hat * log pi_S(a_recorded)).mean() with "
                              "A_hat = per-batch-standardized within-battle TD "
                              "residual of the recorded critic",
                "status": "UNRESOLVED — the estimator returns |cos| > 1 "
                          "(up to 27.8 in policy_head), i.e. the batch noise "
                          "exceeds the signal; reported, never interpreted"}},
        "method": {
            "noise_ceiling": "cosine between two BATTLE-DISJOINT batches of the "
                             "SAME team — the number every between-team cosine "
                             "must be read against",
            "noise_free_estimator": "cross-half Gram: <mean(folds 0,2,4), "
                                    "mean(folds 1,3,5)> is unbiased for "
                                    "<mu_i, mu_j> including i==j, because the "
                                    "two halves' batch noises are independent",
            "uncertainty": "all 10 balanced 3/3 fold splits; a pair counts as "
                           "CONFLICTING only if it is negative in every split",
            "missing_policy": "a team whose own cross-half inner product is "
                              "<= 0 has NO resolvable team-consistent gradient "
                              "at this sample size; every cosine involving it "
                              "is MISSING, never imputed as zero. A group a "
                              "loss cannot reach at all is labelled "
                              "structurally_zero instead.",
            "pcgrad_arithmetic": "PCGrad projects g_i off each conflicting g_j, "
                                 "removing ||g_i||*|cos_ij| of NORM and "
                                 "|cos_ij|^2 of ENERGY; the per-team sum over "
                                 "conflicting j is an UPPER BOUND because the "
                                 "sequential projections interfere",
            "isotropic_null": "PC1 energy fraction for 9 mutually orthogonal "
                              "team gradients = 1/9 = 0.1111"},
        "checkpoints": {
            "END": {"path": idx["student"],
                    "meaning": "the rev-2 fold student, fold complete",
                    "index": strip(idx)},
            "INIT": {"path": idx_i["student"],
                     "meaning": "ai_v9_29_rev1_0823 final — the checkpoint every "
                                "fork AND the student were forked from, i.e. "
                                "where the fold's step 1 happens and where "
                                "PCGrad would first act",
                     "index": strip(idx_i)}},
        "geometry_END": geo,
        "geometry_INIT": geo_i,
    }
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    json.dump(out, open(DEST, "w"), indent=1)
    print("wrote", DEST, os.path.getsize(DEST), "bytes")


if __name__ == "__main__":
    main()
