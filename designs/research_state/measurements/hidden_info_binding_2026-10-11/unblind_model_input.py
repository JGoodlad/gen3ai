"""UNBLIND the model-input read: apply ``arm_key.json``'s labels to ``result_model_input_blinded.json`` (written by
``score_model_input.py``, which never reads the key), then evaluate the REGISTERED RULES
(``designs/endstate/design_hidden_info_binding.md`` §12.8). Pure Python:

    python -I unblind_model_input.py [--key arm_key.json] [--blinded result_model_input_blinded.json]
                                     [--out result_model_input.json]

Relabelling (nothing else is computed from data):
- every per-code table is re-keyed by its LABEL (``TRUE@0.95`` ...); BASE (the un-hooked forward) stays BASE;
- every coded contrast ``Axx-Ayy`` is re-keyed ``<label x> - <label y>``;
- ``registered_contrasts``: TRUE(p) - PLACEBO(p) and TRUE(p) - SELFMAP(p) for every p both families carry, for
  IMP_POLICY and IMP_CRITIC, overall and per stratum. A pair stored in the other order is negated (mean -> -mean,
  CI [lo, hi] -> [-hi, -lo]: exact, the bootstrap draws are paired).

THE RULES (§12.8), per checkpoint (end2, prod: the registered reads; end2_s2002: the REPLICATION) and on the POOLED
end2 + prod read (``score_model_input.py``'s stratified bootstrap: battles resampled within each checkpoint,
state-weighted means). At the PRIMARY dose p = 0.95 and the SECONDARY p = 1.00. Every quantity is the RAW
(unweighted) read. Notation: IMP_POL = IMP_POLICY, IMP_CRIT = IMP_CRITIC, [lo, hi] its 95 % CI; FLIP / KL = TRUE(p)'s
mean flip rate / mean masked KL (point estimates).

RULE 8 (deterministic checks): EDGE = 0.0005, the registered text read LITERALLY ("a bound within +-0.0005 of its
boundary counts as not clear"), applied to EVERY threshold, not only the zero bounds. A comparison with a boundary b
is CLEAR only outside the band [b - EDGE, b + EDGE]: "x > b" means x > b + EDGE, "x < b" means x < b - EDGE, and
"x >= b" is read as "x > b" (clear). So FLIP >= 0.02 needs FLIP > 0.0205, FLIP < 0.02 needs < 0.0195, KL < 0.01 needs
< 0.0095, and "inside +-0.005" needs lo > -0.0045 and hi < +0.0045. A quantity inside a band satisfies neither a
condition nor its opposite, so it falls through to NOT DETECTED.

IDENTITY TEETH: per checkpoint, the TRUE@0.00 code (the hook ON at p = 0) must read exactly as BASE: max |KL| == 0,
FLIP count == 0, and every per-state IMP_POLICY and IMP_CRITIC equal to BASE's (the blinded result's ``identity``
block). ``identity_check`` is PASS / FAIL per checkpoint, and on the pooled read PASS only if every pooled checkpoint
passes. A FAIL VOIDS that read's NET verdicts (printed loudly); the CRITIC verdicts are kept but flagged.

NET (the first rule that holds, in this ORDER: USES, MISUSES, CONFIDENCE-ONLY, IGNORES; else NOT-DETECTED):
- NET-USES: IMP_POL(TRUE) lo > 0 AND [IMP_POL(TRUE) - IMP_POL(PLACEBO)] lo > 0 (paired);
- NET-MISUSES: FLIP >= 0.02 AND IMP_POL(TRUE) hi < 0;
- NET-CONFIDENCE-ONLY: FLIP >= 0.02 AND the CIs of IMP_POL(TRUE) - IMP_POL(SELFMAP) and IMP_POL(TRUE) - IMP_POL(PLACEBO)
  both lie inside (-0.005, +0.005) (lo > -0.005 and hi < +0.005, each clear of its boundary);
- NET-IGNORES: FLIP < 0.02 AND KL < 0.01 nats;
- NET-NOT-DETECTED otherwise.
CRITIC: CRITIC-USES iff [IMP_CRIT(TRUE) - IMP_CRIT(TRUE@0.00)] lo > 0 AND [IMP_CRIT(TRUE) - IMP_CRIT(PLACEBO)] lo > 0;
else CRITIC-NOT-DETECTED. (TRUE@0.00 is the hook-on null arm, bit-identical to BASE by the runner's teeth.)
Each verdict carries the evidence it read and the list of every NET rule whose condition held (``held``).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
BASE = "BASE"
EDGE = 0.0005
FLIP_B = 0.02
KL_B = 0.01
CONF_B = 0.005
DOSES: Tuple[Tuple[float, str], ...] = ((0.95, "primary"), (1.0, "secondary"))
ROLE = {"end2": "registered", "prod": "registered", "end2_s2002": "replication"}


# ------------------------------------------------------------------------------------------------
# relabelling
# ------------------------------------------------------------------------------------------------

def flip(stat: dict) -> dict:
    if stat.get("mean") is None:
        return dict(stat)
    lo, hi = stat["ci"]
    return {**stat, "mean": -stat["mean"], "ci": [-hi, -lo]}


def oriented(contrasts: dict, a: str, b: str, sep: str = "-") -> dict:
    """The contrast a - b from a table keyed ``x<sep>y``; a stored b - a is negated."""
    if f"{a}{sep}{b}" in contrasts:
        return contrasts[f"{a}{sep}{b}"]
    if f"{b}{sep}{a}" in contrasts:
        return {k: flip(v) for k, v in contrasts[f"{b}{sep}{a}"].items()}
    raise KeyError(f"no contrast between {a} and {b}")


def relabel_contrasts(con: dict, lab: Dict[str, str]) -> dict:
    out = {}
    for metric, table in con.items():
        out[metric] = {}
        for pair, v in table.items():
            a, b = pair.split("-")
            out[metric][f"{lab.get(a, a)} - {lab.get(b, b)}"] = v
    return out


def registered(con: dict, key: Dict[str, dict]) -> dict:
    by = {(v["family"], float(v["p"])): c for c, v in key.items()}
    out = {}
    for metric in ("IMP_POLICY", "IMP_CRITIC"):
        out[metric] = {}
        for other in ("PLACEBO", "SELFMAP"):
            for (fam, p), c in sorted(by.items()):
                if fam != "TRUE" or (other, p) not in by:
                    continue
                out[metric][f"TRUE@{p:.2f} - {other}@{p:.2f}"] = oriented(con[metric], c, by[(other, p)])
    return out


# ------------------------------------------------------------------------------------------------
# the registered rules (§12.8)
# ------------------------------------------------------------------------------------------------

def gt(x: Optional[float], b: float) -> bool:
    return x is not None and x > b + EDGE


def lt(x: Optional[float], b: float) -> bool:
    return x is not None and x < b - EDGE


def _ci(stat: dict) -> Tuple[Optional[float], Optional[float]]:
    return tuple(stat["ci"]) if stat.get("ci") else (None, None)  # type: ignore[return-value]


def _contrast(u: dict, metric: str, a: str, b: str) -> dict:
    return oriented(u["contrasts"][metric], a, b, sep=" - ")["raw"]


def net_rule(u: dict, p: float) -> dict:
    T, Pl, Sm = f"TRUE@{p:.2f}", f"PLACEBO@{p:.2f}", f"SELFMAP@{p:.2f}"
    raw = u["per_arm"][T]["raw"]
    imp = raw["IMP_POLICY"]
    fl, kl = raw["FLIP"]["mean"], raw["KL"]["mean"]
    dP = _contrast(u, "IMP_POLICY", T, Pl)
    dS = _contrast(u, "IMP_POLICY", T, Sm)
    ilo, ihi = _ci(imp)
    plo, phi = _ci(dP)
    slo, shi = _ci(dS)

    def inside(lo, hi):
        return gt(lo, -CONF_B) and lt(hi, CONF_B)

    held = {
        "NET-USES": gt(ilo, 0.0) and gt(plo, 0.0),
        "NET-MISUSES": gt(fl, FLIP_B) and lt(ihi, 0.0),
        "NET-CONFIDENCE-ONLY": gt(fl, FLIP_B) and inside(slo, shi) and inside(plo, phi),
        "NET-IGNORES": lt(fl, FLIP_B) and lt(kl, KL_B),
    }
    verdict = next((k for k in ("NET-USES", "NET-MISUSES", "NET-CONFIDENCE-ONLY", "NET-IGNORES") if held[k]),
                   "NET-NOT-DETECTED")
    return {"verdict": verdict, "held": [k for k, v in held.items() if v],
            "evidence": {"IMP_POL(TRUE)": imp, "FLIP(TRUE)": fl, "KL(TRUE)": kl,
                         "IMP_POL(TRUE) - IMP_POL(PLACEBO)": dP, "IMP_POL(TRUE) - IMP_POL(SELFMAP)": dS}}


def critic_rule(u: dict, p: float) -> dict:
    T, Pl, Z = f"TRUE@{p:.2f}", f"PLACEBO@{p:.2f}", "TRUE@0.00"
    dZ = _contrast(u, "IMP_CRITIC", T, Z)
    dP = _contrast(u, "IMP_CRITIC", T, Pl)
    uses = gt(_ci(dZ)[0], 0.0) and gt(_ci(dP)[0], 0.0)
    return {"verdict": "CRITIC-USES" if uses else "CRITIC-NOT-DETECTED",
            "evidence": {"IMP_CRIT(TRUE) - IMP_CRIT(TRUE@0.00)": dZ, "IMP_CRIT(TRUE) - IMP_CRIT(PLACEBO)": dP}}


RULES: List[Tuple[str, Callable[[dict, float], dict]]] = [("NET", net_rule), ("CRITIC", critic_rule)]


def verdicts(u: dict, identity_ok: bool = True) -> dict:
    if "per_arm" not in u:
        return {}
    out = {}
    for p, role in DOSES:
        need = {f"TRUE@{p:.2f}", f"PLACEBO@{p:.2f}", f"SELFMAP@{p:.2f}", "TRUE@0.00"}
        missing = sorted(need - set(u["per_arm"]))
        if missing:
            out[f"p={p:.2f} ({role})"] = {"error": f"arms absent: {missing}"}
            continue
        v = {name: fn(u, p) for name, fn in RULES}
        if not identity_ok:
            v["NET"] = {**v["NET"], "verdict": "VOID (identity_check FAIL)", "void_of": v["NET"]["verdict"]}
            v["CRITIC"] = {**v["CRITIC"], "flag": "identity_check FAIL"}
        out[f"p={p:.2f} ({role})"] = v
    return out


def identity_check(res: dict, key: Dict[str, dict]) -> str:
    """PASS iff the TRUE@0.00 code reads exactly as BASE (the blinded result's ``identity`` block)."""
    null = [c for c, v in key.items() if v["family"] == "TRUE" and float(v["p"]) == 0.0]
    if len(null) != 1 or "identity" not in res or null[0] not in res["identity"]:
        return "FAIL"
    x = res["identity"][null[0]]
    ok = (x.get("max_abs_kl") == 0.0 and x.get("flip_count") == 0
          and x.get("max_abs_d_imp_policy") == 0.0 and x.get("max_abs_d_imp_critic") == 0.0)
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------------------------------------------

def unblind_cp(res: dict, key: Dict[str, dict], identity: Optional[str] = None) -> dict:
    """``identity``: the identity teeth's verdict for this read (default: from ``res``'s own ``identity`` block)."""
    lab = {c: v["label"] for c, v in key.items()}
    lab[BASE] = BASE
    codes = res.get("codes", [])
    if sorted(codes) != sorted(key):
        raise SystemExit(f"{res.get('cp', res.get('pool'))}: the result carries codes {codes}, the key {sorted(key)}")
    out = {k: v for k, v in res.items()
           if k not in ("per_code", "contrasts", "strata", "coverage", "codes", "identity")}
    out["arms"] = {c: lab[c] for c in codes}
    if "identity" in res:
        out["identity"] = {lab[c]: v for c, v in res["identity"].items()}
    out["identity_check"] = identity if identity is not None else identity_check(res, key)
    name = res.get("cp", res.get("pool"))
    if out["identity_check"] != "PASS":
        print(f"!!! identity_check FAIL [{name}]: TRUE@0.00 does not read as BASE -> the NET verdicts are VOID",
              file=sys.stderr, flush=True)
    if "per_code" not in res:
        return out
    out["per_arm"] = {lab[c]: v for c, v in res["per_code"].items()}
    out["contrasts"] = relabel_contrasts(res["contrasts"], lab)
    out["registered_contrasts"] = registered(res["contrasts"], key)
    if "coverage" in res:
        out["coverage"] = {lab[c]: v for c, v in res["coverage"].items()}
    if "strata" in res:
        out["strata"] = {}
        for s, d in res["strata"].items():
            out["strata"][s] = {"n": d["n"], "per_arm": {lab[c]: v for c, v in d["per_code"].items()},
                                "contrasts": relabel_contrasts(d["contrasts"], lab),
                                "registered_contrasts": registered(d["contrasts"], key)}
    out["verdicts"] = verdicts(out, identity_ok=out["identity_check"] == "PASS")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(HERE / "arm_key.json"))
    ap.add_argument("--blinded", default=str(HERE / "result_model_input_blinded.json"))
    ap.add_argument("--out", default=str(HERE / "result_model_input.json"))
    a = ap.parse_args(argv)
    key = json.loads(Path(a.key).read_text())["codes"]
    bl = json.loads(Path(a.blinded).read_text())
    if not bl.get("blinded"):
        raise SystemExit("the input is not a blinded result")
    out = {"blinded": False, "partial": bl.get("partial"), "config": bl.get("config"),
           "rules": {"edge": EDGE, "flip": FLIP_B, "kl": KL_B, "confidence_band": CONF_B,
                     "doses": [list(d) for d in DOSES], "read": "raw (unweighted)"},
           "key": {c: v["label"] for c, v in sorted(key.items())},
           "cps": {cp: {**unblind_cp(r, key), "role": ROLE.get(cp, "other")} for cp, r in bl["cps"].items()}}
    if "pooled" in bl:
        parts = bl["pooled"]["pool"].split("+")
        ok = all(p in bl["cps"] and identity_check(bl["cps"][p], key) == "PASS" for p in parts)
        out["pooled"] = unblind_cp(bl["pooled"], key, identity="PASS" if ok else "FAIL")
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
