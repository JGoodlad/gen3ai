r"""The registered rule ``distill_play``: X65 ExIt-0's read (``designs/research_state/measurements/x65_exit0_2026-11-05/
PREREG.md`` §1, §4, §6, §10). Three P\* STUDENTS, each distilled offline from S′'s banked choices on two panels, each play
greedy with no search against Kakuna on the THIRD panel's deals (``students``: S_H on heldout72_hpfix, S_F on forum72,
S_P on p72), paired by deal with P\* raw (``raw``).

THE DEALS. A deal is (panel, unit seed, game), one lone one-game unit per arm (``per_game`` groups). ``seeds`` registers
each look's unit seeds (look 1: ``{"1": "300-344"}``; look 2 adds ``"2": "600-674"``); a look reads the CUMULATIVE pool,
every game ``0..19`` of every seed. ``groups[panel]`` names the anchor groups whose units make that panel's deals: the
student's and the raw arm's units on that panel, played here or imported (``main.reads.imports``). Each (label, panel,
seed, game) is read from exactly one of them.

THE CONTROL. ``control: "banked"``: the raw arm's look-1 deals are S′-LIVE's BANKED ``pstar`` units, imported, and
G-BANK (below) must pass. ``control: "fresh"`` (G-BANK failed): the raw arm is played (or imported from the read that
played it); no G-BANK.

A DEAL'S STATUS. A side is ``counted`` (``main.anchors.results.unit_counts``), ``exhausted`` (not counted, and either
imported from a closed read or its retries are spent here, ``main.reads.attempts``), ``pending`` (not counted, retries
left) or ``missing`` (no row: unplayed). A deal with a pending or missing side is UNFINISHED (a VOID: the look is
incomplete, never silently short). A finished deal is a PAIR (both counted) or a DROP (a side exhausted). y = win 1,
tie ½, loss 0; D = y(student) − y(raw). The arithmetic is exact, in units of ½ (``2D`` is an integer).

THE ESTIMATE (PREREG §6). Ĝ_X = the mean of D over panel X's deals of the look; Ĝ = (Ĝ_H + Ĝ_F + Ĝ_P) / 3. Three
estimates: PAIRED (drops left out, REPORTED), GO-IMPUTED and NULL-IMPUTED (every drop at the worst case for its class,
§4: for GO the missing student scores 0 and the missing raw 1; for NULL the missing student 1 and the missing raw 0).

THE SE. For each estimate a FRESH ``numpy.random.default_rng(seed)``; for each panel in ``u_panels`` order, its seeds in
ascending order (look 2: look 1's and look 2's as ONE pool) as vectors ``s2`` (Σ 2D of the seed's deals) and ``c`` (its
deal count), ``idx = rng.integers(0, n, size=(B, n))``; replicate b's Ĝ*_X = Σ s2[idx[b]] / (2 Σ c[idx[b]]) (the seeds
drawn with replacement, their deals whole), Ĝ*_b = the mean of the panels' Ĝ*_X. Every replicate is an exact fraction.
SE = the sample standard deviation of the B replicates (divisor B − 1), its variance exact and its square root the
float ``math.sqrt`` of the correctly rounded variance.

THE BOUNDS AND CLASSES (PREREG §1). LB = Ĝ_go − z_go[look] · SE_go (z 2.576 at look 1, 2.0 at look 2); UB = Ĝ_null +
z_null · SE_null (1.96). Compared as exact fractions; a bound within ``rule8_eps`` of its bar is ``near``, never a
crossing (root rule 8), and the decision is flagged ``boundary``. In order:

* VOID if the delivery gate's file is missing or does not match (below);
* NOT-DELIVERED if the gate's ``delivered`` is false (no game is read);
* VOID on any reason below;
* GO iff LB > 0; else NULL iff UB < ``null_bar_pp`` / 100; else CONTINUE at look 1 (look 2 plays) and UNDECIDED at look 2.

THE DELIVERY GATE (PREREG §5; computed by the X65 builder, schema ``gen3_x65_delivery_v1``, read by path, its sha256
recorded): the rule TRUSTS its ``delivered`` but checks the schema and key set, ``B`` and ``seed`` against the registered
``delivery``, each panel's student ``zip_sha256`` against the spec's student model, ``bank_manifest_sha256`` against the
registered one, and that ``delivered`` / ``boundary`` are the file's own LB against 0 under ``rule8_eps``.

G-BANK (PREREG §4; ``control: "banked"`` only). The ``gbank`` group re-plays, for each panel and look-1 unit seed s, the
banked raw deal game ``(s − origin) mod 20`` (the spec's ``games`` subset, exactly). Each replay is compared with the
banked unit of the same deal (the raw arm's imported unit): the START document and both sides' protocol lines with the
``|t:|`` lines removed, the whole game (``main.reads.prefix``'s comparison, with no turn bound). A replay or bank that did
not count is not identical. Fewer than ``min_identical`` (133) of ``n`` (135) identical VOIDs the banked control for the
look; look 1 then plays a fresh raw arm on the same deals (``read_spec_look1_fresh.json``). An unfinished replay VOIDs.

VOID (named reasons): an unfinished deal or G-BANK replay; drops above ``max_drop_frac`` of a panel's deals at the look
(PREREG §4: S′-LIVE's banked raw drops enter as drops, being imported and not counted); a counted unit off the registered
``engine`` (or on none); a counted unit, or a resolved model, off its label's pinned sha256; a counted unit not greedy, or
carrying a search or a belief-prior mix; a counted unit whose ``games.jsonl`` disagrees with its summary; a panel's
counted units (both arms and the G-BANK replays) on more than one regime id, peer code or policy mode; a panel with < 2
seeds; G-BANK as above.

REPORTED, never decided: the three estimates (each Ĝ_X, Ĝ, SE), each arm's mean score over the paired deals per panel
and equal-weight; the discordance (b + c); the counts (deals, pairs, drops by side, unfinished) per panel and look;
G-BANK's comparisons; the delivery file's numbers; at look 2, the same statistic on look 1's deals alone (``look1``).
"""
from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_read_distill_play_v1"
DELIVERY_SCHEMA = "gen3_x65_delivery_v1"
DELIVERY_KEYS = ("schema", "bank_manifest_sha256", "students", "per_panel", "pooled", "lb", "B", "seed", "delivered",
                 "boundary", "reported")
DELIVERY_PANEL_KEYS = ("n_rows", "n_deals", "agree_student", "agree_pstar", "delta")
KEYS = ("look", "u_panels", "students", "raw", "groups", "seeds", "control", "gbank", "engine", "z_go", "z_null",
        "null_bar_pp", "bootstrap", "rule8_eps", "max_drop_frac", "delivery", "bank_manifest_sha256")
GBANK_KEYS = ("group", "origin", "min_identical", "n")
CONTROLS = ("banked", "fresh")
RAW_REGIME = "greedy"
NOT_DELIVERED, GO, NULL, CONTINUE, UNDECIDED, VOID = "NOT-DELIVERED", "GO", "NULL", "CONTINUE", "UNDECIDED", "VOID"
COUNTED, EXHAUSTED, PENDING, MISSING = "counted", "exhausted", "pending", "missing"
ESTIMATES = ("paired", "go_imputed", "null_imputed")
RULE_DOCS = {"distill_play": "X65 ExIt-0: each distilled P* student against P* raw on its held-out panel, paired by deal "
                             "(lone one-game units vs Kakuna; look 1's raw arm banked from S′-LIVE behind G-BANK); "
                             "Ĝ = mean of panel means of D, SE by a within-panel unit-seed bootstrap; NOT-DELIVERED by "
                             "the delivery gate's file, GO iff LB(z by look) > 0, NULL iff UB(1.96) < +2 pp, worst-case "
                             "imputation adverse to each class, exact fractions with rule 8"}
_UNIT_KEYS = ("unit", "label", "panel", "unit_seed", "game", "counted", "wins", "ties", "n", "our_regime", "search",
              "search_played", "belief_prior_mix", "model_sha256", "engine", "games", "regime_id", "peer_code",
              "policy_mode", "peer_pack", "imported_from", "cell_dir")


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _hex64(x: Any) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def unit_name(label: str, panel: str, seed: int, game: int) -> str:
    """A per-game unit's name (``main.x35.kakuna.plan_units``)."""
    return f"{label}__{panel}__{int(seed):02d}__g{int(game):02d}"


def look_seeds(p: Mapping[str, Any]) -> Dict[int, int]:
    """``{unit seed: the look that added it}`` over the looks up to ``p["look"]`` (the look's cumulative pool)."""
    from main.reads.spec import parse_unit_seeds

    out: Dict[int, int] = {}
    for k in range(1, int(p["look"]) + 1):
        for s in parse_unit_seeds(p["seeds"][str(k)], f"seeds.{k}"):
            out[int(s)] = k
    return out


# ------------------------------------------------------------------------------------------------ check (spec load)
def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys
    from main.reads.spec import PENDING, SpecError, parse_unit_seeds

    probs = _check_keys(p, KEYS, KEYS)
    if probs:
        return probs
    look = p["look"]
    if look not in (1, 2) or isinstance(look, bool):
        return [f"look: 1 or 2, not {look!r}"]
    u = p["u_panels"]
    if not (isinstance(u, list) and u and all(isinstance(x, str) for x in u) and len(set(u)) == len(u)):
        return ["u_panels: a non-empty list of distinct team sets"]
    raw, stu, grp = p["raw"], p["students"], p["groups"]
    if not (isinstance(stu, dict) and set(stu) == set(u) and len(set(stu.values())) == len(u)):
        return ["students: {panel: label}, one distinct label for each of u_panels"]
    if not (isinstance(grp, dict) and set(grp) == set(u)
            and all(isinstance(v, list) and v and len(set(v)) == len(v) for v in grp.values())):
        return ["groups: {panel: [anchor group, ...]}, a non-empty list for each of u_panels"]
    for lab in [*stu.values(), raw]:
        m = models.get(lab)
        if m is None:
            probs.append(f"{lab!r} is not a declared model")
        elif m.ref.kind != "pinned":
            probs.append(f"{lab!r}: a pinned ref (its recorded sha256; PREREG §10)")
        elif m.load:
            probs.append(f"{lab!r}: plays greedy with no load option (PREREG §1), has {m.load}")
    if raw in stu.values():
        probs.append(f"raw {raw!r} is also a student")
    sd = p["seeds"]
    if not (isinstance(sd, dict) and set(sd) == {str(k) for k in range(1, look + 1)}):
        return probs + [f"seeds: {{look: unit seeds}} for exactly the looks 1..{look}"]
    pools: Dict[str, Tuple[int, ...]] = {}
    for k, v in sd.items():
        try:
            pools[k] = parse_unit_seeds(v, f"seeds.{k}")
        except SpecError as exc:
            probs.append(str(exc))
    if len(pools) != len(sd):
        return probs
    pool = [x for v in pools.values() for x in v]
    if len(pool) != len(set(pool)):
        probs.append("seeds: the looks' unit seeds overlap")
    if probs:
        return probs
    opponents = set()
    listed = {g for v in grp.values() for g in v}
    for ts in u:
        for lab in (stu[ts], raw):
            cover: List[int] = []
            for gname in grp[ts]:
                a = anchor.get(gname)
                if a is None:
                    probs.append(f"groups.{ts}: {gname!r} is not an anchor group")
                    continue
                opponents.add(a.opponent)
                if not a.per_game or a.games is not None:
                    probs.append(f"group {gname!r}: a deal group plays every game of its seeds as one-game units "
                                 "(per_game, no games subset)")
                if ts not in a.teamsets:
                    probs.append(f"group {gname!r} does not play {ts}")
                if not set(a.models) <= {stu[ts], raw}:
                    probs.append(f"group {gname!r} under {ts} holds {list(a.models)}, not only {stu[ts]!r} / {raw!r}")
                if stu[ts] in a.models and tuple(a.teamsets) != (ts,):
                    probs.append(f"group {gname!r}: student {stu[ts]!r} plays its read panel {ts} only, the group "
                                 f"plays {list(a.teamsets)}")
                if lab in a.models:
                    cover += list(a.unit_seeds)
            if sorted(cover) != sorted(pool):
                probs.append(f"{ts}: {lab!r}'s groups cover unit seeds {sorted(set(cover) ^ set(pool))[:6]} other than "
                             "exactly once each of the look's pool")
    for st in stu.values():
        other = [g for g, a in anchor.items() if st in a.models and g not in grp[next(t for t in u if stu[t] == st)]]
        if other:
            probs.append(f"student {st!r} is in group(s) {other} outside its panel's groups")
    ctl, gb = p["control"], p["gbank"]
    if ctl not in CONTROLS:
        probs.append(f"control: one of {CONTROLS}")
    elif ctl == "fresh" and gb is not None:
        probs.append("gbank: null with a fresh control (G-BANK guards the banked one)")
    elif ctl == "banked":
        if not (isinstance(gb, dict) and set(gb) == set(GBANK_KEYS)):
            probs.append(f"gbank: {{{', '.join(GBANK_KEYS)}}} with a banked control")
        else:
            probs += _check_gbank(gb, anchor, raw, u, pools["1"], listed)
            for ts in u:
                for gname in grp[ts]:
                    a = anchor.get(gname)
                    if a is not None and raw in a.models and set(a.unit_seeds) & set(pools["1"]) and (
                            a.imports is None or raw not in a.imports.models):
                        probs.append(f"group {gname!r}: a banked control imports the raw arm's look-1 deals")
    extra = sorted(set(anchor) - listed - ({gb["group"]} if isinstance(gb, dict) else set()))
    if extra:
        probs.append(f"anchor group(s) {extra} are read by nothing")
    if len(opponents) > 1:
        probs.append(f"the deal groups play {sorted(opponents)}: one opponent")
    if not (isinstance(p["engine"], str) and p["engine"]):
        probs.append("engine: the registered engine stamp (port=..;env=..)")
    z = p["z_go"]
    if not (isinstance(z, dict) and set(z) == {"1", "2"} and all(_num(v) and v > 0 for v in z.values())):
        probs.append('z_go: {"1": z, "2": z}, each > 0')
    for k in ("z_null", "null_bar_pp"):
        if not (_num(p[k]) and p[k] > 0):
            probs.append(f"{k}: a number > 0")
    bs = p["bootstrap"]
    if not (isinstance(bs, dict) and set(bs) == {"B", "seed"} and _int(bs.get("B")) and bs["B"] >= 1000
            and _int(bs.get("seed"))):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    if not (_num(p["rule8_eps"]) and 0 <= p["rule8_eps"] < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01)")
    if not (_num(p["max_drop_frac"]) and 0 <= p["max_drop_frac"] <= 0.2):
        probs.append("max_drop_frac: a number in [0, 0.2]")
    dv = p["delivery"]
    if not (isinstance(dv, dict) and set(dv) == {"path", "B", "seed"} and isinstance(dv.get("path"), str)
            and os.path.isabs(dv["path"]) and _int(dv.get("B")) and _int(dv.get("seed"))):
        probs.append("delivery: {path (absolute), B: int, seed: int}")
    if not (_hex64(p["bank_manifest_sha256"]) or p["bank_manifest_sha256"] == PENDING):
        probs.append("bank_manifest_sha256: 64 lowercase hex (the pin amendment's)")
    return probs


def _check_gbank(gb: Mapping[str, Any], anchor: Any, raw: str, u: Sequence[str], seeds1: Sequence[int],
                 listed: set) -> List[str]:
    from main.x35.kakuna import UNIT_GAMES

    probs: List[str] = []
    a = anchor.get(gb["group"])
    if a is None:
        return [f"gbank.group {gb['group']!r} is not an anchor group"]
    if gb["group"] in listed:
        probs.append(f"gbank.group {gb['group']!r} is also a deal group")
    if tuple(a.models) != (raw,) or set(a.teamsets) != set(u) or tuple(a.unit_seeds) != tuple(sorted(seeds1)):
        probs.append(f"gbank group {gb['group']!r} must play exactly {raw!r} on {list(u)} at look 1's unit seeds")
    if not _int(gb["origin"]):
        return probs + ["gbank.origin: an int (game = (seed - origin) mod 20)"]
    want = {int(s): ((int(s) - gb["origin"]) % UNIT_GAMES,) for s in seeds1}
    if not a.per_game or a.games != want:
        probs.append(f"gbank group {gb['group']!r} must be per_game with games exactly (seed - {gb['origin']}) mod "
                     f"{UNIT_GAMES} for each seed")
    n = len(u) * len(seeds1)
    if gb["n"] != n or not _int(gb["n"]):
        probs.append(f"gbank.n: {n} (one deal a panel and look-1 seed)")
    if not (_int(gb["min_identical"]) and 1 <= gb["min_identical"] <= n):
        probs.append(f"gbank.min_identical: an int in [1, {n}]")
    return probs


# ------------------------------------------------------------------------------------------------ inputs (I/O here)
def _status(u: Mapping[str, Any], spent: set) -> str:
    if u.get("counted"):
        return COUNTED
    return EXHAUSTED if (u.get("imported_from") or str(u["unit"]) in spent) else PENDING


def _row(u: Mapping[str, Any], group: str, role: str, spent: set) -> Dict[str, Any]:
    return dict({k: u.get(k) for k in _UNIT_KEYS}, group=group, role=role, status=_status(u, spent))


def read_delivery(dv: Mapping[str, Any], *, students: Mapping[str, Optional[str]], u_panels: Sequence[str],
                  bank_manifest: str, eps: Any) -> Dict[str, Any]:
    """The delivery gate's file (module docstring): its numbers and every way it does not match (``problems``)."""
    from main.reads.rules_search_headroom import cmp
    from utils.hashing import sha256_file

    path = Path(str(dv["path"]))
    if not path.is_file():
        return {"path": str(path), "problems": [f"{path}: missing (the delivery gate writes it before any game)"]}
    try:
        doc = json.loads(path.read_text())
    except ValueError as exc:
        return {"path": str(path), "problems": [f"{path}: unreadable ({exc})"]}
    out: Dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "problems": []}
    probs: List[str] = out["problems"]
    if not isinstance(doc, dict):
        probs.append("not a JSON object")
        return out
    if doc.get("schema") != DELIVERY_SCHEMA:
        probs.append(f"schema {doc.get('schema')!r}, not {DELIVERY_SCHEMA!r}")
    if set(doc) != set(DELIVERY_KEYS):
        probs.append(f"keys {sorted(set(doc) ^ set(DELIVERY_KEYS))} differ from the schema's {list(DELIVERY_KEYS)}")
    for k, want in (("B", dv["B"]), ("seed", dv["seed"])):
        if doc.get(k) != want or not _int(doc.get(k)):
            probs.append(f"{k} {doc.get(k)!r}, not the registered {want}")
    if not _hex64(doc.get("bank_manifest_sha256")):
        probs.append("bank_manifest_sha256: not 64 lowercase hex")
    elif doc["bank_manifest_sha256"] != bank_manifest:
        probs.append(f"bank_manifest_sha256 {doc['bank_manifest_sha256'][:12]} is not the registered "
                     f"{str(bank_manifest)[:12]}")
    st = doc.get("students")
    if not (isinstance(st, dict) and set(st) == set(u_panels)):
        probs.append(f"students: not exactly {list(u_panels)}")
    else:
        for ts in u_panels:
            e = st[ts]
            got = e.get("zip_sha256") if isinstance(e, dict) and set(e) == {"zip_sha256"} else None
            if not _hex64(got) or got != students[ts]:
                probs.append(f"students.{ts}.zip_sha256 {str(got)[:12]} is not the spec's student "
                             f"{str(students[ts])[:12]}")
    pp = doc.get("per_panel")
    if not (isinstance(pp, dict) and set(pp) == set(u_panels) and all(
            isinstance(v, dict) and set(v) == set(DELIVERY_PANEL_KEYS) and _int(v["n_rows"]) and _int(v["n_deals"])
            and all(_num(v[k]) for k in ("agree_student", "agree_pstar", "delta")) for v in pp.values())):
        probs.append(f"per_panel: {{panel: {{{', '.join(DELIVERY_PANEL_KEYS)}}}}} for exactly {list(u_panels)}")
    for k in ("pooled", "lb"):
        if not _num(doc.get(k)) or not math.isfinite(float(doc[k])):
            probs.append(f"{k}: not a finite number")
    for k in ("delivered", "boundary"):
        if not isinstance(doc.get(k), bool):
            probs.append(f"{k}: not true / false")
    if not probs:
        crosses, near = cmp(float(doc["lb"]), 0, eps, above=True)
        if (doc["delivered"], doc["boundary"]) != (crosses, near):
            probs.append(f"delivered {doc['delivered']} / boundary {doc['boundary']} are not its LB {doc['lb']!r} "
                         f"against 0 with rule8_eps (delivered {crosses}, boundary {near})")
    out.update({k: doc.get(k) for k in ("delivered", "boundary", "pooled", "lb", "per_panel", "B", "seed")})
    return out


def compare_whole(banked: Mapping[str, Any], replay: Mapping[str, Any]) -> Dict[str, Any]:
    """One G-BANK pair, the WHOLE game: the START documents and both sides' protocol without ``|t:|``
    (``main.reads.prefix``'s comparison with no turn bound)."""
    from main.reads import prefix as PX

    a, b = PX._start(banked), PX._start(replay)
    if a != b:
        return {"identical": False, "why": f"the START documents differ in "
                                           f"{sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}"}
    for slot in ("p1", "p2"):
        la, _ = PX.side_prefix(banked, slot, -1)          # |turn|-1 never occurs: every line
        lb, _ = PX.side_prefix(replay, slot, -1)
        if la != lb:
            i = next((k for k, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
            return {"identical": False, "why": (f"{slot}'s protocol differs at line {i}: banked "
                                                f"{(la[i] if i < len(la) else '<end>')[:160]!r} vs replay "
                                                f"{(lb[i] if i < len(lb) else '<end>')[:160]!r}")}
    return {"identical": True, "why": None}


def gbank_compare(replay: Optional[Mapping[str, Any]], banked: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``{identical, why}``: True / False, or None while the replay is unfinished (a VOID until it is)."""
    from main.reads import prefix as PX

    if replay is None or replay["status"] in (PENDING, MISSING):
        return {"identical": None, "why": "the replay is unfinished"}
    if replay["status"] != COUNTED:
        return {"identical": False, "why": "the replay did not count"}
    if banked is None or banked["status"] != COUNTED:
        return {"identical": False, "why": "the banked unit did not count (or is not in this read)"}
    try:
        return compare_whole(PX.capture_of(Path(str(banked["cell_dir"]))),
                             PX.capture_of(Path(str(replay["cell_dir"]))))
    except (PX.PrefixError, TypeError) as exc:
        return {"identical": False, "why": f"unreadable: {exc}"}


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.reads.rules import _group
    from main.x35.kakuna import UNIT_GAMES

    u_panels, raw, stu = list(p["u_panels"]), str(p["raw"]), dict(p["students"])
    labels = [*stu.values(), raw]
    want = {lab: spec.models[lab].ref.sha256 for lab in labels}
    models = cells.get("models") or {}
    units: List[Dict[str, Any]] = []
    by: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    groups: Dict[str, Any] = {}
    for ts in u_panels:
        for gname in p["groups"][ts]:
            g = _group(cells, "anchor", gname)
            groups[gname] = {k: g.get(k) for k in ("engine", "regime_id", "regime_ids", "unplayed", "inconclusive")}
            spent = set(g.get("exhausted_units") or ())
            for u in g["units"]:
                if u["panel"] != ts or u["label"] not in (stu[ts], raw):
                    continue
                if u.get("game") is None:
                    raise _err(f"unit {u['unit']}: not a per-game unit")
                key = (str(u["label"]), ts, int(u["unit_seed"]), int(u["game"]))
                if key in by:
                    raise _err(f"deal {key[1:]} of {key[0]!r} is in two groups ({by[key]['group']}, {gname})")
                by[key] = _row(u, gname, "deal", spent)
                units.append(by[key])
    pool = look_seeds(p)
    deals: List[Dict[str, Any]] = []
    for ts in u_panels:
        for seed in sorted(pool):
            for game in range(UNIT_GAMES):
                d: Dict[str, Any] = {"deal": f"{ts}:{seed}:{game}", "panel": ts, "unit_seed": seed, "game": game,
                                     "look": pool[seed]}
                for side, lab in (("student", stu[ts]), ("raw", raw)):
                    r = by.get((lab, ts, seed, game))
                    d[side] = {"unit": unit_name(lab, ts, seed, game) if r is None else r["unit"],
                               "status": MISSING if r is None else r["status"]}
                deals.append(d)
    gb: Optional[Dict[str, Any]] = None
    if p.get("gbank"):
        gp = p["gbank"]
        g = _group(cells, "anchor", gp["group"])
        groups[gp["group"]] = {k: g.get(k) for k in ("engine", "regime_id", "regime_ids", "unplayed", "inconclusive")}
        spent = set(g.get("exhausted_units") or ())
        reps: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        for u in g["units"]:
            if u["label"] == raw and u.get("game") is not None:
                reps[(str(u["panel"]), int(u["unit_seed"]), int(u["game"]))] = _row(u, gp["group"], "gbank", spent)
        units += list(reps.values())
        comps = []
        for ts in u_panels:
            for seed in sorted(k for k, look in pool.items() if look == 1):
                game = (seed - int(gp["origin"])) % UNIT_GAMES
                rep, bank = reps.get((ts, seed, game)), by.get((raw, ts, seed, game))
                comps.append(dict(gbank_compare(rep, bank), deal=f"{ts}:{seed}:{game}",
                                  replay=None if rep is None else rep["unit"], banked=None if bank is None
                                  else bank["unit"]))
        gb = {"group": gp["group"], "origin": gp["origin"], "n": gp["n"], "min_identical": gp["min_identical"],
              "compared": comps}
    return {"schema": SCHEMA, "look": int(p["look"]), "u_panels": u_panels, "students": stu, "raw": raw,
            "control": p["control"], "seeds": {str(k): sorted(s for s, lk in pool.items() if lk == k)
                                               for k in sorted(set(pool.values()))},
            "engine_registered": p["engine"], "want_sha256": want,
            "resolved_sha256": {lab: (models.get(lab) or {}).get("sha256") for lab in labels},
            "groups": groups, "units": sorted(units, key=lambda r: (r["role"], r["unit"])), "deals": deals,
            "gbank": gb,
            "delivery": read_delivery(p["delivery"], students={ts: want[stu[ts]] for ts in u_panels},
                                      u_panels=u_panels, bank_manifest=p["bank_manifest_sha256"],
                                      eps=p["rule8_eps"])}


# ------------------------------------------------------------------------------------------------ the statistic
def _y2(u: Mapping[str, Any]) -> int:
    """2y of a one-game unit (win 2, tie 1, loss 0)."""
    n, w, t = int(u["n"]), int(u["wins"]), int(u.get("ties") or 0)
    if n != 1:
        raise _err(f"unit {u['unit']}: n = {n}, not a one-game (per_game) unit")
    return 2 * w + t


def tally(doc: Mapping[str, Any], looks: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Per panel: the deals (of ``looks``; default every look in the doc), their counts, and for each estimate
    ``{seed: [Σ 2D, deals]}``. A drop enters the GO / NULL estimates at its worst case (module docstring)."""
    units = {u["unit"]: u for u in doc["units"] if u["role"] == "deal"}
    out: Dict[str, Any] = {}
    for ts in doc["u_panels"]:
        out[ts] = {"deals": 0, "pairs": 0, "unfinished": 0, "student_missing": 0, "raw_missing": 0, "both_missing": 0,
                   "discordant": 0, "y2_student": 0, "y2_raw": 0, "drops": [],
                   "est": {e: {} for e in ESTIMATES}}
    for d in doc["deals"]:
        if looks is not None and int(d["look"]) not in looks:
            continue
        t = out[d["panel"]]
        t["deals"] += 1
        ss, rs = d["student"]["status"], d["raw"]["status"]
        if ss in (PENDING, MISSING) or rs in (PENDING, MISSING):
            t["unfinished"] += 1
            continue
        ys = _y2(units[d["student"]["unit"]]) if ss == COUNTED else None
        yr = _y2(units[d["raw"]["unit"]]) if rs == COUNTED else None
        seed = int(d["unit_seed"])
        if ys is not None and yr is not None:
            t["pairs"] += 1
            t["discordant"] += int(ys != yr)
            t["y2_student"] += ys
            t["y2_raw"] += yr
            for e in ESTIMATES:
                acc = t["est"][e].setdefault(seed, [0, 0])
                acc[0] += ys - yr
                acc[1] += 1
            continue
        cause = "both_missing" if ys is None and yr is None else "student_missing" if ys is None else "raw_missing"
        t[cause] += 1
        t["drops"].append({"deal": d["deal"], "cause": cause})
        for e, (s_miss, r_miss) in (("go_imputed", (0, 2)), ("null_imputed", (2, 0))):
            acc = t["est"][e].setdefault(seed, [0, 0])
            acc[0] += (s_miss if ys is None else ys) - (r_miss if yr is None else yr)
            acc[1] += 1
    return out


def bootstrap_se(per_panel: Mapping[str, Mapping[int, Sequence[int]]], u_panels: Sequence[str], B: int,
                 seed: int) -> float:
    """SE of Ĝ (module docstring): a fresh ``default_rng(seed)``, panels in order, seeds ascending, exact replicates."""
    import numpy as np

    rng = np.random.default_rng(seed)
    reps = [Fraction(0)] * B
    for ts in u_panels:
        seeds = sorted(per_panel[ts])
        s2 = np.array([int(per_panel[ts][s][0]) for s in seeds], dtype=np.int64)
        c = np.array([int(per_panel[ts][s][1]) for s in seeds], dtype=np.int64)
        idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
        S, C = s2[idx].sum(axis=1), c[idx].sum(axis=1)
        reps = [r + Fraction(int(a), 2 * int(b)) for r, a, b in zip(reps, S, C)]
    k = len(u_panels)
    reps = [r / k for r in reps]
    mean = sum(reps, Fraction(0)) / B
    var = sum(((r - mean) ** 2 for r in reps), Fraction(0)) / (B - 1)
    return math.sqrt(var)


def estimate(t: Mapping[str, Any], u_panels: Sequence[str], name: str, B: int, seed: int) -> Dict[str, Any]:
    """One estimate: each Ĝ_X and Ĝ exact, and its SE."""
    per = {ts: {s: v for s, v in t[ts]["est"][name].items() if v[1] > 0} for ts in u_panels}
    g = {ts: Fraction(sum(v[0] for v in per[ts].values()), 2 * sum(v[1] for v in per[ts].values()))
         for ts in u_panels}
    gg = sum(g.values(), Fraction(0)) / len(u_panels)
    se = bootstrap_se(per, u_panels, B, seed)
    return {"G": {ts: str(g[ts]) for ts in u_panels}, "G_float": {ts: float(g[ts]) for ts in u_panels},
            "G_all": str(gg), "G_all_float": float(gg), "se": se, "deals": {ts: sum(v[1] for v in per[ts].values())
                                                                             for ts in u_panels}}


def _bound(e: Mapping[str, Any], z: Any, sign: int) -> Fraction:
    return Fraction(e["G_all"]) + sign * Fraction(str(z)) * Fraction(e["se"])


def classify(lb: Fraction, ub: Fraction, look: int, p: Mapping[str, Any]) -> Dict[str, Any]:
    """GO / NULL / CONTINUE (look 1) / UNDECIDED (look 2) from the exact bounds; rule 8 by ``rule8_eps``."""
    eps = Fraction(str(p["rule8_eps"]))
    bar = Fraction(str(p["null_bar_pp"])) / 100

    def test(stat: Fraction, at: Fraction, above: bool) -> Tuple[bool, bool]:
        d = stat - at
        if abs(d) <= eps:
            return False, True
        return (d > 0 if above else d < 0), False

    t = {"LB > 0": test(lb, Fraction(0), True), "UB < null_bar": test(ub, bar, False)}
    go, null = t["LB > 0"][0], t["UB < null_bar"][0]
    cls = GO if go else NULL if null else CONTINUE if look == 1 else UNDECIDED
    return {"class": cls, "boundary": any(near for _c, near in t.values()),
            "tests": {k: {"crosses": c, "near": n} for k, (c, n) in t.items()},
            "lb": float(lb), "ub": float(ub), "lb_exact": str(lb), "ub_exact": str(ub)}


# ------------------------------------------------------------------------------------------------ VOID
def _games_disagree(u: Mapping[str, Any]) -> bool:
    gs = u.get("games")
    if not gs:
        return False
    want = "win" if int(u["wins"]) else "tie" if int(u.get("ties") or 0) else "loss"
    return len(gs) != 1 or str(gs[0][3]) != want


def gbank_summary(gb: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if gb is None:
        return None
    comps = gb["compared"]
    ident = sum(1 for c in comps if c["identical"] is True)
    unfinished = sum(1 for c in comps if c["identical"] is None)
    return {"n": len(comps), "registered_n": gb["n"], "identical": ident, "unfinished": unfinished,
            "min_identical": gb["min_identical"], "passed": unfinished == 0 and len(comps) == gb["n"]
            and ident >= int(gb["min_identical"]),
            "differing": [{k: c[k] for k in ("deal", "replay", "banked", "why")} for c in comps
                          if c["identical"] is False]}


def void_reasons(doc: Mapping[str, Any], t: Mapping[str, Any], p: Mapping[str, Any]) -> List[str]:
    """Every named VOID reason of a delivered look (empty = it may be decided)."""
    why: List[str] = []
    unf = {ts: int(x["unfinished"]) for ts, x in t.items() if x["unfinished"]}
    if unf:
        why.append(f"{sum(unf.values())} deal(s) unfinished: the look is incomplete ({unf})")
    for ts, x in t.items():
        drops = int(x["student_missing"]) + int(x["raw_missing"]) + int(x["both_missing"])
        if x["deals"] and Fraction(drops, int(x["deals"])) > Fraction(str(p["max_drop_frac"])):
            why.append(f"{ts}: {drops} of {x['deals']} deals dropped (student {x['student_missing']}, raw "
                       f"{x['raw_missing']}, both {x['both_missing']}), above max_drop_frac {p['max_drop_frac']}")
        if len([s for s, v in x["est"]["go_imputed"].items() if v[1]]) < 2:
            why.append(f"{ts}: fewer than 2 unit seeds hold a deal (the unit bootstrap needs >= 2)")
    counted = [u for u in doc["units"] if u["counted"]]
    reg = doc["engine_registered"]
    off = [u["unit"] for u in counted if u.get("engine") != reg]
    if off:
        why.append(f"engine: {len(off)} counted unit(s) not on the registered {reg} (first {off[0]})")
    for lab, want in sorted(doc["want_sha256"].items()):
        if doc["resolved_sha256"].get(lab) != want:
            why.append(f"checkpoint: {lab!r} resolved to {str(doc['resolved_sha256'].get(lab))[:12]}, not its pinned "
                       f"{str(want)[:12]}")
        bad = [u["unit"] for u in counted if u["label"] == lab and u.get("model_sha256") != want]
        if bad:
            why.append(f"checkpoint: {len(bad)} counted unit(s) of {lab!r} played other bytes than {str(want)[:12]} "
                       f"(first {bad[0]})")
    reg_bad = [u["unit"] for u in counted if u.get("our_regime") != RAW_REGIME or u.get("search") is not None
               or u.get("search_played") is not None or u.get("belief_prior_mix") is not None]
    if reg_bad:
        why.append(f"regime: {len(reg_bad)} counted unit(s) not greedy with no search / prior mix (first {reg_bad[0]})")
    dis = [u["unit"] for u in counted if _games_disagree(u)]
    if dis:
        why.append(f"results: {len(dis)} counted unit(s) whose games.jsonl disagrees with the summary (first {dis[0]})")
    for ts in doc["u_panels"]:
        mine = [u for u in counted if u["panel"] == ts]
        for key in ("regime_id", "peer_code", "policy_mode"):
            vals = {u.get(key) for u in mine}
            if len(vals) > 1 or None in vals:
                why.append(f"{ts}: counted units (both arms, G-BANK) span {key} {sorted(map(str, vals))}")
    gs = gbank_summary(doc.get("gbank"))
    if gs is not None and not gs["passed"]:
        if gs["unfinished"] or gs["n"] != gs["registered_n"]:
            why.append(f"G-BANK: {gs['unfinished']} of {gs['n']} replay(s) unfinished (registered {gs['registered_n']})")
        else:
            why.append(f"G-BANK: {gs['identical']} of {gs['n']} replays byte-identical to the banked capture, fewer than "
                       f"{gs['min_identical']}: the banked control is VOID for this look; look 1 plays a fresh raw arm "
                       "on the same deals (PREREG §4; read_spec_look1_fresh.json)")
    return why


# ------------------------------------------------------------------------------------------------ decide
def _arm_scores(t: Mapping[str, Any], u_panels: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in ("student", "raw"):
        per = {ts: Fraction(t[ts][f"y2_{arm}"], 2 * t[ts]["pairs"]) for ts in u_panels if t[ts]["pairs"]}
        out[arm] = {"by_panel": {ts: float(v) for ts, v in per.items()},
                    "equal_weight": float(sum(per.values(), Fraction(0)) / len(per)) if len(per) == len(u_panels)
                    else None}
    return out


def _counts(t: Mapping[str, Any]) -> Dict[str, Any]:
    return {ts: {k: x[k] for k in ("deals", "pairs", "unfinished", "student_missing", "raw_missing", "both_missing",
                                   "discordant")} | {"drops": list(x["drops"])} for ts, x in t.items()}


def look_statistic(doc: Mapping[str, Any], p: Mapping[str, Any], look: int, looks: Sequence[int]) -> Dict[str, Any]:
    """The three estimates and the class at ``look`` over the deals of ``looks``."""
    t = tally(doc, looks)
    b = p["bootstrap"]
    u = list(doc["u_panels"])
    est = {e: estimate(t, u, e, int(b["B"]), int(b["seed"])) for e in ESTIMATES}
    z = p["z_go"][str(look)]
    lb, ub = _bound(est["go_imputed"], z, -1), _bound(est["null_imputed"], p["z_null"], +1)
    return {"tally": t, "estimates": est, "z_go": z, "z_null": p["z_null"], "class": classify(lb, ub, look, p)}


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    look = int(doc["look"])
    dv = doc["delivery"]
    out: Dict[str, Any] = {"schema": SCHEMA, "look": look, "control": doc["control"],
                           "delivery": {k: dv.get(k) for k in ("path", "sha256", "delivered", "boundary", "pooled", "lb",
                                                               "per_panel", "problems")},
                           "gbank": gbank_summary(doc.get("gbank")), "seeds": doc["seeds"]}
    if dv.get("problems"):
        reasons = [f"delivery gate: {x}" for x in dv["problems"]]
        return dict(out, decision={"class": VOID, "verdict": f"{VOID}: {reasons[0]}", "reasons": reasons})
    if not dv["delivered"]:
        played = sum(1 for u in doc["units"] if u["counted"] and u["label"] in doc["students"].values())
        return dict(out, students_counted_units=played, decision={
            "class": NOT_DELIVERED, "boundary": bool(dv["boundary"]),
            "verdict": f"{NOT_DELIVERED}: the delivery gate's LB(Δagree) {dv['lb']!r} is not above 0 "
                       f"(pooled {dv['pooled']!r}); no Kakuna game is read"
                       + ("; flagged boundary" if dv["boundary"] else "")})
    looks = list(range(1, look + 1))
    t = tally(doc, looks)
    out["counts"] = _counts(t)
    reasons = void_reasons(doc, t, p)
    if reasons:
        return dict(out, decision={"class": VOID, "verdict": f"{VOID}: {reasons[0]}", "reasons": reasons})
    st = look_statistic(doc, p, look, looks)
    est, cls = st["estimates"], st["class"]
    flat_pairs = sum(t[ts]["pairs"] for ts in t)
    out.update(estimates=est, bounds={"lb": cls["lb"], "ub": cls["ub"], "lb_exact": cls["lb_exact"],
                                      "ub_exact": cls["ub_exact"], "z_go": st["z_go"], "z_null": st["z_null"]},
               scores=_arm_scores(t, list(doc["u_panels"])),
               discordance={"pairs": flat_pairs, "discordant": sum(t[ts]["discordant"] for ts in t),
                            "share": (sum(t[ts]["discordant"] for ts in t) / flat_pairs) if flat_pairs else None},
               bootstrap={"B": int(p["bootstrap"]["B"]), "seed": int(p["bootstrap"]["seed"]),
                          "scheme": "default_rng(seed) fresh per estimate; u_panels in order; seeds ascending as one "
                                    "pool; integers(0, n, (B, n)); exact replicates; SE = sample SD (B - 1)"})
    if look == 2:
        l1 = look_statistic(doc, p, 1, [1])
        out["look1"] = {"class": l1["class"], "G_go": l1["estimates"]["go_imputed"]["G_all_float"],
                        "G_null": l1["estimates"]["null_imputed"]["G_all_float"]}
    pp = lambda x: f"{100 * x:+.2f}"                                                  # noqa: E731
    go, nu, pa = est["go_imputed"], est["null_imputed"], est["paired"]
    verdict = (f"{cls['class']}: look {look}, Ĝ (student − P* raw, by deal) paired {pp(pa['G_all_float'])} pp "
               f"(SE {100 * pa['se']:.2f}); GO-imputed {pp(go['G_all_float'])}, LB(z {st['z_go']}) {pp(cls['lb'])} pp "
               f"(GO iff > 0); NULL-imputed {pp(nu['G_all_float'])}, UB(z {st['z_null']}) {pp(cls['ub'])} pp (NULL iff "
               f"< {p['null_bar_pp']} pp)" + ("; flagged boundary" if cls["boundary"] else ""))
    return dict(out, decision=dict(cls, verdict=verdict))
