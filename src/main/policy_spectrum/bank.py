"""The TURN BANK (`gen3_policy_spectrum_bank_v1`, M5 Lane S gates ① and ②).

A fixed set of ≥ 10,000 decisions, saved ONCE and read by every policy afterwards (checkpoints over
training, future architectures, search, Q-derived mixes) on the SAME turns.

**Stored as re-encodable INPUTS, never obs vectors.** A banked battle is its sim input log (the
resolved PRNG seed, both players' packed teams, the command log: the eval trace's reconstruction
record); a banked decision is an index into it — ``(battle, side, n)`` — plus its stamps. The obs a
policy reads is re-derived by :mod:`main.policy_spectrum.replay` with the reader's own encoder. What
the bank keeps of the recording is the sha256 of each recorded obs row (gate ①'s reference), the
recorded action and the recording policy's raw logits: 11 numbers, not an observation.

**Source: the lineage's own EVAL TRACES** (justification in the PROGRESS file). Each trace carries
the reconstruction record (the input log), and the obs / logits / mask the recording policy saw —
so gate ① (re-encoding reproduces the recorded obs byte-equal) is checkable on every recorded
decision, and the eval roster supplies the opponent classes (9 scripted bots, 5 pool-snapshot
sentinels) with no new games. The EXPLOITER class comes from the round-0 exploiter's games against
its target K2: there the traced side is the exploiter, and the bank takes the OTHER side — K2's own
decisions against the exploiter (re-encodable; no recorded obs to compare, so gate ① is not
checkable on those rows and the stamp says so).

⚠️ The eval trace quota is LOSS-ENRICHED by design (10 losses, 5 draws, 5 wins per opponent per
cycle). The builder draws a fixed per-outcome quota from each (cycle, opponent) so the bank is not
the quota's mix, and stamps every decision with its battle's outcome, but the bank is not a random
sample of the lineage's games — the RATE of a stratum in the bank is not the rate in play.

Layout of a bank directory: ``manifest.json`` (selection rule, sources, counts, gate ① result, the
bank's content hash), ``battles.jsonl.gz`` (one input log per line), ``decisions.jsonl.gz`` (one
stamped decision per line). Both ``.gz`` files are written with ``mtime=0``, so the same bank is the
same bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BANK_SCHEMA = "gen3_policy_spectrum_bank_v1"
BANK_SEED = "m5-laneS-bank-v1"

#: The scripted eval roster (``agents.training.eval_callback``); an unknown name is REFUSED.
BOTS = ("random", "heuristic", "heuristic2", "staller", "staller_v2", "aggressive",
        "aggressive_v2", "setup_sweep", "setup_sweep_v2")

#: Per (cycle, opponent) OUTCOME quota for a trainee-side source; a short bucket is filled from the
#: others in the order loss → win → draw.
OUTCOME_QUOTA = {"win": 2, "loss": 2, "draw": 1}


@dataclass(frozen=True)
class Source:
    """One eval-trace cycle the bank draws from."""

    run: str
    step_dir: str                 # ``step_<n>`` under ``eval_traces/``
    label: str                    # human label, e.g. ``N0@74M``
    #: ``trainee`` = bank the traced side against every roster opponent; ``other:<opponent dir>`` =
    #: bank the UNTRACED side of that opponent's games (the exploiter's target).
    mode: str = "trainee"
    #: for ``other:``: the policy that played the untraced side, and its label.
    other_policy: Optional[str] = None
    other_label: Optional[str] = None


#: The v1 bank's sources: the fixed lineage N0 → C_fix → K2 → K3, and K2 against the round-0
#: exploiter A′. N0's traces start at 36M (earlier cycles are not on disk).
SOURCES_V1: Tuple[Source, ...] = (
    Source("ai_v14_01_base", "step_36000000", "N0@36M"),
    Source("ai_v14_01_base", "step_46000032", "N0@46M"),
    Source("ai_v14_01_base", "step_56000016", "N0@56M"),
    Source("ai_v14_01_base", "step_66000000", "N0@66M"),
    Source("ai_v14_01_base", "step_74000016", "N0@74M"),
    Source("ai_v14_06_lbat_ctrl_fix", "step_82000032", "C_fix@82M"),
    Source("ai_v14_07_g0p_k2", "step_90000000", "K2@90M"),
    Source("ai_v14_08_g0p_k3", "step_98000016", "K3@98M"),
    Source("ai_v14_09_r0_offense_a", "step_92000016", "K2final_vs_A'@92M",
           mode="other:ext_ai_v14_07_g0p_k2",
           other_policy="ai_v14_07_g0p_k2/final_model.zip", other_label="K2 final (91.1M)"),
)


@dataclass
class BankBattle:
    battle_id: str
    source: dict                  # run, step_dir, label, opponent, prefix, recording commit/config
    format_id: str
    seed: str
    p1: dict
    p2: dict
    commands: List[list]
    banked_side: str              # the side whose decisions are banked
    traced_side: str              # the side the eval trace recorded (obs / logits)
    outcome: str                  # banked side's result: win / loss / draw

    def recorded(self):
        from agents.battle.rust_core_parity import RecordedBattle

        return RecordedBattle(label=self.battle_id, format_id=self.format_id, seed=self.seed,
                              p1=self.p1, p2=self.p2, commands=[list(c) for c in self.commands])


# ---------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------

def _rank_key(path: str) -> str:
    return hashlib.sha256(f"{BANK_SEED}:{path}".encode()).hexdigest()


def outcome_of(prefix: str) -> str:
    head = prefix.split("_", 1)[0]
    if head not in ("win", "loss", "draw"):
        raise ValueError(f"trace prefix {prefix!r} carries no outcome")
    return head


def pick_traces(prefixes: Iterable[str], quota: Dict[str, int] = OUTCOME_QUOTA) -> List[str]:
    """The deterministic per-outcome draw from one (cycle, opponent)'s traces: each outcome's
    traces in hash order, ``quota[o]`` of each, short buckets filled loss → win → draw."""
    by: Dict[str, List[str]] = {"win": [], "loss": [], "draw": []}
    for p in sorted(prefixes, key=_rank_key):
        by[outcome_of(p)].append(p)
    total = sum(quota.values())
    chosen = []
    for o in ("win", "loss", "draw"):
        chosen += by[o][: quota.get(o, 0)]
    for o in ("loss", "win", "draw"):
        for p in by[o][quota.get(o, 0):]:
            if len(chosen) >= total:
                break
            chosen.append(p)
    return sorted(chosen, key=_rank_key)


def trace_prefixes(opp_dir: Path) -> List[str]:
    return sorted(p.name[: -len("_reconstruction.json")]
                  for p in opp_dir.glob("*_reconstruction.json"))


def opponent_class(opp: str) -> str:
    if opp in BOTS:
        return "bot"
    if opp.startswith("sentinel_"):
        return "pool_snapshot"
    if opp.startswith("ext_"):
        return "exploiter_target"
    raise KeyError(f"unknown eval opponent {opp!r} (not a roster bot, sentinel_* or ext_*)")


def load_battle(models: Path, src: Source, opp: str, prefix: str) -> Tuple[BankBattle, dict]:
    """One trace → (its bank battle, the recorded arrays of the traced side)."""
    from utils.bridge.reconstruction import ReconstructionRecord

    cyc = models / src.run / "eval_traces" / src.step_dir
    manifest = json.loads((cyc / "eval_manifest.json").read_text())
    base = cyc / opp / prefix
    rec = ReconstructionRecord.from_dict(json.loads(Path(f"{base}_reconstruction.json").read_text()))
    players = rec.players()
    traced = rec.side_of(rec.trainee_username)
    other = "p2" if traced == "p1" else "p1"
    out_traced = outcome_of(prefix)
    if src.mode == "trainee":
        banked, outcome = traced, out_traced
    else:
        banked = other
        outcome = {"win": "loss", "loss": "win", "draw": "draw"}[out_traced]
    z = np.load(f"{base}_states.npz")
    arrays = {k: z[k] for k in ("obs", "logits", "actions", "action_mask", "has_state")}
    source = {"run": src.run, "step_dir": src.step_dir, "label": src.label, "opponent": opp,
              "prefix": prefix, "recording_commit": manifest.get("git_hash"),
              "config_version": manifest.get("config_version"),
              "arch_signature": manifest.get("arch_signature")}
    if src.mode != "trainee":
        source["banked_policy"] = src.other_policy
        source["banked_label"] = src.other_label
        source["opponent_policy"] = f"{src.run} eval snapshot {src.step_dir}"
    else:
        source["banked_policy"] = f"{src.run}/eval_traces/{src.step_dir}/snapshot.zip"
        source["banked_label"] = src.label
    bid = f"{src.run}/{src.step_dir}/{opp}/{prefix}"
    return BankBattle(battle_id=bid, source=source, format_id=rec.format_id, seed=rec.prng_seed,
                      p1=players["p1"], p2=players["p2"], commands=[list(c) for c in rec.commands],
                      banked_side=banked, traced_side=traced, outcome=outcome), arrays


def select(models: Path, sources: Sequence[Source] = SOURCES_V1) -> List[Tuple[Source, str, str]]:
    """Every (source, opponent, prefix) the bank takes, in a fixed order."""
    out = []
    for src in sources:
        cyc = models / src.run / "eval_traces" / src.step_dir
        if not cyc.is_dir():
            raise FileNotFoundError(f"eval-trace cycle {cyc} is missing")
        manifest = json.loads((cyc / "eval_manifest.json").read_text())
        if src.mode == "trainee":
            opps = [o for o in manifest["opponents"] if not o.startswith("ext_")]
            for opp in opps:
                opponent_class(opp)
                for p in pick_traces(trace_prefixes(cyc / opp)):
                    out.append((src, opp, p))
        else:
            opp = src.mode.split(":", 1)[1]
            for p in trace_prefixes(cyc / opp):          # every trace: the exploiter class is small
                out.append((src, opp, p))
    return out


# ---------------------------------------------------------------------------------------------
# stamps
# ---------------------------------------------------------------------------------------------

def team_hash(packed: str) -> str:
    return hashlib.sha256(packed.encode()).hexdigest()[:12]


def team_species(packed: str) -> List[str]:
    return [m.split("|", 1)[0].lower() for m in packed.split("]") if m]


def phase_of(turn: int, our_alive: int, opp_alive: int) -> str:
    """``opening`` (turn ≤ 3) / ``endgame`` (either side has ≤ 2 mons left) / ``midgame``."""
    if turn <= 3:
        return "opening"
    if min(our_alive, opp_alive) <= 2:
        return "endgame"
    return "midgame"


def legal_bucket(n_legal: int) -> str:
    if n_legal <= 3:
        return "2-3"
    if n_legal <= 6:
        return "4-6"
    return "7+"


def stamp(battle: BankBattle, d, rec: Optional[dict]) -> dict:
    """One decision's bank row. ``d`` is a ``replay.ReplayDecision``; ``rec`` the traced side's
    recorded arrays (``None`` when the banked side was not traced)."""
    from main.policy_spectrum.categories import token_category, token_subtype

    side = battle.banked_side
    me = battle.p1 if side == "p1" else battle.p2
    them = battle.p2 if side == "p1" else battle.p1
    req_side = d.request.get("side", {})
    our_alive = sum(1 for m in req_side.get("pokemon", []) if not str(m.get("condition", "")).endswith("fnt"))
    opp_size = len(team_species(them["team"]))
    opp_alive = opp_size - d.opp_fainted
    moves_legal = [i for i, t in d.tokens.items() if t.startswith("move ")]
    if bool(d.request.get("forceSwitch")):
        kind = "forced_switch"
    elif not moves_legal:
        kind = "switch_only"            # no move is legal but it is not a forced switch (rare)
    else:
        kind = "free"
    opp = battle.source["opponent"]
    ocls = opponent_class(opp)
    if ocls == "exploiter_target":
        ocls, oname = "exploiter", battle.source["run"]
    else:
        oname = opp
    cats = {str(i): token_category(t) for i, t in sorted(d.tokens.items())}
    subs = {str(i): s for i, t in sorted(d.tokens.items()) if (s := token_subtype(t))}
    row = {
        "id": f"{battle.battle_id}#{side}#{d.n}",
        "battle": battle.battle_id,
        "side": side,
        "n": d.n,
        "turn": d.turn,
        "kind": kind,
        "n_legal": len(d.tokens),
        "legal_bucket": legal_bucket(len(d.tokens)),
        "mask": "".join(str(int(x)) for x in d.mask),
        "tokens": {str(i): t for i, t in sorted(d.tokens.items())},
        "cats": cats,
        "status_sub": subs,
        "phase": phase_of(d.turn, our_alive, opp_alive),
        "our_alive": our_alive,
        "opp_alive": opp_alive,
        "opp_class": ocls,
        "opp_name": oname,
        "team": team_hash(me["team"]),
        "outcome": battle.outcome,
        "source": battle.source["label"],
        "played": d.choice,
        "rec_action": None,
        "rec_obs_sha256": None,
        "rec_logits": None,
    }
    if rec is not None:
        row["rec_action"] = int(rec["actions"][d.n])
        row["rec_obs_sha256"] = hashlib.sha256(np.ascontiguousarray(rec["obs"][d.n], dtype="<f4").tobytes()).hexdigest()
        # the SHORTEST decimal that round-trips each float32 (exact, about half the bytes)
        row["rec_logits"] = [float(np.format_float_positional(x, unique=True))
                             for x in np.asarray(rec["logits"][d.n], dtype=np.float32)]
    return row


# ---------------------------------------------------------------------------------------------
# gate ① + build
# ---------------------------------------------------------------------------------------------

class GateOneFailure(RuntimeError):
    """A re-encoded row, mask or action disagrees with the recording."""


def check_recorded(battle: BankBattle, decisions: list, rec: dict) -> dict:
    """Gate ① on one battle's TRACED side: the re-encoded rows are byte-equal to the recorded obs,
    the masks equal, and every recorded action is the token the log played. Returns counts; raises
    :class:`GateOneFailure` on the first disagreement."""
    obs, masks, acts, has = rec["obs"], rec["action_mask"], rec["actions"], rec["has_state"]
    if len(decisions) != len(obs):
        raise GateOneFailure(f"{battle.battle_id}: {len(decisions)} re-encoded decisions for "
                             f"{len(obs)} recorded")
    for d in decisions:
        if not int(has[d.n]):
            raise GateOneFailure(f"{battle.battle_id} n={d.n}: the recording has no state here")
        if d.row.tobytes() != np.ascontiguousarray(obs[d.n], dtype="<f4").tobytes():
            diff = np.nonzero(d.row != obs[d.n])[0]
            raise GateOneFailure(f"{battle.battle_id} n={d.n}: obs differs at {diff[:8].tolist()} "
                                 f"({len(diff)} cells)")
        if d.mask.astype(bool).tolist() != np.asarray(masks[d.n], dtype=bool).tolist():
            raise GateOneFailure(f"{battle.battle_id} n={d.n}: mask {d.mask.tolist()} != recorded "
                                 f"{np.asarray(masks[d.n], dtype=int).tolist()}")
        a = int(acts[d.n])
        if d.tokens.get(a) != d.choice:
            raise GateOneFailure(f"{battle.battle_id} n={d.n}: recorded action {a} "
                                 f"({d.tokens.get(a)!r}) is not the played {d.choice!r}")
    return {"decisions": len(decisions)}


def op_semantics() -> str:
    """This checkout's damage-op feature-semantics identity (`damage_tables.OP_SEMANTICS`)."""
    from agents.model.damage_tables import OP_SEMANTICS
    return OP_SEMANTICS


def encoder_identity() -> dict:
    """What the recorded rows are a property of, at THIS checkout: the obs golden's hash (it moves
    with every deliberate encoder change), the obs width and the architecture signature. Gate ①'s
    byte-equality is REQUIRED while these equal the bank's; after a deliberate change it is
    checkable only at the recording commit (a pinned worktree)."""
    from agents.battle.core_obs import obs_dim
    from agents.model.model_version import ARCH_SIGNATURE
    from utils.paths import src_path

    golden = src_path("agents", "training", "golden_obs_fixture.json")
    return {"obs_golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
            "obs_dim": int(obs_dim()), "arch_signature": ARCH_SIGNATURE}


def _gz_bytes(lines: Iterable[str]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as f:
        for line in lines:
            f.write(line.encode())
            f.write(b"\n")
    return buf.getvalue()


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class Bank:
    manifest: dict
    battles: List[BankBattle]
    decisions: List[dict]
    battle_index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.battle_index = {b.battle_id: i for i, b in enumerate(self.battles)}


def content_sha(battles: Sequence[BankBattle], decisions: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for b in battles:
        h.update(_dumps(asdict(b)).encode() + b"\n")
    h.update(b"--\n")
    for d in decisions:
        h.update(_dumps(d).encode() + b"\n")
    return h.hexdigest()


def write_bank(out: Path, manifest: dict, battles: Sequence[BankBattle], decisions: Sequence[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "battles.jsonl.gz").write_bytes(_gz_bytes(_dumps(asdict(b)) for b in battles))
    (out / "decisions.jsonl.gz").write_bytes(_gz_bytes(_dumps(d) for d in decisions))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def load_bank(path: Path, verify: bool = True) -> Bank:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != BANK_SCHEMA:
        raise ValueError(f"{path}: schema {manifest.get('schema')!r} is not {BANK_SCHEMA!r}")
    with gzip.open(path / "battles.jsonl.gz", "rt") as f:
        battles = [BankBattle(**json.loads(line)) for line in f if line.strip()]
    with gzip.open(path / "decisions.jsonl.gz", "rt") as f:
        decisions = [json.loads(line) for line in f if line.strip()]
    if verify:
        sha = content_sha(battles, decisions)
        if sha != manifest["content_sha256"]:
            raise ValueError(f"{path}: content sha {sha} != manifest {manifest['content_sha256']} "
                             "— the bank was edited; a bank is written once")
    return Bank(manifest, battles, decisions)


def summarize(decisions: Sequence[dict]) -> dict:
    """Counts per stratum (the manifest's table)."""
    from collections import Counter

    def count(key):
        return dict(sorted(Counter(key(d) for d in decisions).items()))

    cat_legal = Counter()
    for d in decisions:
        for c in set(d["cats"].values()):
            cat_legal[c] += 1
    return {
        "decisions": len(decisions),
        "battles": len({d["battle"] for d in decisions}),
        "kind": count(lambda d: d["kind"]),
        "phase": count(lambda d: d["phase"]),
        "legal_bucket": count(lambda d: d["legal_bucket"]),
        "opp_class": count(lambda d: d["opp_class"]),
        "opp_name": count(lambda d: d["opp_name"]),
        "source": count(lambda d: d["source"]),
        "outcome": count(lambda d: d["outcome"]),
        "teams": len({d["team"] for d in decisions}),
        "category_legal": dict(sorted(cat_legal.items())),
        "gate1_checkable": sum(1 for d in decisions if d["rec_obs_sha256"]),
    }


def build(models: Path, out: Path, sources: Sequence[Source] = SOURCES_V1, workers: int = 2,
          commit: str = "unknown", log=print) -> dict:
    """Select, re-encode (gate ①), stamp and write a bank. Returns the manifest."""
    from main.policy_spectrum.replay import core_events_identity, replay

    picks = select(models, sources)
    log(f"[policy_spectrum] {len(picks)} traces selected from {len(sources)} cycles")
    battles, recs = [], []
    for src, opp, prefix in picks:
        b, arrays = load_battle(models, src, opp, prefix)
        battles.append(b)
        recs.append(arrays)
    results = replay([b.recorded() for b in battles], workers=workers)
    decisions: List[dict] = []
    gate = {"battles_checked": 0, "decisions_checked": 0, "decisions_unrecorded": 0,
            "failures": []}
    dropped_single = 0
    kept_battles = []
    for b, arrays, res in zip(battles, recs, results):
        if not res.ok:
            raise RuntimeError(f"{b.battle_id}: the core refused the replay: {res.error}")
        traced_ix = 0 if b.traced_side == "p1" else 1
        banked_ix = 0 if b.banked_side == "p1" else 1
        # gate ①: the traced side, EVERY recorded decision (banked or not)
        try:
            gate["decisions_checked"] += check_recorded(b, res.decisions[traced_ix], arrays)["decisions"]
            gate["battles_checked"] += 1
        except GateOneFailure as exc:
            gate["failures"].append(str(exc))
            continue
        rec = arrays if banked_ix == traced_ix else None
        rows = []
        for d in res.decisions[banked_ix]:
            if len(d.tokens) < 2:
                dropped_single += 1
                continue
            rows.append(stamp(b, d, rec))
            if rec is None:
                gate["decisions_unrecorded"] += 1
        if rows:
            kept_battles.append(b)
            decisions.extend(rows)
    if gate["failures"]:
        raise GateOneFailure(f"gate ① FAILED on {len(gate['failures'])} battles; first: "
                             f"{gate['failures'][0]}")
    manifest = {
        "schema": BANK_SCHEMA,
        "bank_seed": BANK_SEED,
        "built_at_commit": commit,
        "sources": [asdict(s) for s in sources],
        "outcome_quota": OUTCOME_QUOTA,
        "selection_rule": ("per (eval cycle, roster opponent): the traces of each outcome in "
                           "sha256(bank_seed:prefix) order, 2 wins + 2 losses + 1 draw, a short "
                           "bucket filled loss -> win -> draw; the exploiter source takes every "
                           "trace and banks the UNTRACED (target) side"),
        "dropped_single_legal": dropped_single,
        "gate1": {**gate, "result": "PASS", "encoder": core_events_identity(),
                  "rule": "every recorded decision of every banked battle's traced side re-encodes "
                          "byte-equal (float32 row), with an equal mask and the recorded action "
                          "equal to the token the log played"},
        "counts": summarize(decisions),
        "encoder_identity": encoder_identity(),
        # The forward's feature semantics the recorded logits were produced under is the RECORDING's,
        # but a re-read on THIS checkout reproduces them only while the op's semantics are the same.
        "op_semantics": op_semantics(),
        "content_sha256": content_sha(kept_battles, decisions),
    }
    write_bank(out, manifest, kept_battles, decisions)
    return manifest
