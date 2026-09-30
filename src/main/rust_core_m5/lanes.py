"""The M5 LANE REGISTRY — every lane's parity gate as ONE declared row (M5 Lane J).

A row names the lane (the program doc's lane id), a human description, what its gate proves, and
the pytest files / node ids that ARE its gate — the lane's own tests, delegated to, never
re-implemented. The tier is chosen by MARKER, exactly as the repo's tiers are: COMMIT = the routine
gate's ``not slow and not e2e``, MILESTONE = everything but ``e2e`` (the ``slow`` tests included).
GPU tests (``gpu_tests``) additionally need ``GEN3AI_TEST_ALLOW_GPU=1`` under the GPU lock; without
``--gpu`` they read NOT RUN, never PASS.

**Adding a lane is one row.** ``lanes_test.py`` (routine) fails when the program doc marks a lane
BUILT and it has no BUILT row here, when a lane of the doc's lane table has no row at all, when a
declared file or node id does not exist, and when a row claims BUILT for a lane the doc does not
without saying why (``doc_note``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: The program doc the registry is checked against (repo-relative).
PROGRAM_DOC = ("designs", "endstate", "program_rust_core.md")

COMMIT_MARKERS = "not slow and not e2e"
MILESTONE_MARKERS = "not e2e"


@dataclass(frozen=True)
class LaneGate:
    lane: str                     # the program doc's lane id ("0", "A", …, "T2")
    title: str                    # a human description of the lane
    gate: str                     # what its gate proves, in one line
    built: bool                   # False ⇒ the row reads NOT BUILT and nothing runs
    m5_gate: bool = True          # counts toward M5's verdict (S and K are reported beside it)
    tests: Tuple[str, ...] = ()   # repo-relative pytest files or node ids (``file::name``)
    gpu_tests: Tuple[str, ...] = ()   # node ids that need an idle GPU (run only with ``--gpu``)
    pending: str = ""             # NOT BUILT: what the gate will be when it lands
    doc_note: str = ""            # why BUILT here differs from the doc's own marking, if it does


_CARGO_SUITE = "src/utils/rust_env/core_cargo_test.py::test_the_rust_env_core_suite_passes"

LANES: Tuple[LaneGate, ...] = (
    LaneGate(
        "0", "the shared core boundary — the column contract, the pool of N envs, the refusal policy, the build stamp",
        "① core rows byte-equal to sim_bridge's __OBS__ (ladder commit routine; ladder milestone, pool and procedural "
        "slow); ② seed → bytes, thread-count-invariant; ③ the generated column schema; ④ quarantine + bank; ⑤ the stamp",
        built=True,
        tests=("src/utils/rust_env/core_cargo_test.py", "src/utils/rust_env/columns_test.py",
               "src/utils/rust_env/protocol_test.py", "src/utils/rust_env/stamp_test.py")),
    LaneGate(
        "A", "the FFI front end — the core loaded in-process through a generated C ABI",
        "the Lane-0 corpus replayed through ctypes byte-equal to the in-Rust run; a panic is a typed error, not a crash",
        built=True, tests=("src/utils/rust_env/ffi_test.py", "src/utils/rust_env/ffi_integration_test.py")),
    LaneGate(
        "B", "the process front end — the core in a child process over one shared memory mapping",
        "FFI == process byte-identical on recorded battles; SIGKILL → typed error + respawn; no leaked segment",
        built=True, tests=("src/utils/rust_env/proc_test.py", "src/utils/rust_env/proc_integration_test.py")),
    LaneGate(
        "C", "training labels — the 18 production label keys the core computes",
        "label columns == Gen3Env's production keys per decision, byte for byte (commit routine, milestone slow)",
        built=True,
        tests=("src/utils/rust_env/label_columns_test.py", "src/agents/training/rust_env_label_inventory_test.py",
               "src/agents/training/rust_env_labels_parity_test.py")),
    LaneGate(
        "D", "episodes and reward — terminal reward, terminated / truncated, the stall forfeit, ties",
        "reward / terminated / truncated == Gen3Env on the same battles, through both front ends",
        built=True,
        tests=("src/agents/training/rust_env_episode_parity_test.py", _CARGO_SUITE)),
    LaneGate(
        "E", "opponent routing — per-episode route table, policy opponents through T2, bots played in the core",
        "a POLICY opponent's decisions through T2 == the per-env RLPlayer path (greedy and sampled, bit for bit)",
        built=True,
        tests=("src/agents/training/rust_env_opponents_test.py", "src/agents/training/rust_env_opponents_parity_test.py",
               _CARGO_SUITE),
        gpu_tests=("src/agents/training/rust_env_opponents_parity_test.py::"
                   "test_milestone_gpu_graph_backend_vs_compiled_cpu_on_a_real_pool",)),
    LaneGate(
        "F", "scripted bots — all ten pooled bot classes ported into the core",
        "per-bot action, view hash and RNG-stream offset equal on a banked decision corpus (commit 55 episodes "
        "routine, milestone 520 slow)",
        built=True,
        tests=("src/utils/rust_env/bot_inventory_test.py", "src/utils/rust_env/bot_tables_test.py",
               "src/utils/rust_env/bots_gate_test.py")),
    LaneGate(
        "I", "search on successors() in process — the search tree and play-to-the-end without the JSON child",
        "the depth-3 successor slice byte-equal to search_driver's rows; clone independence; decision equality",
        built=True, tests=("src/utils/rust_env/successors_integration_test.py", _CARGO_SUITE)),
    LaneGate(
        "T2", "the inference service — every network forward through fixed GPU weight slots and buckets",
        "greedy actions equal + legal log-prob / V bars against eager, per slot × bucket (CPU routine; CUDA slow)",
        built=True,
        tests=("src/agents/inference/service/service_test.py", "src/agents/inference/service/service_cuda_test.py"),
        gpu_tests=("src/agents/inference/service/service_cuda_test.py",)),
    LaneGate(
        "G", "training integration — the Rust env as the trainer's vec env, behind --env-core",
        "", built=False,
        pending="BUILDING — units 1–3 landed: the complete-game collector, the trainer wiring behind --env-core "
                "rust (the --debug smoke and a real GPU launch completed), and slice N at the ROLLOUT level + the "
                "learner-level check (src/agents/training/rust_rollout/parity_test.py, COMMIT routine / MILESTONE "
                "slow); still to come: the opponent sampling change (F-LE-8) and the throughput A/B at --n-envs 48 "
                "through this harness's hooks"),
    LaneGate(
        "H", "eval on the core — the eval callback and its traces on the Rust env",
        "", built=False,
        pending="the same seed set played on both paths gives equal greedy results; traces load in the prober"),
    LaneGate(
        "S", "the policy-spectrum instrument — a fixed re-encodable turn bank and its reader",
        "① the bank re-encodes byte-equal; ② stratified + stamped; ③ the reader reproduces recorded probabilities",
        built=True, m5_gate=False,
        tests=("src/main/policy_spectrum/bank_test.py", "src/main/policy_spectrum/categories_test.py",
               "src/main/policy_spectrum/spectrum_test.py", "src/main/policy_spectrum/truth_test.py",
               "src/main/policy_spectrum/policy_spectrum_integration_test.py",
               "src/main/policy_spectrum/truth_integration_test.py")),
    LaneGate(
        "K", "the learner pipeline — torch 2.8 env, diagnostics cadence, compile lifecycle (K1–K10)",
        "K1: the real-obs compile parity gate; K2: learning bit-identical with diagnostics on vs skipped",
        built=True, m5_gate=False,
        tests=("src/agents/model/compile_control_test.py", "src/agents/training/compiled_train_probes_test.py",
               "src/agents/training/diagnostics_cadence_test.py"),
        doc_note="K1 and K2 are BUILT (their paragraphs); K3–K10 are not — the learner benchmark is K's gate"),
)

#: Lane J is the harness itself — its components are the M5 gate's own rows, not a lane gate.
HARNESS_LANE = "J"


def by_lane() -> Dict[str, LaneGate]:
    return {r.lane: r for r in LANES}


def file_of(test: str) -> str:
    return test.split("::", 1)[0]


# ---------------------------------------------------------------------------- the program doc

_TABLE_ROW = re.compile(r"^\|\s*\**\s*([0-9A-Z][0-9A-Z]?)\s+—")
_PARAGRAPH = re.compile(r"^\*\*Lane ([0-9A-Z]+)\b[^*\n]*?\bBUILT\b")


def program_doc_text(root: Optional[Path] = None) -> str:
    from utils.paths import repo_path

    p = (root.joinpath(*PROGRAM_DOC) if root is not None else repo_path(*PROGRAM_DOC))
    return p.read_text()


def m5_section(text: str) -> str:
    """The M5 section of the program doc (from its heading to the next ``### M``)."""
    start = text.index("### M5 ")
    end = text.index("\n### M6", start)
    return text[start:end]


def doc_lanes(text: str) -> Dict[str, bool]:
    """``{lane id: BUILT per the doc}`` for every lane of the M5 lane table. A lane is BUILT when
    its table row's first cell says BUILT (and not NOT BUILT), or a ``**Lane X … BUILT`` paragraph
    exists; a sub-lane paragraph (``**Lane K1 BUILT``) marks its table lane (``K``)."""
    sec = m5_section(text)
    lanes: Dict[str, bool] = {}
    for line in sec.splitlines():
        m = _TABLE_ROW.match(line)
        if m and line.count("|") >= 5:
            first = line.split("|")[1]
            lanes[m.group(1)] = "BUILT" in first and "NOT BUILT" not in first
    for line in sec.splitlines():
        m = _PARAGRAPH.match(line)
        if not m:
            continue
        lane = m.group(1)
        if lane not in lanes:
            lane = next((t for t in lanes if lane.startswith(t) and lane[len(t):].isdigit()), lane)
        if lane in lanes:
            lanes[lane] = True
    return lanes


def registry_problems(text: str, root: Optional[Path] = None) -> List[str]:
    """Every way the registry disagrees with the program doc or the tree (empty = consistent)."""
    import ast

    from utils.paths import repo_root

    root = root or repo_root()
    doc = doc_lanes(text)
    reg = by_lane()
    out = []
    if len(reg) != len(LANES):
        out.append("a lane id is declared twice")
    for lane, built in sorted(doc.items()):
        if lane == HARNESS_LANE:
            continue
        row = reg.get(lane)
        if row is None:
            out.append(f"lane {lane} is in the program doc's lane table and has no row")
        elif built and not row.built:
            out.append(f"lane {lane} is BUILT per the program doc and its row says NOT BUILT")
        elif built and not row.tests:
            out.append(f"lane {lane} is BUILT per the program doc and its row declares no gate tests")
        elif row.built and not built and not row.doc_note:
            out.append(f"lane {lane}: the row says BUILT, the doc does not, and no doc_note says why")
    for row in LANES:
        if row.lane not in doc:
            out.append(f"row {row.lane} names no lane of the program doc's lane table")
        if not row.built and (row.tests or not row.pending):
            out.append(f"row {row.lane}: a NOT BUILT row runs nothing and states its pending gate")
        for t in (*row.tests, *row.gpu_tests):
            f = root / file_of(t)
            if not f.is_file():
                out.append(f"row {row.lane}: {file_of(t)} does not exist")
                continue
            if "::" in t:
                name = t.split("::", 1)[1].split("[")[0]
                defs = {n.name for n in ast.walk(ast.parse(f.read_text())) if isinstance(n, ast.FunctionDef)}
                if name not in defs:
                    out.append(f"row {row.lane}: {t} names no test function in its file")
        for g in row.gpu_tests:
            if file_of(g) not in {file_of(t) for t in row.tests}:
                out.append(f"row {row.lane}: GPU test {g} is not inside the row's tests")
    return out
