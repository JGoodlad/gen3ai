"""WHERE a traced op, a graph break or an eager op belongs — the component map, and the static
scan of the one frame dynamo cannot enter (`train()`'s own fold loop).

A COMPONENT is decided from a Python STACK (outermost frame first), by the FIRST rule in `RULES`
that any frame of the stack satisfies. The order is the point: an op inside the extractor that a
loss term reached through `features_extractor(obs)` (TD-aux, the CF block) is an EXTRACTOR op
called from the loss, so the extractor rule is tested before the loss rule; the masking code sits
under the policy's `evaluate_actions`, so the distribution rule is tested before the heads rule.

`scan_fold_host_syncs` is STATIC (an AST walk), and it says so in every row it emits: dynamo skips
`train()` as a whole at its first in-loop break, so it never reports the second one. The scan lists
the constructs a DECLARED loss region would have to hoist (host reads, logging, numpy, per-row
Python branching) — the input to K8's functional-rewrite estimate, not a dynamo verdict.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Frame:
    file: str
    line: int
    fn: str

    def short(self) -> str:
        return f"{short_path(self.file)}:{self.line} {self.fn}"


_FILE_LINE_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_FRAME_SUMMARY_RE = re.compile(r"<FrameSummary file (.+?), line (\d+) in (\S+)>")
# `name.py(123): fn` — the python_function event names of a `torch.profiler` chrome trace.
_PROFILER_FRAME_RE = re.compile(r"^(.+?)\((\d+)\): (\S+)$")


def parse_stack(text: str) -> List[Frame]:
    """Every `File "x", line N, in fn` (a formatted traceback) or `<FrameSummary ...>` in ``text``,
    in order (outermost first, as Python prints them)."""
    out = [Frame(f, int(n), fn) for f, n, fn in _FILE_LINE_RE.findall(text or "")]
    if not out:
        out = [Frame(f, int(n), fn) for f, n, fn in _FRAME_SUMMARY_RE.findall(text or "")]
    return out


def parse_profiler_frame(name: str) -> Optional[Frame]:
    m = _PROFILER_FRAME_RE.match(name or "")
    if not m:
        return None
    return Frame(m.group(1), int(m.group(2)), m.group(3))


def short_path(path: str) -> str:
    """Repo- or site-packages-relative, so a readout is stable across worktrees and envs."""
    p = path.replace("\\", "/")
    for anchor in ("/site-packages/", "/src/"):
        i = p.rfind(anchor)
        if i >= 0:
            return p[i + len(anchor):]
    return p


def _is_torch_internal(f: Frame) -> bool:
    p = _p(f)
    return ("/site-packages/torch/" in p or "/torch/_" in p or p.startswith(("/<", "/torch/"))
            or p.endswith(("/main/compile_inventory/capture.py",
                           "/main/compile_inventory/worker.py")))


def user_frames(frames: Sequence[Frame]) -> List[Frame]:
    """The frames that are not torch's own machinery (nn.Module call wrappers included)."""
    return [f for f in frames if not _is_torch_internal(f)]


def innermost_user(frames: Sequence[Frame]) -> Optional[Frame]:
    u = user_frames(frames)
    return u[-1] if u else (frames[-1] if frames else None)


# ------------------------------------------------------------------------------------------------
# The component rules (first match wins; each is a predicate over ONE frame, tested on all frames)
# ------------------------------------------------------------------------------------------------

def _p(f: Frame) -> str:
    """Normalised path: forward slashes and a leading "/" — a profiler trace names frames relative
    to their `sys.path` entry (`agents/model/policy.py(303): evaluate_actions`)."""
    p = f.file.replace("\\", "/")
    return p if p.startswith("/") else "/" + p


def _extractor(f: Frame) -> bool:
    p = _p(f)
    if f.fn == "extract_features" and "stable_baselines3" in p:
        return True
    return (p.endswith(("/agents/model/features_extractor.py", "/agents/model/extractor_forward.py"))
            and f.fn in ("forward", "forward_internal", "_forward_impl", "__call__"))


def _distribution(f: Frame) -> bool:
    p = _p(f)
    return p.endswith("/distributions.py") and ("stable_baselines3" in p or "sb3_contrib" in p)


def _heads(f: Frame) -> bool:
    return _p(f).endswith(("/agents/model/policy.py", "/agents/model/pointer_head.py"))


def _buffer(f: Frame) -> bool:
    p = _p(f)
    return p.endswith("/buffers.py") or "/common/maskable/buffers.py" in p


def _optimizer(f: Frame) -> bool:
    p = _p(f)
    return "/torch/optim/" in p or p.endswith("/torch/nn/utils/clip_grad.py")


def _logging(f: Frame) -> bool:
    p = _p(f)
    return p.endswith("/stable_baselines3/common/logger.py") or "/utils/logging/" in p


def _loss(f: Frame) -> bool:
    return "/agents/training/" in _p(f)


def _model_other(f: Frame) -> bool:
    return "/agents/model/" in _p(f)


#: (component, predicate). The names are the readout's row labels.
RULES: Tuple[Tuple[str, Callable[[Frame], bool]], ...] = (
    ("extractor", _extractor),
    ("distribution+masking", _distribution),
    ("heads", _heads),
    ("rollout buffer", _buffer),
    ("optimizer", _optimizer),
    ("logging", _logging),
    ("loss+diagnostics", _loss),
    ("model (other)", _model_other),
)

#: The component of a stack no rule claims (torch-only frames, the tool's own harness).
OTHER = "other"


def classify(frames: Sequence[Frame], rules: Sequence[Tuple[str, Callable[[Frame], bool]]] = RULES
             ) -> str:
    for name, pred in rules:
        if any(pred(f) for f in frames):
            return name
    return OTHER


def loss_detail(frames: Sequence[Frame]) -> Optional[str]:
    """For a loss-side stack: the OUTERMOST frame under `train()` in agents/training — the term's
    own method (`_value_loss_from_se`, a belief-bank row, ...), which is what a region would call."""
    seen_train = False
    for f in frames:
        if not _loss(f):
            continue
        if f.fn == "train" and not seen_train:
            seen_train = True
            continue
        return f"{short_path(f.file)}:{f.fn}"
    return None


# ------------------------------------------------------------------------------------------------
# The static scan of train()'s fold loop
# ------------------------------------------------------------------------------------------------

@dataclass
class FoldSite:
    line: int
    kind: str
    text: str
    in_loop_depth: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: kind -> what K8 has to do about it (the fix-plan class, see the readout's legend).
SITE_KINDS = {
    "host read (.item)": "hoist: return the tensor, read after the region",
    "host read (float/int/bool of a value)": "hoist: return the tensor, read after the region",
    "host copy (.cpu/.numpy/.tolist)": "hoist: return the tensor, read after the region",
    "logging (logger.record)": "hoist: record outside the region",
    "numpy call": "hoist or rewrite in torch",
    "python branch on a value": "make static (flag resolved before the region) or torch.where",
    "inner python loop": "unrolled by dynamo if its trip count is static; else a region boundary",
}

_HOST_COPY = {"cpu", "numpy", "tolist"}


def _call_kind(node: ast.Call) -> Optional[str]:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        if fn.attr == "item" and not node.args:
            return "host read (.item)"
        if fn.attr in _HOST_COPY:
            return "host copy (.cpu/.numpy/.tolist)"
        if fn.attr == "record":
            base = fn.value
            if (isinstance(base, ast.Attribute) and base.attr == "logger") or (
                    isinstance(base, ast.Name) and base.id == "logger"):
                return "logging (logger.record)"
        root = fn.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in ("np", "numpy"):
            return "numpy call"
    if isinstance(fn, ast.Name) and fn.id in ("float", "int", "bool") and node.args:
        a0 = node.args[0]
        if not isinstance(a0, ast.Constant):
            return "host read (float/int/bool of a value)"
    return None


def _loop_target_names(loop: ast.For) -> List[str]:
    return [n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)]


def _find_fold_loop(tree: ast.AST, target: str) -> Optional[ast.For]:
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and target in _loop_target_names(node):
            return node
    return None


def scan_fold_host_syncs(fn: Callable[..., Any], loop_target: str = "rollout_data"
                         ) -> Dict[str, Any]:
    """AST-scan the body of the ``for <loop_target> in ...`` loop of ``fn`` (``train()``'s minibatch
    loop by default). Returns ``{"source": file, "loop_line": N, "sites": [...], "counts": {...}}``
    with ABSOLUTE line numbers. Branches (`if` on a non-constant test) are listed only when the
    test contains a host read or a call — a pure flag test (`if belief_aux_on:`) is static."""
    src_lines, start = inspect.getsourcelines(fn)
    src = textwrap.dedent("".join(src_lines))
    tree = ast.parse(src)
    loop = _find_fold_loop(tree, loop_target)
    file = inspect.getsourcefile(fn) or "?"
    if loop is None:
        return {"source": file, "loop_line": None, "sites": [], "counts": {},
                "error": f"no `for {loop_target} in ...` loop in {getattr(fn, '__qualname__', fn)}"}
    sites: List[FoldSite] = []
    lines = src.splitlines()

    def _text(n: ast.AST) -> str:
        ln = getattr(n, "lineno", 1)
        return lines[ln - 1].strip()[:140] if 0 < ln <= len(lines) else ""

    def _walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            d = depth
            if isinstance(child, (ast.For, ast.While)):
                sites.append(FoldSite(start + child.lineno - 1, "inner python loop", _text(child),
                                      depth))
                d = depth + 1
            elif isinstance(child, ast.If):
                test_calls = [c for c in ast.walk(child.test) if isinstance(c, ast.Call)]
                if any(_call_kind(c) for c in test_calls):
                    sites.append(FoldSite(start + child.lineno - 1, "python branch on a value",
                                          _text(child), depth))
            elif isinstance(child, ast.Call):
                k = _call_kind(child)
                if k is not None:
                    sites.append(FoldSite(start + child.lineno - 1, k, _text(child), depth))
            _walk(child, d)

    _walk(loop, 0)
    seen = set()
    uniq: List[FoldSite] = []
    for st in sites:                     # `.cpu().numpy()` is ONE site, not two
        if (st.line, st.kind) not in seen:
            seen.add((st.line, st.kind))
            uniq.append(st)
    sites = uniq
    counts: Dict[str, int] = {}
    for s in sites:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    return {"source": file, "loop_line": start + loop.lineno - 1,
            "sites": [s.as_dict() for s in sorted(sites, key=lambda s: (s.line, s.kind))],
            "counts": counts, "static": True}
