"""The three custom argparse pieces every flag family shares.

`optional_float` / `str2bool` / `BoolFlag` are imported by name from `main.train.parser`
(the hub re-exports them) and by every family module that declares a boolean flag.
"""
import argparse

__all__ = ["optional_float", "str2bool", "BoolFlag", "retired_choice", "_BOOL_TRUE", "_BOOL_FALSE"]


def optional_float(s: str) -> float | None:
    """argparse `type=` converter for an optional float (`float | None`).

    Returns `None` for the sentinels `none`/`null`/`""` (case-insensitive),
    otherwise parses a float. A bad value raises `ValueError`, which argparse
    turns into a clean usage error. Used by `--clip-range-vf` so `none`
    disables value-function clipping (SB3 branches on `clip_range_vf is None`).
    """
    if s.strip().lower() in ("none", "null", ""):
        return None
    return float(s)


def retired_choice(flag: str, legal: tuple, why: str):
    """argparse `type=` converter for a flag whose OTHER values were deleted with the code that served them.

    ``choices=`` alone would answer a typed ``--env-core python`` with "invalid choice: 'python'
    (choose from 'rust')" and say nothing about WHY, which reads as a typo. This raises the reason
    (``why``) instead, so a command copied from a pre-deletion runbook fails at parse time with the
    explanation and the way out. A legal value passes through unchanged (pair it with ``choices=legal``
    so ``--help`` still lists it).
    """
    def convert(value: str) -> str:
        if value in legal:
            return value
        raise argparse.ArgumentTypeError(
            f"{flag} {value!r} was DELETED ({why}); the only legal value is "
            f"{' / '.join(repr(x) for x in legal)}")
    convert.__name__ = f"retired_choice({flag})"
    return convert


_BOOL_TRUE = ("true", "t", "yes", "y", "1", "on")
_BOOL_FALSE = ("false", "f", "no", "n", "0", "off")


def str2bool(s: str) -> bool:
    """Parse a human boolean: true/false, yes/no, 1/0, on/off (case-insensitive)."""
    v = s.strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean ({'/'.join(_BOOL_TRUE)} or {'/'.join(_BOOL_FALSE)}), got {s!r}")


class BoolFlag(argparse.Action):
    """Boolean flag accepting BOTH the bare/`--no-` form AND an explicit value.

    Registers a generated `--no-<flag>` for every `--<flag>` (like
    argparse.BooleanOptionalAction) but ALSO takes an optional value:
        --foo               -> True
        --no-foo            -> False
        --foo true | false  -> parsed (also yes/no, 1/0, on/off; --foo=false too)
    Passing a value to the negation (`--no-foo true`) is a usage error.
    """

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        opts, self._negatives = [], set()
        for opt in option_strings:
            opts.append(opt)
            if opt.startswith("--"):
                neg = "--no-" + opt[2:]
                opts.append(neg)
                self._negatives.add(neg)
        super().__init__(option_strings=opts, dest=dest, nargs="?", default=default,
                         required=required, help=help, metavar="{true,false}")

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self._negatives:
            if values is not None:
                raise argparse.ArgumentError(
                    self, f"{option_string} is a negation and does not take a value")
            setattr(namespace, self.dest, False)
        elif values is None:            # bare `--foo`
            setattr(namespace, self.dest, True)
        else:                           # `--foo <value>` / `--foo=<value>`
            setattr(namespace, self.dest, str2bool(values))
