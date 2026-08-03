"""The bridge's seed vocabulary — ONE definition, shared by every Python producer.

A bridge ``START`` may carry a PRNG seed (``seed``) and the counterfactual re-roll may
carry a second one (``resumeReseed.seed``). Both are handed straight to the sim's
``new PRNG(seed)`` (node) / :func:`pokesim::prng::Prng::try_new` (rust), which accept
exactly four spellings:

======================  ==============================================================
``[m, n, o, p]``        a 4-int list — ``PRNG``'s constructor ``join(",")``s it
``"m,n,o,p"``           the same, already joined (the form ``prober.falsifier``
                        emits, and the ONLY form ``new PRNG()`` takes for a re-roll)
``"gen5,<16 hex>"``     the Gen-5 LCG backend, hex-packed
``"sodium,<hex>"``      the ChaCha20 backend (Showdown's DEFAULT, and what a
                        seedless ``START`` mints)
======================  ==============================================================

Anything else is a producer bug, and this module's job is to make it LOUD at the
producer instead of letting it reach the child. That matters because the failure mode is
silent: a seed the child cannot parse used to be dropped, and the battle then ran on some
*other* dice stream while every log said it was seeded (``gen3_bridge_seed_forms_v1``).
The rust child now rejects the same set with an ``__ERR__``; this is the mirrored
producer-side half, so the error names the CALLER rather than surfacing as a dead child.

**No seed is a legitimate, and the DEFAULT, choice.** Training and eval deliberately pass
none: a run's reproducibility does not come from the sim seed (policy sampling, the
teambuilder draw, PFSP opponent choice and the async-rollout interleave all sit outside
it), while one shared seed across N env workers would correlate their dice — strictly
worse than independent streams. The reproducibility that IS wanted — replaying one
recorded battle — comes from the child RESOLVING a seed and reporting it in ``__RECON__``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

SeedSpec = Union[str, Sequence[int], None]


def _valid_seed_str(s: str) -> bool:
    """Mirror ``PRNG.setSeed`` / ``Prng::try_new``'s three string cases."""
    if s.startswith("sodium,"):
        hex_part = s[len("sodium,"):]
        return bool(hex_part) and all(c in "0123456789abcdefABCDEF" for c in hex_part)
    if s.startswith("gen5,"):
        rest = s[len("gen5,"):]
        return len(rest) >= 16 and all(c in "0123456789abcdefABCDEF" for c in rest[:16])
    if s[:1].isdigit():
        parts = s.split(",")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p.strip()) <= 0xFFFF for p in parts)
        except ValueError:
            return False
    return False


def validate_seed_spec(seed: SeedSpec, *, what: str = "seed") -> Optional[str]:
    """Raise ``ValueError`` unless ``seed`` is a form both bridge impls accept.

    Returns the canonical comma-string (useful for logging / the record), or ``None``
    for a genuinely absent seed — which is allowed, and means "the child mints one".
    """
    if seed is None:
        return None
    if isinstance(seed, str):
        s = seed.strip()
        if not _valid_seed_str(s):
            raise ValueError(
                f"bridge {what}={seed!r} is not a PRNG seed. Accepted: [m,n,o,p], "
                f'"m,n,o,p", "gen5,<16 hex>", "sodium,<hex>", or None (the child mints one).'
            )
        return s
    if isinstance(seed, (list, tuple)):
        if len(seed) != 4 or not all(isinstance(x, int) and not isinstance(x, bool)
                                     and 0 <= x <= 0xFFFF for x in seed):
            raise ValueError(
                f"bridge {what}={seed!r} must be exactly 4 ints in [0, 65535] "
                f"(the gen-5 seed quadruple)."
            )
        return ",".join(str(int(x)) for x in seed)
    raise ValueError(
        f"bridge {what}={seed!r} has type {type(seed).__name__}; expected a seed string, "
        f"a 4-int list, or None."
    )
