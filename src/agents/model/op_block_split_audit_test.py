"""`--site assembler` must REFUSE, not measure nothing (the positional-binding sweep).

The arm bound the assembler's LAST POSITIONAL argument and compared it by identity against the
op's `incoming_rows` view. `ProjectionAssembler.forward` has taken no op tensor since v61
(gen3_no_concat_v1 deleted the head concat) / v96 (the critic-route wave deleted the seed
readout that replaced it), so that comparison was unconditionally False: the pre-hook returned
its arguments untouched, every arm's forward equalled the baseline, and the report printed
0.0000 KL / flips / |dV| on every row — which reads as "the block was worthless", not as "the
block is gone". Same failure as `edge_ablation_audit`'s deleted `concat` arm, one file over.

These tests pin the two halves of the cure: the subject is resolved BY NAME against the live
signature, and its absence RAISES.
"""
import pytest

from agents.model.op_block_split_audit import (assembler_site_subject, check_assembler_site,
                                               _ASSEMBLER_BLOCK_ARG_NAMES)


class _Assembler:
    """Stands in for `ProjectionAssembler` — only `forward`'s SIGNATURE is under test."""

    def __init__(self, fn):
        self.forward = fn


def _live_assembler():
    from agents.model.features_extractor import ProjectionAssembler
    return ProjectionAssembler({})


def test_the_live_assembler_has_no_op_block_argument():
    """The fact the arm's silence depended on. If this ever fails, an op tensor came BACK to the
    assembler and the legacy site becomes measurable again — re-read the arm before trusting it."""
    assert assembler_site_subject(_live_assembler()) is None


def test_the_assembler_site_refuses_instead_of_reporting_zeros():
    with pytest.raises(RuntimeError, match="no subject on this architecture"):
        check_assembler_site(_live_assembler())
    # and it names the replacement, so the refusal ends the investigation rather than starting one
    with pytest.raises(RuntimeError, match=r"--site op"):
        check_assembler_site(_live_assembler())


@pytest.mark.parametrize("name", _ASSEMBLER_BLOCK_ARG_NAMES)
def test_a_block_argument_is_found_by_NAME_wherever_it_sits(name):
    """The other half: resolution must not depend on POSITION. The same argument is found first,
    last and in the middle — which is exactly what a positional bind cannot promise."""
    for build in (
        lambda: eval(f"lambda a, b, {name}=None: None"),          # noqa: S307 — local literal
        lambda: eval(f"lambda {name}=None, a=None, b=None: None"),  # noqa: S307
        lambda: eval(f"lambda a, {name}=None, b=None: None"),       # noqa: S307
    ):
        asm = _Assembler(build())
        assert assembler_site_subject(asm) == name
        assert check_assembler_site(asm) == name


def test_an_unrelated_trailing_argument_is_NOT_taken_for_the_block():
    """The literal regression: `hidden_opp_belief` is the assembler's trailing argument today.
    A positional bind would patch it and report a hidden-opp number under the block's name."""
    asm = _Assembler(lambda a, b, ctx=None, hidden_opp_belief=None: None)
    assert assembler_site_subject(asm) is None
