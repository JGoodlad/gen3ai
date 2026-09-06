"""Pure unit tests for `cf_audit_render` — the bias map's markdown presentation.

Moved here with `render_markdown` when it left `cf_audit.py` (2026-09-06). The fixtures are
imported from `cf_audit_test` rather than copied: a renderer test that built its own bias map
would be checking the formatting of a shape the audit no longer emits.

The one rule worth a test at all is the ABSENT-vs-ZERO one — a checkpoint with no evidential head
and a head that claims no uncertainty are opposite findings, and a row of zeros renders them
identically. The rendered markdown is additionally pinned byte-for-byte by `cf_audit_test`'s
extraction-parity golden.
"""

from __future__ import annotations

from agents.training.cf_audit_test import _dec, _evid_labels


def test_the_markdown_says_ABSENT_rather_than_rendering_zeros():
    from agents.training.cf_audit import bias_map, render_markdown
    labels = _evid_labels(lambda dec: 0.05 + 0.03 * dec)
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 18.0], "sampler_version": "test", "seed": 0}
    headless = [{k: v for k, v in r.items() if not k.startswith("evid_")} for r in labels]
    md = render_markdown(bias_map(headless, frame, n_rollouts=8, design=design, accounting={}),
                         run_dir="/t", step=1, ckpt=None)
    assert "carries no `cf_evid_head`" in md
    assert "0.000" not in md.split("## EVIDENTIAL")[1].split("## Caveats")[0]

    md2 = render_markdown(bias_map(labels, frame, n_rollouts=8, design=design, accounting={}),
                          run_dir="/t", step=1, ckpt=None)
    assert "width_vs_blur_spearman" in md2 and "Beta width" in md2
