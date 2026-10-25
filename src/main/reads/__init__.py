"""REGISTERED READS: one module for every read of a trained model against a registered rule
(``designs/tools/reads.md``, the design of record). A read is a SPEC (``read_spec.json``, committed with its PREREG)
and a STATE directory; the verbs are ``python -m main.reads plan | run | resume | status | assemble | decide | record``.

* :mod:`main.reads.spec` -- the spec: models (pinned files or SETTLED finals), anchor groups, S groups, the rule;
* :mod:`main.reads.models` -- the settled-file rule, the sha-checked copy with its provenance sidecars;
* :mod:`main.reads.runner` -- plan / run / status over the short lane (Kakuna units) and the heavy slots (S cells);
* :mod:`main.reads.assemble` -- units and ledger rows -> cells, the regime id ALWAYS recorded;
* :mod:`main.reads.rules` -- the registered interval and decision rules, by name (reusing ``main.x35.readout`` /
  ``main.x35.d1`` / ``main.anchors.results``);
* :mod:`main.reads.record` -- the schema-checked record (X35-stage records load in ``main.x35.batch.read_record``);
* :mod:`main.reads.api` / :mod:`main.reads.cli` -- the verbs.

Nothing here plays on import.
"""
