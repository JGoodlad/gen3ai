"""REGISTERED READS: one module for every read of a trained model against a registered rule
(``designs/tools/reads.md``, the design of record). A read is a SPEC (``read_spec.json``, committed with its PREREG)
and a STATE directory; the verbs are ``python -m main.reads check-spec | plan | run | resume | hold | pid | status |
repin | assemble | decide | release-guard-hold | record | check-record | search-cost`` (``main.reads.cli`` is the list;
``s-unit`` is the S lane's internal child, not an operator verb). Every module, by the job it does:

The spec and the models:

* :mod:`main.reads.spec` -- the spec: models (pinned files or SETTLED finals), anchor groups, S groups, the rule;
* :mod:`main.reads.speccheck` -- ``check-spec``: a spec validates, and its ``pin`` names a commit (the one-commit
  registration flow);
* :mod:`main.reads.models` -- the settled-file rule, the sha-checked copy with its provenance sidecars;
* :mod:`main.reads.layout` -- the state directory's files, and the refusal of a state inside the repo or ``models/``.

Playing (the lanes and their claims):

* :mod:`main.reads.runner` -- plan / run / status over the short lane (Kakuna units) and the heavy slots (S cells);
* :mod:`main.reads.anchor_pool` -- the anchor lane's bounded concurrency (``--concurrency K``; scheduling only);
* :mod:`main.reads.supervisors` -- ONE supervisor per lane per state PATH (a lock outside the state, a ``/proc`` scan, the
  dir-identity stop), and no second supervisor on the same kakuna state through a nested path;
* :mod:`main.reads.lanectl` -- a supervisor's own pid file, ``hold`` / ``resume``, the registered-stop evidence;
* :mod:`main.reads.attempts` -- a unit's failed attempts from its kakuna log: EXHAUSTED (retries spent) versus cut off
  by the stop;
* :mod:`main.reads.playpath` / :mod:`main.reads.repin` -- the tracked play path at a commit, and ``repin`` (another
  playing commit, only when that path is byte-identical);
* :mod:`main.reads.prefix` -- G-PREFIX, the paired-prefix tooth of a searched read; :mod:`main.reads.search_cost` -- its cost.

Reading (cells, rules, the record):

* :mod:`main.reads.assemble` -- units and ledger rows -> cells, the regime id ALWAYS recorded;
* :mod:`main.reads.engine` -- the Rust engine a unit played on (part of every regime); :mod:`main.reads.side_identity` --
  the Metamon peer's code and our policy's load mode it played with (likewise); :mod:`main.reads.imports` -- cells taken
  from an EARLIER read's state, sha-, regime-, engine- and side-checked;
* :mod:`main.reads.rederive` -- the forfeit-win guard and the H21 peer-fault check RE-DERIVED from each unit's captures
  and peer logs (units of pre-guard pins too): a hit voids the unit, or HOLDS ``decide`` for a read registered before;
* :mod:`main.reads.rules` (with :mod:`main.reads.rules_n4_screen`, :mod:`main.reads.rules_search_headroom`,
  :mod:`main.reads.rules_deviation_sign` and :mod:`main.reads.rules_x59s`) -- the registered interval and decision
  rules, by name (reusing ``main.x35.readout`` / ``main.x35.d1`` / ``main.anchors.results``), and the registered VOID
  gate;
* :mod:`main.reads.record` -- the schema-checked record (X35-stage records load in ``main.x35.batch.read_record``);
* :mod:`main.reads.api` / :mod:`main.reads.cli` -- the verbs as functions / as a command line.

On the PLAY PATH (an edit here blocks a ``repin``, ``designs/tools/reads.md`` §3): only this ``__init__`` and
``supervisors``; every other module is off it. Nothing here plays on import.
"""
