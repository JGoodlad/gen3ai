"""BOUNDARY-1's kit: the fresh 2 x 2 (trunk width x team breadth), from its launch block to its verdict.

``main.b1.arms`` is the ONE source of the arms, seeds, run names, launch order and phase targets;
``main.b1.resume_check`` proves (or names what breaks) "stop at 50M, resume to 100M" against one launch to 100M;
``main.b1.cross`` is the registered head-to-head statistic and every class read from it; ``main.b1.look`` plans
one look of that cross from the runs' own records and reads it from the eval ledger;
``main.b1.team_bootstrap`` REPORTS, beside a look's verdict and never in it, how far each factor's statistic moves
when the panel's TEAMS are resampled; ``main.b1.lr_report`` REPORTS, the same way, the learning rate each run
trained at and how far its weights moved (TensorBoard rows and checkpoint zips only), and with ``--fresh-arm`` a
fresh cell's rungs, movement label and REPORTED training meters (clip fraction, entropy, value loss, the LR bounds,
dose, step time) beside its banked cell. ``main.b1.kakuna_rule`` is the
confirming Kakuna meter, the ``main.reads`` rule ``b1_kakuna`` (registered in ``main.reads.rules``); it lives HERE
because ``main/reads/__init__.py`` is on the anchor lane's play path and its docstring must name every module of
that package: a rule module added there changes the play-path digest of every read in flight. The pin is an argument of
every command: the registration (the measurement folder's ``PREREG.md``) is the one place it is written.

After BOUNDARY-1, ``main.b1.fresh`` plans, guards, plays and reads ONE fresh three-seed cell against ONE banked B1 cell
(``measurements/fresh_cell_frame_2026-11-17/FRAME.md``), and a card's REPORTED matched-GPU-time cross
(``plan-gpu-time``), reusing the modules above unchanged, and
``main.b1.fresh_rule`` is its statistic and classes, the ``main.reads`` rule ``b1_fresh_cell`` (it lives here for
the reason ``kakuna_rule`` does). Unlike B1's own, the fresh-cell kit names B1's launch pin as a constant (an older
commit) and takes the arm's pin from the card's arm file.
"""
