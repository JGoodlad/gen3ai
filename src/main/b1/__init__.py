"""BOUNDARY-1's kit: the fresh 2 x 2 (trunk width x team breadth), from its launch block to its verdict.

``main.b1.arms`` is the ONE source of the arms, seeds, run names, launch order and phase targets;
``main.b1.resume_check`` proves (or names what breaks) "stop at 50M, resume to 100M" against one launch to 100M;
``main.b1.cross`` is the registered head-to-head statistic and every class read from it; ``main.b1.look`` plans
one look of that cross from the runs' own records and reads it from the eval ledger;
``main.b1.team_bootstrap`` REPORTS, beside a look's verdict and never in it, how far each factor's statistic moves
when the panel's TEAMS are resampled. ``main.b1.kakuna_rule`` is the
confirming Kakuna meter, the ``main.reads`` rule ``b1_kakuna`` (registered in ``main.reads.rules``); it lives HERE
because ``main/reads/__init__.py`` is on the anchor lane's play path and its docstring must name every module of
that package: a rule module added there changes the play-path digest of every read in flight. The pin is an argument of
every command: the registration (the measurement folder's ``PREREG.md``) is the one place it is written.
"""
