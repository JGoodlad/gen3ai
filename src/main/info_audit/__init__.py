"""THE INFORMATION-LEVER AUDIT SUITE: one set of checks run against all three consumers of the information-lever
spec (``agents.info_levers``; ``designs/endstate/design_information_levers.md`` §10).

``checks`` holds the checks (each returns a verdict dict: ``check``, ``pass``, counts, up to 10 examples); ``python -m
main.info_audit`` runs them over decisions of the committed Lane S bank and writes one JSON report (the per-class
leakage of the honest world builder against the pilot's, the log-consistency counts, every check's verdict). The
routine gate runs a small fixed corpus (``info_audit_integration_test.py``); the slow tier a larger one
(``info_audit_slow_test.py``). Every check passes or fails DETERMINISTICALLY: fixed battles, fixed seeds, and a
statistical bound only where the planted defect it must catch sits far beyond it (rule 8).
"""
