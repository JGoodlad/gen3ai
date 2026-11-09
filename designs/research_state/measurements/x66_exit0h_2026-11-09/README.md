# X66 ExIt-0H (2026-11-09): can P\* learn the honest search's choices? — REGISTERED

**Status: REGISTERED 2026-11-09** ([`PREREG.md`](PREREG.md)). Nothing trained, scored or played. The roadmap's MF-1
(`../../EXPERIMENT_BACKLOG.md` §2.0): three P\* students, each fine-tuned toward S′-HONEST's honest piKL targets on two
panels' banked decisions, scored on the third panel by the delivery gate (Δagree, game-clustered LB > 0) before any
Kakuna game; only on DELIVERED, a two-look read against Kakuna on the held-out panel's S′-HONEST deals with S′-HONEST's
banked P\* raw arm as the control.

| file | what |
|---|---|
| [`PREREG.md`](PREREG.md) | the registration: question, classes, bank, recipe, gate, deals, statistic, power, cost, launch, VOID |
| [`bank_build.sh`](bank_build.sh) | builds the honest bank at the side pin, outcome-blind, in ≤ 44-min resumable chunks (CPU) |
| [`gpu_students.sh`](gpu_students.sh) | the GPU kit for the lease holder: preconditions, then S_H, S_F, S_P (resumable) |
| [`gate.sh`](gate.sh) | the read operator's CPU steps: pick, export at the play pin, delivery score and gate, by hand, the report |
| [`delivery_by_hand.py`](delivery_by_hand.py) | the gate recomputed with loops of its own (X65's) |
| [`read_spec_look1.json`](read_spec_look1.json), [`read_spec_look1_fresh.json`](read_spec_look1_fresh.json), [`read_spec_look2.json`](read_spec_look2.json) | the Kakuna read's specs (PENDING: the students, the pin, the bank sha256), written by [`make_specs.py`](make_specs.py) |
| [`power.py`](power.py) → [`power.txt`](power.txt) | the two-look power of the Kakuna read |
| [`noise_ceiling.py`](noise_ceiling.py) → [`noise_ceiling.json`](noise_ceiling.json) | S′h's self-agreement under another key (stage 1b's 157 pairs): 61.8 %; a third of its departures reproduce |

State: `~/gen3ai_archive/x66_exit0h_2026-11-09/` (`bank/`, `students/`, `delivery/`, `delivery.json`, `look1/`, …).
