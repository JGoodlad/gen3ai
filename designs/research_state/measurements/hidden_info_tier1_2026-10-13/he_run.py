"""Run the hidden-info study's hand-evaluator driver UNCHANGED (`../hidden_info_binding_2026-10-11/hand_eval_playouts.py`,
§6.5) through this folder's `run_pin.sh`, for Tier 1's R-EVAL:

    run_pin.sh <cp>     2 8 24h he_run.py --world O --out-root <archive>/<cp>/hand_eval --identity
    run_pin.sh <cp>     2 8 24h he_run.py --world P --worlds-dir ~/gen3ai_archive/value_disc_pilot/<cp>/det2 \
                                          --out-root <archive>/<cp>/hand_eval --identity
    run_pin.sh <cp>_pad 2 8 24h he_run.py --world F --worlds-dir <archive>/<cp>/worlds/<F's code> \
                                          --out-root <archive>/<cp>/hand_eval

P is the pilot's Phase 2 base RE-BUILT on the fixed builder (`det2`, Amendment 1 (b)), not the driver's default `det`
(v1). F's worlds carry the gender pad, so HE@F runs at the PAD pin.
"""
import sys
from pathlib import Path

HIB = Path(__file__).resolve().parent.parent / "hidden_info_binding_2026-10-11"
sys.path.insert(0, str(HIB))
import hand_eval_playouts as H  # noqa: E402

if __name__ == "__main__":
    H.main()
