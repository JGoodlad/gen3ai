"""Gate (c)'s peer plays the RUST port of a roster bot (P6): every bot it offers is one the env core plays, the
seven P4 bots are all there, and the parser carries no poke-env role (the shadow role was deleted with the legacy
client)."""
from main.live.gate_peer import ALL_BOTS, BOTS, build_parser
from utils.rust_env.bot_inventory import ported


def test_every_peer_bot_is_a_ported_rust_bot():
    rust = {r.name for r in ported()}
    assert set(ALL_BOTS) == rust, (sorted(set(ALL_BOTS) - rust), sorted(rust - set(ALL_BOTS)))
    assert set(BOTS) <= set(ALL_BOTS) and len(BOTS) == 7


def test_the_parser_has_no_poke_env_role():
    opts = {o for a in build_parser()._actions for o in a.option_strings}
    assert "--role" not in opts and "--model" not in opts
    a = build_parser().parse_args(["--port", "9551", "--username", "p", "--opponent", "o", "--n", "2"])
    assert a.bot == "heuristic2" and a.seed == 0
