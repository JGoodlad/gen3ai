"""LIVE websocket play on the Rust stack (poke-env retirement P4) and the T28 parse-panic HALT.

* :mod:`main.live.halt` — the durable HALT marker every live entry point refuses to start past
  (owner 2026-10-07, ``designs/endstate/design_ladder_campaign.md`` Decision record), and its
  ``python -m main.live.halt status|clear`` CLI;
* :mod:`main.live.reader` — the reader SESSION: protocol lines in, the training core's row / mask /
  choice tokens out (the ``live_reader`` Rust binary, the SAME parse chain ``sim_bridge``'s core
  observation mode ships to training);
* :mod:`main.live.client` — the thin websocket client (login, challenge / accept, the request /
  choice cycle, ``rqid``, the forfeit turn limit) that drives a policy over the session.
"""
