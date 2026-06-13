"""Behavioural-cloning / human-replay tooling (designs/ai_v6 Step 2).

The first concrete slice is the offline ``log_reader`` — it replays a saved
spectator Showdown ``.log`` into the same per-decision ``(obs, action, mask)``
triples training produces. The same pipeline backs the *human-agreement probe*
(a measure-before-you-build diagnostic) and, later, BC dataset extraction.
"""
