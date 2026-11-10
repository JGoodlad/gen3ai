"""Human-replay data as an OFFLINE measurement and training-row source (``designs/endstate/human_data_architecture.md``).

Nothing here is the switch ``--human-data`` (``agents.info_levers.human_data``): these modules read a registered
SNAPSHOT through its manifest (:mod:`.snapshots`), build a registered ROWSET from it (:mod:`.rowset`), train and export
the X59-S matched students (:mod:`.student`), and train, export and register the human-replay CLONE the switch's
``init`` / ``opp`` heads name (:mod:`.clone`; the registered model is read through ``utils.human_models``). No run,
eval or play path imports them.
"""
