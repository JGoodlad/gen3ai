"""Human-replay data as an OFFLINE measurement and training-row source (``designs/endstate/human_data_architecture.md``).

Nothing here is the switch ``--human-data`` (``agents.info_levers.human_data``, PROPOSED, not built): these modules
read a registered SNAPSHOT through its manifest (:mod:`.snapshots`), build a registered ROWSET from it
(:mod:`.rowset`), and train and export the X59-S matched students (:mod:`.student`). No run, eval or play path imports
them.
"""
