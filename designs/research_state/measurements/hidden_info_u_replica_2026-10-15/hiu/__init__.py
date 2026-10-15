"""The blind U-arm replication (red team E1): the replica's builder (``replica``) EXTENDED with the U arms.

Import order: ``replica.env.setup(cp)`` first (it puts the checkpoint's pin first on the path), then
``hiu.fix.install()`` (the learnset fix and the uniformly-placed never-seen draw), then anything else here.
"""
