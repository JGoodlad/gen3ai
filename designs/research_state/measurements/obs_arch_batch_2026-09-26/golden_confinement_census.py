"""The golden-obs CONFINEMENT census for `gen3_event_record_v2` (the observation-architecture batch).

Capture the 991 golden decision vectors at the base commit (56837827, obs 2501) and at the batch
(obs 2761) with `golden_obs_capture.capture_vectors()` saved as .npy, then:

    python golden_confinement_census.py BASE.npy HEAD.npy BASE_FIXTURE.json

It checks that the base capture reproduces the committed fixture, maps HEAD back to the old layout
(drops the four field-sport columns and the event window), requires every non-event cell bit-identical,
and explains every differing event window by the batch's intended changes alone.
"""
import sys
import json, numpy as np
from agents.observation import constants as C
from agents.training.golden_obs_capture import vector_hashes
b = np.load(sys.argv[1]); h = np.load(sys.argv[2])
fix = json.load(open(sys.argv[3]))
print("base reproduces committed fixture:", vector_hashes(list(b)) == fix["hashes"] if "hashes" in fix else list(fix)[:3])
# map HEAD -> base layout: drop the 2 sport cols at the end of each active ctx, and cut the event window
old_ctx = 58; new_ctx = 60
ctx0 = C.OFFSET_CONTEXT
pre = h[:, :ctx0]
ours = h[:, ctx0:ctx0+new_ctx]; opp = h[:, ctx0+new_ctx:ctx0+2*new_ctx]
sport = np.concatenate([ours[:, old_ctx:], opp[:, old_ctx:]], axis=1)
mid = h[:, ctx0+2*new_ctx:C.OFFSET_EVENT_WINDOW]
mapped = np.concatenate([pre, ours[:, :old_ctx], opp[:, :old_ctx], mid], axis=1)
b_nonev = b[:, :mapped.shape[1]]
print("non-event cells bit-identical:", np.array_equal(mapped.view(np.uint32), b_nonev.view(np.uint32)), "sport cells nonzero:", int((sport != 0).sum()))
# event window: compare old 22 columns of the rows, row-aligned from the END is not valid when rows are inserted; count decisions whose window differs
ew_b = b[:, b_nonev.shape[1]:].reshape(len(b), 32, 22)
ew_h = h[:, C.OFFSET_EVENT_WINDOW:].reshape(len(h), 32, 30)
diff = [(i) for i in range(len(b)) if not np.array_equal(ew_h[i, :, :22], ew_b[i])]
print("decisions whose event window (old 22 cols) differs:", len(diff), "of", len(b))
# of those: are the only differences explained by new DENIED rows / changed ITEM_TR / FAINT_CAUSE?
T = C.EventCol
def rows(ew): return [tuple(r) for r in ew if r[T.VALID] > 0]
expl = 0; unexpl = []
for i in diff:
    rb = [r for r in ew_b[i] if r[T.VALID] > 0]
    rh = [r[:22] for r in ew_h[i] if r[T.VALID] > 0 and r[T.TYPE] != C.EVENT_T_DENIED]
    # drop recency/forced cols? compare all 22; allow ITEM_TR (21) 4->5 and FAINT_CAUSE 8->9/10 and magnitude on switch-in rows
    def norm(r, head):
        r = np.array(r, dtype=np.float64).copy()
        if r[T.TYPE] == C.EVENT_T_SWITCH_IN: r[T.MAGNITUDE] = 0
        if r[T.TYPE] == C.EVENT_T_ITEM_REVEAL and r[T.ITEM_TRANSITION] in (4, 5): r[T.ITEM_TRANSITION] = 4
        if r[T.TYPE] == C.EVENT_T_FAINT and r[T.FAINT_CAUSE] in (8, 9, 10): r[T.FAINT_CAUSE] = 8
        if r[T.TYPE] in (C.EVENT_T_SWITCH_IN, C.EVENT_T_DENIED): r[T.MOVE] = 0
        if r[T.TYPE] == C.EVENT_T_SWITCH_REJECTED: r[T.TARGET_SPECIES] = 0
        return tuple(r)
    nb = [norm(r, False) for r in rb]; nh = [norm(r, True) for r in rh]
    k = min(len(nb), len(nh))
    if nb[len(nb)-k:] == nh[len(nh)-k:]: expl += 1
    else: unexpl.append(i)
print("explained by the intended E12 changes (new DENIED rows, switch-in chip/move, item-transfer direction, faint causes, E4 target):", expl, "unexplained:", len(unexpl), unexpl[:5])
