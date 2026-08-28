# Learning note — "geometry", unpacked (2026-08-28)

The word bundles THREE distinct technical objects living in different spaces.

## 1. Loss-landscape geometry (weight space)

Every setting of the network's ~millions of weights is a point; training loss is an altitude over
that space; training walks downhill; a checkpoint is a position. What matters at a position is
**curvature**: steep directions (moving the weights changes outputs the objective punishes) vs
flat directions (nothing the objective cares about changes). "The loss is starting to plateau" is
genuinely geometric: plateau = gradient near zero = bottom of a valley — and the valley's *shape*
is what then governs fork behavior. Steep walls around existing competence channel new learning
into flat directions (the annex); a coefficient scales step size, the walls control direction.
Practical proxies: the Fisher information matrix (the basis of EWC), perturbation response —
the true Hessian (~10⁸ × 10⁸) is never computed.

## 2. Representation geometry (activation space)

Not weights — features. Feed ten thousand board states through the net and collect the feature
vectors; those points form a shape: which game distinctions get their own directions, how many
directions are in real use (effective rank — the "pi_features rank 12.4" numbers), what clusters
near what. This is the model's *coordinate system for the game*. It matters for distillation
because teaching-by-examples writes content into the student's existing coordinates: if the
coordinates keep moving under the student's own training, the writing smears — labeling a map of
a city that is still moving its streets. CKA (Centered Kernel Alignment) measures whether two
networks' (or two checkpoints') coordinate systems match.

## 3. Gradient geometry (directions of learning pressure)

Each objective pushes the weights in some direction; the angles between those directions (cosine
similarity) decide interference: orthogonal ≈ non-interfering, opposed = conflict. This is the
space PCGrad operates in — shelved here because we *measured* the angle (Δcos −0.030, essentially
orthogonal): no conflict to remove.

## How they interlock

Curvature (1) decides WHERE new learning is allowed to land; representation stability (2) decides
whether what lands stays ADDRESSABLE; gradient angles (3) decide whether ongoing training FIGHTS
the transplant. "Consolidation" = (1) sharpening walls around acquired competence while (2) stops
moving.

## The honesty paragraph

Everything we say about these geometries is measured through proxies — rank, CKA, cosines,
Fisher diagonals — shadows of shapes too large to see directly. That is the normal condition of
the field. The epistemically precise version of "the geometry consolidated" is "several cheap
shadows all moved the way consolidation predicts" — which is why the plasticity forensics scored
four INDEPENDENT shadows (P1–P4) rather than trusting one. (Postscript, same day: two of the four
came back opposite for our v8-vs-gen comparison — the discipline of using several shadows is
exactly what caught it.)
