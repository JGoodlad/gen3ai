---
description: Teach a concept in the context of *our* Gen3 models, then capture it as a durable learning note under designs/learning/. Use whenever the user wants to LEARN or UNDERSTAND a concept (ML, RL, game-theory, Pokémon mechanics, or an architecture choice) rather than change code — "explain X", "help me understand X", "what is X / why do we X", "teach me about the damage op / belief heads / critic". The deliverable is a two-level explanation (intuitive then technical, no code) AND a maintained markdown file: one file per major concept, grounded in our actual architecture, updated in place rather than duplicated.
---

# /gen3ai-learning

Turn an explanation into a **durable, growing knowledge base**. Two jobs, always both:

1. **Explain well, right now** — intuitive level first, then technical, **no code** (unless the
   user explicitly asks). Ground every concept in *our* models (the `DamageOperator`, belief
   heads, `ValueDistHead`, obs blocks, reward registry, the version map) — not generic
   textbook framing. Cite the real flags / `ARCH_SIGNATURE` / file names so the note stays
   connected to the codebase.
2. **Capture it** — write or update a markdown file under `designs/learning/`, **one file per
   major concept**. A whole conversation usually maps to **one** file. Prefer **updating an
   existing file** over creating a near-duplicate.

## The explanation pattern (what "good" looks like)

Follow the shape the user asked for and that the seed note (`marginalization_and_uncertainty.md`)
demonstrates:

- **Intuitive level first.** A plain-language mental model + a concrete toy example that makes
  the idea *click* (e.g. the 50/50 Choice-Band example for marginalization). Show *why it
  matters for a decision*, not just what it is.
- **Then technical.** The formal statement (the inequality, the loss, the estimator), *why*
  it's true, and the precise mechanism — still no code.
- **Then "where this lives in our architecture."** Name the flags, versions, obs blocks, and
  files. This is what separates a learning note from a blog post.
- **A TL;DR at the top and a one-paragraph synthesis at the end.** The reader should get the
  gist from the first screen and the closure from the last.

Match the seed file's structure unless the user wants something different.

## File conventions

- Location: `designs/learning/<concept>.md` (kebab-case, concept-named, not date-named).
- **One file per *major concept*** — a cluster of tightly-related ideas the user explored in
  one sitting. Marginalization + Jensen + "how a model carries uncertainty" is **one** concept
  cluster → one file. A genuinely separate concept (e.g. PBRS reward shaping, PopArt, GAE) →
  its own file.
- **Update in place.** Before writing a new file, list `designs/learning/` and read any file
  whose topic overlaps. If the new material extends an existing concept, **edit that file**
  (add a section, refine, append to "See also") rather than spawning a duplicate. Keep the
  TL;DR and synthesis current when you do.
- **Always-current, like the CLAUDE.md docs.** If the architecture changes such that a
  learning note is now wrong (a flag renamed, a version bumped, a mechanism replaced), fix the
  affected note in the same pass — these are reference docs, not a dated transcript. Do NOT
  reproduce code; link to the file/flag instead so the note can't rot against an implementation
  detail.
- **Each file is self-contained but cross-linked.** End with a **See also** section pointing at
  the relevant root/leaf `CLAUDE.md` sections, `designs/ai_vN/design_*.md`, and memory files.

## Recipe

1. **Explain** the concept to the user in the two-level pattern above. This is the primary
   deliverable — do it in the chat response, in full.
2. **Find the home.** `ls designs/learning/` and skim any overlapping file. Decide: extend an
   existing file, or create a new concept file?
3. **Write / update** the markdown to match the seed file's shape (TL;DR → intuitive →
   technical → where-it-lives → synthesis → see-also). Capture what was actually discussed,
   distilled into a durable explainer — not a verbatim chat log.
4. **Register if new.** The `designs/learning/` folder is noted in `designs/CLAUDE.md`; no
   per-file index is required, but keep that pointer accurate if the folder's purpose grows.
5. **Ship only on request.** Do NOT commit unless the user's current message contains
   `/gen3ai-ship` (per the project git rule). Finishing the note is not permission to commit.

## Notes

- This skill is for *understanding*, not changing behaviour. If the user actually wants a code
  change, drop the skill and do the work (or use the relevant build/probe skill).
- If the concept is best understood by probing the live model, pair with `/gen3ai-probe` —
  capture the *insight* here, run the investigation there.
- Honesty: if an explanation rests on something unverified about our code, say so in the note
  (a short "open / unverified" caveat) rather than asserting it as fact.
