---
description: Teach a concept in the context of *our* Gen3 models, then capture it as a durable learning note under designs/learning/. Use whenever the user wants to LEARN or UNDERSTAND a concept (ML, RL, game-theory, Pokémon mechanics, or an architecture choice) rather than change code — "explain X", "help me understand X", "what is X / why do we X", "teach me about the damage op / belief heads / critic". The deliverable is a two-level explanation (intuitive then technical, no code) AND a maintained markdown file: one file per major concept, grounded in our actual architecture, updated in place rather than duplicated.
---

# /gen3ai-learning

Two jobs, always both: explain the concept now, then capture it in `designs/learning/`. These notes teach (the owner's "explain to teach" preference), so unlike instruction files they may explain at length.

## 1. Explain, in the chat response

- **Intuitive first**: a plain-language model and a concrete toy example that makes it click, showing why it matters for a decision.
- **Then technical**: the formal statement (inequality, loss, estimator), why it holds, the mechanism. No code unless asked.
- **Then where it lives in our architecture**: ground it in the real models (damage operator, belief heads, value heads, obs blocks, rewards) and cite the actual flags, `ARCH_SIGNATURE` and file names; read live values from the code, never from memory.
- A TL;DR at the top and a one-paragraph synthesis at the end.

## 2. Capture

- File: `designs/learning/<concept>.md`, kebab-case and concept-named. One file per major concept cluster; a conversation usually maps to one.
- Run `ls designs/learning/` and read any overlapping note first; extend it in place (keep TL;DR and synthesis current) rather than spawning a near-duplicate.
- Shape: TL;DR, intuitive, technical, where-it-lives, synthesis, See also (the relevant `CLAUDE.md` sections and design docs). Distil what was discussed; no code, link the file or flag so the note cannot rot. State what is unverified about our code as a short caveat.
- Add a row for a new note to `designs/learning/README.md`, and keep an existing row accurate. When an architecture change makes a note wrong, fix it in the same pass.
- Do not commit unless the user's current message contains `/gen3ai-ship`.

This command is for understanding, not changing behaviour. To investigate the live model, use `/gen3ai-probe` and capture the insight here.
