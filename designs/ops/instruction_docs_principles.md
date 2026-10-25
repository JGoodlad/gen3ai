# Instruction docs — principles

**Always-current.** How we write the files an agent reads because of *where it is working*, not
because it chose to open them: the root and leaf `CLAUDE.md` files, the user-memory index
`MEMORY.md`, the subagent definitions in `~/.claude/agents/`, the slash commands in
`.claude/commands/`, and agent briefs. The audit that applied these principles first, with the
sweep plan, is [`instruction_docs_audit_2026-10-19.md`](instruction_docs_audit_2026-10-19.md).

**Why this exists (owner, 2026-10-19):** the files grow append-only. Two big cuts have not held.
The root went from 37 KB (2026-09-08, after the first cut) to 71 KB (2026-10-09), was cut to 33 KB
(2026-10-10), and is 38.6 KB nine days later. The training leaf did the same: 59 KB → 133 KB →
26 KB → 35.5 KB. A one-off sweep does not fix this; a budget the gates enforce, and a habit of
replacing rather than appending, do.

---

## 0. The ten principles

1. **The deletion test.** For each line, ask: *would a capable agent that has not read it do the
   work wrong, and fail to find the right way within a minute from the code, a `--help`, or the
   tool's own refusal?* If not, cut the line. (This is Anthropic's test, and the one our 2026-09-06
   census used.)
2. **Rules, not stories.** State the current rule in the imperative. No incident narratives, no
   "previously …", no measurement tables. The history goes to the ledger, `CHANGELOG.md` or
   `designs/research_state/claude_md_archive/`, and the rule may point there in a few words.
3. **What, then a one-clause why.** Every rule says what to do and, briefly, why. The why lets the
   model generalise and lets a later editor judge whether the rule is obsolete. A why longer than
   the rule is a story (principle 2).
4. **One home per fact.** A rule, command or number lives in exactly one file, and everyone else
   points at it. A leaf never restates a root rule. A brief never restates a standing rule. Never
   copy a figure that a generator prints (a test count, a gate count, a version): point at the
   generator. The one exception is a copy that a test pins (§4).
5. **The right altitude.** Root = rules binding every session, plus the map. Leaf = the
   non-obvious hazards and conventions for that directory's code, plus pointers. Procedures,
   rationale and reference detail go in `designs/<tree>/` and the SOPs. Volatile project state
   (what is live, what is frozen) goes in `MEMORY.md`. See §2.
6. **Enforce in code, point in prose.** Anything that must hold with zero exceptions belongs in a
   gate, a guard that refuses with its reason, or a hook. Once it is enforced, the doc keeps one
   line, and keeps even that only if hitting the refusal would be expensive (a lost run, an
   hours-long gate).
7. **Budgets.** Every file type has a byte budget (§3), and a static gate enforces its cap
   (`src/claude_md_freshness_gate_test.py`). A file over budget may only shrink: the next edit
   removes more than it adds.
8. **Replace, don't append.** When a fact changes, rewrite the sentence and delete the old one.
   A new rule names what it replaces or retires. The past lives in git and the history files.
9. **Write for current models.** Plain, specific, checkable sentences ("run X before Y"). State the
   general principle once instead of listing cases. Say what to do, not only what to avoid.
   Emphasis (🚨, bold, CAPS) on at most five lines per file: if everything is emphasised, nothing
   is. Do not explain standard tools (git, pytest, argparse) that the model already knows.
10. **Mechanically fresh.** Every path, flag, module and command a doc names must exist. The
    freshness gate checks paths, flags, `python -m` modules and `file § section` pointers
    (`utils.section_refs`). Words like "now", "currently" or "live" about volatile state are a smell:
    that state belongs in `MEMORY.md` or an SOP, not in a `CLAUDE.md`.

---

## 1. Who loads what, and what it costs

| file | loaded | cost today (2026-10-25, ≈ bytes / 4) |
|---|---|---|
| root `CLAUDE.md` | every session and every subagent (Explore / Plan excepted) | 15.0 KB ≈ 3.8k tokens |
| `MEMORY.md` | every session; **observed** in our `opus-*` subagents' context too (see the FINDING below) | 7.8 KB ≈ 1.9k tokens |
| `~/.claude/agents/<type>.md` body | the system prompt of every subagent of that type | 1.8-2.0 KB ≈ 0.5k tokens |
| a leaf `CLAUDE.md` | on demand, the first time the agent reads or edits a file in that directory | 0.8–10.1 KB |
| `.claude/commands/*.md` | when invoked (the description line always) | 2.2–5.7 KB |
| `@path` imports | at launch, with the file that imports them; **imports do not reduce cost** | — |

Sources (fetched 2026-10-19; §6): Claude Code loads `CLAUDE.md` from the working directory and its
ancestors at launch, and a subdirectory's file "once Claude reads, writes, or edits another file in
that subdirectory" [CC-MEM]. A non-fork subagent loads "every level of the CLAUDE.md hierarchy the
main conversation loads" [CC-SUB]. `MEMORY.md` loads up to "the first 200 lines … or the first
25KB" [CC-MEM]. The recommended size is "under 200 lines per CLAUDE.md file. Longer files consume
more context and reduce adherence" [CC-MEM]. "Bloated CLAUDE.md files cause Claude to ignore your
actual instructions" [CC-BP].

So the real cost of a line in the root is not its tokens (cache reads are cheap); it is that every
line competes for the model's attention with the rules that matter, in every session.

> **FINDING (unverified mechanism):** [CC-MEM] says "the main conversation's auto memory isn't
> loaded into subagents", yet the full `MEMORY.md` arrives in our `opus-high` subagents' context
> (observed 2026-10-19 in the audit agent itself). Treat `MEMORY.md` as loaded everywhere until
> the mechanism is pinned down (`/context` in a subagent, or an `InstructionsLoaded` hook).

---

## 2. Where each kind of content lives

| content | home | not here |
|---|---|---|
| a rule that binds every session (git, GPU lease, kill-by-PID, CPU confinement, the env) | root `CLAUDE.md` | any leaf, any brief |
| the map: "about to do X → read Y" | root `CLAUDE.md` (one table) | — |
| a hazard or convention specific to one directory's code | that leaf `CLAUDE.md` | root |
| a multi-step procedure (launch a run, ship, resume) | an SOP / runbook in `designs/ops/`, or a slash command | any `CLAUDE.md` beyond a pointer |
| reference detail (schemas, op contracts, flag notes, panel lists) | `designs/<tree>/<topic>.md` (the eight leaf-detail trees) | the leaf, beyond a one-line pointer |
| what the model *is* | `designs/ARCHITECTURE.md` | everywhere else (point) |
| what we believe about the research | `designs/research_state/UNDERSTANDING.md` + the ledger | any `CLAUDE.md` |
| how something came to be; incidents; measurements | ledger, `CHANGELOG.md`, `claude_md_archive/`, the measurement file | any instruction file |
| owner preferences; volatile project state (what is live, frozen, the quota) | `MEMORY.md` (one line each) + its topic files | any `CLAUDE.md` |
| an owner ruling on how sessions operate | the owning section of `ORCHESTRATOR_SOP.md` / `TRAINING_RUN_SOP.md` | `MEMORY.md` beyond a one-line pointer |
| zero-exception enforcement | a gate, a guard, a hook | prose alone |
| the task, its scope, its lease and CPU allowance, its ship authority | the brief | `CLAUDE.md` |

**Human-facing docs are different.** `docs/RUNNING.md`, `docs/DEVELOPING.md`, `designs/learning/`
and reports to the owner are written to *teach* (the owner's "explain to teach" preference), so
they may explain at length. These principles govern what agents load automatically.

---

## 3. Budgets

Bytes, because our lines are long (the root averages 128 bytes a line, so "200 lines" would allow
26 KB). 1 KB = 1,000 bytes; one token ≈ 4 bytes for this prose. The **cap** is what the budget
gate enforces; the **target** is what a sweep aims for.

**The ratchet** (the freshness gate; numbers in
[`instruction_docs_budget.json`](instruction_docs_budget.json), rules in `utils.instruction_budget`,
census `python -m utils.instruction_budget`). A file over its type cap when the ratchet landed
(2026-10-19) is grandfathered at its size then. A file may not grow past its cap. A sweep that
shrinks a grandfathered file lowers its entry to the new size in the same commit (the gate fails at
more than 1,000 bytes of headroom), and removes the entry once the file is within its type cap.
A cap is never raised. Audit U11 ends with the grandfathered list empty. `MEMORY.md` and the
subagent definitions are machine-local: checked only where they exist, and only against their cap
(their owner lowers their entries; U9). The gate checks this table against the budget file.

| file | target | cap |
|---|---:|---:|
| root `CLAUDE.md` | 12 KB (≈ 3k tokens) | 16 KB |
| `MEMORY.md` | 8 KB | 10 KB (the owner's own ceiling; the platform drops past 25 KB) |
| a code leaf `CLAUDE.md` | 8 KB (≈ 2k tokens) | 12 KB |
| `designs/CLAUDE.md` (a map) | 8 KB | 12 KB |
| a `designs/<topic>/` tree doc | 40 KB | 120 KB |
| a subagent definition file | 1.5 KB | 2.5 KB |
| a slash command | 8 KB | 12 KB |

A `designs/<topic>/` tree doc is read on demand, never loaded, so its cap (120 KB) only stops a dump; a tree doc
over the 40 KB target is a sweep candidate (a TECH_DEBT row), and a story in one moves to the archive.

**Always-loaded total** (root + `MEMORY.md` + one agent definition): today ≈ 25 KB ≈ 6.2k tokens;
target ≤ 21.5 KB ≈ 5.4k tokens.

---

## 4. How these principles meet the existing rules and gates

- **The ALWAYS-CURRENT rule** (root § Documentation Maintenance) stands unchanged. These principles
  make it cheaper to honour: fewer facts per file means fewer facts to keep current.
- **`src/claude_md_freshness_gate_test.py`** checks that every path under the five owned prefixes,
  every `--flag` and every `python -m <module>` that a `CLAUDE.md`, a `README.md`, an always-current doc or a
  `designs/<topic>/` tree doc names resolves, and enforces the byte-budget ratchet (§3). It stays green through
  every sweep. A closed round's log is history: it goes to `designs/research_state/claude_md_archive/`, out of the
  always-current set, not into a tree.
- **Tests that pin doc text.** A few tests assert that a doc still contains a sentence:
  `src/utils/static_gates_test.py` (the root's "**Eighteen static gates**" count),
  `src/agents/training/tb_relevance_test.py` (`designs/training/telemetry_scalars.md`),
  `src/main/train/disk_guard_wiring_test.py` (`designs/launcher/crashes_and_exit_codes.md`) and
  `src/main/launcher/restart_interval_default_test.py` (`designs/launcher/flags.md`, `designs/ops/flag_census.md`).
  A pinned copy cannot go stale, but it does make the doc keep the text.
  When a sweep moves pinned text, the test's target moves in the same unit. A new pin goes on the
  text's one home, never on a summary of it.
- **`designs/deleted_flags.md`** is a history file. When a sweep deletes a doc's mention of a
  deleted flag, the entry stays. Never add an entry so a doc can keep naming a flag it no longer
  needs to name: delete the mention instead.
- **`designs/research_state/claude_md_archive/`** is append-only history. A wholesale rewrite of a
  `CLAUDE.md` may freeze the old text there (the existing convention); a routine edit relies on git.
- **The leaf-detail trees** (`designs/model/`, `designs/training/`, …) are always-current and own
  the detail a leaf points at. Moving text from a leaf to its tree is a move, not a deletion: the
  tree doc must say it.
- **`MEMORY.md`** is the user's auto-memory index, outside the repo and machine-local. It follows
  the same discipline: one line per memory (title + hook, ≤ ~150 characters), no inline incident
  retellings, no repo facts (those belong in the docs), no owner ruling that an SOP already
  states (point instead). Anthropic's memory guidance for Fable says the same: "Don't save what
  the repo or chat history already records; update an existing note rather than creating a
  duplicate; delete notes that turn out to be wrong" [P-FABLE5].
- **Agent briefs** carry the task, its scope and the grants that change per dispatch (GPU lease, CPU
  allowance, ship delegation). They never restate a standing rule; they may name one when the
  brief changes it.
- **Subagent definitions** (`~/.claude/agents/`) hold only what differs by tier (model, effort,
  which tiers it may spawn). A rule every agent needs lives in the root, which every subagent loads
  anyway.

---

## 5. Reviewing a change to an instruction file

Every sweep unit, and every later edit that adds more than a line, shows:

1. **Before / after bytes** for each file touched.
2. **A disposition for every deleted rule:** `OBSOLETE` (cite why: the code is gone, a guard now
   refuses it, a newer rule supersedes it), `MOVED → <path § section>`, or `ENFORCED BY <gate /
   guard / hook>`. Nothing load-bearing disappears without one of the three.
3. **The diff read by a fresh-context reviewer** against that disposition list.
4. **The static tier green**, the freshness gate included.

---

## 6. Sources

Fetched 2026-10-19. Quotations are from these pages; anything not quoted is our own reasoning.

- **[CC-MEM]** Claude Code docs, *How Claude remembers your project* —
  <https://code.claude.com/docs/en/memory>. The 200-line target; on-demand subdirectory loading;
  `MEMORY.md`'s 200-line / 25 KB window; "Imports … don't reduce its context cost"; "if two
  instructions contradict each other, Claude may pick one arbitrarily"; "Write instructions that
  are concrete enough to verify"; Claude treats `CLAUDE.md` as "context, not enforced
  configuration" (use a hook to block an action); `/doctor prompt-audit` finds "instructions
  written for older models, references to files or commands that don't exist, and files that
  contradict each other" (Claude Code ≥ 2.1.283; this box runs 2.1.292).
- **[CC-BP]** Claude Code docs, *Best practices* — <https://code.claude.com/docs/en/best-practices>.
  "For each line, ask: *Would removing this cause Claude to make mistakes?* If not, cut it."; the
  include / exclude table (exclude "Anything Claude can figure out by reading code", "Information
  that changes frequently", "Long explanations or tutorials", "File-by-file descriptions of the
  codebase"); "If you emphasize many lines, none of them stands out"; "Ruthlessly prune. If Claude
  already does something correctly without the instruction, delete it or convert it to a hook";
  hooks "are deterministic", `CLAUDE.md` instructions "are advisory".
- **[CC-LARGE]** Claude Code docs, *Monorepos and large codebases* —
  <https://code.claude.com/docs/en/large-codebases>. Root vs per-directory split; "Revisit after
  major model releases: instructions that worked around an older model's limitation may become
  overhead once a newer model handles the case on its own".
- **[CC-SUB]** Claude Code docs, *Subagents* — <https://code.claude.com/docs/en/sub-agents>. A
  subagent loads the whole `CLAUDE.md` hierarchy; `omitClaudeMd: true` opts one out.
- **[P-BP]** Claude platform docs, *Prompting best practices* —
  <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices>.
  "Providing context or motivation behind your instructions … Claude is smart enough to generalize
  from the explanation"; "dial back any aggressive language. Where you might have said 'CRITICAL:
  You MUST use this tool when...', you can use more normal prompting"; "Tell Claude what to do
  instead of what not to do".
- **[P-FABLE5]** Claude platform docs, *Prompting Claude Fable 5* —
  <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5>.
  "Instruction-following is improved enough that you can steer most behaviors with a brief
  instruction rather than enumerating each behavior by name"; "Capability improvements at this
  level are also a good prompt to re-evaluate which instructions, tools, and guardrails are still
  needed"; "Skills developed for prior models are often too prescriptive … Review and consider
  removing older instructions if default performance is better"; "Give the reason, not only the
  request". The Fable 5.1, Sonnet 5.5 and Opus 5.5 pages say existing prompts carry over and add
  nothing specific to instruction files.
