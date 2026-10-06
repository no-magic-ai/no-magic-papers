# Contributing

`no-magic-papers` accepts paper cards and optional lessons that are written from primary sources. Do not copy from tutorials, blog posts, course material, third-party repositories, or generated summaries of those sources.

## Paper Cards

1. Read the paper directly from arXiv, DOI, or the open venue page.
2. Search existing cards by title, arXiv ID, DOI, and slug before drafting.
3. Create one file at `papers/{paper-slug}.md`.
4. Use a paper-canonical slug: lowercase ASCII, hyphen-separated, no `micro*` prefix.
5. Fill every required field in `SCHEMA.md`.
6. Set exactly one primary theme and zero to two secondary themes from `THEMES.md`.
7. Keep the card body concise: 800 words or less.
8. Regenerate `INDEX.md` with `python3 scripts/generate_index.py --write`.

## Lessons

Lessons are optional companions at `lessons/{paper-slug}.md`. They use the same slug as the paper card and stay under 1500 words.

Use this section order:

1. Paper summary
2. Intuition
3. Code walkthrough
4. Exercises

Reference the paper card by slug and any implementation by its explicit repository path. Script slugs keep the `micro*` prefix because they refer to pedagogical miniature implementations; paper and lesson slugs do not.

## Review Rules

- Every PR is reviewed before merge. No auto-merge, review-bot solicitation or administrative bypass.
- Scoped enhancement-phase technical merges ([issue #51](https://github.com/no-magic-ai/no-magic-papers/issues/51)): the canonical runner may mechanically merge a technical PR only after current design approval, separate independent verification, independent per-PR and coordinated-stack review, matching canonical and independent technical classifications of the actual head/base diff, every applicable current-head provider/CI gate, and no unresolved blocker. Technical scope is tooling, CI, deterministic link/metadata repairs and control/security plumbing that do not change scientific or math behavior. Reclassify and rerun stale gates when the head, base or either classification changes.
- Paper cards, lessons, scientific/math behavior, research claims and other content require actual human approval naming the exact PR, current head SHA and content scope. Mixed, ambiguous or disputed classifications are human-gated. AI review, an agent-posted approval or stale approval is not human approval. Provider-required reviewers and branch protections remain binding.
- This exception grants no automatic scientific review, factory activation, experimental spend, release/version authority or weakened acceptance/safety gate.
- CI validates slug namespaces, the strict card and lesson schema (`scripts/generate_index.py --validate`), byte-exact `INDEX.md` freshness (`--check`), the validator tests and the cross-repo cohort (`scripts/validate_invariants.py`): implementation paths and ownership against the `no-magic` catalog and committed files, lesson lifecycle and declared `no-magic-viz` media. Pushes to `main` check the published cohort; PRs and other pushes check the exact head as a candidate cohort, which makes no publication claim.
- `INDEX.md` is generated output. Hand edits are reverted.
- Commits use conventional commit format, imperative mood, and one logical change per commit.

## Contributor Pledge

By contributing, you affirm that the card or lesson was written from the paper itself and that any implementation references are explicit, source-controlled paths rather than inferred slug mappings.
