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

- Every PR is manually reviewed.
- No auto-merge.
- CI validates slug namespaces, required frontmatter, and generated index freshness.
- `INDEX.md` is generated output. Hand edits are reverted.
- Commits use conventional commit format, imperative mood, and one logical change per commit.

## Contributor Pledge

By contributing, you affirm that the card or lesson was written from the paper itself and that any implementation references are explicit, source-controlled paths rather than inferred slug mappings.
