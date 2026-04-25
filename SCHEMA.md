# Paper Card Schema

Every paper card lives at `papers/{paper-slug}.md` and uses YAML frontmatter followed by markdown body sections.

## Required Frontmatter

```yaml
---
slug: turboquant
title: "TurboQuant: ..."
authors:
  - First Author
venue: arXiv
year: 2025
arxiv_id: "2504.19874"
doi: null
url: https://arxiv.org/abs/2504.19874
discovered_via: maintainer
discovered_date: 2026-04-25
status: summarized
themes:
  primary: efficient-inference
  secondary: []
tags:
  - quantization
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microturboquant
  target_path: 03-systems/microturboquant.py
  target_tier: 03-systems
  batch_label: efficient-inference-seed
  review_date: null
implementations: []
lesson:
  path: no-magic-papers/lessons/turboquant.md
  status: planned
dependencies_on_other_papers: []
---
```

## Required Body Sections

```markdown
## TL;DR

## Problem

## Contribution

## Method summary

## Key results

## Relation to existing work

## Implementation notes
```

`## Open questions` and `## Further reading` are optional.

## Field Rules

| Field | Rule |
|---|---|
| `slug` | Must match the filename stem and must not start with `micro`. |
| `authors` | YAML list of author names. |
| `status` | One of `triaged`, `summarized`, `backlog-implement`, `implemented`, `deprecated`, `replaced`, `archived`, `reference-only`. |
| `themes.primary` | One value from `THEMES.md`. |
| `themes.secondary` | YAML list with zero to two values from `THEMES.md`. |
| `implementations` | YAML list. Empty when no implementation has landed. Each entry uses `repo`, `path`, `script_slug`, `commit`, and `release`. |
| `lesson.status` | One of `none`, `planned`, `drafted`, `published`, or `null`. |

## Implementation Entry Shape

```yaml
implementations:
  - repo: no-magic
    path: 03-systems/microturboquant.py
    script_slug: microturboquant
    commit: null
    release: null
```

One paper card may list multiple implementations when one paper introduces multiple distinct algorithms. Bibliographic metadata remains one card; implementation artifacts are list entries.
