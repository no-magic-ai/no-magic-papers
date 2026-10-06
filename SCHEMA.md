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
| `slug` | Must match the filename stem, use lowercase ASCII words joined by hyphens and must not start with `micro`. |
| `authors` | Non-empty YAML list of author names. |
| `year` | Four-digit year. |
| `discovered_date` | `YYYY-MM-DD`. |
| `status` | One of `triaged`, `summarized`, `backlog-implement`, `implemented`, `deprecated`, `replaced`, `archived`, `reference-only`. `implemented` requires at least one `implementations[]` entry. |
| `themes.primary` | One value from `THEMES.md`. |
| `themes.secondary` | YAML list with zero to two values from `THEMES.md`. |
| `implementations` | YAML list. Empty when no implementation has landed. Each entry uses exactly the keys shown below, in that order. |
| `lesson.status` | One of `none`, `planned`, `drafted`, `published`, or `null`. |
| `lesson.path` | `null` for `none`/`null`; `null` or `no-magic-papers/lessons/{slug}.md` for `planned` (the file must not exist yet); exactly `no-magic-papers/lessons/{slug}.md` for `drafted`/`published`, and that lesson must exist. |
| `dependencies_on_other_papers` | `[]` or a list of `- slug: {paper-slug}` entries naming existing cards. |

## Implementation Entry Shape

```yaml
implementations:
  - repo: no-magic
    path: 03-systems/microturboquant.py
    script_slug: microturboquant
    commit: null
    release: null
    media_repo: no-magic-viz
    media_status: linked
    scene_path: scenes/scene_microturboquant.py
    preview_path: previews/microturboquant.gif
    media_note: null
```

One paper card may list multiple implementations when one paper introduces multiple distinct algorithms. Bibliographic metadata remains one card; implementation artifacts are list entries.

| Key | Rule |
|---|---|
| `repo` | `no-magic`. |
| `path` | Repo-relative `{tier}/{script_slug}.py` in one of the four tier directories; no absolute, `..` or symlinked component. The cross-repo validator requires it to match the catalog entry and a committed regular file. |
| `script_slug` | Lowercase letters, digits or underscores; one owner across all cards. |
| `commit` | 40-hex commit or `null`. |
| `release` | `vMAJOR.MINOR.PATCH` or `null`. |
| `media_status` | `linked` or `omitted`. |
| `media_repo`, `scene_path`, `preview_path` | Linked: `no-magic-viz`, `scenes/scene_{script_slug}.py` and `previews/{script_slug}.gif`, which must be committed in `no-magic-viz` as valid Python and a GIF with nonzero dimensions. Omitted: all `null`. |
| `media_note` | Linked: `null` or a note. Omitted: a non-empty explanation. Omission is allowed only for catalog `teaching_kind: comparison` scripts; missing media is declared, not backfilled. |

The media checks confirm the declared files exist and are well formed. They do not certify rendering, playback or conceptual fidelity.

## Lessons

`lessons/{paper-slug}.md` exists only for a card whose `lesson.status` is `drafted` or `published`. It references `papers/{paper-slug}.md`, uses exactly the `## Paper summary`, `## Intuition`, `## Code walkthrough`, `## Exercises` sections in that order and stays under 1500 words.

## Supported Frontmatter Syntax

`scripts/generate_index.py` holds the one frontmatter parser; `scripts/validate_invariants.py` imports it. It accepts the subset used above, not general YAML: two-space indentation; `key: value`, `key: []` or a `key:` block containing a list of scalars, a list of flat records or a mapping of scalars and scalar lists. Each value is `null`, a quoted string or a plain value, checked in this order. An empty value is rejected; write `null` or a block. A raw value with leading or trailing whitespace is rejected; whitespace meant as content belongs inside quotes, where it is kept and the field rules still apply. Only the exact lowercase spelling `null` means null. A `"double-quoted"` or `'single-quoted'` value must end with the same quote and contain neither that quote nor a backslash (there are no escapes); its content is kept as written. A plain value is rejected when it starts with `[`, `]`, `{`, `}`, `&`, `*`, `!`, `|`, `>`, `%`, `@`, `` ` ``, `#`, `,`, `?` or `-`, contains `: ` or ` #`, or ends with `:`; quote such values. Unquoted `true`, `false`, `yes`, `no`, `on`, `off`, `~` and any other capitalisation of these or of `null` (for example `True`, `NO`, `Null`) are rejected as ambiguous; quote them (`"True"`, `'off'`) to store them as strings. Every remaining plain value is kept as literal text: the parser does not interpret numbers, dates or other YAML forms, so `2024`, `y`, `0x10`, `.inf` and `1e3` are strings, and the field rules above decide whether they are valid. Tabs, blank lines, comments, flow collections and duplicate keys are rejected.
