# no-magic-papers

Paper cards in `papers/`, optional prose lessons in `lessons/`, paper-canonical slugs.

Status: `v0.1`

`no-magic-papers` is the primary-source prose layer for the `no-magic-ai` organization. It stores concise paper cards, optional lessons, a generated index and generated card metadata JSON without coupling itself to implementation repositories.

## Constraints

- Read and summarize papers from primary sources only.
- Use paper-canonical slugs for files in `papers/` and `lessons/`; never use the `micro*` script prefix here.
- Keep implementation references explicit in card frontmatter under `implementations:`.
- Regenerate `INDEX.md` (`--write`) and `data/papers.json` (`--format json --write`) with `scripts/generate_index.py`; do not hand-edit them.

## References

- [Schema](SCHEMA.md)
- [Themes](THEMES.md)
- [Contributing](CONTRIBUTING.md)
- [Organization strategy](https://github.com/no-magic-ai/.github/blob/main/docs/no-magic-ai-expansion-strategy.md)
- [Paper ingestion SOP](https://github.com/no-magic-ai/.github/blob/main/docs/paper-ingestion-process.md)
