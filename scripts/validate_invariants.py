#!/usr/bin/env python3
"""
Cross-repo validator enforcing SOP §7.3 atomic integrity invariants 1-3.

This script reads:
  - no-magic-papers/papers/*.md (this repo) — paper cards with frontmatter
  - no-magic/docs/catalog.json (sibling repo, --catalog path) — script registry

And enforces:

  Invariant 1: Every script in catalog.json has exactly one paper card whose
  implementations[] contains a script_slug matching the catalog entry's name,
  and that card's status is `implemented`.

  Invariant 2: Every paper card with status `implemented` has implementations[]
  entries whose script_slug values resolve to a real entry in catalog.json
  (the proxy here for "the file exists in the named repo on main").

  Invariant 3 (no-magic v3.0+): Every catalog entry must have a paper_slug
  field that points at a real paper card; the paper card's implementations[]
  must reference back. (This is the symmetric form of Invariant 1.)

Usage:
    python scripts/validate_invariants.py --catalog ../no-magic/docs/catalog.json

Exit codes:
    0  all invariants hold
    1  one or more invariants violated (details printed to stderr)
    2  bad arguments or missing files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_card(path: Path) -> dict[str, object]:
    """Parse a paper card and return a dict of fields needed for validation."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        _, frontmatter, _ = text.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed YAML frontmatter fence") from exc

    slug = path.stem
    status = ""
    implementations: list[dict[str, str]] = []

    in_impl_block = False
    current_entry: dict[str, str] | None = None

    for line in frontmatter.splitlines():
        if not in_impl_block:
            # Top-level keys have no leading whitespace. Nested keys (e.g.
            # `lesson.status`) are indented and must NOT be matched here.
            m = re.match(r"^status:\s*(.+)$", line)
            if m:
                status = m.group(1).strip().strip('"').strip("'")
                continue
            if line == "implementations: []":
                in_impl_block = False
                continue
            if line.startswith("implementations:"):
                in_impl_block = True
                continue
        else:
            if line and not line.startswith(" "):
                in_impl_block = False
                if current_entry is not None:
                    implementations.append(current_entry)
                    current_entry = None
                continue
            if line.lstrip().startswith("- "):
                if current_entry is not None:
                    implementations.append(current_entry)
                current_entry = {}
                rest = line.lstrip()[2:]
                if ":" in rest:
                    key, _, val = rest.partition(":")
                    current_entry[key.strip()] = val.strip().strip('"').strip("'")
                continue
            m = re.match(r"\s+([a-z_]+):\s*(.*)$", line)
            if m and current_entry is not None:
                current_entry[m.group(1)] = m.group(2).strip().strip('"').strip("'")

    if current_entry is not None:
        implementations.append(current_entry)

    return {"slug": slug, "status": status, "implementations": implementations}


def load_papers(papers_dir: Path) -> list[dict[str, object]]:
    """Load and parse every paper card under papers/."""
    cards = []
    for card_path in sorted(papers_dir.glob("*.md")):
        cards.append(parse_card(card_path))
    return cards


def load_catalog(catalog_path: Path) -> list[dict[str, object]]:
    """Load no-magic catalog.json."""
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def check_invariants(
    catalog: list[dict[str, object]],
    cards: list[dict[str, object]],
    enforce_invariant_3: bool = True,
) -> list[str]:
    """Return a list of invariant-violation messages (empty if all pass).

    Invariants 1 and 2 are always enforced. Invariant 3 (the catalog must carry
    paper_slug pointing back at the card) is enforced only when
    enforce_invariant_3 is True — set False for pre-v3.0 catalogs that have
    not yet adopted the paper_slug field.
    """
    errors: list[str] = []

    # Build (script_slug -> [card_slugs that point at it]) for invariant 1
    script_to_cards: dict[str, list[str]] = {}
    # Build (script_slug -> set of card_slugs whose status is implemented and reference it)
    script_to_implemented_cards: dict[str, list[str]] = {}
    # Build (card_slug -> status) and (card_slug -> [script_slug ...])
    card_status: dict[str, str] = {}
    card_to_scripts: dict[str, list[str]] = {}

    for card in cards:
        slug = str(card["slug"])
        status = str(card["status"])
        impls = card["implementations"]
        if not isinstance(impls, list):
            errors.append(f"card {slug}: implementations must be a list")
            continue
        card_status[slug] = status
        scripts_in_card: list[str] = []
        for entry in impls:
            if not isinstance(entry, dict):
                continue
            script_slug = entry.get("script_slug", "")
            if not script_slug or script_slug == "null":
                continue
            scripts_in_card.append(script_slug)
            script_to_cards.setdefault(script_slug, []).append(slug)
            if status == "implemented":
                script_to_implemented_cards.setdefault(script_slug, []).append(slug)
        card_to_scripts[slug] = scripts_in_card

    catalog_names = {str(entry["name"]) for entry in catalog}

    # Invariant 1: every catalog script has exactly one implemented paper card.
    for entry in catalog:
        name = str(entry["name"])
        impl_cards = script_to_implemented_cards.get(name, [])
        if len(impl_cards) == 0:
            errors.append(
                f"invariant 1: script {name!r} in catalog has no paper card with "
                f"status=implemented and implementations[].script_slug={name!r}"
            )
        elif len(impl_cards) > 1:
            errors.append(
                f"invariant 1: script {name!r} has multiple paper cards: {impl_cards}"
            )

    # Invariant 2: every implemented card's implementations[] script_slugs
    # resolve to a real catalog entry.
    for card_slug, status in card_status.items():
        if status != "implemented":
            continue
        for script_slug in card_to_scripts.get(card_slug, []):
            if script_slug not in catalog_names:
                errors.append(
                    f"invariant 2: paper card {card_slug!r} (status implemented) "
                    f"references script_slug {script_slug!r} not present in catalog.json"
                )

    # Invariant 3 (v3.0+): every catalog entry has a paper_slug field that
    # matches an existing paper card AND that card's implementations[] references
    # the script back. Skip when enforce_invariant_3 is False (pre-v3.0).
    if not enforce_invariant_3:
        return errors

    paper_slugs = set(card_status.keys())
    for entry in catalog:
        name = str(entry["name"])
        paper_slug = entry.get("paper_slug")
        if paper_slug is None:
            errors.append(
                f"invariant 3: catalog entry {name!r} is missing paper_slug field "
                f"(required from no-magic v3.0)"
            )
            continue
        if not isinstance(paper_slug, str) or not paper_slug:
            errors.append(
                f"invariant 3: catalog entry {name!r} paper_slug must be a non-empty string"
            )
            continue
        if paper_slug not in paper_slugs:
            errors.append(
                f"invariant 3: catalog entry {name!r} paper_slug={paper_slug!r} "
                f"does not match any paper card in papers/"
            )
            continue
        if name not in card_to_scripts.get(paper_slug, []):
            errors.append(
                f"invariant 3: catalog entry {name!r} paper_slug={paper_slug!r} "
                f"but the card's implementations[] does not reference {name!r}"
            )

    return errors


def detect_no_magic_major_version(catalog_path: Path) -> int:
    """Return the no-magic major version by reading sibling VERSION file.

    Defaults to 0 if VERSION cannot be located, which suppresses invariant 3
    enforcement on pre-v3 catalogs.
    """
    version_path = catalog_path.resolve().parent.parent / "VERSION"
    if not version_path.is_file():
        return 0
    raw = version_path.read_text(encoding="utf-8").strip()
    head = raw.split(".", 1)[0]
    if not head.isdigit():
        return 0
    return int(head)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="path to no-magic/docs/catalog.json",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "papers",
        help="path to papers/ directory (default: ../papers relative to script)",
    )
    parser.add_argument(
        "--require-paper-slug",
        choices=("auto", "yes", "no"),
        default="auto",
        help=(
            "enforce SOP §7.3 invariant 3 (catalog must carry paper_slug). "
            "'auto' enables it when the no-magic VERSION file reports major >= 3."
        ),
    )
    args = parser.parse_args()

    if not args.catalog.is_file():
        print(f"catalog file not found: {args.catalog}", file=sys.stderr)
        return 2
    if not args.papers.is_dir():
        print(f"papers directory not found: {args.papers}", file=sys.stderr)
        return 2

    try:
        cards = load_papers(args.papers)
        catalog = load_catalog(args.catalog)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"failed to load inputs: {exc}", file=sys.stderr)
        return 2

    if args.require_paper_slug == "yes":
        enforce_inv3 = True
    elif args.require_paper_slug == "no":
        enforce_inv3 = False
    else:
        enforce_inv3 = detect_no_magic_major_version(args.catalog) >= 3

    errors = check_invariants(catalog, cards, enforce_invariant_3=enforce_inv3)
    if errors:
        print(
            f"FAIL: {len(errors)} invariant violation(s) across {len(catalog)} "
            f"catalog scripts and {len(cards)} paper cards:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"OK: SOP §7.3 invariants 1-3 hold across {len(catalog)} catalog "
        f"scripts and {len(cards)} paper cards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
