#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# fmt: off
THEMES = tuple(
    "efficient-inference long-context alignment reasoning architecture "
    "training-dynamics parameter-efficient interpretability retrieval "
    "safety-robustness agents multimodal".split()
)
STATUSES = set(
    "triaged summarized backlog-implement implemented deprecated replaced archived reference-only".split()
)
LESSON_STATUSES = {"none", "planned", "drafted", "published", "null"}
REQUIRED_FIELDS = tuple(
    "slug title authors venue year arxiv_id doi url discovered_via discovered_date status "
    "themes tags routing implementations lesson dependencies_on_other_papers".split()
)
REQUIRED_SECTIONS = (
    "## TL;DR",
    "## Problem",
    "## Contribution",
    "## Method summary",
    "## Key results",
    "## Relation to existing work",
    "## Implementation notes",
)


def clean(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def split_card(path: Path) -> tuple[list[str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        _, frontmatter, body = text.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed YAML frontmatter fence") from exc
    return frontmatter.splitlines(), body


def scalar(lines: list[str], key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*)$")
    for line in lines:
        if match := pattern.match(line):
            return clean(match.group(1))
    return None


def block(lines: list[str], key: str) -> list[str]:
    start = next((i for i, line in enumerate(lines) if line == f"{key}:" or line.startswith(f"{key}:")), -1)
    if start < 0:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        out.append(line)
    return out


def list_items(lines: list[str], key: str) -> list[str]:
    if scalar(lines, key) == "[]":
        return []
    return [clean(line.strip()[2:].strip()) for line in block(lines, key) if line.strip().startswith("- ")]


def nested_scalar(lines: list[str], parent: str, child: str) -> str:
    for line in block(lines, parent):
        stripped = line.strip()
        if stripped.startswith(f"{child}:"):
            return clean(stripped.split(":", 1)[1])
    return "null"


def parse_themes(lines: list[str]) -> tuple[str, tuple[str, ...]]:
    primary = ""
    secondary: list[str] = []
    in_secondary = False
    for line in block(lines, "themes"):
        stripped = line.strip()
        if stripped.startswith("primary:"):
            primary = clean(stripped.split(":", 1)[1])
            in_secondary = False
        elif stripped.startswith("secondary:"):
            in_secondary = True
        elif in_secondary and stripped.startswith("- "):
            secondary.append(clean(stripped[2:]))
    if not primary:
        raise ValueError("themes.primary is required")
    return primary, tuple(secondary)


def parse_card(path: Path) -> dict[str, object]:
    lines, body = split_card(path)
    missing = [field for field in REQUIRED_FIELDS if scalar(lines, field) is None and not block(lines, field)]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    slug = scalar(lines, "slug") or ""
    status = scalar(lines, "status") or ""
    primary, secondary = parse_themes(lines)
    lesson_status = nested_scalar(lines, "lesson", "status")
    errors: list[str] = []
    checks = (
        (path.stem.startswith("micro") or slug.startswith("micro"), "paper slug must not start with micro"),
        (slug != path.stem, f"slug {slug!r} must match filename stem {path.stem!r}"),
        (not list_items(lines, "authors"), "authors must be a non-empty YAML list"),
        (status not in STATUSES, f"status {status!r} is not allowed"),
        (primary not in THEMES, f"themes.primary {primary!r} is not allowed"),
        (len(secondary) > 2 or any(theme not in THEMES for theme in secondary), "themes.secondary must contain zero to two known themes"),
        (lesson_status not in LESSON_STATUSES, f"lesson.status {lesson_status!r} is not allowed"),
    )
    errors.extend(message for failed, message in checks if failed)
    errors.extend(f"missing body section {section}" for section in REQUIRED_SECTIONS if section not in body)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return {
        "path": path,
        "slug": slug,
        "title": scalar(lines, "title") or "",
        "year": scalar(lines, "year") or "",
        "status": status,
        "primary": primary,
        "secondary": secondary,
        "lesson": lesson_status,
    }


def load_cards(root: Path) -> list[dict[str, object]]:
    return [parse_card(path) for path in sorted((root / "papers").glob("*.md"))]


def render(cards: list[dict[str, object]]) -> str:
    lines = ["# no-magic-papers Index", "", "Generated by `scripts/generate_index.py`. Do not hand-edit.", ""]
    if not cards:
        return "\n".join([*lines, "No paper cards have been added yet.", ""])
    by_theme: dict[str, list[dict[str, object]]] = {theme: [] for theme in THEMES}
    for card in cards:
        by_theme[str(card["primary"])].append(card)
    for theme in THEMES:
        theme_cards = sorted(by_theme[theme], key=lambda card: str(card["slug"]))
        if not theme_cards:
            continue
        lines += [f"## {theme}", "", "| Paper | Year | Status | Secondary themes | Lesson |", "|---|---:|---|---|---|"]
        for card in theme_cards:
            secondary_values = card["secondary"]
            secondary = ", ".join(secondary_values) if isinstance(secondary_values, tuple) and secondary_values else "-"
            lesson = str(card["lesson"])
            lesson = "-" if lesson == "null" else lesson
            path = card["path"]
            if not isinstance(path, Path):
                raise TypeError("card path must be a Path")
            lines.append(f"| [{card['title']}]({path.as_posix()}) | {card['year']} | `{card['status']}` | {secondary} | {lesson} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate INDEX.md.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        output = render(load_cards(root))
    except (TypeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    index_path = root / "INDEX.md"
    if args.write:
        index_path.write_text(output, encoding="utf-8")
    if args.check:
        current = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        if current != output:
            print("INDEX.md is stale; run scripts/generate_index.py --write", file=sys.stderr)
            return 1
    if not args.write and not args.check and not args.validate:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
