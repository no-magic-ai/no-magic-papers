#!/usr/bin/env python3
"""Generate and validate INDEX.md and data/papers.json from paper-card frontmatter.

This module is the single frontmatter parser and card/lesson validator for
no-magic-papers; scripts/validate_invariants.py imports it rather than parsing
cards itself.

The parser accepts exactly the frontmatter subset that SCHEMA.md documents, not
general YAML: two-space indentation; top-level `key: scalar`, `key: []` or a
`key:` block holding a list of scalars, a list of flat records or a mapping of
scalars/lists. Scalars are plain text, "double-quoted" or 'single-quoted' text
without escapes, or `null`. Tabs, blank lines, comments, flow collections,
duplicate keys and anything else outside that subset are rejected.

Usage:
    python scripts/generate_index.py --write     # regenerate INDEX.md
    python scripts/generate_index.py --check     # exit 1 unless INDEX.md bytes match
    python scripts/generate_index.py --validate  # validate cards and lessons only
    python scripts/generate_index.py --format json          # print metadata JSON
    python scripts/generate_index.py --format json --write  # regenerate data/papers.json
    python scripts/generate_index.py --format json --check  # exit 1 unless its bytes match

Every mode validates all cards and lessons before producing any output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

THEMES = (
    "efficient-inference",
    "long-context",
    "alignment",
    "reasoning",
    "architecture",
    "training-dynamics",
    "parameter-efficient",
    "interpretability",
    "retrieval",
    "safety-robustness",
    "agents",
    "multimodal",
)
STATUSES = {
    "triaged",
    "summarized",
    "backlog-implement",
    "implemented",
    "deprecated",
    "replaced",
    "archived",
    "reference-only",
}
LESSON_STATUSES = {"none", "planned", "drafted", "published"}
REQUIRED_FIELDS = (
    "slug",
    "title",
    "authors",
    "venue",
    "year",
    "arxiv_id",
    "doi",
    "url",
    "discovered_via",
    "discovered_date",
    "status",
    "themes",
    "tags",
    "routing",
    "implementations",
    "lesson",
    "dependencies_on_other_papers",
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
IMPLEMENTATION_KEYS = (
    "repo",
    "path",
    "script_slug",
    "commit",
    "release",
    "media_repo",
    "media_status",
    "scene_path",
    "preview_path",
    "media_note",
)
LESSON_KEYS = ("path", "status")
LESSON_SECTIONS = (
    "## Paper summary",
    "## Intuition",
    "## Code walkthrough",
    "## Exercises",
)
LESSON_WORD_LIMIT = 1500
IMPLEMENTATION_REPO = "no-magic"
MEDIA_REPO = "no-magic-viz"
TIER_DIRS = ("01-foundations", "02-alignment", "03-systems", "04-agents")
INDEX_PATH = "INDEX.md"
METADATA_PATH = "data/papers.json"
METADATA_SCHEMA_VERSION = 1

Scalar = str | None
Record = dict[str, Scalar]
MappingValue = Scalar | list[str]
FieldValue = Scalar | list[str] | list[Record] | dict[str, MappingValue]
Frontmatter = dict[str, FieldValue]
Reader = Callable[[str], bytes]

KEY_LINE = re.compile(r"^([a-z_]+):(?: (.*))?$")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SCRIPT_SLUG = re.compile(r"^[a-z0-9_]+$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
RELEASE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")


class FrontmatterError(ValueError):
    """The frontmatter is outside the supported subset or malformed."""


class ValidationError(ValueError):
    """One or more cards or lessons violate the schema; message lists every error."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Implementation:
    repo: str
    path: str
    script_slug: str
    commit: str | None
    release: str | None
    media_repo: str | None
    media_status: str
    scene_path: str | None
    preview_path: str | None
    media_note: str | None


@dataclass(frozen=True)
class Lesson:
    path: str | None
    status: str | None


@dataclass(frozen=True)
class Card:
    path: PurePosixPath
    slug: str
    title: str
    year: str
    status: str
    primary: str
    secondary: tuple[str, ...]
    lesson: Lesson
    implementations: tuple[Implementation, ...]
    dependencies: tuple[str, ...]
    frontmatter: Frontmatter


@dataclass(frozen=True)
class PaperRepository:
    cards: tuple[Card, ...]
    lessons: tuple[str, ...]


def parse_scalar(raw: str, where: str) -> Scalar:
    if raw == "":
        raise FrontmatterError(f"{where}: empty value; use null or a block")
    if raw != raw.strip():
        raise FrontmatterError(f"{where}: leading or trailing whitespace")
    if raw == "null":
        return None
    for quote in ('"', "'"):
        if raw[0] == quote:
            inner = raw[1:-1]
            if len(raw) < 2 or raw[-1] != quote or quote in inner or "\\" in inner:
                raise FrontmatterError(f"{where}: unsupported quoted scalar {raw!r}")
            return inner
    if raw[0] in "[]{}&*!|>%@`#,?-" or ": " in raw or " #" in raw or raw.endswith(":"):
        raise FrontmatterError(f"{where}: ambiguous plain scalar {raw!r}; quote it")
    if raw.lower() in {"~", "null", "true", "false", "yes", "no", "on", "off"}:
        raise FrontmatterError(f"{where}: ambiguous plain scalar {raw!r}; quote it")
    return raw


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_key(text: str, where: str) -> tuple[str, str | None]:
    match = KEY_LINE.match(text)
    if not match:
        raise FrontmatterError(f"{where}: expected 'key: value', got {text!r}")
    return match.group(1), match.group(2)


def parse_list(
    lines: list[str], start: int, indent: int, where: str
) -> tuple[list[str] | list[Record], int]:
    """Parse `- item` lines at `indent`; items are all scalars or all flat records."""
    scalars: list[str] = []
    records: list[Record] = []
    index = start
    while index < len(lines) and indent_of(lines[index]) >= indent:
        line = lines[index]
        if indent_of(line) != indent or not line[indent:].startswith("- "):
            raise FrontmatterError(
                f"{where}: expected '- item' at indent {indent}, got {line!r}"
            )
        item = line[indent + 2 :]
        if KEY_LINE.match(item):
            if scalars:
                raise FrontmatterError(f"{where}: list mixes scalars and records")
            key, raw = split_key(item, where)
            record: Record = {key: parse_scalar(raw or "", f"{where}.{key}")}
            index += 1
            while index < len(lines) and indent_of(lines[index]) > indent:
                if indent_of(lines[index]) != indent + 2:
                    raise FrontmatterError(
                        f"{where}: record field must be at indent {indent + 2}"
                    )
                key, raw = split_key(lines[index][indent + 2 :], where)
                if key in record:
                    raise FrontmatterError(f"{where}: duplicate record key {key!r}")
                record[key] = parse_scalar(raw or "", f"{where}.{key}")
                index += 1
            records.append(record)
        else:
            if records:
                raise FrontmatterError(f"{where}: list mixes scalars and records")
            value = parse_scalar(item, f"{where}[]")
            if value is None:
                raise FrontmatterError(f"{where}: list items must not be null")
            scalars.append(value)
            index += 1
    if records:
        return records, index
    return scalars, index


def parse_mapping(
    lines: list[str], start: int, where: str
) -> tuple[dict[str, MappingValue], int]:
    mapping: dict[str, MappingValue] = {}
    index = start
    while index < len(lines) and indent_of(lines[index]) >= 2:
        line = lines[index]
        if indent_of(line) != 2:
            raise FrontmatterError(f"{where}: unexpected indentation in {line!r}")
        key, raw = split_key(line[2:], where)
        if key in mapping:
            raise FrontmatterError(f"{where}: duplicate key {key!r}")
        index += 1
        if raw == "[]":
            mapping[key] = []
        elif raw is None:
            items, index = parse_list(lines, index, 4, f"{where}.{key}")
            if not items or not all(isinstance(item, str) for item in items):
                raise FrontmatterError(
                    f"{where}.{key}: nested block must be a non-empty scalar list"
                )
            mapping[key] = [item for item in items if isinstance(item, str)]
        else:
            mapping[key] = parse_scalar(raw, f"{where}.{key}")
    return mapping, index


def parse_frontmatter(text: str) -> Frontmatter:
    lines = text.split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise FrontmatterError(f"line {number}: blank lines are not supported")
        if "\t" in line or "\r" in line:
            raise FrontmatterError(
                f"line {number}: tabs and carriage returns are not supported"
            )
    result: Frontmatter = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if indent_of(line) != 0:
            raise FrontmatterError(
                f"line {index + 1}: unexpected indentation in {line!r}"
            )
        key, raw = split_key(line, f"line {index + 1}")
        if key in result:
            raise FrontmatterError(f"line {index + 1}: duplicate key {key!r}")
        index += 1
        if raw == "[]":
            result[key] = []
        elif raw is not None:
            result[key] = parse_scalar(raw, key)
        elif index < len(lines) and lines[index].startswith("  - "):
            items, index = parse_list(lines, index, 2, key)
            result[key] = items
        elif index < len(lines) and indent_of(lines[index]) == 2:
            mapping, index = parse_mapping(lines, index, key)
            result[key] = mapping
        else:
            raise FrontmatterError(f"{key}: block has no content; use [] or null")
    return result


def split_card(raw: bytes, name: str) -> tuple[Frontmatter, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrontmatterError(f"{name}: not valid UTF-8") from exc
    if not text.startswith("---\n"):
        raise FrontmatterError(f"{name}: missing frontmatter fence")
    end = text.find("\n---\n", 3)
    if end < 0:
        raise FrontmatterError(f"{name}: unterminated frontmatter fence")
    return parse_frontmatter(text[4:end]), text[end + 5 :]


def safe_relative_path(value: str) -> str | None:
    """Return an error unless value is a plain repo-relative POSIX path."""
    if (
        not value
        or "\\" in value
        or "\0" in value
        or value.startswith("/")
        or ":" in value
    ):
        return f"path {value!r} must be repo-relative POSIX"
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return f"path {value!r} has an empty, '.' or '..' component"
    return None


def lesson_path_for(slug: str) -> str:
    return f"no-magic-papers/lessons/{slug}.md"


class CardChecker:
    """Typed validation of one card's parsed frontmatter; accumulates errors."""

    def __init__(self, name: str, fields: Frontmatter) -> None:
        self.name = name
        self.fields = fields
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(f"{self.name}: {message}")

    def scalar(self, key: str, *, nullable: bool = False) -> str | None:
        value = self.fields.get(key)
        if value is None and nullable and key in self.fields:
            return None
        if not isinstance(value, str) or not value:
            self.error(
                f"{key} must be a non-empty {'string or null' if nullable else 'string'}"
            )
            return None
        return value

    def scalar_list(self, key: str, *, non_empty: bool) -> list[str]:
        value = self.fields.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            self.error(f"{key} must be a list of strings")
            return []
        if non_empty and not value:
            self.error(f"{key} must be a non-empty list")
        return [item for item in value if isinstance(item, str)]

    def records(self, key: str) -> list[Record]:
        value = self.fields.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            self.error(f"{key} must be [] or a list of records")
            return []
        return [item for item in value if isinstance(item, dict)]

    def mapping(self, key: str) -> dict[str, MappingValue]:
        value = self.fields.get(key)
        if not isinstance(value, dict):
            self.error(f"{key} must be a mapping")
            return {}
        return value


def check_themes(checker: CardChecker) -> tuple[str, tuple[str, ...]]:
    themes = checker.mapping("themes")
    primary = themes.get("primary")
    secondary = themes.get("secondary", [])
    if not isinstance(primary, str) or primary not in THEMES:
        checker.error(f"themes.primary {primary!r} is not allowed")
        primary = ""
    if (
        not isinstance(secondary, list)
        or len(secondary) > 2
        or any(t not in THEMES for t in secondary)
    ):
        checker.error("themes.secondary must contain zero to two known themes")
        secondary = []
    return primary, tuple(secondary)


def check_implementation(
    checker: CardChecker, index: int, record: Record
) -> Implementation | None:
    where = f"implementations[{index}]"
    keys = tuple(record)
    if keys != IMPLEMENTATION_KEYS:
        checker.error(
            f"{where} keys must be exactly {', '.join(IMPLEMENTATION_KEYS)} in order; got {', '.join(keys)}"
        )
        return None
    repo, path, script_slug = record["repo"], record["path"], record["script_slug"]
    commit, release = record["commit"], record["release"]
    media_repo, media_status = record["media_repo"], record["media_status"]
    scene_path, preview_path, media_note = (
        record["scene_path"],
        record["preview_path"],
        record["media_note"],
    )
    problems: list[str] = []
    if repo != IMPLEMENTATION_REPO:
        problems.append(f"repo must be {IMPLEMENTATION_REPO!r}, got {repo!r}")
    if not isinstance(script_slug, str) or not SCRIPT_SLUG.match(script_slug):
        problems.append(
            f"script_slug {script_slug!r} must be lowercase letters, digits or underscores"
        )
    unsafe = (
        safe_relative_path(path) if isinstance(path, str) else "path must be a string"
    )
    if unsafe:
        problems.append(unsafe)
    elif isinstance(path, str) and isinstance(script_slug, str):
        parts = path.split("/")
        if (
            len(parts) != 2
            or parts[0] not in TIER_DIRS
            or parts[1] != f"{script_slug}.py"
        ):
            problems.append(
                f"path {path!r} must be {{tier}}/{script_slug}.py in a tier directory"
            )
    if commit is not None and not COMMIT.match(commit):
        problems.append(f"commit {commit!r} must be a 40-hex commit or null")
    if release is not None and not RELEASE.match(release):
        problems.append(f"release {release!r} must be vMAJOR.MINOR.PATCH or null")
    if media_status == "linked":
        if media_repo != MEDIA_REPO:
            problems.append(f"linked media_repo must be {MEDIA_REPO!r}")
        if scene_path != f"scenes/scene_{script_slug}.py":
            problems.append(f"linked scene_path must be scenes/scene_{script_slug}.py")
        if preview_path != f"previews/{script_slug}.gif":
            problems.append(f"linked preview_path must be previews/{script_slug}.gif")
        if media_note is not None and not media_note.strip():
            problems.append("linked media_note must be null or non-empty")
    elif media_status == "omitted":
        if media_repo is not None or scene_path is not None or preview_path is not None:
            problems.append(
                "omitted media requires media_repo, scene_path and preview_path to be null"
            )
        if media_note is None or not media_note.strip():
            problems.append(
                "omitted media requires a non-empty media_note explaining the omission"
            )
    else:
        problems.append(f"media_status {media_status!r} must be 'linked' or 'omitted'")
    for problem in problems:
        checker.error(f"{where}: {problem}")
    if (
        problems
        or not isinstance(path, str)
        or not isinstance(script_slug, str)
        or media_status is None
    ):
        return None
    return Implementation(
        repo=IMPLEMENTATION_REPO,
        path=path,
        script_slug=script_slug,
        commit=commit,
        release=release,
        media_repo=media_repo,
        media_status=media_status,
        scene_path=scene_path,
        preview_path=preview_path,
        media_note=media_note,
    )


def check_lesson_fields(checker: CardChecker, slug: str) -> Lesson:
    lesson = checker.mapping("lesson")
    if tuple(lesson) != LESSON_KEYS:
        checker.error(f"lesson keys must be exactly {', '.join(LESSON_KEYS)}")
        return Lesson(path=None, status=None)
    path, status = lesson["path"], lesson["status"]
    if isinstance(path, list) or isinstance(status, list):
        checker.error("lesson.path and lesson.status must be scalars")
        return Lesson(path=None, status=None)
    if status is not None and status not in LESSON_STATUSES:
        checker.error(f"lesson.status {status!r} is not allowed")
    elif status in {None, "none"} and path is not None:
        checker.error(f"lesson.status {status or 'null'} requires lesson.path null")
    elif status == "planned" and path is not None and path != lesson_path_for(slug):
        checker.error(f"planned lesson.path must be null or {lesson_path_for(slug)}")
    elif status in {"drafted", "published"} and path != lesson_path_for(slug):
        checker.error(f"{status} lesson.path must be {lesson_path_for(slug)}")
    return Lesson(path=path, status=status)


def check_card(relative: str, raw: bytes) -> Card | list[str]:
    """Validate one card's bytes; return the Card or the list of errors."""
    try:
        fields, body = split_card(raw, relative)
    except FrontmatterError as exc:
        return [f"{relative}: {exc}"]
    checker = CardChecker(relative, fields)
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        checker.error(f"missing required fields: {', '.join(missing)}")
        return checker.errors
    stem = PurePosixPath(relative).stem
    slug = checker.scalar("slug") or ""
    title = checker.scalar("title") or ""
    year = checker.scalar("year") or ""
    status = checker.scalar("status") or ""
    for key in ("venue", "url", "discovered_via"):
        checker.scalar(key)
    for key in ("arxiv_id", "doi"):
        checker.scalar(key, nullable=True)
    checker.scalar_list("authors", non_empty=True)
    checker.scalar_list("tags", non_empty=False)
    checker.mapping("routing")
    if stem.startswith("micro") or slug.startswith("micro"):
        checker.error("paper slug must not start with micro")
    if slug != stem:
        checker.error(f"slug {slug!r} must match filename stem {stem!r}")
    elif not SLUG.match(slug):
        checker.error(f"slug {slug!r} must be lowercase ASCII words joined by hyphens")
    if not re.fullmatch(r"[0-9]{4}", year):
        checker.error(f"year {year!r} must be a four-digit year")
    discovered = checker.scalar("discovered_date") or ""
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", discovered):
        checker.error(f"discovered_date {discovered!r} must be YYYY-MM-DD")
    if status not in STATUSES:
        checker.error(f"status {status!r} is not allowed")
    primary, secondary = check_themes(checker)
    implementations = [
        impl
        for index, record in enumerate(checker.records("implementations"))
        if (impl := check_implementation(checker, index, record)) is not None
    ]
    if status == "implemented" and not checker.fields.get("implementations"):
        checker.error(
            "status implemented requires at least one implementations[] entry"
        )
    dependencies: list[str] = []
    for index, record in enumerate(checker.records("dependencies_on_other_papers")):
        dependency = record.get("slug")
        if tuple(record) != ("slug",) or not isinstance(dependency, str):
            checker.error(
                f"dependencies_on_other_papers[{index}] must be exactly '- slug: <paper-slug>'"
            )
        else:
            dependencies.append(dependency)
    lesson = check_lesson_fields(checker, slug)
    for section in REQUIRED_SECTIONS:
        if section not in body:
            checker.error(f"missing body section {section}")
    if checker.errors:
        return checker.errors
    return Card(
        path=PurePosixPath(relative),
        slug=slug,
        title=title,
        year=year,
        status=status,
        primary=primary,
        secondary=secondary,
        lesson=lesson,
        implementations=tuple(implementations),
        dependencies=tuple(dependencies),
        frontmatter=fields,
    )


def check_lesson_body(slug: str, relative: str, raw: bytes) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{relative}: not valid UTF-8"]
    errors: list[str] = []
    headings = tuple(
        line.rstrip() for line in text.split("\n") if line.startswith("## ")
    )
    if headings != LESSON_SECTIONS:
        errors.append(
            f"{relative}: sections must be exactly {', '.join(LESSON_SECTIONS)} in order"
        )
    words = len(text.split())
    if words >= LESSON_WORD_LIMIT:
        errors.append(
            f"{relative}: {words} words; lessons stay under {LESSON_WORD_LIMIT}"
        )
    if f"papers/{slug}.md" not in text:
        errors.append(f"{relative}: must reference its paper card papers/{slug}.md")
    return errors


def read_file(root: Path) -> Reader:
    def read(relative: str) -> bytes:
        return (root / relative).read_bytes()

    return read


def load_repository(root: Path, read: Reader) -> PaperRepository:
    """Validate every card and lesson under root; raise ValidationError listing all errors.

    `read` supplies the bytes of a repo-relative path, so callers decide where
    the bytes come from (the invariant validator binds them to committed blobs).
    """
    errors: list[str] = []
    cards: list[Card] = []
    for path in sorted((root / "papers").glob("*.md")):
        relative = f"papers/{path.name}"
        if path.is_symlink() or not path.is_file():
            errors.append(f"{relative}: paper card must be a regular file")
            continue
        try:
            result = check_card(relative, read(relative))
        except (OSError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        if isinstance(result, Card):
            cards.append(result)
        else:
            errors.extend(result)
    by_slug = {card.slug: card for card in cards}
    owners: dict[str, list[str]] = {}
    for card in cards:
        for dependency in card.dependencies:
            if dependency not in by_slug:
                errors.append(
                    f"{card.path}: dependency {dependency!r} names no paper card"
                )
        for impl in card.implementations:
            owners.setdefault(impl.script_slug, []).append(card.slug)
            owners.setdefault(f"path:{impl.path}", []).append(card.slug)
    for owned, claimants in sorted(owners.items()):
        if len(claimants) > 1:
            errors.append(
                f"{owned} is declared by more than one implementation record: {claimants}"
            )
    lesson_dir = root / "lessons"
    lesson_files = sorted(p.name for p in lesson_dir.glob("*.md"))
    lessons: list[str] = []
    for name in lesson_files:
        relative = f"lessons/{name}"
        stem = PurePosixPath(name).stem
        owner = by_slug.get(stem)
        if (lesson_dir / name).is_symlink() or not (lesson_dir / name).is_file():
            errors.append(f"{relative}: lesson must be a regular file")
        elif owner is None:
            errors.append(f"{relative}: orphan lesson; no paper card {stem!r}")
        elif owner.lesson.status not in {"drafted", "published"}:
            errors.append(
                f"{relative}: lesson file exists but {owner.path} lesson.status is {owner.lesson.status or 'null'}"
            )
    for card in cards:
        if card.lesson.status not in {"drafted", "published"}:
            continue
        relative = f"lessons/{card.slug}.md"
        if f"{card.slug}.md" not in lesson_files:
            errors.append(
                f"{card.path}: {card.lesson.status} lesson {relative} does not exist"
            )
            continue
        try:
            errors.extend(check_lesson_body(card.slug, relative, read(relative)))
        except (OSError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        lessons.append(relative)
    if errors:
        raise ValidationError(errors)
    return PaperRepository(cards=tuple(cards), lessons=tuple(lessons))


def render(cards: tuple[Card, ...]) -> bytes:
    lines = [
        "# no-magic-papers Index",
        "",
        "Generated by `scripts/generate_index.py`. Do not hand-edit.",
        "",
    ]
    if not cards:
        return "\n".join([*lines, "No paper cards have been added yet.", ""]).encode(
            "utf-8"
        )
    by_theme: dict[str, list[Card]] = {theme: [] for theme in THEMES}
    for card in cards:
        by_theme[card.primary].append(card)
    for theme in THEMES:
        theme_cards = sorted(by_theme[theme], key=lambda card: card.slug)
        if not theme_cards:
            continue
        lines += [
            f"## {theme}",
            "",
            "| Paper | Year | Status | Secondary themes | Lesson |",
            "|---|---:|---|---|---|",
        ]
        for card in theme_cards:
            secondary = ", ".join(card.secondary) if card.secondary else "-"
            lesson = card.lesson.status or "-"
            lines.append(
                f"| [{card.title}]({card.path.as_posix()}) | {card.year} | `{card.status}` | {secondary} | {lesson} |"
            )
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def render_json(cards: tuple[Card, ...]) -> bytes:
    """Serialize every card's validated frontmatter, sorted by card path.

    The JSON is a derivative of the cards, not a second metadata authority:
    values are exactly the parsed frontmatter (strings, null, lists and
    mappings; never numbers or booleans) and nothing is inferred from bodies.
    """
    document = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "papers": [
            {"card_path": card.path.as_posix(), "frontmatter": card.frontmatter}
            for card in sorted(cards, key=lambda card: card.path.as_posix())
        ],
    }
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{text}\n".encode()


def output_file(root: Path, relative: str) -> Path:
    """Return root/relative, refusing symlinked components and non-regular files."""
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{relative}: {part!r} is a symlink")
    if current.exists() and not current.is_file():
        raise ValueError(f"{relative} is not a regular file")
    return current


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and validate INDEX.md or data/papers.json."
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        repository = load_repository(root, read_file(root))
    except ValidationError as exc:
        print(f"FAIL: {len(exc.errors)} card/lesson error(s):", file=sys.stderr)
        for error in exc.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        output, relative, write_flags = (
            render_json(repository.cards),
            METADATA_PATH,
            "--format json --write",
        )
        try:
            target = output_file(root, relative)
            if args.write:
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(output)
        except (OSError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    else:
        output, relative, write_flags = render(repository.cards), INDEX_PATH, "--write"
        target = root / relative
        if args.write:
            target.write_bytes(output)
    if args.check:
        current = target.read_bytes() if target.is_file() else None
        if current != output:
            print(
                f"{relative} is stale; run scripts/generate_index.py {write_flags}",
                file=sys.stderr,
            )
            return 1
    if not args.write and not args.check and not args.validate:
        if args.format == "json":
            sys.stdout.buffer.write(output)
        else:
            sys.stdout.write(output.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
