#!/usr/bin/env python3
"""
Cross-repo validator for one explicit no-magic / no-magic-papers / no-magic-viz cohort.

It reads paper cards and lessons through scripts/generate_index.py (the single
frontmatter parser), the no-magic catalog, VERSION and implementation files,
and the declared no-magic-viz scenes and previews. Every byte it reads must be
the committed blob at the selected HEAD of its owning repository.

It enforces SOP §7.3 invariants 1-3 plus path, lifecycle and media rules:

  Invariant 1: every catalog script has exactly one implemented paper card
  whose implementations[] names it.
  Invariant 2: every implementations[] record, whatever its card's status,
  resolves to a catalog entry and a committed no-magic file at
  {tier}/{script_slug}.py.
  Invariant 3: every catalog paper_slug names a card that references the
  script back. Enforced unconditionally; VERSION must be a valid MAJOR.MINOR.PATCH.

  Media: a linked record names a committed scene (valid Python) and GIF preview
  (valid header, nonzero size) in no-magic-viz; an omitted record is allowed
  only for catalog teaching_kind `comparison`. This is not render, playback or
  conceptual-fidelity certification.

  Generated files: committed INDEX.md and data/papers.json must equal the
  bytes scripts/generate_index.py renders from the committed cards
  (index-fresh, metadata-json-fresh).

Cohorts:
  published (default)  each selected commit must be an ancestor of the public
                       repository's main, resolved by unauthenticated read-only
                       `git ls-remote` and checked against locally present
                       history. The validator never fetches.
  candidate            all three --*-revision arguments are required and must
                       equal the worktree HEADs. No publication claim is made.

On success a schema-version-1 JSON receipt is written to --receipt (outside
the checked repositories) or printed on stdout; diagnostics go to stderr.

Usage:
    python scripts/validate_invariants.py --catalog ../no-magic/docs/catalog.json \\
        --core ../no-magic --viz ../no-magic-viz --papers papers \\
        --require-paper-slug yes --cohort candidate \\
        --core-revision SHA --papers-revision SHA --viz-revision SHA --receipt FILE

Exit codes:
    0  the whole cohort is valid (receipt emitted)
    1  one or more checks failed (details on stderr; no receipt)
    2  bad arguments or missing input directories
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import generate_index
from generate_index import Card, ValidationError, safe_relative_path

RECEIPT_SCHEMA_VERSION = 1
GITHUB_ORG = "no-magic-ai"
REPOSITORIES = ("no-magic", "no-magic-papers", "no-magic-viz")
PUBLIC_MAIN_REF = "refs/heads/main"
PROVIDER_TIMEOUT_SECONDS = 60
VERSION_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\n?$")
FULL_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
CATALOG_STRING_FIELDS = ("tier", "name", "paper_slug", "teaching_kind")
INVARIANTS = (
    "repository-identity",
    "revision-binding",
    "committed-input-binding",
    "validator-authority-committed",
    "frontmatter-schema",
    "card-body-sections",
    "dependency-links",
    "lesson-lifecycle",
    "implementation-path-shape",
    "duplicate-ownership",
    "version-valid",
    "catalog-shape",
    "invariant-1-catalog-script-owned",
    "invariant-2-implementation-resolves",
    "invariant-3-paper-slug-backref",
    "media-declaration",
    "media-omission-comparison-only",
    "media-assets",
    "index-fresh",
    "metadata-json-fresh",
)


def public_url(name: str) -> str:
    return f"https://github.com/{GITHUB_ORG}/{name}.git"


class CohortError(Exception):
    """A cohort-level precondition (identity, revision, publication) failed."""


def git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise CohortError(f"{root}: git {' '.join(args)} failed: {detail}")
    return proc.stdout


def blob_oid(data: bytes, object_format: str) -> str:
    header = f"blob {len(data)}\0".encode()
    digest = hashlib.sha256() if object_format == "sha256" else hashlib.sha1()
    digest.update(header + data)
    return digest.hexdigest()


def normalized_remote(url: str) -> str:
    """Reduce a GitHub remote URL to 'org/name' (lowercase); '' when not GitHub."""
    value = url.strip()
    for prefix in ("https://github.com/", "ssh://git@github.com/", "git@github.com:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
            break
    else:
        return ""
    return value.rstrip("/").removesuffix(".git").lower()


@dataclass
class Repository:
    """One selected repository: its identity, HEAD and the committed inputs read."""

    name: str
    root: Path
    commit: str = ""
    tree: str = ""
    object_format: str = "sha1"
    entries: dict[str, tuple[str, str]] = field(default_factory=dict)
    inputs: dict[str, tuple[str, str]] = field(default_factory=dict)
    publication: dict[str, str] = field(default_factory=dict)

    def open(self) -> None:
        if not self.root.is_dir():
            raise CohortError(f"{self.name} root {self.root} is not a directory")
        toplevel = Path(
            os.fsdecode(git(self.root, "rev-parse", "--show-toplevel").strip())
        )
        if toplevel.resolve() != self.root.resolve():
            raise CohortError(
                f"{self.name} root {self.root} is not its Git worktree top level ({toplevel})"
            )
        origin = git(self.root, "config", "--get", "remote.origin.url").decode().strip()
        if normalized_remote(origin) != f"{GITHUB_ORG}/{self.name}":
            raise CohortError(
                f"{self.root} origin {origin!r} is not {GITHUB_ORG}/{self.name}"
            )
        self.commit = (
            git(self.root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        )
        self.tree = (
            git(self.root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
        )
        self.object_format = (
            git(self.root, "rev-parse", "--show-object-format").decode().strip()
        )
        listing = git(self.root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
        for record in listing.split(b"\0"):
            if not record:
                continue
            meta, _, path = record.partition(b"\t")
            mode, kind, oid = meta.decode().split(" ")
            self.entries[os.fsdecode(path)] = (mode, f"{kind}:{oid}")

    def read(self, relative: str) -> bytes:
        """Return working-tree bytes of relative, which must equal its committed blob."""
        unsafe = safe_relative_path(relative)
        if unsafe:
            raise ValueError(f"{self.name}: {unsafe}")
        current = self.root
        for part in relative.split("/"):
            current = current / part
            if current.is_symlink():
                raise ValueError(
                    f"{self.name}:{relative} has a symlink component {current.name!r}"
                )
        if not current.is_file():
            raise ValueError(f"{self.name}:{relative} is not a regular file")
        entry = self.entries.get(relative)
        if (
            entry is None
            or entry[0] not in {"100644", "100755"}
            or not entry[1].startswith("blob:")
        ):
            raise ValueError(
                f"{self.name}:{relative} is not a committed regular file at {self.commit}"
            )
        data = current.read_bytes()
        oid = blob_oid(data, self.object_format)
        if entry[1] != f"blob:{oid}":
            raise ValueError(
                f"{self.name}:{relative} differs from its committed blob at {self.commit}"
            )
        self.inputs[relative] = (hashlib.sha256(data).hexdigest(), oid)
        return data

    def committed_names(self, directory: str, suffix: str) -> set[str]:
        prefix = f"{directory}/"
        return {
            path[len(prefix) :]
            for path in self.entries
            if path.startswith(prefix)
            and "/" not in path[len(prefix) :]
            and path.endswith(suffix)
        }


def provider_main(url: str) -> str:
    """Resolve refs/heads/main of a public repository with no credentials or config."""
    with tempfile.TemporaryDirectory(prefix="no-magic-provider-") as home:
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": home,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "",
            "SSH_ASKPASS": "",
            "LC_ALL": "C",
        }
        try:
            proc = subprocess.run(
                ["git", "ls-remote", "--exit-code", url, PUBLIC_MAIN_REF],
                cwd=home,
                env=env,
                capture_output=True,
                timeout=PROVIDER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CohortError(
                f"{url}: ls-remote timed out after {PROVIDER_TIMEOUT_SECONDS}s"
            ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise CohortError(
            f"{url}: cannot resolve {PUBLIC_MAIN_REF}: {detail or 'no such ref'}"
        )
    lines = [line.split("\t") for line in proc.stdout.decode().splitlines()]
    oids = [
        parts[0] for parts in lines if len(parts) == 2 and parts[1] == PUBLIC_MAIN_REF
    ]
    if len(oids) != 1 or not FULL_OID.match(oids[0]):
        raise CohortError(f"{url}: unexpected ls-remote output for {PUBLIC_MAIN_REF}")
    return oids[0]


def publication_evidence(repository: Repository, provider: str) -> dict[str, str]:
    """Prove repository.commit is published ancestry of the provider's main."""
    main = provider_main(provider)
    present = subprocess.run(
        ["git", "-C", str(repository.root), "cat-file", "-e", f"{main}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if present.returncode != 0:
        raise CohortError(
            f"{repository.name}: provider main {main} is not in local history; "
            f"fetch it before validating (the validator never fetches)"
        )
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(repository.root),
            "merge-base",
            "--is-ancestor",
            repository.commit,
            main,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestry.returncode == 1:
        raise CohortError(
            f"{repository.name}: {repository.commit} is not published on {provider} main {main}"
        )
    if ancestry.returncode != 0:
        raise CohortError(
            f"{repository.name}: ancestry check failed: {ancestry.stderr.decode(errors='replace').strip()}"
        )
    return {
        "status": "published",
        "provider": provider,
        "ref": PUBLIC_MAIN_REF,
        "main_commit": main,
        "evidence": "selected commit is an ancestor of provider main in local history",
    }


@dataclass
class Findings:
    errors: list[str] = field(default_factory=list)
    omissions: list[dict[str, str]] = field(default_factory=list)
    linked: int = 0


def check_version(core: Repository, findings: Findings) -> None:
    try:
        raw = core.read("VERSION")
    except ValueError as exc:
        findings.errors.append(f"version-valid: {exc}")
        return
    if not VERSION_PATTERN.match(raw.decode("utf-8", errors="replace")):
        findings.errors.append(
            f"version-valid: no-magic VERSION {raw!r} is not MAJOR.MINOR.PATCH"
        )


def load_catalog(core: Repository, findings: Findings) -> dict[str, dict[str, str]]:
    try:
        entries = json.loads(core.read("docs/catalog.json").decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        findings.errors.append(f"catalog-shape: {exc}")
        return {}
    if not isinstance(entries, list):
        findings.errors.append("catalog-shape: catalog.json must be a JSON array")
        return {}
    catalog: dict[str, dict[str, str]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry.get(key)
            for key in CATALOG_STRING_FIELDS
        ):
            findings.errors.append(
                f"catalog-shape: entry {position} needs non-empty string {', '.join(CATALOG_STRING_FIELDS)}"
            )
            continue
        name = entry["name"]
        if name in catalog:
            findings.errors.append(f"catalog-shape: duplicate catalog name {name!r}")
            continue
        catalog[name] = {key: entry[key] for key in CATALOG_STRING_FIELDS}
    return catalog


def check_ownership(
    cards: tuple[Card, ...],
    catalog: dict[str, dict[str, str]],
    core: Repository,
    findings: Findings,
) -> None:
    implemented: dict[str, list[str]] = {}
    references: dict[str, set[str]] = {}
    slugs = {card.slug for card in cards}
    for card in cards:
        for impl in card.implementations:
            references.setdefault(card.slug, set()).add(impl.script_slug)
            if card.status == "implemented":
                implemented.setdefault(impl.script_slug, []).append(card.slug)
            # Every declared record must resolve, whatever the card's status.
            entry = catalog.get(impl.script_slug)
            if entry is None:
                findings.errors.append(
                    f"invariant 2: {card.path} references script_slug {impl.script_slug!r} not present in catalog.json"
                )
                continue
            expected = f"{entry['tier']}/{impl.script_slug}.py"
            if impl.path != expected:
                findings.errors.append(
                    f"invariant 2: {card.path} path {impl.path!r} is not the catalog location {expected!r}"
                )
                continue
            try:
                core.read(impl.path)
            except ValueError as exc:
                findings.errors.append(f"invariant 2: {card.path}: {exc}")
    for name, entry in sorted(catalog.items()):
        owners = implemented.get(name, [])
        if len(owners) != 1:
            findings.errors.append(
                f"invariant 1: catalog script {name!r} has {len(owners)} implemented paper cards {owners}; expected exactly one"
            )
        paper_slug = entry["paper_slug"]
        if paper_slug not in slugs:
            findings.errors.append(
                f"invariant 3: catalog {name!r} paper_slug={paper_slug!r} names no paper card"
            )
        elif name not in references.get(paper_slug, set()):
            findings.errors.append(
                f"invariant 3: catalog {name!r} paper_slug={paper_slug!r} but that card does not reference {name!r}"
            )


def check_media(
    cards: tuple[Card, ...],
    catalog: dict[str, dict[str, str]],
    viz: Repository,
    findings: Findings,
) -> None:
    for card in cards:
        for impl in card.implementations:
            label = f"{card.path} {impl.script_slug}"
            if impl.media_status == "omitted":
                kind = catalog.get(impl.script_slug, {}).get("teaching_kind")
                if kind != "comparison":
                    findings.errors.append(
                        f"media: {label} declares omitted media but teaching_kind is {kind!r}, not 'comparison'"
                    )
                    continue
                findings.omissions.append(
                    {
                        "paper_slug": card.slug,
                        "script_slug": impl.script_slug,
                        "teaching_kind": kind,
                        "media_status": "omitted",
                        "media_note": impl.media_note or "",
                    }
                )
                continue
            if impl.scene_path is None or impl.preview_path is None:
                findings.errors.append(
                    f"media: {label} linked media lacks scene_path or preview_path"
                )
                continue
            try:
                scene = viz.read(impl.scene_path)
                preview = viz.read(impl.preview_path)
            except ValueError as exc:
                findings.errors.append(f"media: {label}: {exc}")
                continue
            try:
                ast.parse(scene.decode("utf-8"), filename=impl.scene_path)
            except (SyntaxError, UnicodeDecodeError) as exc:
                findings.errors.append(
                    f"media: {label} scene {impl.scene_path} is not valid Python: {exc}"
                )
                continue
            width = int.from_bytes(preview[6:8], "little") if len(preview) >= 10 else 0
            height = (
                int.from_bytes(preview[8:10], "little") if len(preview) >= 10 else 0
            )
            if preview[:6] not in GIF_SIGNATURES or width == 0 or height == 0:
                findings.errors.append(
                    f"media: {label} preview {impl.preview_path} is not a GIF with nonzero dimensions"
                )
                continue
            findings.linked += 1


def check_listing(papers: Repository, directory: str, findings: Findings) -> None:
    on_disk = {p.name for p in (papers.root / directory).glob("*.md")}
    committed = papers.committed_names(directory, ".md")
    for name in sorted(on_disk - committed):
        findings.errors.append(
            f"committed-input-binding: {directory}/{name} is not committed at {papers.commit}"
        )
    for name in sorted(committed - on_disk):
        findings.errors.append(
            f"committed-input-binding: committed {directory}/{name} is missing from the worktree"
        )


def check_authority(papers: Repository, findings: Findings) -> None:
    """The parser/validator that ran must be the committed authority of the selected papers revision."""
    parser_file = generate_index.__file__
    if parser_file is None:
        raise CohortError("cannot locate the imported generate_index module file")
    running = {
        "scripts/generate_index.py": Path(parser_file).resolve(),
        "scripts/validate_invariants.py": Path(__file__).resolve(),
    }
    for relative, path in running.items():
        try:
            committed = papers.read(relative)
        except ValueError as exc:
            findings.errors.append(f"validator-authority-committed: {exc}")
            continue
        if path.read_bytes() != committed:
            findings.errors.append(
                f"validator-authority-committed: running {path} differs from papers {relative} at {papers.commit}"
            )
    try:
        papers.read("SCHEMA.md")
    except ValueError as exc:
        findings.errors.append(f"validator-authority-committed: {exc}")


def validate_cohort(repos: dict[str, Repository]) -> Findings:
    core, papers, viz = (
        repos["no-magic"],
        repos["no-magic-papers"],
        repos["no-magic-viz"],
    )
    findings = Findings()
    check_authority(papers, findings)
    check_listing(papers, "papers", findings)
    check_listing(papers, "lessons", findings)
    try:
        repository = generate_index.load_repository(papers.root, papers.read)
    except ValidationError as exc:
        findings.errors.extend(f"papers: {error}" for error in exc.errors)
        return findings
    try:
        if papers.read("INDEX.md") != generate_index.render(repository.cards):
            findings.errors.append(
                "index-fresh: INDEX.md bytes differ from scripts/generate_index.py output"
            )
    except ValueError as exc:
        findings.errors.append(f"index-fresh: {exc}")
    try:
        metadata = papers.read(generate_index.METADATA_PATH)
    except ValueError as exc:
        findings.errors.append(f"metadata-json-fresh: {exc}")
    else:
        if metadata != generate_index.render_json(repository.cards):
            findings.errors.append(
                f"metadata-json-fresh: {generate_index.METADATA_PATH} bytes differ from scripts/generate_index.py --format json output"
            )
    check_version(core, findings)
    catalog = load_catalog(core, findings)
    try:
        core.read("scripts/generate_catalog.py")
    except ValueError as exc:
        findings.errors.append(f"catalog-shape: {exc}")
    check_ownership(repository.cards, catalog, core, findings)
    check_media(repository.cards, catalog, viz, findings)
    return findings


def build_receipt(
    cohort: str, repos: dict[str, Repository], findings: Findings
) -> dict[str, object]:
    invariants: list[str] = list(INVARIANTS)
    if cohort == "published":
        invariants.append("publication-ancestry")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "cohort": cohort,
        "repositories": {
            name: {
                "identity": f"{GITHUB_ORG}/{name}",
                "commit": repo.commit,
                "tree": repo.tree,
                "object_format": repo.object_format,
                "publication": repo.publication,
                "inputs": [
                    {"path": path, "sha256": digest, "blob": oid}
                    for path, (digest, oid) in sorted(repo.inputs.items())
                ],
            }
            for name, repo in repos.items()
        },
        "omissions": sorted(
            findings.omissions,
            key=lambda item: (item["paper_slug"], item["script_slug"]),
        ),
        "media": {"linked": findings.linked, "omitted": len(findings.omissions)},
        "invariants": invariants,
    }


def receipt_target(path: Path, repos: dict[str, Repository]) -> Path:
    target = path.resolve()
    for repo in repos.values():
        if target.is_relative_to(repo.root.resolve()):
            raise CohortError(
                f"--receipt {path} must be outside the checked repository {repo.root}"
            )
    return target


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate one explicit no-magic repository cohort."
    )
    parser.add_argument(
        "--catalog", type=Path, required=True, help="path to no-magic/docs/catalog.json"
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=script_root / "papers",
        help="no-magic-papers papers/ directory",
    )
    parser.add_argument(
        "--core",
        type=Path,
        help="no-magic worktree (default: sibling of no-magic-papers)",
    )
    parser.add_argument(
        "--viz",
        type=Path,
        help="no-magic-viz worktree (default: sibling of no-magic-papers)",
    )
    parser.add_argument(
        "--require-paper-slug",
        choices=("yes",),
        default="yes",
        help="SOP §7.3 invariant 3 is always enforced; 'yes' is the only accepted value",
    )
    parser.add_argument(
        "--cohort", choices=("candidate", "published"), default="published"
    )
    for name in ("core", "papers", "viz"):
        parser.add_argument(
            f"--{name}-revision",
            metavar="SHA",
            help=f"expected full commit of the {name} HEAD",
        )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="write the JSON receipt here (outside the checked repositories)",
    )
    args = parser.parse_args(argv)
    revisions = (args.core_revision, args.papers_revision, args.viz_revision)
    if args.cohort == "candidate" and None in revisions:
        parser.error(
            "--cohort candidate requires --core-revision, --papers-revision and --viz-revision"
        )
    for value in revisions:
        if value is not None and not FULL_OID.match(value):
            parser.error(f"revision {value!r} must be a full lowercase commit id")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    papers_dir = args.papers.resolve()
    papers_root = papers_dir.parent
    if not papers_dir.is_dir() or papers_dir.name != "papers":
        print(f"papers directory not found: {args.papers}", file=sys.stderr)
        return 2
    roots = {
        "no-magic": (args.core or papers_root.parent / "no-magic").resolve(),
        "no-magic-papers": papers_root,
        "no-magic-viz": (args.viz or papers_root.parent / "no-magic-viz").resolve(),
    }
    expected = {
        "no-magic": args.core_revision,
        "no-magic-papers": args.papers_revision,
        "no-magic-viz": args.viz_revision,
    }
    repos = {name: Repository(name=name, root=root) for name, root in roots.items()}
    try:
        if args.catalog.resolve() != roots["no-magic"] / "docs" / "catalog.json":
            raise CohortError(
                f"--catalog {args.catalog} is not docs/catalog.json of the no-magic root {roots['no-magic']}"
            )
        for name, repo in repos.items():
            repo.open()
            if expected[name] is not None and expected[name] != repo.commit:
                raise CohortError(
                    f"{name}: expected revision {expected[name]} but {repo.root} HEAD is {repo.commit}"
                )
        target = receipt_target(args.receipt, repos) if args.receipt else None
        findings = validate_cohort(repos)
        if findings.errors:
            print(
                f"FAIL: {len(findings.errors)} violation(s) in the {args.cohort} cohort:",
                file=sys.stderr,
            )
            for error in findings.errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        for name, repo in repos.items():
            if args.cohort == "published":
                repo.publication = publication_evidence(repo, public_url(name))
            else:
                repo.publication = {
                    "status": "not-asserted",
                    "classification": "candidate",
                }
    except CohortError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    receipt = (
        json.dumps(
            build_receipt(args.cohort, repos, findings), indent=2, sort_keys=True
        )
        + "\n"
    )
    if target is None:
        sys.stdout.write(receipt)
    else:
        target.write_text(receipt, encoding="utf-8")
    print(
        f"OK: {args.cohort} cohort valid — {len(repos['no-magic-papers'].inputs)} papers, "
        f"{len(repos['no-magic'].inputs)} core and {len(repos['no-magic-viz'].inputs)} viz inputs bound to committed blobs",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
