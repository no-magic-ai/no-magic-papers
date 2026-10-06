"""Builders for isolated three-repository cohorts used by the validator tests.

Each cohort is a temporary directory holding real Git repositories named
no-magic, no-magic-papers and no-magic-viz, with the papers repository carrying
copies of this checkout's scripts so the validator under test is the code that
runs. Mutations are committed, so a failure proves the protected invariant
rather than the committed-input binding.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_index

__all__ = [
    "SCRIPTS",
    "Cohort",
    "CohortTestCase",
    "card",
    "generate_index",
    "git",
    "implementation",
]

GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}
GIF_BYTES = (
    b"GIF89a" + (4).to_bytes(2, "little") + (3).to_bytes(2, "little") + b"\x00" * 16
)
BODY = "\n".join(
    f"{section}\n\nFixture text.\n" for section in generate_index.REQUIRED_SECTIONS
)
LESSON = "# Alpha Lesson\n\nPaper card: `papers/alpha.md`\n\n" + "\n".join(
    f"{section}\n\nFixture text.\n" for section in generate_index.LESSON_SECTIONS
)


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        env=GIT_ENV,
        capture_output=True,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


def implementation(
    script_slug: str,
    tier: str = "01-foundations",
    *,
    linked: bool = True,
    path: str | None = None,
) -> str:
    lines = [
        "  - repo: no-magic",
        f"    path: {path or f'{tier}/{script_slug}.py'}",
        f"    script_slug: {script_slug}",
        "    commit: null",
        "    release: null",
    ]
    if linked:
        lines += [
            "    media_repo: no-magic-viz",
            "    media_status: linked",
            f"    scene_path: scenes/scene_{script_slug}.py",
            f"    preview_path: previews/{script_slug}.gif",
            "    media_note: null",
        ]
    else:
        lines += [
            "    media_repo: null",
            "    media_status: omitted",
            "    scene_path: null",
            "    preview_path: null",
            '    media_note: "Comparison script without a scene or preview."',
        ]
    return "\n".join(lines)


def card(
    slug: str,
    implementations: list[str],
    *,
    status: str = "implemented",
    lesson_status: str = "none",
    lesson_path: str = "null",
    dependencies: str = "dependencies_on_other_papers: []",
) -> str:
    impl_block = (
        "implementations:\n" + "\n".join(implementations)
        if implementations
        else "implementations: []"
    )
    return (
        "---\n"
        f"slug: {slug}\n"
        f'title: "{slug.title()}: A Fixture Paper"\n'
        "authors:\n  - Ada Fixture\n"
        "venue: arXiv\nyear: 2024\n"
        'arxiv_id: "2401.00001"\ndoi: null\nurl: https://arxiv.org/abs/2401.00001\n'
        "discovered_via: maintainer\ndiscovered_date: 2026-01-01\n"
        f"status: {status}\n"
        "themes:\n  primary: architecture\n  secondary: []\n"
        "tags:\n  - fixture\n"
        "routing:\n  decision: backlog-implement\n  target_repo: no-magic\n"
        "  target_script_slug: null\n  target_path: null\n  target_tier: null\n"
        "  batch_label: fixture\n  review_date: null\n"
        f"{impl_block}\n"
        f"lesson:\n  path: {lesson_path}\n  status: {lesson_status}\n"
        f"{dependencies}\n"
        "---\n\n" + BODY
    )


class Cohort:
    """A valid committed cohort: two cards, one linked and one omitted comparison."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.core = base / "no-magic"
        self.papers = base / "no-magic-papers"
        self.viz = base / "no-magic-viz"
        self.catalog = [
            {
                "tier": "01-foundations",
                "name": "microalpha",
                "paper_slug": "alpha",
                "teaching_kind": "train_infer",
            },
            {
                "tier": "02-alignment",
                "name": "beta_vs_gamma",
                "paper_slug": "beta",
                "teaching_kind": "comparison",
            },
        ]
        self.files: dict[Path, dict[str, bytes]] = {
            self.core: {
                "VERSION": b"3.0.0\n",
                "docs/catalog.json": self.catalog_bytes(),
                "scripts/generate_catalog.py": b"# catalog producer fixture\n",
                "01-foundations/microalpha.py": b"print('alpha')\n",
                "02-alignment/beta_vs_gamma.py": b"print('beta')\n",
            },
            self.papers: {
                "papers/alpha.md": card(
                    "alpha",
                    [implementation("microalpha")],
                    lesson_status="drafted",
                    lesson_path="no-magic-papers/lessons/alpha.md",
                ).encode(),
                "papers/beta.md": card(
                    "beta",
                    [implementation("beta_vs_gamma", "02-alignment", linked=False)],
                    dependencies="dependencies_on_other_papers:\n  - slug: alpha",
                ).encode(),
                "lessons/alpha.md": LESSON.encode(),
                "SCHEMA.md": b"# Paper Card Schema\n",
                "scripts/generate_index.py": (
                    SCRIPTS / "generate_index.py"
                ).read_bytes(),
                "scripts/validate_invariants.py": (
                    SCRIPTS / "validate_invariants.py"
                ).read_bytes(),
            },
            self.viz: {
                "scenes/scene_microalpha.py": b"class AlphaScene:\n    pass\n",
                "previews/microalpha.gif": GIF_BYTES,
            },
        }

    def catalog_bytes(self) -> bytes:
        return (json.dumps(self.catalog, indent=2) + "\n").encode()

    def build(self) -> Cohort:
        for root, files in self.files.items():
            root.mkdir(parents=True)
            git(root, "init", "-q", "-b", "main")
            git(
                root,
                "remote",
                "add",
                "origin",
                f"https://github.com/no-magic-ai/{root.name}.git",
            )
            for relative, data in files.items():
                self.write(root, relative, data)
            if root == self.papers:
                self.reindex()
            self.commit(root, "fixture: initial cohort")
        return self

    def write(self, root: Path, relative: str, data: bytes) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def reindex(self) -> None:
        repository = generate_index.load_repository(
            self.papers, generate_index.read_file(self.papers)
        )
        (self.papers / "INDEX.md").write_bytes(generate_index.render(repository.cards))

    def commit(self, root: Path, message: str) -> str:
        git(root, "add", "-A")
        git(root, "commit", "-q", "--allow-empty", "-m", message)
        return git(root, "rev-parse", "HEAD")

    def head(self, root: Path) -> str:
        return git(root, "rev-parse", "HEAD")

    def candidate_args(self, receipt: Path | None = None) -> list[str]:
        args = [
            "--catalog",
            str(self.core / "docs" / "catalog.json"),
            "--core",
            str(self.core),
            "--viz",
            str(self.viz),
            "--papers",
            str(self.papers / "papers"),
            "--require-paper-slug",
            "yes",
            "--cohort",
            "candidate",
            "--core-revision",
            self.head(self.core),
            "--papers-revision",
            self.head(self.papers),
            "--viz-revision",
            self.head(self.viz),
        ]
        if receipt is not None:
            args += ["--receipt", str(receipt)]
        return args

    def validate(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.papers / "scripts" / "validate_invariants.py"),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def generate_index(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.papers / "scripts" / "generate_index.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )


class CohortTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="no-magic-cohort-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cohort = Cohort(self.tmp / "cohort").build()

    def commit_change(
        self, root: Path, relative: str, data: bytes, *, reindex: bool = False
    ) -> None:
        self.cohort.write(root, relative, data)
        if reindex:
            self.cohort.reindex()
        self.cohort.commit(root, f"fixture: change {relative}")

    def assert_rejected(
        self, result: subprocess.CompletedProcess[str], fragment: str
    ) -> None:
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(fragment, result.stderr)
        self.assertEqual(result.stdout, "", "no receipt may be emitted on failure")
