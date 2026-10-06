"""validate_invariants.py binds a committed cohort, enforces cross-repo invariants and emits receipts."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

from support import GIF_BYTES, SCRIPTS, CohortTestCase, card, git, implementation

sys.path.insert(0, str(SCRIPTS))

import validate_invariants


class CandidateReceiptTests(CohortTestCase):
    def test_candidate_valid_cohort_writes_receipt_bound_to_committed_bytes(
        self,
    ) -> None:
        receipt_path = self.tmp / "receipt.json"

        result = self.cohort.validate(self.cohort.candidate_args(receipt_path))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["schema_version"], 1)
        self.assertEqual(receipt["cohort"], "candidate")
        for name, root in [
            ("no-magic", self.cohort.core),
            ("no-magic-papers", self.cohort.papers),
            ("no-magic-viz", self.cohort.viz),
        ]:
            repo = receipt["repositories"][name]
            self.assertEqual(repo["commit"], self.cohort.head(root))
            self.assertEqual(repo["tree"], git(root, "rev-parse", "HEAD^{tree}"))
            self.assertEqual(
                repo["publication"],
                {"status": "not-asserted", "classification": "candidate"},
            )
            for record in repo["inputs"]:
                data = (root / record["path"]).read_bytes()
                self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(
                    record["blob"], git(root, "rev-parse", f"HEAD:{record['path']}")
                )
        papers_inputs = [
            r["path"] for r in receipt["repositories"]["no-magic-papers"]["inputs"]
        ]
        self.assertEqual(papers_inputs, sorted(papers_inputs))
        self.assertTrue(
            {
                "INDEX.md",
                "SCHEMA.md",
                "lessons/alpha.md",
                "papers/alpha.md",
                "papers/beta.md",
            }
            <= set(papers_inputs)
        )
        core_inputs = {r["path"] for r in receipt["repositories"]["no-magic"]["inputs"]}
        self.assertEqual(
            core_inputs,
            {
                "VERSION",
                "docs/catalog.json",
                "scripts/generate_catalog.py",
                "01-foundations/microalpha.py",
                "02-alignment/beta_vs_gamma.py",
            },
        )
        viz_inputs = {
            r["path"] for r in receipt["repositories"]["no-magic-viz"]["inputs"]
        }
        self.assertEqual(
            viz_inputs, {"scenes/scene_microalpha.py", "previews/microalpha.gif"}
        )
        self.assertEqual(receipt["media"], {"linked": 1, "omitted": 1})
        self.assertEqual(
            [o["script_slug"] for o in receipt["omissions"]], ["beta_vs_gamma"]
        )

    def test_candidate_without_receipt_prints_receipt_on_stdout(self) -> None:
        result = self.cohort.validate(self.cohort.candidate_args())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["cohort"], "candidate")

    def test_candidate_missing_revision_argument_exits_usage_error(self) -> None:
        args = self.cohort.candidate_args()
        position = args.index("--viz-revision")
        del args[position : position + 2]

        result = self.cohort.validate(args)

        self.assertEqual(result.returncode, 2)
        self.assertIn("--cohort candidate requires", result.stderr)

    def test_candidate_revision_not_matching_head_is_rejected(self) -> None:
        previous = self.cohort.head(self.cohort.core)
        self.cohort.commit(self.cohort.core, "fixture: advance core")
        args = self.cohort.candidate_args()
        args[args.index("--core-revision") + 1] = previous

        self.assert_rejected(
            self.cohort.validate(args), f"expected revision {previous}"
        )

    def test_candidate_receipt_inside_checked_repository_is_rejected(self) -> None:
        target = self.cohort.papers / "receipt.json"

        result = self.cohort.validate(self.cohort.candidate_args(target))

        self.assert_rejected(result, "must be outside the checked repository")
        self.assertFalse(target.exists())

    def test_candidate_removed_auto_bypass_value_exits_usage_error(self) -> None:
        args = self.cohort.candidate_args()
        args[args.index("--require-paper-slug") + 1] = "no"

        self.assertEqual(self.cohort.validate(args).returncode, 2)


class CommittedInputTests(CohortTestCase):
    def test_uncommitted_card_edit_is_rejected_as_unbound_input(self) -> None:
        path = self.cohort.papers / "papers" / "beta.md"
        path.write_bytes(
            path.read_bytes().replace(b"Fixture text.", b"Edited text.", 1)
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "papers/beta.md differs from its committed blob",
        )

    def test_untracked_card_is_rejected_as_uncommitted(self) -> None:
        self.cohort.write(
            self.cohort.papers,
            "papers/gamma.md",
            card("gamma", [], status="summarized").encode(),
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "papers/gamma.md is not committed",
        )

    def test_catalog_outside_core_root_is_rejected(self) -> None:
        args = self.cohort.candidate_args()
        args[args.index("--catalog") + 1] = str(self.tmp / "catalog.json")

        self.assert_rejected(
            self.cohort.validate(args), "is not docs/catalog.json of the no-magic root"
        )

    def test_wrong_repository_identity_is_rejected(self) -> None:
        git(
            self.cohort.viz,
            "remote",
            "set-url",
            "origin",
            "https://github.com/someone/no-magic-viz.git",
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "is not no-magic-ai/no-magic-viz",
        )

    def test_symlinked_implementation_file_is_rejected(self) -> None:
        target = self.cohort.core / "01-foundations" / "microalpha.py"
        real = self.cohort.core / "01-foundations" / "real_alpha.py"
        target.rename(real)
        target.symlink_to(real.name)
        self.cohort.commit(self.cohort.core, "fixture: symlink implementation")

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "has a symlink component",
        )


class CrossRepoInvariantTests(CohortTestCase):
    def test_unresolved_record_on_non_implemented_card_is_rejected(self) -> None:
        baseline = self.cohort.validate(self.cohort.candidate_args())
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.commit_change(self.cohort.viz, "scenes/scene_microgamma.py", b"x = 1\n")
        self.commit_change(self.cohort.viz, "previews/microgamma.gif", GIF_BYTES)
        self.commit_change(
            self.cohort.papers,
            "papers/gamma.md",
            card(
                "gamma",
                [implementation("microgamma", "03-systems")],
                status="summarized",
            ).encode(),
            reindex=True,
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "invariant 2: papers/gamma.md references script_slug 'microgamma' "
            "not present in catalog.json",
        )

    def test_invalid_version_is_rejected_not_downgraded(self) -> None:
        self.commit_change(self.cohort.core, "VERSION", b"three\n")

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "VERSION b'three\\n' is not MAJOR.MINOR.PATCH",
        )

    def test_missing_version_is_rejected(self) -> None:
        (self.cohort.core / "VERSION").unlink()
        self.cohort.commit(self.cohort.core, "fixture: remove VERSION")

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "VERSION is not a regular file",
        )

    def test_missing_implementation_file_is_rejected(self) -> None:
        (self.cohort.core / "01-foundations" / "microalpha.py").unlink()
        self.cohort.commit(self.cohort.core, "fixture: remove implementation")

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "01-foundations/microalpha.py is not a regular file",
        )

    def test_implementation_in_wrong_tier_is_rejected(self) -> None:
        self.commit_change(
            self.cohort.papers,
            "papers/alpha.md",
            card(
                "alpha",
                [implementation("microalpha", "03-systems")],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/alpha.md",
            ).encode(),
            reindex=True,
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "is not the catalog location '01-foundations/microalpha.py'",
        )

    def test_catalog_paper_slug_mismatch_is_rejected(self) -> None:
        self.cohort.catalog[0]["paper_slug"] = "beta"
        self.commit_change(
            self.cohort.core, "docs/catalog.json", self.cohort.catalog_bytes()
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "paper_slug='beta' but that card does not reference 'microalpha'",
        )

    def test_catalog_script_without_card_is_rejected(self) -> None:
        self.cohort.catalog.append(
            {
                "tier": "03-systems",
                "name": "microomega",
                "paper_slug": "alpha",
                "teaching_kind": "train_infer",
            }
        )
        self.commit_change(
            self.cohort.core, "docs/catalog.json", self.cohort.catalog_bytes()
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "catalog script 'microomega' has 0 implemented paper cards",
        )

    def test_omitted_media_for_non_comparison_is_rejected(self) -> None:
        self.commit_change(
            self.cohort.papers,
            "papers/alpha.md",
            card(
                "alpha",
                [implementation("microalpha", linked=False)],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/alpha.md",
            ).encode(),
            reindex=True,
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "teaching_kind is 'train_infer', not 'comparison'",
        )

    def test_missing_preview_is_rejected(self) -> None:
        (self.cohort.viz / "previews" / "microalpha.gif").unlink()
        self.cohort.commit(self.cohort.viz, "fixture: remove preview")

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "previews/microalpha.gif is not a regular file",
        )

    def test_preview_with_zero_width_is_rejected(self) -> None:
        self.commit_change(
            self.cohort.viz,
            "previews/microalpha.gif",
            b"GIF89a\x00\x00\x03\x00" + b"\x00" * 8,
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "is not a GIF with nonzero dimensions",
        )

    def test_scene_with_invalid_python_is_rejected(self) -> None:
        self.commit_change(
            self.cohort.viz, "scenes/scene_microalpha.py", b"class Broken(:\n"
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()), "is not valid Python"
        )

    def test_stale_committed_index_is_rejected(self) -> None:
        index = self.cohort.papers / "INDEX.md"
        self.commit_change(
            self.cohort.papers, "INDEX.md", index.read_bytes() + b"extra\n"
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "index-fresh: INDEX.md bytes differ",
        )

    def test_committed_invalid_lesson_state_is_rejected(self) -> None:
        self.commit_change(
            self.cohort.papers,
            "papers/alpha.md",
            card(
                "alpha", [implementation("microalpha")], lesson_status="none"
            ).encode(),
        )

        self.assert_rejected(
            self.cohort.validate(self.cohort.candidate_args()),
            "lesson file exists but papers/alpha.md lesson.status is none",
        )


class PublicationProtocolTests(CohortTestCase):
    """Offline publication-protocol fixtures: a local bare repository stands in for the provider.

    These prove the acceptance/rejection boundary of publication_evidence; they
    are not a published-cohort pass against the real provider.
    """

    def provider_for(self, root: Path) -> str:
        bare = self.tmp / f"provider-{root.name}.git"
        git(self.tmp, "clone", "-q", "--bare", str(root), str(bare))
        return bare.as_uri()

    def opened(self, name: str, root: Path) -> validate_invariants.Repository:
        repo = validate_invariants.Repository(name=name, root=root)
        repo.open()
        return repo

    def test_publication_selected_commit_on_provider_main_is_accepted(self) -> None:
        provider = self.provider_for(self.cohort.core)

        evidence = validate_invariants.publication_evidence(
            self.opened("no-magic", self.cohort.core), provider
        )

        self.assertEqual(evidence["status"], "published")
        self.assertEqual(evidence["main_commit"], self.cohort.head(self.cohort.core))

    def test_publication_unpushed_commit_is_rejected(self) -> None:
        provider = self.provider_for(self.cohort.core)
        self.cohort.commit(self.cohort.core, "fixture: local only")

        with self.assertRaisesRegex(
            validate_invariants.CohortError, "is not published on"
        ):
            validate_invariants.publication_evidence(
                self.opened("no-magic", self.cohort.core), provider
            )

    def test_publication_provider_history_missing_locally_is_rejected(self) -> None:
        provider = self.provider_for(self.cohort.core)
        clone = self.tmp / "advance"
        git(self.tmp, "clone", "-q", provider, str(clone))
        git(clone, "commit", "-q", "--allow-empty", "-m", "fixture: provider advances")
        git(clone, "push", "-q", "origin", "HEAD:main")

        with self.assertRaisesRegex(
            validate_invariants.CohortError, "not in local history"
        ):
            validate_invariants.publication_evidence(
                self.opened("no-magic", self.cohort.core), provider
            )

    def test_publication_missing_provider_ref_is_rejected(self) -> None:
        empty = self.tmp / "empty.git"
        git(self.tmp, "init", "-q", "--bare", str(empty))

        with self.assertRaisesRegex(
            validate_invariants.CohortError, "cannot resolve refs/heads/main"
        ):
            validate_invariants.publication_evidence(
                self.opened("no-magic", self.cohort.core), empty.as_uri()
            )


if __name__ == "__main__":
    unittest.main()
