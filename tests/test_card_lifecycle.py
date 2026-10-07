"""generate_index.py rejects invalid cards and lessons and keeps its generated files exact."""

from __future__ import annotations

import json
import unittest

from support import CohortTestCase, card, implementation


class CardValidationTests(CohortTestCase):
    def validate_alpha(self, text: str) -> str:
        self.cohort.write(self.cohort.papers, "papers/alpha.md", text.encode())
        result = self.cohort.generate_index("--validate")
        self.assertEqual(result.returncode, 1, result.stderr)
        return result.stderr

    def test_validate_valid_fixture_cohort_succeeds(self) -> None:
        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validate_traversing_implementation_path_reports_component(self) -> None:
        stderr = self.validate_alpha(
            card(
                "alpha",
                [
                    implementation(
                        "microalpha", path="../no-magic/01-foundations/microalpha.py"
                    )
                ],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/alpha.md",
            )
        )

        self.assertIn("has an empty, '.' or '..' component", stderr)

    def test_validate_absolute_implementation_path_reports_relative_rule(self) -> None:
        stderr = self.validate_alpha(
            card(
                "alpha",
                [implementation("microalpha", path="/01-foundations/microalpha.py")],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/alpha.md",
            )
        )

        self.assertIn("must be repo-relative POSIX", stderr)

    def test_validate_wrong_repo_reports_repo(self) -> None:
        text = card(
            "alpha",
            [implementation("microalpha")],
            lesson_status="drafted",
            lesson_path="no-magic-papers/lessons/alpha.md",
        ).replace("  - repo: no-magic\n", "  - repo: no-magic-viz\n", 1)

        stderr = self.validate_alpha(text)

        self.assertIn("repo must be 'no-magic'", stderr)

    def test_validate_path_slug_mismatch_reports_expected_location(self) -> None:
        stderr = self.validate_alpha(
            card(
                "alpha",
                [implementation("microalpha", path="01-foundations/microbeta.py")],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/alpha.md",
            )
        )

        self.assertIn("must be {tier}/microalpha.py", stderr)

    def test_validate_card_slug_filename_mismatch_reports_slug(self) -> None:
        stderr = self.validate_alpha(
            card(
                "gamma",
                [implementation("microalpha")],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/alpha.md",
            )
        )

        self.assertIn("slug 'gamma' must match filename stem 'alpha'", stderr)

    def test_validate_duplicate_script_ownership_reports_both_cards(self) -> None:
        self.cohort.write(
            self.cohort.papers,
            "papers/delta.md",
            card("delta", [implementation("microalpha")]).encode(),
        )

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "microalpha is declared by more than one implementation record: ['alpha', 'delta']",
            result.stderr,
        )

    def test_validate_linked_media_with_wrong_scene_reports_expected_scene(
        self,
    ) -> None:
        text = card(
            "alpha",
            [implementation("microalpha")],
            lesson_status="drafted",
            lesson_path="no-magic-papers/lessons/alpha.md",
        ).replace(
            "scene_path: scenes/scene_microalpha.py",
            "scene_path: scenes/scene_microbeta.py",
        )

        stderr = self.validate_alpha(text)

        self.assertIn("linked scene_path must be scenes/scene_microalpha.py", stderr)

    def test_validate_omitted_media_with_scene_path_reports_contradiction(self) -> None:
        text = card(
            "alpha",
            [implementation("microalpha", linked=False)],
            lesson_status="drafted",
            lesson_path="no-magic-papers/lessons/alpha.md",
        ).replace("scene_path: null", "scene_path: scenes/scene_microalpha.py")

        stderr = self.validate_alpha(text)

        self.assertIn(
            "omitted media requires media_repo, scene_path and preview_path to be null",
            stderr,
        )

    def test_validate_unknown_media_status_reports_allowed_values(self) -> None:
        text = card(
            "alpha",
            [implementation("microalpha")],
            lesson_status="drafted",
            lesson_path="no-magic-papers/lessons/alpha.md",
        ).replace("media_status: linked", "media_status: pending")

        stderr = self.validate_alpha(text)

        self.assertIn("media_status 'pending' must be 'linked' or 'omitted'", stderr)

    def test_validate_missing_media_field_reports_exact_keys(self) -> None:
        text = card(
            "alpha",
            [implementation("microalpha")],
            lesson_status="drafted",
            lesson_path="no-magic-papers/lessons/alpha.md",
        ).replace("    media_note: null\n", "")

        stderr = self.validate_alpha(text)

        self.assertIn("keys must be exactly repo, path, script_slug", stderr)

    def test_validate_dependency_on_missing_card_reports_broken_link(self) -> None:
        self.cohort.write(
            self.cohort.papers,
            "papers/beta.md",
            card(
                "beta",
                [implementation("beta_vs_gamma", "02-alignment", linked=False)],
                dependencies="dependencies_on_other_papers:\n  - slug: omega",
            ).encode(),
        )

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn("dependency 'omega' names no paper card", result.stderr)


class LessonLifecycleTests(CohortTestCase):
    def test_validate_planned_lesson_with_existing_file_reports_state_mismatch(
        self,
    ) -> None:
        self.cohort.write(
            self.cohort.papers,
            "papers/alpha.md",
            card(
                "alpha",
                [implementation("microalpha")],
                lesson_status="planned",
                lesson_path="no-magic-papers/lessons/alpha.md",
            ).encode(),
        )

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "lesson file exists but papers/alpha.md lesson.status is planned",
            result.stderr,
        )

    def test_validate_drafted_lesson_without_file_reports_missing_lesson(self) -> None:
        (self.cohort.papers / "lessons" / "alpha.md").unlink()

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn("drafted lesson lessons/alpha.md does not exist", result.stderr)

    def test_validate_orphan_lesson_reports_orphan(self) -> None:
        self.cohort.write(self.cohort.papers, "lessons/omega.md", b"# Omega\n")

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "lessons/omega.md: orphan lesson; no paper card 'omega'", result.stderr
        )

    def test_validate_none_lesson_with_path_reports_null_path_rule(self) -> None:
        self.cohort.write(
            self.cohort.papers,
            "papers/beta.md",
            card(
                "beta",
                [implementation("beta_vs_gamma", "02-alignment", linked=False)],
                lesson_path="no-magic-papers/lessons/beta.md",
            ).encode(),
        )

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn("lesson.status none requires lesson.path null", result.stderr)

    def test_validate_drafted_lesson_wrong_path_reports_expected_path(self) -> None:
        self.cohort.write(
            self.cohort.papers,
            "papers/alpha.md",
            card(
                "alpha",
                [implementation("microalpha")],
                lesson_status="drafted",
                lesson_path="no-magic-papers/lessons/other.md",
            ).encode(),
        )

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "drafted lesson.path must be no-magic-papers/lessons/alpha.md",
            result.stderr,
        )

    def test_validate_lesson_sections_out_of_order_reports_order(self) -> None:
        lesson = (self.cohort.papers / "lessons" / "alpha.md").read_text()
        swapped = (
            lesson.replace("## Intuition", "## TEMP")
            .replace("## Exercises", "## Intuition")
            .replace("## TEMP", "## Exercises")
        )
        self.cohort.write(self.cohort.papers, "lessons/alpha.md", swapped.encode())

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "sections must be exactly ## Paper summary, ## Intuition, ## Code walkthrough, ## Exercises in order",
            result.stderr,
        )

    def test_validate_lesson_over_word_cap_reports_word_count(self) -> None:
        lesson = (
            self.cohort.papers / "lessons" / "alpha.md"
        ).read_text() + "word " * 1500
        self.cohort.write(self.cohort.papers, "lessons/alpha.md", lesson.encode())

        result = self.cohort.generate_index("--validate")

        self.assertEqual(result.returncode, 1)
        self.assertIn("lessons stay under 1500", result.stderr)


class IndexCheckTests(CohortTestCase):
    def test_check_line_ending_only_difference_fails(self) -> None:
        index = self.cohort.papers / "INDEX.md"
        index.write_bytes(index.read_bytes().replace(b"\n", b"\r\n"))

        result = self.cohort.generate_index("--check")

        self.assertEqual(result.returncode, 1)
        self.assertIn("INDEX.md is stale", result.stderr)

    def test_check_current_index_succeeds(self) -> None:
        result = self.cohort.generate_index("--check")

        self.assertEqual(result.returncode, 0, result.stderr)


class MetadataJsonTests(CohortTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metadata = self.cohort.papers / "data" / "papers.json"
        self.index = self.cohort.papers / "INDEX.md"

    def assert_rejected_without_output(self, path: str, data: str) -> None:
        metadata, index = self.metadata.read_bytes(), self.index.read_bytes()
        self.cohort.write(self.cohort.papers, path, data.encode())

        printed = self.cohort.generate_index("--format", "json")
        written = self.cohort.generate_index("--format", "json", "--write")

        for result in (printed, written):
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("card/lesson error(s)", result.stderr)
            self.assertEqual(result.stdout, "")
        self.assertEqual(self.metadata.read_bytes(), metadata)
        self.assertEqual(self.index.read_bytes(), index)

    def test_format_json_stdout_preserves_card_frontmatter_values(self) -> None:
        alpha = card(
            "alpha",
            [
                implementation("microalpha"),
                implementation("alpha_vs_beta", "02-alignment", linked=False),
            ],
            lesson_status="drafted",
            lesson_path="no-magic-papers/lessons/alpha.md",
        ).replace("  - Ada Fixture\n", "  - Christopher Ré\n  - Łukasz Kaiser\n")
        self.cohort.write(self.cohort.papers, "papers/alpha.md", alpha.encode())

        result = self.cohort.generate_index("--format", "json")

        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(
            [paper["card_path"] for paper in document["papers"]],
            ["papers/alpha.md", "papers/beta.md"],
        )
        fields = document["papers"][0]["frontmatter"]
        self.assertEqual(fields["authors"], ["Christopher Ré", "Łukasz Kaiser"])
        self.assertEqual((fields["year"], fields["doi"]), ("2024", None))
        self.assertEqual(
            fields["lesson"],
            {"path": "no-magic-papers/lessons/alpha.md", "status": "drafted"},
        )
        self.assertEqual(
            [(i["script_slug"], i["media_status"]) for i in fields["implementations"]],
            [("microalpha", "linked"), ("alpha_vs_beta", "omitted")],
        )
        self.assertEqual(
            fields["implementations"][1]["media_note"],
            "Comparison script without a scene or preview.",
        )

    def test_format_json_check_stale_bytes_fails_without_mutation(self) -> None:
        stale = self.metadata.read_bytes().replace(b"Alpha:", b"Omega:")
        self.metadata.write_bytes(stale)

        result = self.cohort.generate_index("--format", "json", "--check")

        self.assertEqual(result.returncode, 1)
        self.assertIn("data/papers.json is stale", result.stderr)
        self.assertEqual(self.metadata.read_bytes(), stale)

    def test_format_json_check_missing_file_fails_without_creating_it(self) -> None:
        self.metadata.unlink()

        result = self.cohort.generate_index("--format", "json", "--check")

        self.assertEqual(result.returncode, 1)
        self.assertIn("data/papers.json is stale", result.stderr)
        self.assertFalse(self.metadata.exists())

    def test_format_json_malformed_first_card_emits_nothing(self) -> None:
        self.assert_rejected_without_output("papers/aardvark.md", "---\nslug: x\n")

    def test_format_json_malformed_last_card_emits_nothing(self) -> None:
        self.assert_rejected_without_output("papers/zeta.md", "no frontmatter\n")

    def test_format_json_malformed_lesson_emits_nothing(self) -> None:
        self.assert_rejected_without_output("lessons/alpha.md", "# Alpha\n")

    def test_format_json_write_through_symlink_is_refused(self) -> None:
        outside = self.tmp / "outside.json"
        outside.write_bytes(b"untouched\n")
        self.metadata.unlink()
        self.metadata.symlink_to(outside)

        result = self.cohort.generate_index("--format", "json", "--write")

        self.assertEqual(result.returncode, 1)
        self.assertIn("is a symlink", result.stderr)
        self.assertEqual(outside.read_bytes(), b"untouched\n")


if __name__ == "__main__":
    unittest.main()
