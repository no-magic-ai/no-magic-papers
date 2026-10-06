"""generate_index.py rejects invalid card paths, media declarations and lesson states."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
