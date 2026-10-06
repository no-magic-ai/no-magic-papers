"""The shared frontmatter parser accepts the documented subset and rejects the rest."""

from __future__ import annotations

import unittest

from support import generate_index

FrontmatterError = generate_index.FrontmatterError
parse = generate_index.parse_frontmatter


class ParseFrontmatterTests(unittest.TestCase):
    def test_parse_frontmatter_supported_shapes_returns_typed_values(self) -> None:
        text = (
            'title: "LLM.int8(): 8-bit Matrix Multiplication"\n'
            "doi: null\n"
            "authors:\n"
            "  - Tim Dettmers\n"
            "tags: []\n"
            "themes:\n"
            "  primary: efficient-inference\n"
            "  secondary:\n"
            "    - architecture\n"
            "implementations:\n"
            "  - repo: no-magic\n"
            "    path: 03-systems/microquant.py"
        )

        fields = parse(text)

        self.assertEqual(fields["title"], "LLM.int8(): 8-bit Matrix Multiplication")
        self.assertIsNone(fields["doi"])
        self.assertEqual(fields["authors"], ["Tim Dettmers"])
        self.assertEqual(fields["tags"], [])
        self.assertEqual(
            fields["themes"],
            {"primary": "efficient-inference", "secondary": ["architecture"]},
        )
        self.assertEqual(
            fields["implementations"],
            [{"repo": "no-magic", "path": "03-systems/microquant.py"}],
        )

    def test_parse_frontmatter_duplicate_top_level_key_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "duplicate key 'status'"):
            parse("status: implemented\nstatus: summarized")

    def test_parse_frontmatter_duplicate_record_key_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "duplicate record key 'path'"):
            parse(
                "implementations:\n  - repo: no-magic\n    path: a.py\n    path: b.py"
            )

    def test_parse_frontmatter_ambiguous_plain_scalar_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "ambiguous plain scalar"):
            parse("title: Attention: Is All You Need")

    def test_parse_frontmatter_flow_mapping_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "ambiguous plain scalar"):
            parse("lesson: {path: null, status: none}")

    def test_parse_frontmatter_tab_indentation_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "tabs"):
            parse("authors:\n\t- Ada")

    def test_parse_frontmatter_misindented_record_field_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "indent"):
            parse("implementations:\n  - repo: no-magic\n      path: a.py")

    def test_parse_frontmatter_mixed_list_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "mixes scalars and records"):
            parse("authors:\n  - Ada\n  - name: Grace")

    def test_parse_frontmatter_unterminated_quote_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "unsupported quoted scalar"):
            parse('title: "Unterminated')

    def test_parse_frontmatter_unquoted_boolean_or_null_spelling_raises(self) -> None:
        for spelling in [
            "True",
            "TRUE",
            "False",
            "FALSE",
            "Yes",
            "NO",
            "On",
            "OFF",
            "Null",
        ]:
            with (
                self.subTest(spelling=spelling),
                self.assertRaisesRegex(FrontmatterError, "ambiguous plain scalar"),
            ):
                parse(f"media_note: {spelling}")

    def test_parse_frontmatter_quoted_boolean_spelling_returns_string(self) -> None:
        fields = parse("media_note: \"True\"\ntitle: 'OFF'")

        self.assertEqual(fields, {"media_note": "True", "title": "OFF"})

    def test_parse_frontmatter_empty_block_raises(self) -> None:
        with self.assertRaisesRegex(FrontmatterError, "block has no content"):
            parse("authors:\nyear: 2024")


if __name__ == "__main__":
    unittest.main()
