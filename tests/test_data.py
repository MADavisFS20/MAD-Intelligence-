import tempfile
import unittest
from pathlib import Path

from mad import data


class CollectFilesTests(unittest.TestCase):
    def test_plain_source_file_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "A.kt").write_text("class A { fun f() = 1 }\n", encoding="utf-8")
            records, skipped = data.collect_files([d])
            self.assertEqual(len(records), 1, skipped)

    def test_replacement_character_heavy_file_is_binary(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "B.kt").write_text("\ufffd" * 50 + "x" * 10, encoding="utf-8")
            records, skipped = data.collect_files([d])
            self.assertEqual(records, [])
            self.assertEqual(skipped[0][1], "looks binary")

    def test_jsonl_output_field_is_collected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.jsonl").write_text('{"output": "fun a() {}", "license": "mit"}\n', encoding="utf-8")
            records, _ = data.collect_files([d])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["license"], "mit")


if __name__ == "__main__":
    unittest.main()
