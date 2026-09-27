#!/usr/bin/env python3
"""Verify that public declarations and the export manifest remain independent."""

import argparse
from pathlib import Path
import sys
import unittest

from check_naming import check_naming
from check_quality import inventory, read_source, violation


class NamingFixtureTests(unittest.TestCase):
    root: Path

    @classmethod
    def setUpClass(cls):
        inventory_report = {"violations": [], "excluded_generated": []}
        cls.sources = inventory(cls.root, inventory_report)
        if inventory_report["violations"]:
            raise ValueError(inventory_report["violations"])
        project = cls.root / "LMMC" if (cls.root / "LMMC").is_dir() else cls.root
        cls.manifest = project / "cmake" / "lmmc_public_symbols.txt"
        cls.original = read_source(cls.manifest)

    def evaluate(self, manifest_text):
        report = {"files": [], "violations": []}

        def fixture_source(path):
            return manifest_text if Path(path) == self.manifest else read_source(path)

        check_naming(self.root, self.sources, report, fixture_source, violation)
        return [item for item in report["violations"]
                if item["code"].startswith("naming.export_")]

    def test_missing_export_is_rejected(self):
        lines = self.original.splitlines()
        self.assertIn("lmmc_beta", lines)
        lines.remove("lmmc_beta")
        violations = self.evaluate("\n".join(lines) + "\n")
        self.assertEqual(
            [(item["code"], item["symbol"]) for item in violations],
            [("naming.export_missing", "lmmc_beta")])

    def test_undeclared_export_is_rejected(self):
        violations = self.evaluate(self.original + "lmmc_not_declared\n")
        self.assertEqual(
            [(item["code"], item["symbol"]) for item in violations],
            [("naming.export_undeclared", "lmmc_not_declared")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    NamingFixtureTests.root = args.root.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NamingFixtureTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
