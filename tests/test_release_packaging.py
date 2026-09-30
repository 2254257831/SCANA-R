"""Regression checks for release boundaries and encoded language content."""
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_source import package
from release_files import INVENTORY, validate_path
from check_language import inspect


class ReleasePackagingTests(unittest.TestCase):
    def test_only_listed_sources_are_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'validation').mkdir()
            (root / 'README.md').write_text('Public source', encoding='utf-8')
            (root / '.env').write_text('LOCAL_ONLY=example', encoding='utf-8')
            (root / 'private-notes.txt').write_text('Not for export', encoding='utf-8')
            (root / INVENTORY).write_text(json.dumps({'files': ['README.md', INVENTORY]}))
            first = package(root, root / 'outputs/first.zip')
            second = package(root, root / 'outputs/second.zip')
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(set(archive.namelist()), {'SCANA-R/README.md', 'SCANA-R/' + INVENTORY})

    def test_private_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['../outside', '/absolute', '.env', '.env.local', 'paper.pdf',
                         'paper.docx', 'private.bundle', 'artifacts/model.pt', 'a\\b']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_path(Path(tmp), name)

    def test_nested_archives_and_escaped_text_are_checked(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('metadata.json', json.dumps({'label': '\u4e2d\u6587'}))
            archive.writestr('page.html', '<p>&#20013;</p>')
        errors, counts = [], {'files': 0, 'archives': 0}
        inspect('example.zip', stream.getvalue(), errors, counts)
        self.assertEqual(len(errors), 2)
        self.assertEqual(counts['archives'], 1)


if __name__ == '__main__':
    unittest.main()
