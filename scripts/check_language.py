"""Reject CJK text in current public source, documents and nested archives.

This covers text and filenames, including XML inside PPTX and JSON escapes.
It does not OCR raster media or rewrite explicitly retained Git history.
"""
from pathlib import Path
import argparse
import html
import io
import json
import re
import zipfile
from release_files import check_inventory

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]')
TEXT = {'.md', '.txt', '.html', '.xml', '.json', '.csv', '.py', '.js', '.mjs', '.css',
        '.svg', '.yml', '.yaml', '.toml', '.tex', '.bib', '.rst', '.cfg', '.ini'}


def inspect(name, data, errors, counts, depth=0):
    counts['files'] += 1
    if CJK.search(name):
        errors.append(name + ': CJK filename')
    suffix = Path(name).suffix.lower()
    if suffix in {'.zip', '.pptx'}:
        if depth >= 5:
            raise ValueError('Archive nesting exceeds the release limit')
        counts['archives'] += 1
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.comment:
                inspect(name + '::comment.txt', archive.comment, errors, counts, depth + 1)
            for member in archive.infolist():
                if not member.is_dir():
                    inspect(name + '::' + member.filename, archive.read(member), errors, counts, depth + 1)
        return
    if suffix not in TEXT and Path(name).name not in {'LICENSE', 'NOTICE', '.gitignore'}:
        return
    text = data.decode('utf-8-sig')
    if suffix == '.json':
        text = json.dumps(json.loads(text), ensure_ascii=False)
    if suffix in {'.html', '.xml', '.svg'}:
        text = html.unescape(text)
    if CJK.search(text):
        errors.append(name + ': CJK text')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, help='Also inspect an exported source package')
    args = ap.parse_args()
    errors, counts = [], {'files': 0, 'archives': 0}
    for name in check_inventory(ROOT):
        inspect(name, (ROOT / name).read_bytes(), errors, counts)
    if args.archive:
        inspect(args.archive.name, args.archive.read_bytes(), errors, counts)
    print(json.dumps({'passed': not errors, 'checked': counts, 'errors': errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
