"""Select reviewed source files from Git, or from the exported inventory."""
from pathlib import Path, PurePosixPath
import json
import subprocess

INVENTORY = 'validation/source_inventory.json'
EXCLUDED = {'.git', '.venv', 'venv', 'artifacts', 'outputs', 'build', 'dist',
            'node_modules', '__pycache__'}
WEB_PDFS = {f'docs/assets/figures/figure-{n}.pdf' for n in (1, 2, 3, 4, 5, 9)}


def validate_path(root, name):
    rel = PurePosixPath(name)
    if (not name or rel.is_absolute() or '..' in rel.parts or '\\' in name
            or ':' in name or any(p in EXCLUDED or p.endswith('.egg-info') for p in rel.parts)
            or rel.name == '.env' or rel.name.startswith('.env.')
            or rel.suffix.lower() in {'.docx', '.doc', '.bundle', '.pyc', '.log', '.tmp'}):
        raise ValueError('Private or unsafe release path: ' + name)
    if rel.suffix.lower() == '.pdf' and not (rel.parts[0] == 'figures' or name in WEB_PDFS):
        raise ValueError('Only approved individual figure PDFs can be released: ' + name)
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or unsafe release file: ' + name)
    return path


def release_names(root):
    root = Path(root)
    if (root / '.git').exists():
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode('utf-8').split('\0')
        names = [n for n in names if n]
    else:
        names = json.loads((root / INVENTORY).read_text(encoding='utf-8'))['files']
    if len(names) != len(set(names)):
        raise ValueError('Duplicate release inventory entry')
    for name in names:
        validate_path(root, name)
    return sorted(names)


def check_inventory(root):
    names = release_names(root)
    recorded = json.loads((root / INVENTORY).read_text(encoding='utf-8'))['files']
    if names != recorded:
        raise ValueError('Refresh the source inventory after staging intended file additions/removals')
    return names
