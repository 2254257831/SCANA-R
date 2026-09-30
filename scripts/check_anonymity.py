"""Check common identity leaks in the review release, including nested archives.

Hosting accounts remain visible in URLs. This is a content check, not a claim
of complete anonymity. Upstream license and copyright attribution is retained.
"""
from pathlib import Path, PurePosixPath
import argparse
import io
import json
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', 'artifacts', 'outputs', 'build', 'dist', 'node_modules', '__pycache__'}
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
HOME_PATH = re.compile(r'(?:[A-Z]:[/\\]+(?:Users|Documents and Settings)[/\\]+|/(?:home|Users)/)[^/\\\s"<>]+', re.I)
NEUTRAL_NAME = 'Anonymous Contributors'
NEUTRAL_EMAIL = 'anonymous@example.invalid'


def inspect(name, data, errors, counts):
    counts['files'] += 1
    parts = PurePosixPath(name.replace('::', '/')).parts
    # Upstream attribution is not project-author identity and must be kept.
    if 'third_party' in parts:
        return
    if name.lower().endswith('.zip'):
        counts['archives'] += 1
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.comment:
                inspect(name + '::archive-comment.txt', archive.comment, errors, counts)
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if '.git' in PurePosixPath(member.filename).parts or member.filename.lower().endswith(('.bundle', '.docx')):
                    errors.append(f'{name} contains private history or manuscript material: {member.filename}')
                inspect(name + '::' + member.filename, archive.read(member), errors, counts)
        return
    if name.lower().endswith('.pdf') and 'figures' not in parts:
        errors.append(f'{name}: PDF outside the permitted figure directory')
    if name.lower().endswith(('.mp4', '.stl', '.png', '.jpg', '.jpeg', '.pdf')):
        return  # Image/PDF/video metadata is independently inspected before release.
    text = data.decode('utf-8', errors='ignore')
    emails = set(EMAIL.findall(text)) - {NEUTRAL_EMAIL}
    if emails:
        errors.append(f'{name}: non-neutral email address')
    if HOME_PATH.search(text):
        errors.append(f'{name}: user-profile path')
    if name.lower().endswith('.html'):
        if re.search(r'<meta\b[^>]*name=[\"\']author[\"\']', text, re.I):
            errors.append(f'{name}: author meta tag')
        if 'mailto:' in text.lower():
            errors.append(f'{name}: email contact link')
        if re.search(r'[\"\']@type[\"\']\s*:\s*[\"\']Person[\"\']', text):
            errors.append(f'{name}: person structured data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Check new commit identities after the explicitly retained historical baseline')
    args = parser.parse_args()
    errors, counts = [], {'files': 0, 'archives': 0, 'commits': 0}
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p in EXCLUDED or p.endswith('.egg-info') for p in rel.parts):
            continue
        if path.suffix in {'.pyc', '.log', '.tmp'}:
            continue
        inspect(rel.as_posix(), path.read_bytes(), errors, counts)
    if args.history:
        policy = json.loads((ROOT/'validation/review_identity_policy.json').read_text(encoding='utf-8'))
        baseline = policy['retained_history_through']
        if not re.fullmatch(r'[0-9a-f]{40}', baseline):
            raise ValueError('Invalid retained-history baseline')
        subprocess.run(['git', 'merge-base', '--is-ancestor', baseline, 'HEAD'], cwd=ROOT, check=True)
        lines = subprocess.check_output(['git', 'log', f'{baseline}..HEAD', '--format=%an%x09%ae%x09%cn%x09%ce'], cwd=ROOT, text=True).splitlines()
        for line in lines:
            counts['commits'] += 1
            if line.split('\t') != [NEUTRAL_NAME, NEUTRAL_EMAIL, NEUTRAL_NAME, NEUTRAL_EMAIL]:
                errors.append('New public history contains a non-neutral author or committer identity')
                break
    print(json.dumps({'checked': counts, 'passed': not errors, 'errors': errors,
                      'limitation': 'Earlier Git history is explicitly retained; hosting account and external URLs remain visible. Full anonymity is not guaranteed.'}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
