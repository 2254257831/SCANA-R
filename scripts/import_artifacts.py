"""Verify and import the separate local evidence bundle; performs no download."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, zipfile

ROOT=Path(__file__).resolve().parents[1]

def checked_path(root,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path')
    dest=(root/Path(*rel.parts)).resolve()
    if not dest.is_relative_to(root.resolve()):raise ValueError('Archive path escapes destination')
    return dest

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('archive',type=Path)
    ap.add_argument('--destination',type=Path,default=ROOT/'artifacts')
    ap.add_argument('--manifest',type=Path,default=ROOT/'configs/artifact_bundle.json')
    ap.add_argument('--profile',choices=['smoke','all'],default='all');args=ap.parse_args()
    expected=json.loads(args.manifest.read_text(encoding='utf-8'))
    h=hashlib.sha256()
    with args.archive.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected['sha256']:raise ValueError('Artifact archive SHA-256 mismatch')
    count=0
    with zipfile.ZipFile(args.archive) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        if len(manifest['files'])!=expected['files']:raise ValueError('Manifest length mismatch')
        for row in manifest['files']:
            name=row['path']
            if args.profile=='smoke' and name not in expected['smoke_files']:continue
            dest=checked_path(args.destination,name);data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Entry hash mismatch: '+name)
            if dest.exists() and dest.read_bytes()!=data:raise FileExistsError('Refusing to replace changed artifact: '+str(dest))
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);count+=1
    print(f'Verified and imported {count} files to {args.destination.resolve()}')

if __name__=='__main__':main()
