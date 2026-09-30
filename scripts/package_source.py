"""Build a local source ZIP from reviewed files, never from a recursive directory scan."""
from pathlib import Path
import argparse
import json
import zipfile
from release_files import INVENTORY, check_inventory, release_names

ROOT = Path(__file__).resolve().parents[1]


def package(root, target):
    names = check_inventory(root)
    if target.resolve() in {(root / n).resolve() for n in names}:
        raise ValueError('The output must not overwrite a release input')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = zipfile.ZipInfo('SCANA-R/' + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (root / name).read_bytes(), compresslevel=9)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refresh-inventory', action='store_true',
                    help='After staging intended changes, record the reviewed Git file list')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/scana-r-source.zip')
    args = ap.parse_args()
    if args.refresh_inventory:
        if not (ROOT / '.git').exists():
            raise ValueError('Inventory updates require a Git checkout')
        names = sorted(set(release_names(ROOT) + [INVENTORY]))
        (ROOT / INVENTORY).write_text(json.dumps({'files': names}, indent=2) + '\n', encoding='utf-8')
        print(f'Recorded {len(names)} source paths. Stage the updated inventory before packaging.')
        return
    target = package(ROOT, args.output)
    print(f'{target.name}: {target.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
