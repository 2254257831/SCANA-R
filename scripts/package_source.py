"""Create a downloadable source zip from tracked, non-generated inputs."""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','artifacts','outputs','build','dist','node_modules','__pycache__'}
target=ROOT/'docs/assets/scana-r-source.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or p==target or any(x in EXCLUDE or x.endswith('.egg-info') for x in rel.parts):continue
        if p.suffix in {'.pyc','.log','.tmp'}:continue
        # The author has not approved public manuscript distribution.
        # Figure PDFs are exported plots, not the manuscript.
        if p.suffix.lower() in {'.pdf','.docx'} and rel.parts[0]!='figures':continue
        z.write(p,'SCANA-R/'+rel.as_posix())
print(f'{target.name}: {target.stat().st_size:,} bytes')
