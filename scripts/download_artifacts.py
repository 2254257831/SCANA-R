"""Download immutable release assets, verify SHA-256, and optionally import them."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1]
BUNDLES={'original':'artifact_bundle.json','extension':'mujoco_extension_bundle.json',
         'current-act':'independent_act_bundle.json','current-metaworld':'independent_metaworld_bundle.json'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def fetch(config,destination):
    spec=json.loads(config.read_text(encoding='utf-8'));destination.mkdir(parents=True,exist_ok=True)
    target=destination/spec['filename']
    if target.exists():
        if target.stat().st_size!=spec['bytes'] or sha(target)!=spec['sha256']:
            raise ValueError('Existing file does not match the manifest; refusing to overwrite '+str(target))
        return target
    url=spec.get('public_url')
    if not url or not url.startswith('https://github.com/2254257831/SCANA-R/releases/download/'):
        raise ValueError('Missing or unexpected public release URL')
    partial=target.with_suffix('.zip.part');h=hashlib.sha256();size=0
    # A failed attempt leaves only .part; a retry restarts that incomplete download.
    request=urllib.request.Request(url,headers={'User-Agent':'SCANA-R-artifact-downloader'})
    with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as out:
        while block:=response.read(2**20):out.write(block);h.update(block);size+=len(block)
    if size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:raise ValueError('Downloaded archive fails size or SHA-256 verification')
    partial.replace(target);return target

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',choices=[*BUNDLES,'all'],default='all')
    ap.add_argument('--downloads',type=Path,default=ROOT/'artifacts/downloads')
    ap.add_argument('--import',dest='import_files',action='store_true',help='Verify each entry and extract into artifacts; never execute checkpoint objects.')
    args=ap.parse_args()
    for key in BUNDLES if args.bundle=='all' else [args.bundle]:
        config=ROOT/'configs'/BUNDLES[key];path=fetch(config,args.downloads)
        print(f'Verified {key}: {path.name}',flush=True)
        if args.import_files:subprocess.run([sys.executable,str(ROOT/'scripts/import_artifacts.py'),str(path),'--manifest',str(config)],check=True)

if __name__=='__main__':main()
