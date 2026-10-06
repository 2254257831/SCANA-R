"""Fetch the ten predeclared official task files; verify each LFS digest."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, hashlib, requests, time, os, argparse

ROOT=Path(__file__).resolve().parent
def session():
    s=requests.Session()
    if os.environ.get('SCANA_HTTP_PROXY'):
        s.proxies={'https':os.environ['SCANA_HTTP_PROXY'],'http':os.environ['SCANA_HTTP_PROXY']}
    return s

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def fetch(item,commit):
    s=session(); target=ROOT/'data'/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
    expected=item['lfs']['oid']
    if target.exists():
        assert target.stat().st_size==item['size']
        assert digest(target)==expected,'Existing dataset LFS SHA256 mismatch'
        return {'file':item['path'],'sha256':expected,'bytes':item['size'],'existing':True}
    part=target.with_suffix('.hdf5.part'); last=0
    for attempt in range(10):
        offset=part.stat().st_size if part.exists() else 0
        try:
            url=f'https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{commit}/'+item['path']
            r=s.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},timeout=(30,120),stream=True)
            r.raise_for_status()
            if offset and r.status_code!=206:
                raise RuntimeError('server did not honor resume range')
            with part.open('ab' if offset else 'wb') as f:
                for b in r.iter_content(1024*1024):
                    f.write(b);offset+=len(b)
                    if time.time()-last>30:
                        print(target.name,round(offset/1e6,1),'/',round(item['size']/1e6,1),'MB',flush=True);last=time.time()
            if offset==item['size']:break
        except Exception as e:
            print('retry',target.name,attempt,str(e),flush=True)
            time.sleep(min(20,2+attempt*2))
    assert part.stat().st_size==item['size'], 'incomplete data download'
    h=hashlib.sha256()
    with part.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==expected,'LFS SHA256 mismatch'
    part.replace(target)
    print('verified',target.name,expected,flush=True)
    return {'file':item['path'],'sha256':expected,'bytes':item['size']}

def main(verify_only=False):
    frozen=json.loads((ROOT/'dataset_selection.json').read_text())
    commit=frozen['commit'];selected=frozen['files']
    assert commit=='f13aa24a3da8c43c7225569f28c562979fa0e35a'
    assert frozen['task_ids']==list(range(10)) and len(selected)==10
    if verify_only:
        for item in selected:
            target=ROOT/'data'/item['path']
            assert target.stat().st_size==item['size'] and digest(target)==item['lfs']['oid']
            print('Verified existing dataset',item['path'],flush=True)
        return
    with ThreadPoolExecutor(max_workers=3) as pool:
        verified=list(pool.map(lambda x:fetch(x,commit),selected))
    (ROOT/'data_manifest.json').write_text(json.dumps({'dataset_commit':commit,'files':verified},indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();main(a.verify_only)
