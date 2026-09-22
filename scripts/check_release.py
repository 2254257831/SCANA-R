"""Static release checks: source syntax, page resources, evidence and provenance."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

class Page(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.ids=set();self.missing_alt=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        for key in ['href','src','poster']:
            if key in a:self.refs.append(a[key])
        if tag=='img' and not a.get('alt'):self.missing_alt+=1

def main():
    pyfiles=[]
    for folder in ['src','scripts','experiments','tests','legacy']:
        for p in (ROOT/folder).rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));pyfiles.append(p)
    pages={}
    for p in (ROOT/'docs').glob('*.html'):
        parser=Page();parser.feed(p.read_text(encoding='utf-8'));pages[p.resolve()]=parser
    refs=0
    for p,page in pages.items():
        assert page.missing_alt==0,f'Missing image alt text: {p}'
        for link in page.refs:
            u=urlsplit(link)
            if u.scheme or u.netloc:continue
            target=(p.parent/unquote(u.path)).resolve() if u.path else p
            assert target.is_file(),f'Missing {link} in {p.name}'
            assert target.is_relative_to((ROOT/'docs').resolve()),'Pages asset outside published docs directory'
            if u.fragment and target in pages:assert unquote(u.fragment) in pages[target].ids,f'Missing anchor {link}'
            refs+=1
    rows=json.loads((ROOT/'provenance/source_files.json').read_text(encoding='utf-8'))
    for row in rows:
        got=hashlib.sha256((ROOT/row['destination']).read_bytes()).hexdigest()
        assert got==row['release_sha256'],f'Update provenance for {row["destination"]}'
    media=json.loads((ROOT/'docs/assets/videos/provenance.json').read_text(encoding='utf-8'))
    for item in media['clips']+media['compositions']:
        video=ROOT/'docs/assets/videos'/item['video']
        expected=item.get('video_sha256',item.get('sha256'))
        assert hashlib.sha256(video.read_bytes()).hexdigest()==expected,f'Video hash mismatch: {video.name}'
    extension=json.loads((ROOT/'docs/assets/mujoco-extension/provenance.json').read_text(encoding='utf-8'))
    for item in extension['clips']:
        video=ROOT/'docs/assets/mujoco-extension'/item['video']
        assert hashlib.sha256(video.read_bytes()).hexdigest()==item['video_sha256'],video.name
        assert item['state_max_error']==0 and item['rewards_identical']
    frozen=json.loads((ROOT/'results/metaworld_extension_v1/protocol_frozen.json').read_text(encoding='utf-8'))
    for name,expected in frozen['code_sha256'].items():
        assert hashlib.sha256((ROOT/'experiments/metaworld_extension'/name).read_bytes()).hexdigest()==expected,name
    act=json.loads((ROOT/'results/act_extension_v1/protocol_frozen.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'experiments/scana_r/robustness_extension.py').read_bytes()).hexdigest()==act['script_sha256']
    info=dict(python_files=len(pyfiles),html_pages=len(pages),local_links_verified=refs,provenance_files=len(rows),video_hashes_verified=len(media['clips'])+len(media['compositions']),static_checks_passed=True,browser_visual_qa='Not performed: local file navigation blocked by browser security policy')
    info.update(extension_video_hashes_verified=len(extension['clips']),frozen_extension_scripts_unchanged=True)
    print(json.dumps(info,indent=2))
    return info

if __name__=='__main__':main()
