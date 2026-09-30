"""Static release checks: source syntax, page resources, evidence and provenance."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import ast,hashlib,json,zipfile,argparse,re
from release_files import check_inventory
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
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,help='Also inspect a local source ZIP')
    args=ap.parse_args()
    inventory=check_inventory(ROOT)
    assert not (ROOT/'docs/assets/v46/manuscript.pdf').exists(),'Manuscript PDF must remain private'
    assert not (ROOT/'docs/assets/scana-r-source.zip').exists(),'Do not duplicate the Git tree in a tracked ZIP'
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            for name in archive.namelist():
                if name.lower().endswith(('.pdf','.docx')):
                    web_figures={f'SCANA-R/docs/assets/figures/figure-{n}.pdf' for n in [1,2,3,4,5,9]}
                    assert name.lower().endswith('.pdf') and (name.startswith('SCANA-R/figures/') or name in web_figures),f'Unapproved manuscript in source download: {name}'
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
    current=json.loads((ROOT/'docs/assets/current-videos/provenance.json').read_text())
    assert len(current['clips'])==20 and len(current['compositions'])==6
    for item in current['clips']+current['compositions']:
        path=ROOT/'docs/assets/current-videos'/item['video']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item.get('video_sha256',item.get('sha256'))
        if 'state_max_error' in item:assert item['state_max_error']==0 and item['rewards_identical']
    figures=json.loads((ROOT/'docs/assets/figures/manifest.json').read_text())
    assert {f['figure'] for f in figures}=={1,2,3,4,5,9}
    for fig in figures:
        assert fig['pages']==1 and not fig['metadata'].get('/Author')
        for kind in ['pdf','svg']:assert hashlib.sha256((ROOT/'docs/assets/figures'/fig[kind]).read_bytes()).hexdigest()==fig[kind+'_sha256']
    home=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    assert 'id="results"' in home and home.index('id="results"')<home.index('id="original-results"')
    assert 'href="study-protocol.html"' in home and 'python scripts/reproduce.py' in home
    assert 'Nothing has been uploaded' not in (ROOT/'docs/mujoco-extension.html').read_text()
    assert 'https://github.com/2254257831/SCANA-R/archive/refs/heads/main.zip' in home
    for page in pages:
        content=page.read_text(encoding='utf-8')
        if '<video' in content:assert 'assets/site.js' in content,f'Missing shared video controls: {page.name}'
    markdown_links=0
    for name in inventory:
        if not name.endswith('.md'):continue
        p=ROOT/name
        for link in re.findall(r'\[[^\]\n]*\]\(([^\s)]+)\)',p.read_text(encoding='utf-8')):
            u=urlsplit(link)
            if u.scheme or u.netloc:continue
            target=(p.parent/unquote(u.path)).resolve() if u.path else p
            assert target.is_relative_to(ROOT.resolve()) and target.exists(),f'Missing Markdown link {link} in {name}'
            if u.fragment and target in pages:assert unquote(u.fragment) in pages[target].ids,f'Missing anchor {link} in {name}'
            markdown_links+=1
    info=dict(inventory_files=len(inventory),markdown_links_verified=markdown_links,python_files=len(pyfiles),html_pages=len(pages),local_links_verified=refs,provenance_files=len(rows),video_hashes_verified=len(media['clips'])+len(media['compositions']),current_video_hashes_verified=26,vector_figure_pairs_verified=6,static_checks_passed=True,browser_visual_qa='See separate release verification; static checks do not assert browser rendering.')
    info.update(extension_video_hashes_verified=len(extension['clips']),frozen_extension_scripts_unchanged=True)
    print(json.dumps(info,indent=2))
    return info

if __name__=='__main__':main()
