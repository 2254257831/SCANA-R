// Optional documentation build. The website is already prebuilt and runs offline.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const {marked}=await import(process.env.SCANA_MARKED_MODULE || 'marked');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
for(const name of ['protocol','evidence','migration','template-references','release-checklist','videos','mujoco-extension','study-protocol','reproduction','current-videos','libero-spatial']) {
  const markdown=await fs.readFile(path.join(root,'docs',name+'.md'),'utf8');
  const title=markdown.split('\n')[0].replace(/^# /,'');
  const body=marked.parse(markdown).replaceAll(/href="([^":]+)\.md(#[^"]*)?"/g,'href="$1.html$2"')
    .replaceAll('<table>', '<div class="table-scroll" role="region" aria-label="Scrollable data table" tabindex="0"><table>')
    .replaceAll('</table>', '</table></div>');
  const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${title} · SCANA-R</title><link rel="stylesheet" href="assets/style.css"><style>main{max-width:1100px;margin:40px auto;padding:0 24px}main h1{font-size:36px;margin-bottom:26px}main table{font-size:16px;margin:25px 0}main h2{margin-top:35px}li{margin-bottom:10px}main pre{background:#f3f7f8}main code{overflow-wrap:anywhere}</style><script defer src="assets/site.js"></script></head><body><header class="topbar"><a href="index.html" class="wordmark">SCANA-R</a><nav><a href="index.html#reproduce">Back to project</a></nav></header><main>${body}</main><footer>SCANA-R · Release documentation</footer></body></html>`;
  await fs.writeFile(path.join(root,'docs',name+'.html'),html);
}
const index=path.join(root,'docs/index.html');
await fs.writeFile(index,(await fs.readFile(index,'utf8')).replaceAll('href="protocol.md"','href="protocol.html"').replaceAll('href="evidence.md"','href="evidence.html"'));
console.log('Built eleven documentation pages.');
