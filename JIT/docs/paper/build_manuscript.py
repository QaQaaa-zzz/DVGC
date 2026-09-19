#!/usr/bin/env python3
"""Build an offline Chinese manuscript with native MathML and vector figures.

Authoring only. Dependencies: Markdown 3.8.2, latex2mathml 3.78.1, local Chrome.
No simulator imports, server, remote scripts, or GPU execution.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess, tempfile
import markdown
from latex2mathml.converter import convert

HERE=Path(__file__).resolve().parent
STYLE='''
@page { size:A4; margin:17mm 16mm 18mm; @bottom-center { content:counter(page); font-size:9pt; color:#69747e; } }
* {box-sizing:border-box} body {font-family:"Noto Sans CJK SC","Noto Sans CJK",sans-serif;color:#202c35;line-height:1.67;font-size:10.5pt;margin:0 auto;max-width:900px;padding:32px;background:#fff}
h1 {text-wrap:balance;font-size:24px;line-height:1.4;color:#174f70;margin-bottom:14px} h2 {font-size:18px;color:#174f70;border-bottom:1px solid #bdd0db;padding:12px 0 5px;margin-top:25px;break-after:avoid} h3 {font-size:14px;color:#2a4859;margin-top:19px;break-after:avoid}
p {margin:9px 0;orphans:3;widows:3} a {color:#0072b2;text-decoration:none;overflow-wrap:anywhere} strong {font-weight:700} blockquote {margin:12px 0;padding:10px 14px;border-left:3px solid #0072b2;background:#f3f7fa;color:#425260}
table {border-collapse:collapse;width:100%;font-size:9pt;line-height:1.5;margin:13px 0;overflow-wrap:anywhere} th {background:#eaf1f6;color:#174f70;border-bottom:1px solid #809eb1;text-align:left} th,td {padding:6px 7px;border-bottom:1px solid #dce3e7;vertical-align:top} tr {break-inside:avoid} thead {display:table-header-group}
code {font-family:"DejaVu Sans Mono",monospace;font-size:.82em;overflow-wrap:anywhere;background:#f4f5f6;padding:1px 3px} pre {background:#f5f7f8;padding:12px;border:1px solid #d8e2e8;white-space:pre-wrap;font-size:9pt;line-height:1.65;break-inside:avoid} pre code {background:none;padding:0}
.equation {display:flex;align-items:center;justify-content:center;gap:8px;position:relative;margin:16px 0;padding:7px 24px 7px 0;break-inside:avoid;font-size:11pt;overflow:visible} .equation math {max-width:100%} .eqno {position:absolute;right:0;top:50%;transform:translateY(-50%);font-family:serif;font-size:10pt;color:#43525e}
math {font-family:"STIX Two Math","Latin Modern Math",math} img {width:100%;height:auto} .paperfigure {margin:17px 0;break-inside:avoid} .paperfigure p {font-size:9pt;line-height:1.55;margin:7px 0} .paperfigure img {display:block;width:100%}
li {margin:4px 0} .toc {padding:12px 18px;background:#f6f9fb;margin-bottom:18px}.toc ul {padding-left:18px}.toc li {margin:3px 0} footer {font-size:9pt;color:#65727b;margin-top:25px;border-top:1px solid #dce3e7}
@media print { body {max-width:none;padding:0;font-size:10pt} h1 {font-size:22px} h2 {font-size:16px} h3 {font-size:12px} a {color:inherit} .toc {display:none} .equation {font-size:10.5pt} table {font-size:8.4pt} }
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--html-only',action='store_true');args=ap.parse_args()
 source=HERE/'JIT_PAPER_DRAFT.md';raw=source.read_text();equations=[]
 def math_block(m):
  latex=m.group(1).strip();tag=re.search(r'\\tag\{([^}]+)\}',latex)
  latex=re.sub(r'\\tag\{[^}]+\}','',latex).strip()
  rendered=convert(latex,display='block')
  assert 'merror' not in rendered
  eqno=f'<span class="eqno">({tag.group(1)})</span>' if tag else ''
  equations.append(f'<div class="equation">{rendered}{eqno}</div>')
  return '\n\nJITEQUATION'+str(len(equations)-1)+'TOKEN\n\n'
 prose=re.sub(r'\$\$(.*?)\$\$',math_block,raw,flags=re.S)
 body=markdown.markdown(prose,extensions=['tables','fenced_code','toc','sane_lists'],extension_configs={'toc':{'toc_depth':'2-2'}})
 for i,rendered in enumerate(equations):body=body.replace(f'<p>JITEQUATION{i}TOKEN</p>',rendered)
 assert 'JITEQUATION' not in body
 body=re.sub(r'src="figures/([^" ]+)\.png"',r'src="figures/\1.svg"',body)
 body=re.sub(r'(<p><img [\s\S]*?</p>)\s*(<p><strong>图\d+\.[\s\S]*?</p>)',r'<div class="paperfigure">\1\2</div>',body)
 html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>JIT 论文草稿 · 2026-09-12</title><style>'+STYLE+'</style></head><body>'+body+'<footer>JIT · Evidence-audited development manuscript · 2026-09-12 · Source tables and editable figures accompany this document.</footer></body></html>'
 out=HERE/'JIT_PAPER_DRAFT.html';out.write_text(html)
 record={'schema':'jit_manuscript_build_v1','equation_count':len(equations),'offline_native_mathml':True,
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'html_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'new_simulator_interactions':0,'new_ppo_transitions':0,'pdf_built':False,
 'figure_manifest_sha256':hashlib.sha256((HERE/'figures_manifest.json').read_bytes()).hexdigest()}
 if not args.html_only:
  chrome=shutil.which('google-chrome') or shutil.which('chromium')
  if not chrome:raise RuntimeError('Local Chrome/Chromium required for PDF; HTML already built.')
  pdf=HERE/'JIT_PAPER_DRAFT.pdf'
  with tempfile.TemporaryDirectory(prefix='jit-paper-chrome-') as profile:
   cmd=[chrome,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-pdf-header-footer','--run-all-compositor-stages-before-draw','--virtual-time-budget=8000',f'--user-data-dir={profile}',f'--print-to-pdf={pdf}',out.as_uri()]
   result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
   record['chrome_exit_code']=result.returncode
   if result.returncode or not pdf.exists():raise RuntimeError(result.stderr[-3000:])
  record['pdf_built']=True;record['pdf_sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
  record['pdfinfo']=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
 (HERE/'build_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ['pdfinfo']},ensure_ascii=False))
if __name__=='__main__':main()
