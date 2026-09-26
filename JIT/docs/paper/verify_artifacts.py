#!/usr/bin/env python3
"""Check manuscript, table provenance and figure consistency without simulation."""
from pathlib import Path
from urllib.parse import unquote
import argparse, csv, datetime, hashlib, json, re, subprocess, sys
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--record-visual-review',action='store_true',help='Record only after an actual human/agent view of every figure and selected PDF pages.');args=ap.parse_args()
 subprocess.run([sys.executable,str(HERE/'data/rebuild.py'),'--check'],check=True)
 ev=json.loads((HERE/'data/evidence.json').read_text());checked=0;unavailable=[]
 for key,src in ev['sources'].items():
  path=ROOT/src['original_path']
  if path.exists():
   assert sha(path)==src['original_sha256'],f'Original changed: {key}';checked+=1
  else:unavailable.append(src['original_path'])
 fm=json.loads((HERE/'figures_manifest.json').read_text())
 assert len(fm['figures'])==5
 for fig in fm['figures']:
  for path,digest in {**fig['source_sha256'],**fig['files']}.items():assert sha(HERE/path)==digest,path
  svg=HERE/'figures'/f"{fig['name']}.svg";ET.parse(svg)
 ck=(HERE/'figures/fig03_campaign_checkpoints.svg').read_text()
 for label in ['32k','64k','128k','192k','256k']:assert f'>{label}<' in ck,label
 assert 'checkpok' not in ck
 build=json.loads((HERE/'build_manifest.json').read_text())
 for name,key in [('JIT_PAPER_DRAFT.md','source_sha256'),('JIT_PAPER_DRAFT.html','html_sha256'),('JIT_PAPER_DRAFT.pdf','pdf_sha256')]:assert sha(HERE/name)==build[key],name
 assert sha(HERE/'figures_manifest.json')==build['figure_manifest_sha256']
 html=(HERE/'JIT_PAPER_DRAFT.html').read_text();assert html.count('class="equation"')==30
 assert '<merror' not in html and '<script src=' not in html
 assert html.count('<div class="paperfigure">')==5
 texts=subprocess.run(['pdftotext','-layout',str(HERE/'JIT_PAPER_DRAFT.pdf'),'-'],capture_output=True,text=True,check=True).stdout
 for t in ['348,371','128,407','9,296','320,265','参考文献','pending']:assert t in texts,t
 for n in range(1,31):assert re.search(r'\('+str(n)+r'\)',texts),f'Equation {n} missing'
 info=subprocess.run(['pdfinfo',str(HERE/'JIT_PAPER_DRAFT.pdf')],capture_output=True,text=True,check=True).stdout
 pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages>=12
 docs=[ROOT/p for p in ['README.md','PROJECT.md','AGENTS.md','JIT/README.md','JIT/AGENTS.md','JIT/docs/CURRENT_STATUS.md','JIT/docs/CODE_ORGANIZATION.md','JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md','JIT/docs/JIT_TRAINING_ROADMAP.md','JIT/docs/JIT_PAPER_OUTLINE.md','JIT/docs/VERIFICATION.md','docs/EXPERIMENT_STATE.md','docs/REPOSITORY_LAYOUT.md']]
 docs += list(HERE.glob('*.md'))
 broken=[];links=0
 for doc in docs:
  content=re.sub(r'```[\s\S]*?```','',doc.read_text())
  for target in re.findall(r'\]\(([^)]+)\)',content):
   target=target.split('#')[0].strip('<>')
   if not target or re.match(r'^[a-zA-Z]+://',target):continue
   resolved=(doc.parent/unquote(target)).resolve()
   # This report is created only after all checks succeed; permit its own link on first build.
   if resolved != HERE/'validation.json' and not resolved.exists():broken.append((str(doc.relative_to(ROOT)),target))
   links+=1
 assert not broken,broken
 record={'schema':'jit_paper_validation_v1','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'data_rebuild':'passed','original_source_hashes_checked':checked,'original_sources_unavailable':unavailable,
 'figures_checked':5,'figure_formats':['SVG','PDF','PNG'],'equations_checked':30,'pdf_pages':pages,
 'active_document_links_checked':links,'broken_active_links':broken,'manuscript_sha256':sha(HERE/'JIT_PAPER_DRAFT.md'),
 'pdf_sha256':sha(HERE/'JIT_PAPER_DRAFT.pdf'),'figure_manifest_sha256':sha(HERE/'figures_manifest.json'),
 'new_simulator_interactions':0,'new_ppo_transitions':0,
 'scientific_scope':'Source/formula audit and CPU artifact consistency, not new GPU validation or final TEST.',
 'authoring_cost':'CPU plotting and HTML/PDF rendering only; no new environment interactions. Existing experiment costs are in data/cost_breakdown.csv.'}
 if args.record_visual_review:
  record['visual_review']={'all_five_png_figures':'viewed; corrected checkpoint labels and residual-bound notation',
   'pdf_pages_sampled':[1,7,17],'checked':'Chinese glyphs, equations, tables, clipping, figure annotations; first-page title balanced after initial review'}
 (HERE/'validation.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
 assert (HERE/'validation.json').exists()
 print(json.dumps(record,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
