"""Build an immutable stage-expert registry from frozen policy bundles."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from dvgc.experts import StageExpertRegistry

def pairs(values): return {k:v for k,v in (x.split("=",1) for x in values)}

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--expert",action="append",required=True); p.add_argument("--entry-set",action="append",default=[]); p.add_argument("--runtime-gate",default="docs/RUNTIME_GATE.json"); p.add_argument("--output",required=True); a=p.parse_args()
 gate=json.loads(Path(a.runtime_gate).read_text());
 if gate.get("status")!="PASS": raise SystemExit("Runtime gate is not PASS")
 registry=StageExpertRegistry.build(pairs(a.expert),pairs(a.entry_set),runtime_source_fingerprint=gate["source_fingerprint"]); registry.save(a.output); print(json.dumps(registry.to_dict(),indent=2))

if __name__=="__main__": main()
