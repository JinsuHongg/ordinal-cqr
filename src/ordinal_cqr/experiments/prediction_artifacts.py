"""Shared JSONL prediction schema and metrics for conference experiments."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

REQUIRED={"sample_id","Y_ord","Z","point_prediction","prediction_set_raw","prediction_set_final"}
def _set(value: Any,k:int, field:str)->list[int]:
 if not isinstance(value,list) or value!=sorted(value) or len(value)!=len(set(value)) or any(not isinstance(x,int) or x<0 or x>=k for x in value): raise ValueError(f"invalid {field}")
 return value
def load_predictions(path:Path,k:int)->list[dict[str,Any]]:
 rows=[]; ids=set()
 for line in path.read_text().splitlines():
  r=json.loads(line); missing=REQUIRED-r.keys()
  if missing: raise ValueError(f"missing fields {missing}")
  if r["sample_id"] in ids or not isinstance(r["Y_ord"],int) or r["Y_ord"] not in range(k) or not isinstance(r["Z"],(int,float)): raise ValueError("invalid sample id or target")
  ids.add(r["sample_id"]); r["prediction_set_raw"]=_set(r["prediction_set_raw"],k,"prediction_set_raw"); r["prediction_set_final"]=_set(r["prediction_set_final"],k,"prediction_set_final"); rows.append(r)
 return rows
def _components(s:list[int])->int: return sum(i==0 or s[i-1]!=v-1 for i,v in enumerate(s))
def _max_disjoint_jump(s:list[int])->int:
 return max((right-left-1 for left,right in zip(s,s[1:])),default=0)
def evaluate(rows:list[dict[str,Any]], k:int, alpha:float, ocqr:bool=False)->dict[str,Any]:
 if not rows: raise ValueError("empty predictions")
 target=1-alpha; by=defaultdict(list)
 for r in rows: by[r["Y_ord"]].append(r)
 final=[r["prediction_set_final"] for r in rows]; raw=[r["prediction_set_raw"] for r in rows]
 def coverage(sets): return sum(r["Y_ord"] in s for r,s in zip(rows,sets))/len(rows)
 def summary(sets):
  sizes=[len(s) for s in sets]; comps=[_components(s) for s in sets]; jumps=[_max_disjoint_jump(s) for s in sets]; return {"mean_set_size":sum(sizes)/len(sizes),"median_set_size":median(sizes),"mean_ordinal_span":sum((s[-1]-s[0]+1) if s else 0 for s in sets)/len(sizes),"singleton_rate":sum(x==1 for x in sizes)/len(sizes),"full_set_rate":sum(x==k for x in sizes)/len(sizes),"contiguous_set_rate":sum(x==1 for x in comps)/len(sizes),"fragmented_set_rate":sum(x>1 for x in comps)/len(sizes),"mean_connected_components":sum(comps)/len(comps),"avg_sfs":sum(comps)/len(comps),"avg_mdj":sum(jumps)/len(jumps)}
 pc={str(c):sum(r["Y_ord"] in r["prediction_set_final"] for r in rs)/len(rs) for c,rs in by.items()}; counts={str(c):len(rs) for c,rs in by.items()}; gaps={c:v-target for c,v in pc.items()}
 out={"aggregate":{"marginal_coverage":coverage(final),**summary(final),"ccr":sum(r["Y_ord"] in s and _components(s)==1 for r,s in zip(rows,final))/len(rows),"macro_class_coverage":sum(pc.values())/len(pc),"worst_class_coverage":min(pc.values()),"worst_undercoverage":min(gaps.values()),"mean_absolute_class_coverage_deviation":sum(abs(v) for v in gaps.values())/len(gaps),"classes_below_target":sum(v<0 for v in gaps.values()),"classes_more_than_002_below_target":sum(v<-0.02 for v in gaps.values())},"per_class":[{"class_id":int(c),"count":counts[c],"coverage":pc[c],"coverage_gap":gaps[c],"mean_set_size":sum(len(r["prediction_set_final"]) for r in by[int(c)])/counts[c]} for c in sorted(pc,key=int)]}
 if ocqr:
  fallback=[list(range(k)) if r.get("fallback_activated",not s) else s for r,s in zip(rows,raw)]; out["aggregate"].update({"raw_marginal_coverage":coverage(raw),"final_marginal_coverage":coverage(final),"raw_mean_set_size":sum(map(len,raw))/len(rows),"final_mean_set_size":sum(map(len,final))/len(rows),"raw_empty_rate":sum(not s for s in raw)/len(rows),"fragmented_raw_set_rate":sum(_components(s)>1 for s in raw)/len(rows),"fallback_activation_rate":sum(r.get("fallback_activated",not s) for r,s in zip(rows,raw))/len(rows),"hull_activation_rate":sum(r.get("hull_activated",len(f)!=len(b)) for r,f,b in zip(rows,final,fallback))/len(rows),"fallback_inflation":sum(len(f)-len(s) for f,s in zip(fallback,raw))/len(rows),"hull_inflation":sum(len(f)-len(b) for f,b in zip(final,fallback))/len(rows),"total_inflation":sum(len(f)-len(s) for f,s in zip(final,raw))/len(rows)})
 return out
