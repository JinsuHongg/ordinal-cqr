"""Freeze the canonical RetinaMNIST conference split manifest."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

THRESHOLDS = (0.5, 1.5, 2.5, 3.5)
DATASET_VERSION = "retinamnist.npz:254915f5f0a2074665c4676356824cf4ef4a3bcab233894b4bafcaf48962bd69"

def canonical_bytes(records):
    return b"".join((json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode() for r in records)

def build(npz_path: Path, seed: int):
    source=np.load(npz_path)
    records=[]
    train_y=source["train_labels"].reshape(-1).astype(int)
    idx=np.arange(len(train_y))
    train_idx, cal_idx=train_test_split(idx, test_size=.30, stratify=train_y, random_state=seed)
    specs=(("train",train_idx,train_y,"train"),("calibration",cal_idx,train_y,"train"),
           ("validation",np.arange(len(source["val_labels"])),source["val_labels"].reshape(-1).astype(int),"val"),
           ("test",np.arange(len(source["test_labels"])),source["test_labels"].reshape(-1).astype(int),"test"))
    for split, indices, labels, source_split in specs:
        for i in sorted(map(int,indices)):
            y=int(labels[i]); records.append({"sample_id":f"retinamnist:{source_split}:{i}","source_index":i,"source_split":source_split,"canonical_split":split,"Z":float(y),"Y_ord":y,"dataset_version":DATASET_VERSION})
    records.sort(key=lambda r:r["sample_id"])
    return records

def validate(records):
    assert len({r["sample_id"] for r in records})==len(records)
    assert all(r["Y_ord"] in range(5) and r["Z"]==float(r["Y_ord"]) for r in records)
    assert len(records)==1600

def main():
 p=argparse.ArgumentParser(); p.add_argument("--npz",type=Path,default=Path("/mnt/storage/medmnist/retinamnist.npz")); p.add_argument("--output",type=Path,default=Path("data/manifests/conference_v0_3/retinamnist")); p.add_argument("--seed",type=int,default=0); p.add_argument("--overwrite",action="store_true"); a=p.parse_args()
 manifest=a.output/"manifest.jsonl"
 if manifest.exists() and not a.overwrite: raise SystemExit(f"{manifest} exists; pass --overwrite after verifying its provenance.")
 records=build(a.npz,a.seed); validate(records); payload=canonical_bytes(records); digest=hashlib.sha256(payload).hexdigest()
 a.output.mkdir(parents=True,exist_ok=True); manifest.write_bytes(payload)
 counts={s:dict(sorted(Counter(r["Y_ord"] for r in records if r["canonical_split"]==s).items())) for s in ("train","validation","calibration","test")}
 summary={"dataset":"retinamnist","dataset_contract_version":"0.2.0","dataset_version":DATASET_VERSION,"source_npz_sha256":hashlib.sha256(a.npz.read_bytes()).hexdigest(),"split_seed":a.seed,"source_subdivision":{"source_split":"train","rule":"stratified 70/30 train/calibration","reason":"independent split-conformal calibration"},"manifest_sha256":digest,"split_counts":{s:sum(v.values()) for s,v in counts.items()},"class_counts":counts,"validation":{"unique_sample_ids":True,"no_overlap":True,"target_label_bin_consistent":True,"deterministic_regeneration":hashlib.sha256(canonical_bytes(build(a.npz,a.seed))).hexdigest()==digest}}
 for name in ("manifest_metadata.json","split_summary.json"): (a.output/name).write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()
