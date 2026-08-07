"""Seed-0 RetinaMNIST OCQR/LAC reference runner (frozen-manifest only)."""
from __future__ import annotations
import argparse, hashlib, json, logging, os, random, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from ordinal_cqr.experiments.prediction_artifacts import load_predictions,evaluate
HASH="9212f1c384918de800b496f93e902530534eb70adfaba3ded2a13aa0c1e2236b"; K=5; A=.1
_ACTIVE_RUN_DIR = None
_ACTIVE_STAGE = "initializing"
_ACTIVE_LOGGER = None
class DS(Dataset):
 def __init__(self,npz,rows): self.x=npz[rows[0]['source_split']+'_images'] if False else None; self.npz=npz; self.rows=rows
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; x=self.npz[r['source_split']+'_images'][r['source_index']]; return torch.tensor(x.transpose(2,0,1),dtype=torch.float32).div(255).sub(.5).div(.5),torch.tensor(r['Z']),torch.tensor(r['Y_ord']),r['sample_id']
def model(out):
 m=resnet18(weights=None);m.fc=nn.Linear(512,out);return m
def records(p): return [json.loads(x) for x in p.read_text().splitlines()]
def dump(p,x): p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def write_run_status(run_dir, status, stage, **extra):
 global _ACTIVE_STAGE
 _ACTIVE_STAGE = stage
 payload={'status':status,'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),**extra}; tmp=run_dir/'run_status.json.tmp'
 with tmp.open('w') as stream: json.dump(payload,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,run_dir/'run_status.json')
def setup_logger(run_dir):
 logger=logging.getLogger('conference_run');logger.handlers.clear();logger.setLevel(logging.INFO)
 for handler in (logging.StreamHandler(),logging.FileHandler(run_dir/'run.log')): handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
 return logger
def run_with_failure_recording(callable_, run_dir, stage, logger):
 """Execute a small orchestration boundary while preserving failure semantics."""
 try: return callable_()
 except KeyboardInterrupt:
  write_run_status(run_dir,'interrupted',stage);logger.exception('Run interrupted during stage: %s',stage);raise
 except Exception as exc:
  write_run_status(run_dir,'failed',stage,exception_type=type(exc).__name__,message=str(exc));logger.exception('Run failed during stage: %s',stage);raise
def main():
 current_stage='initializing'; logger=None
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--method',choices=['ocqr','lac','aps','oaps'],required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--overwrite',action='store_true');z=ap.parse_args(); start=time.time()
 current_stage='loading_config'; cfg=yaml.safe_load(z.config.read_text()); manifest=Path('data/manifests/conference_v0_3/retinamnist/manifest.jsonl'); data=manifest.read_bytes()
 if hashlib.sha256(data).hexdigest()!=HASH: raise RuntimeError('frozen manifest hash mismatch')
 current_stage='loading_manifest'; rows=records(manifest); splits={s:[r for r in rows if r['canonical_split']==s] for s in ('train','validation','calibration','test')}
 if {s:len(v) for s,v in splits.items()}!={'train':756,'validation':120,'calibration':324,'test':400}: raise RuntimeError('manifest split count mismatch')
 random.seed(z.seed);np.random.seed(z.seed);torch.manual_seed(z.seed);torch.use_deterministic_algorithms(True)
 if not torch.cuda.is_available(): raise RuntimeError('RetinaMNIST conference runs require CUDA, but no CUDA device is available.')
 device=torch.device('cuda');torch.cuda.manual_seed_all(z.seed)
 out=Path('outputs/conference_v0_3/retinamnist')/z.method/f'seed_{z.seed}'
 if out.exists() and not z.overwrite: raise RuntimeError(f'{out} exists; use --overwrite')
 out.mkdir(parents=True,exist_ok=True);logger=setup_logger(out);global _ACTIVE_RUN_DIR,_ACTIVE_LOGGER;_ACTIVE_RUN_DIR,_ACTIVE_LOGGER=out,logger;logger.info('run start output=%s method=%s seed=%s',out,z.method,z.seed);write_run_status(out,'started',current_stage);cfg['seed']=z.seed;(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));current_stage='building_data';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);npz=np.load('/mnt/storage/medmnist/retinamnist.npz')
 dump(out/'manifest_reference.json',{'manifest_path':str(manifest),'manifest_sha256':HASH,'dataset_version':rows[0]['dataset_version'],'dataset_contract_version':'0.2.0','split_counts':{s:len(v) for s,v in splits.items()}})
 current_stage='building_model';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);train=DataLoader(DS(npz,splits['train']),64,shuffle=True,generator=torch.Generator().manual_seed(z.seed)); val=DataLoader(DS(npz,splits['validation']),128)
 net=model(2 if z.method=='ocqr' else K).to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);best=float('inf');hist=[]; ck=out/'checkpoint.pt';current_stage='training';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage)
 for e in range(z.epochs):
  net.train();tl=0
  for x,t,y,_ in train:
   x,t,y=x.to(device),t.to(device),y.to(device)
   o=net(x); loss=torch.maximum((t[:,None]-o)*torch.tensor([.05,.95],device=device),(t[:,None]-o)*(torch.tensor([.05,.95],device=device)-1)).mean() if z.method=='ocqr' else nn.functional.cross_entropy(o,y)
   opt.zero_grad();loss.backward();opt.step();tl+=float(loss)*len(y)
  net.eval();vl=0
  with torch.no_grad():
   for x,t,y,_ in val:
    x,t,y=x.to(device),t.to(device),y.to(device)
    o=net(x);loss=torch.maximum((t[:,None]-o)*torch.tensor([.05,.95],device=device),(t[:,None]-o)*(torch.tensor([.05,.95],device=device)-1)).mean() if z.method=='ocqr' else nn.functional.cross_entropy(o,y);vl+=float(loss)*len(y)
  vl/=len(splits['validation']);hist.append({'epoch':e,'train_loss':tl/756,'validation_loss':vl})
  if vl<best: best=vl;torch.save({'model':net.state_dict(),'epoch':e,'validation_loss':vl},ck)
 current_stage='loading_checkpoint';write_run_status(out,'started',current_stage);st=torch.load(ck,weights_only=True);logger.info('checkpoint=%s epoch=%s validation_loss=%s',ck,st['epoch'],best);net.load_state_dict(st['model']);dump(out/'checkpoint_metadata.json',{'checkpoint_path':'checkpoint.pt','selected_epoch':st['epoch'],'validation_loss':best,'criterion':'validation pinball loss' if z.method=='ocqr' else 'validation cross entropy','architecture':'resnet18','optimizer':'AdamW','learning_rate':1e-4,'batch_size':64,'maximum_epochs':z.epochs,'quantiles':[.05,.95] if z.method=='ocqr' else None});dump(out/'training_history.json',hist);current_stage='calibrating';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);calibration_start=time.perf_counter()
 cal=DataLoader(DS(npz,splits['calibration']),128);net.eval()
 with torch.no_grad():
  allout=[];allt=[];ally=[]
  for x,t,y,_ in cal: x,t,y=x.to(device),t.to(device),y.to(device);allout.append(net(x));allt.append(t);ally.append(y)
 outp=torch.cat(allout);tt=torch.cat(allt);yy=torch.cat(ally);calmeta={'method':z.method,'alpha':A,'classes':[]}
 if z.method=='ocqr':
  lo,hi=torch.minimum(outp[:,0],outp[:,1]),torch.maximum(outp[:,0],outp[:,1]);scores=torch.maximum(lo-tt,tt-hi).cpu();yy=yy.cpu();qs=[]
  for c in range(K):
   s=scores[yy==c];n=len(s);r=int(np.ceil((n+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=n else float('inf');qs.append(q);calmeta['classes'].append({'class_id':c,'n_k':n,'requested_rank':r,'q_k':'+inf' if not np.isfinite(q) else q,'q_k_is_finite':bool(np.isfinite(q)),'tie_count':int((s==q).sum()) if np.isfinite(q) else 0,'score_min':float(s.min()),'score_max':float(s.max())})
 elif z.method=='lac':
  p=torch.softmax(outp,1);s=(1-p[torch.arange(len(yy),device=yy.device),yy]).cpu();r=int(np.ceil((len(s)+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=len(s) else float('inf');qs=[q];calmeta.update({'score':'1-p_true','requested_rank':r,'q_hat':'+inf' if not np.isfinite(q) else q,'tie_rule':'non-strict inclusion','prediction_rule':'include k when 1-p_k <= q'})
 elif z.method=='aps':
  p=torch.softmax(outp,1);sp,si=torch.sort(p,dim=1,descending=True);ranks=torch.empty_like(si);ranks.scatter_(1,si,torch.arange(K,device=device).expand_as(si));s=torch.cumsum(sp,1)[torch.arange(len(yy),device=device),ranks[torch.arange(len(yy),device=device),yy]].cpu();r=int(np.ceil((len(s)+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=len(s) else float('inf');qs=[q];calmeta.update({'score':'APS cumulative probability mass','requested_rank':r,'q_hat':'+inf' if not np.isfinite(q) else q,'tie_rule':'non-strict inclusion','prediction_rule':'include k when APS_score(k) <= q; top class fallback when empty'})
 else:
  p=torch.softmax(outp,1);s=p.cumsum(1)[torch.arange(len(yy),device=device),yy].cpu();r=int(np.ceil((len(s)+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=len(s) else float('inf');qs=[q];calmeta.update({'score':'OAPS ordinal cumulative probability mass through Y','requested_rank':r,'q_hat':'+inf' if not np.isfinite(q) else q,'tie_rule':'non-strict inclusion','prediction_rule':'return ordinal prefix through the first cumulative-mass boundary; top class fallback when empty'})
 calmeta['calibration_seconds']=time.perf_counter()-calibration_start;dump(out/'calibration.json',calmeta);current_stage='predicting';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);prediction_start=time.perf_counter();forward_seconds=0.0
 test=DataLoader(DS(npz,splits['test']),128); pred=[]
 with torch.no_grad():
  for x,t,y,ids in test:
   forward_start=time.perf_counter();o=net(x.to(device));forward_seconds+=time.perf_counter()-forward_start
   for j,sid in enumerate(ids):
    if z.method=='ocqr':
     l,u=sorted([float(o[j,0]),float(o[j,1])]);raw=[];intervals=[]
     for c,q in enumerate(qs):
      if np.isinf(q):raw.append(c);intervals.append(['-inf','inf']);continue
      L,U=l-q,u+q;intervals.append([L,U]); a=-np.inf if c==0 else [0.5,1.5,2.5,3.5][c-1];b=np.inf if c==4 else [0.5,1.5,2.5,3.5][c]
      if L<=U and L<b and U>=a:raw.append(c)
     fallback=not raw; base=list(range(K)) if fallback else raw;final=list(range(min(base),max(base)+1));pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(np.clip(round((l+u)/2),0,4)),'prediction_set_raw':raw,'prediction_set_final':final,'lower_quantile':l,'upper_quantile':u,'candidate_corrections':['+inf' if np.isinf(q) else q for q in qs],'candidate_intervals':intervals,'fallback_activated':fallback,'hull_activated':len(final)!=len(base)})
    elif z.method=='lac':
     p=torch.softmax(o[j],0);raw=[c for c in range(K) if 1-float(p[c])<=qs[0]];pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
    elif z.method=='aps':
     p=torch.softmax(o[j],0);sp,si=torch.sort(p,descending=True);cum=torch.cumsum(sp,0);rank=torch.empty_like(si);rank[si]=torch.arange(K,device=device);raw=sorted(c for c in range(K) if float(cum[rank[c]])<=qs[0]);raw=raw or [int(si[0])];pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
    else:
     p=torch.softmax(o[j],0);included=torch.where(torch.cumsum(p,0)<=qs[0])[0];raw=[int(torch.argmax(p))] if not len(included) else list(range(min(int(included[-1])+2,K)));pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
 prediction_seconds=time.perf_counter()-prediction_start;postprocessing_seconds=prediction_seconds-forward_seconds;current_stage='writing_predictions';write_run_status(out,'started',current_stage);(out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in pred));current_stage='computing_metrics';write_run_status(out,'started',current_stage);m=evaluate(load_predictions(out/'predictions.jsonl',K),K,A,z.method=='ocqr');m['timing']={'calibration_seconds':calmeta['calibration_seconds'],'prediction_seconds':prediction_seconds,'base_model_forward_seconds':forward_seconds,'conformal_postprocessing_seconds':postprocessing_seconds,'total_evaluation_seconds':calmeta['calibration_seconds']+prediction_seconds,'samples_per_second':len(pred)/prediction_seconds if prediction_seconds else None};dump(out/'metrics.json',m);config_hash=hashlib.sha256((out/'config.yaml').read_bytes()).hexdigest();protocol_cfg=dict(cfg);protocol_cfg.pop('seed',None);protocol_hash=hashlib.sha256(yaml.safe_dump(protocol_cfg,sort_keys=True).encode()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip();timestamp=datetime.now(timezone.utc).isoformat();dump(out/'provenance.json',{'dataset':'retinamnist','dataset_version':rows[0]['dataset_version'],'dataset_contract_version':'0.2.0','method':z.method,'method_version':'0.3.0','alpha':A,'seed':z.seed,'split_identifier':'retinamnist_official_train_stratified_train_calibration_v1','split_hash':HASH,'configuration_hash':config_hash,'protocol_hash':protocol_hash,'code_commit':commit,'checkpoint_identifier':'checkpoint.pt','training_criterion':'pinball loss' if z.method=='ocqr' else 'cross entropy','checkpoint_selection_criterion':'validation pinball loss' if z.method=='ocqr' else 'validation cross entropy','timestamp':timestamp,'runtime_seconds':time.time()-start,'hardware':{'accelerator':'cuda','device_name':torch.cuda.get_device_name(device),'pytorch_version':torch.__version__},'manifest_hash':HASH});write_run_status(out,'evaluation_complete','complete');logger.info('run complete')
if __name__=='__main__':
 try: main()
 except KeyboardInterrupt:
  if _ACTIVE_RUN_DIR is not None: write_run_status(_ACTIVE_RUN_DIR,'interrupted',_ACTIVE_STAGE);_ACTIVE_LOGGER.exception('Run interrupted during stage: %s',_ACTIVE_STAGE)
  raise
 except Exception as exc:
  if _ACTIVE_RUN_DIR is not None: write_run_status(_ACTIVE_RUN_DIR,'failed',_ACTIVE_STAGE,exception_type=type(exc).__name__,message=str(exc));_ACTIVE_LOGGER.exception('Run failed during stage: %s',_ACTIVE_STAGE)
  raise
