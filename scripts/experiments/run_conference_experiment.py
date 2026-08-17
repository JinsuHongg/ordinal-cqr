"""Frozen-manifest conference runner for RetinaMNIST and UTKFace."""
from __future__ import annotations
import argparse, hashlib, json, logging, os, random, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ordinal_cqr.models.backbone import ResNet18COPOC,is_unimodal_probabilities
from ordinal_cqr.experiments.prediction_artifacts import load_predictions,evaluate
from ordinal_cqr.explainability import aps_prediction_sets,oaps_entry_scores,oaps_prediction_sets
K=5; A=.1
OCQR_METHODS=('ocqr','ocqr_pooled','ocqr_no_hull','ocqr_no_fallback')
DATASETS={
 'retinamnist':{'manifest':Path('data/manifests/conference_v0_3/retinamnist/manifest.jsonl'),'hash':'9212f1c384918de800b496f93e902530534eb70adfaba3ded2a13aa0c1e2236b','counts':{'train':756,'validation':120,'calibration':324,'test':400},'thresholds':(.5,1.5,2.5,3.5),'split_identifier':'retinamnist_official_train_stratified_train_calibration_v1'},
 'utkface':{'manifest':Path('data/manifests/conference_v0_3/utkface/manifest.jsonl'),'hash':'3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83','counts':{'train':14224,'validation':2371,'calibration':4742,'test':2371},'thresholds':(20.,40.,60.,80.),'split_identifier':'sorted_filename_stratified_60_10_20_10_v1'},
}
_ACTIVE_RUN_DIR = None
_ACTIVE_STAGE = "initializing"
_ACTIVE_LOGGER = None
class RetinaDS(Dataset):
 def __init__(self,npz,rows): self.x=npz[rows[0]['source_split']+'_images'] if False else None; self.npz=npz; self.rows=rows
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; x=self.npz[r['source_split']+'_images'][r['source_index']]; return torch.tensor(x.transpose(2,0,1),dtype=torch.float32).div(255).sub(.5).div(.5),torch.tensor(r['Z']),torch.tensor(r['Y_ord']),r['sample_id']
class UTKFaceManifestDS(Dataset):
 def __init__(self,data_dir,rows,train):
  self.data_dir=Path(data_dir);self.rows=rows;self.transform=transforms.Compose(([transforms.Resize((128,128)),transforms.RandomHorizontalFlip()] if train else [transforms.Resize((128,128))])+[transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];name=r['sample_id'].split(':',1)[1]
  with Image.open(self.data_dir/name) as image: x=self.transform(image.convert('RGB'))
  return x,torch.tensor(r['Z'],dtype=torch.float32),torch.tensor(r['Y_ord']),r['sample_id']
def model(method,out):
 if method=='copoc': return ResNet18COPOC(in_channels=3,time_steps=1,num_classes=K,dropout=.5)
 m=resnet18(weights=None);m.fc=nn.Linear(512,out);return m
def forward_model(net, x, method): return net(x.unsqueeze(2) if method=='copoc' else x)
def ordinal_bin(value,thresholds):
 return next((index for index,threshold in enumerate(thresholds) if value<threshold),len(thresholds))
def candidate_bounds(class_id,thresholds):
 return (-float('inf') if class_id==0 else thresholds[class_id-1],float('inf') if class_id==len(thresholds) else thresholds[class_id])
def ocqr_variant_options(method):
 if method not in OCQR_METHODS: raise ValueError(f'not an OCQR method: {method}')
 return {'mondrian':method!='ocqr_pooled','hull':method!='ocqr_no_hull','fallback':method!='ocqr_no_fallback'}
def aps_prefix_cutoff(cumulative,q_hat):
 """Return the first APS prefix reaching ``q_hat``, with exact-mass fallback."""
 if np.isinf(q_hat): return len(cumulative)-1
 reached=torch.where(cumulative>=q_hat)[0]
 return int(reached[0]) if len(reached) else len(cumulative)-1
def aps_candidate_set(probabilities,q_hat):
 """Apply the deterministic boundary-including APS prefix rule."""
 return torch.where(aps_prediction_sets(probabilities.unsqueeze(0),q_hat)[0])[0].tolist()
def validate_manifest_rows(rows,thresholds):
 """Validate canonical target/label/bin consistency and sample uniqueness."""
 sample_ids=set()
 for index,row in enumerate(rows):
  sample_id=row.get('sample_id');z=row.get('Z');y=row.get('Y_ord')
  if sample_id in sample_ids: raise RuntimeError(f'duplicate manifest sample_id at row {index}: {sample_id!r}')
  sample_ids.add(sample_id)
  if isinstance(y,bool) or not isinstance(y,int) or not 0<=y<=len(thresholds): raise RuntimeError(f'invalid Y_ord at manifest row {index}: {y!r}')
  if isinstance(z,bool) or not isinstance(z,(int,float)) or not np.isfinite(z): raise RuntimeError(f'nonfinite or nonnumeric Z at manifest row {index}: {z!r}')
  if ordinal_bin(float(z),thresholds)!=y: raise RuntimeError(f'target-label-bin inconsistency at manifest row {index}')
def validate_config(cfg,spec):
 """Reject resolved settings that conflict with the conference v0.3 contract."""
 if cfg.get('method')!={'name':'ocqr','version':'0.3.0'}: raise RuntimeError('canonical method metadata must be ocqr v0.3.0')
 if float(cfg.get('alpha',float('nan')))!=A: raise RuntimeError('conference driver requires alpha=0.10')
 if cfg.get('data',{}).get('class_count')!=K: raise RuntimeError('conference driver requires five ordinal classes')
 if tuple(float(v) for v in cfg.get('data',{}).get('thresholds',()))!=tuple(spec['thresholds']): raise RuntimeError('configuration thresholds do not match the frozen dataset contract')
 if [float(v) for v in cfg.get('ocqr',{}).get('quantiles',())]!=[A/2,1-A/2]: raise RuntimeError('OCQR quantiles must equal alpha/2 and 1-alpha/2')
def prepare_output_directory(out,overwrite):
 """Create a fresh run directory, preserving an explicitly replaced run."""
 if out.exists():
  if not overwrite: raise RuntimeError(f'{out} exists; use --overwrite')
  relative=out.relative_to('outputs/conference_v0_3')
  stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  archive=Path('outputs/legacy/conference_v0_3')/relative.parent/f'{relative.name}_{stamp}'
  archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(archive))
 out.mkdir(parents=True,exist_ok=False)
def git_source_is_dirty():
 """Report source-tree changes while ignoring untracked runtime artifacts."""
 result=subprocess.run(['git','status','--porcelain'],check=True,capture_output=True,text=True)
 for line in result.stdout.splitlines():
  path=line[3:]
  if line.startswith('?? ') and (path.startswith(('outputs/','results/')) or path.endswith(('.log','_exit_code.txt'))): continue
  return True
 return False
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
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--method',choices=[*OCQR_METHODS,'lac','aps','oaps','copoc'],required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--checkpoint-source',type=Path);ap.add_argument('--overwrite',action='store_true');z=ap.parse_args(); start=time.time(); is_ocqr=z.method in OCQR_METHODS; ocqr_options=ocqr_variant_options(z.method) if is_ocqr else None
 if z.checkpoint_source is not None:
  if not is_ocqr: raise RuntimeError('--checkpoint-source is supported only for OCQR variants')
  if not z.checkpoint_source.is_file(): raise RuntimeError(f'checkpoint source does not exist: {z.checkpoint_source}')
 current_stage='loading_config'; cfg=yaml.safe_load(z.config.read_text()); dataset=cfg.get('experiment',{}).get('dataset')
 if dataset not in DATASETS: raise RuntimeError(f'unsupported conference dataset: {dataset!r}')
 spec=DATASETS[dataset];validate_config(cfg,spec);manifest=spec['manifest'];data=manifest.read_bytes();manifest_hash=spec['hash'];thresholds=spec['thresholds']
 if hashlib.sha256(data).hexdigest()!=manifest_hash: raise RuntimeError('frozen manifest hash mismatch')
 current_stage='loading_manifest'; rows=records(manifest);validate_manifest_rows(rows,thresholds); splits={s:[r for r in rows if r['canonical_split']==s] for s in ('train','validation','calibration','test')}
 if {s:len(v) for s,v in splits.items()}!=spec['counts']: raise RuntimeError('manifest split count mismatch')
 random.seed(z.seed);np.random.seed(z.seed);torch.manual_seed(z.seed);torch.use_deterministic_algorithms(True)
 if not torch.cuda.is_available(): raise RuntimeError('Conference runs require CUDA, but no CUDA device is available.')
 device=torch.device('cuda');torch.cuda.manual_seed_all(z.seed)
 out=Path('outputs/conference_v0_3')/dataset/z.method/f'seed_{z.seed}';prepare_output_directory(out,z.overwrite)
 logger=setup_logger(out);global _ACTIVE_RUN_DIR,_ACTIVE_LOGGER;_ACTIVE_RUN_DIR,_ACTIVE_LOGGER=out,logger;logger.info('run start output=%s dataset=%s method=%s seed=%s',out,dataset,z.method,z.seed);write_run_status(out,'started',current_stage);cfg['seed']=z.seed;(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));current_stage='building_data';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage)
 dump(out/'manifest_reference.json',{'manifest_path':str(manifest),'manifest_sha256':manifest_hash,'dataset_version':rows[0]['dataset_version'],'dataset_contract_version':cfg['experiment']['dataset_contract_version'],'split_counts':{s:len(v) for s,v in splits.items()}})
 if dataset=='retinamnist':
  npz=np.load('/mnt/storage/medmnist/retinamnist.npz');make_dataset=lambda split,train=False:RetinaDS(npz,splits[split])
 else:
  data_root=Path('/mnt/storage/data/utkface/UTKFace')
  missing=[r['sample_id'].split(':',1)[1] for r in rows if not (data_root/r['sample_id'].split(':',1)[1]).is_file()]
  if missing: raise RuntimeError(f'UTKFace manifest references {len(missing)} missing source files; first: {missing[0]}')
  make_dataset=lambda split,train=False:UTKFaceManifestDS(data_root,splits[split],train)
 current_stage='building_model';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);train=DataLoader(make_dataset('train',True),64,shuffle=True,generator=torch.Generator().manual_seed(z.seed)); val=DataLoader(make_dataset('validation'),128)
 net=model(z.method,2 if is_ocqr else K).to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);best=float('inf');hist=[]; ck=out/'checkpoint.pt'
 if z.checkpoint_source is None:
  current_stage='training';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage)
  for e in range(z.epochs):
   net.train();tl=0
   for x,t,y,_ in train:
    x,t,y=x.to(device),t.to(device),y.to(device)
    o=forward_model(net,x,z.method); loss=torch.maximum((t[:,None]-o)*torch.tensor([.05,.95],device=device),(t[:,None]-o)*(torch.tensor([.05,.95],device=device)-1)).mean() if is_ocqr else nn.functional.cross_entropy(o,y)
    opt.zero_grad();loss.backward();opt.step();tl+=float(loss)*len(y)
   net.eval();vl=0
   with torch.no_grad():
    for x,t,y,_ in val:
     x,t,y=x.to(device),t.to(device),y.to(device)
     o=forward_model(net,x,z.method);loss=torch.maximum((t[:,None]-o)*torch.tensor([.05,.95],device=device),(t[:,None]-o)*(torch.tensor([.05,.95],device=device)-1)).mean() if is_ocqr else nn.functional.cross_entropy(o,y);vl+=float(loss)*len(y)
   vl/=len(splits['validation']);hist.append({'epoch':e,'train_loss':tl/len(splits['train']),'validation_loss':vl})
   if vl<best: best=vl;torch.save({'model':net.state_dict(),'epoch':e,'validation_loss':vl},ck)
  checkpoint_path=ck
 else: checkpoint_path=z.checkpoint_source
 current_stage='loading_checkpoint';write_run_status(out,'started',current_stage);st=torch.load(checkpoint_path,weights_only=True);best=float(st['validation_loss']);logger.info('checkpoint=%s epoch=%s validation_loss=%s',checkpoint_path,st['epoch'],best);net.load_state_dict(st['model']);dump(out/'checkpoint_metadata.json',{'checkpoint_path':str(checkpoint_path),'reused_checkpoint':z.checkpoint_source is not None,'selected_epoch':st['epoch'],'validation_loss':best,'criterion':'validation pinball loss' if is_ocqr else 'validation cross entropy','architecture':'resnet18','optimizer':'AdamW','learning_rate':1e-4,'batch_size':64,'maximum_epochs':z.epochs if z.checkpoint_source is None else None,'quantiles':[.05,.95] if is_ocqr else None});dump(out/'training_history.json',hist);current_stage='calibrating';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);calibration_start=time.perf_counter()
 cal=DataLoader(make_dataset('calibration'),128);net.eval()
 with torch.no_grad():
  allout=[];allt=[];ally=[]
  for x,t,y,_ in cal: x,t,y=x.to(device),t.to(device),y.to(device);allout.append(forward_model(net,x,z.method));allt.append(t);ally.append(y)
 outp=torch.cat(allout);tt=torch.cat(allt);yy=torch.cat(ally);calmeta={'method':z.method,'alpha':A,'classes':[]}
 if not torch.isfinite(outp).all(): raise RuntimeError('model emitted nonfinite calibration outputs')
 if not torch.isfinite(tt).all(): raise RuntimeError('calibration numeric targets must be finite')
 if z.method=='copoc' and not is_unimodal_probabilities(torch.softmax(outp,1)).all(): raise RuntimeError('canonical COPOC head emitted non-unimodal calibration probabilities')
 if is_ocqr:
  lo,hi=torch.minimum(outp[:,0],outp[:,1]),torch.maximum(outp[:,0],outp[:,1]);scores=torch.maximum(lo-tt,tt-hi).cpu();yy=yy.cpu();qs=[]
  if ocqr_options['mondrian']:
   for c in range(K):
    s=scores[yy==c];n=len(s);r=int(np.ceil((n+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=n else float('inf');qs.append(q);calmeta['classes'].append({'class_id':c,'n_k':n,'requested_rank':r,'q_k':'+inf' if not np.isfinite(q) else q,'q_k_is_finite':bool(np.isfinite(q)),'tie_count':int((s==q).sum()) if np.isfinite(q) else 0,'score_min':float(s.min()) if n else None,'score_max':float(s.max()) if n else None})
  else:
   n=len(scores);r=int(np.ceil((n+1)*(1-A)));q=float(torch.kthvalue(scores,r).values) if r<=n else float('inf');qs=[q]*K;calmeta['pooled_correction']={'n':n,'requested_rank':r,'q':'+inf' if not np.isfinite(q) else q,'q_is_finite':bool(np.isfinite(q))}
 elif z.method=='lac':
  p=torch.softmax(outp,1);s=(1-p[torch.arange(len(yy),device=yy.device),yy]).cpu();r=int(np.ceil((len(s)+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=len(s) else float('inf');qs=[q];calmeta.update({'lac_method_version':'1.0.0-exact-split','score':'one_minus_true_class_probability','requested_rank':r,'q_hat':'+inf' if not np.isfinite(q) else q,'tie_rule':'non-strict inclusion','prediction_rule':'include k when 1-p_k <= q','calibration':'pooled_exact_augmented_rank'})
 elif z.method in ('aps','copoc'):
  p=torch.softmax(outp,1);sp,si=torch.sort(p,dim=1,descending=True,stable=True);ranks=torch.empty_like(si);ranks.scatter_(1,si,torch.arange(K,device=device).expand_as(si));s=torch.cumsum(sp,1)[torch.arange(len(yy),device=device),ranks[torch.arange(len(yy),device=device),yy]].cpu();r=int(np.ceil((len(s)+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=len(s) else float('inf');qs=[q];calmeta.update({'aps_method_version':'1.0.0-nonrandomized-boundary','score':'cumulative_probability_through_true_label','requested_rank':r,'q_hat':'+inf' if not np.isfinite(q) else q,'tie_rule':'stable descending probability sort, ascending class index within ties','prediction_rule':'include the smallest probability-ranked prefix whose cumulative mass reaches q_hat','calibration':'pooled_exact_augmented_rank'})
 elif z.method=='oaps':
  p=torch.softmax(outp,1);s=oaps_entry_scores(p).gather(1,yy.view(-1,1)).squeeze(1).cpu();r=int(np.ceil((len(s)+1)*(1-A)));q=float(torch.kthvalue(s,r).values) if r<=len(s) else float('inf');qs=[q];calmeta.update({'oaps_method_version':'1.0.0-lu2022-algorithm1','score':'probability mass of the greedy mode-centered interval before Y enters','requested_rank':r,'q_hat':'+inf' if not np.isfinite(q) else q,'tie_rule':'non-strict threshold inclusion; modal ties choose the lowest class; equal adjacent probabilities choose the upper/right class','prediction_rule':'start at the modal class and greedily add the higher-probability adjacent class while current interval mass <= q_hat','calibration':'pooled exact augmented order statistic'})
 calmeta['calibration_seconds']=time.perf_counter()-calibration_start;dump(out/'calibration.json',calmeta);current_stage='predicting';write_run_status(out,'started',current_stage);logger.info('stage %s',current_stage);prediction_start=time.perf_counter();forward_seconds=0.0
 test=DataLoader(make_dataset('test'),128); pred=[]
 with torch.no_grad():
  for x,t,y,ids in test:
   forward_start=time.perf_counter();o=forward_model(net,x.to(device),z.method);forward_seconds+=time.perf_counter()-forward_start
   if not torch.isfinite(o).all(): raise RuntimeError('model emitted nonfinite test outputs')
   if z.method=='copoc' and not is_unimodal_probabilities(torch.softmax(o,1)).all(): raise RuntimeError('canonical COPOC head emitted non-unimodal test probabilities')
   for j,sid in enumerate(ids):
    if is_ocqr:
     l,u=sorted([float(o[j,0]),float(o[j,1])]);raw=[];intervals=[]
     for c,q in enumerate(qs):
      if np.isinf(q):raw.append(c);intervals.append(['-inf','inf']);continue
      L,U=l-q,u+q;intervals.append([L,U]);a,b=candidate_bounds(c,thresholds)
      if L<=U and L<b and U>=a:raw.append(c)
     fallback=bool(not raw and ocqr_options['fallback']);base=list(range(K)) if fallback else raw;final=list(range(min(base),max(base)+1)) if base and ocqr_options['hull'] else base;pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':ordinal_bin((l+u)/2,thresholds),'prediction_set_raw':raw,'prediction_set_final':final,'lower_quantile':l,'upper_quantile':u,'candidate_corrections':['+inf' if np.isinf(q) else q for q in qs],'candidate_intervals':intervals,'fallback_activated':fallback,'hull_activated':bool(base and ocqr_options['hull'] and len(final)!=len(base))})
    elif z.method=='lac':
     p=torch.softmax(o[j],0);raw=[c for c in range(K) if 1-float(p[c])<=qs[0]];pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
    elif z.method=='aps':
     p=torch.softmax(o[j],0);raw=aps_candidate_set(p,qs[0]);pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
    elif z.method=='copoc':
     p=torch.softmax(o[j],0);raw=aps_candidate_set(p,qs[0]);pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
    elif z.method=='oaps':
     p=torch.softmax(o[j],0);raw=torch.where(oaps_prediction_sets(p.unsqueeze(0),qs[0])[0])[0].tolist();pred.append({'sample_id':sid,'Y_ord':int(y[j]),'Z':float(t[j]),'point_prediction':int(torch.argmax(p)),'prediction_set_raw':raw,'prediction_set_final':raw,'class_probabilities':[float(v) for v in p]})
 prediction_seconds=time.perf_counter()-prediction_start;postprocessing_seconds=prediction_seconds-forward_seconds;current_stage='writing_predictions';write_run_status(out,'started',current_stage);(out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in pred));current_stage='computing_metrics';write_run_status(out,'started',current_stage);m=evaluate(load_predictions(out/'predictions.jsonl',K),K,A,is_ocqr);m['timing']={'calibration_seconds':calmeta['calibration_seconds'],'prediction_seconds':prediction_seconds,'base_model_forward_seconds':forward_seconds,'conformal_postprocessing_seconds':postprocessing_seconds,'total_evaluation_seconds':calmeta['calibration_seconds']+prediction_seconds,'samples_per_second':len(pred)/prediction_seconds if prediction_seconds else None};dump(out/'metrics.json',m)
 config_hash=hashlib.sha256((out/'config.yaml').read_bytes()).hexdigest();protocol_cfg=dict(cfg);protocol_cfg.pop('seed',None);protocol_hash=hashlib.sha256(yaml.safe_dump(protocol_cfg,sort_keys=True).encode()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip();git_dirty=git_source_is_dirty();timestamp=datetime.now(timezone.utc).isoformat()
 lac={'lac_method_version':'1.0.0-exact-split','score':'one_minus_true_class_probability','calibration':'pooled_exact_augmented_rank','prediction_rule':'probability_superlevel_set','inclusion':'non_strict'} if z.method=='lac' else None
 aps={'aps_method_version':'1.0.0-nonrandomized-boundary','score':'cumulative_probability_through_true_label','calibration':'pooled_exact_augmented_rank','prediction_rule':'smallest_stable_probability_prefix_reaching_q','probability_tie_rule':'ascending_class_index'} if z.method=='aps' else None
 copoc={'copoc_method_version':'1.0.0-eq5-aps','model_type':'resnet18_copoc_nonparametric_eq5','phi':'abs','psi_even':'negative_abs','conformal_procedure':'aps','checkpoint_selection_metric':'validation_cross_entropy'} if z.method=='copoc' else None
 oaps={'oaps_method_version':'1.0.0-lu2022-algorithm1','set_family':'greedy_mode_centered_adjacent_expansion','calibration':'pooled_exact_augmented_rank','mode_tie_rule':'lowest_class','adjacent_tie_rule':'upper_right'} if z.method=='oaps' else None
 dump(out/'provenance.json',{'dataset':dataset,'dataset_version':rows[0]['dataset_version'],'dataset_contract_version':cfg['experiment']['dataset_contract_version'],'method':z.method,'method_version':'0.3.0','alpha':A,'seed':z.seed,'split_identifier':spec['split_identifier'],'split_hash':manifest_hash,'configuration_hash':config_hash,'protocol_hash':protocol_hash,'code_commit':commit,'git_dirty':git_dirty,'checkpoint_identifier':str(checkpoint_path),'reused_checkpoint':z.checkpoint_source is not None,'training_criterion':'pinball loss' if is_ocqr else 'cross entropy','checkpoint_selection_criterion':'validation pinball loss' if is_ocqr else 'validation cross entropy','timestamp':timestamp,'runtime_seconds':time.time()-start,'hardware':{'accelerator':'cuda','device_name':torch.cuda.get_device_name(device),'pytorch_version':torch.__version__},'manifest_hash':manifest_hash,**({'ocqr_ablation':ocqr_options} if is_ocqr else {}),**({'lac':lac} if lac else {}),**({'aps':aps} if aps else {}),**({'copoc':copoc} if copoc else {}),**({'oaps':oaps} if oaps else {})})
 write_run_status(out,'evaluation_complete','complete');logger.info('run complete')
if __name__=='__main__':
 try: main()
 except KeyboardInterrupt:
  if _ACTIVE_RUN_DIR is not None: write_run_status(_ACTIVE_RUN_DIR,'interrupted',_ACTIVE_STAGE);_ACTIVE_LOGGER.exception('Run interrupted during stage: %s',_ACTIVE_STAGE)
  raise
 except Exception as exc:
  if _ACTIVE_RUN_DIR is not None: write_run_status(_ACTIVE_RUN_DIR,'failed',_ACTIVE_STAGE,exception_type=type(exc).__name__,message=str(exc));_ACTIVE_LOGGER.exception('Run failed during stage: %s',_ACTIVE_STAGE)
  raise
