import importlib.util, json, logging
from pathlib import Path
import pytest
SPEC=importlib.util.spec_from_file_location('conference_driver','scripts/experiments/run_conference_experiment.py');driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)
def read(p): return json.loads(p.read_text())
def logger(path):
 l=logging.getLogger(str(path));l.handlers.clear();l.setLevel(logging.INFO);l.addHandler(logging.FileHandler(path/'run.log'));return l
def test_status_variants(tmp_path):
 for status,stage,extra in [('started','training',{}),('failed','calibrating',{'exception_type':'ValueError','message':'example failure'}),('interrupted','training',{}),('evaluation_complete','complete',{})]:
  driver.write_run_status(tmp_path,status,stage,**extra); value=read(tmp_path/'run_status.json');assert value['status']==status and value['stage']==stage and value['updated_at']
def test_repeated_atomic_replacement(tmp_path):
 for status in ('started','training','failed'): driver.write_run_status(tmp_path,status,'training')
 assert read(tmp_path/'run_status.json')['status']=='failed';assert not (tmp_path/'run_status.json.tmp').exists()
def test_exception_wrapper(tmp_path):
 def fail(): raise ValueError('intentional test failure')
 with pytest.raises(ValueError): driver.run_with_failure_recording(fail,tmp_path,'calibrating',logger(tmp_path))
 value=read(tmp_path/'run_status.json');assert value['status']=='failed' and value['exception_type']=='ValueError' and value['message']=='intentional test failure';assert 'Traceback' in (tmp_path/'run.log').read_text()
def test_interrupt_wrapper(tmp_path):
 def stop(): raise KeyboardInterrupt
 with pytest.raises(KeyboardInterrupt): driver.run_with_failure_recording(stop,tmp_path,'training',logger(tmp_path))
 assert read(tmp_path/'run_status.json')['status']=='interrupted'

def test_numeric_target_uses_right_bin_boundaries():
 assert [driver.ordinal_bin(value,(20.,40.,60.,80.)) for value in (19.9,20.,40.,60.,80.)]==[0,1,2,3,4]
 assert driver.candidate_bounds(0,(20.,40.,60.,80.))==(-float('inf'),20.)
 assert driver.candidate_bounds(4,(20.,40.,60.,80.))==(80.,float('inf'))

def test_ocqr_ablation_variants_change_only_the_named_component():
 assert driver.ocqr_variant_options('ocqr')=={'mondrian':True,'hull':True,'fallback':True}
 assert driver.ocqr_variant_options('ocqr_pooled')=={'mondrian':False,'hull':True,'fallback':True}
 assert driver.ocqr_variant_options('ocqr_no_hull')=={'mondrian':True,'hull':False,'fallback':True}
 assert driver.ocqr_variant_options('ocqr_no_fallback')=={'mondrian':True,'hull':True,'fallback':False}

def test_conference_copoc_pipeline_selects_canonical_head():
 from ordinal_cqr.models.backbone import ResNet18COPOC
 assert isinstance(driver.model('copoc', driver.K), ResNet18COPOC)

def test_aps_prefix_uses_full_set_when_float_sum_misses_qhat():
 import torch
 cumulative=torch.tensor([.5,.8,.95,.99999982])
 assert driver.aps_prefix_cutoff(cumulative,.99999988)==3

def test_aps_candidate_set_uses_boundary_including_prefix():
 import torch
 assert driver.aps_candidate_set(torch.tensor([.6,.3,.1]),.6)==[0]
 assert driver.aps_candidate_set(torch.tensor([.6,.3,.1]),.95)==[0,1,2]
 assert driver.aps_candidate_set(torch.tensor([.5,.5,0.]),.5)==[0]

def test_manifest_validation_rejects_nonfinite_and_inconsistent_targets():
 valid=[{'sample_id':'a','Z':20.0,'Y_ord':1}]
 driver.validate_manifest_rows(valid,(20.,40.,60.,80.))
 with pytest.raises(RuntimeError,match='target-label-bin inconsistency'):
  driver.validate_manifest_rows([{'sample_id':'a','Z':20.0,'Y_ord':0}],(20.,40.,60.,80.))
 with pytest.raises(RuntimeError,match='nonfinite'):
  driver.validate_manifest_rows([{'sample_id':'a','Z':float('nan'),'Y_ord':0}],(20.,40.,60.,80.))

def test_overwrite_archives_instead_of_mixing_stale_artifacts(tmp_path,monkeypatch):
 monkeypatch.chdir(tmp_path);out=Path('outputs/conference_v0_3/retinamnist/aps/seed_0');out.mkdir(parents=True);(out/'stale.txt').write_text('stale')
 with pytest.raises(RuntimeError,match='use --overwrite'): driver.prepare_output_directory(out,False)
 driver.prepare_output_directory(out,True)
 assert out.is_dir() and not (out/'stale.txt').exists()
 archived=list(Path('outputs/legacy/conference_v0_3/retinamnist/aps').glob('seed_0_*'))
 assert len(archived)==1 and (archived[0]/'stale.txt').read_text()=='stale'
