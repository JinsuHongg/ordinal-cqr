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

def test_conference_copoc_pipeline_selects_canonical_head():
 from ordinal_cqr.models.backbone import ResNet18COPOC
 assert isinstance(driver.model('copoc', driver.K), ResNet18COPOC)

def test_aps_prefix_uses_full_set_when_float_sum_misses_qhat():
 import torch
 cumulative=torch.tensor([.5,.8,.95,.99999982])
 assert driver.aps_prefix_cutoff(cumulative,.99999988)==3

def test_aps_candidate_set_inverts_cumulative_through_label_score():
 import torch
 assert driver.aps_candidate_set(torch.tensor([.6,.3,.1]),.6)==[0]
 assert driver.aps_candidate_set(torch.tensor([.6,.3,.1]),.95)==[0,1]
 assert driver.aps_candidate_set(torch.tensor([.5,.5,0.]),.5)==[0]
