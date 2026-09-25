import importlib.util
from pathlib import Path
import numpy as np
import pytest
path=Path(__file__).resolve().parents[1]/'scripts/evaluate_spectra.py'
spec=importlib.util.spec_from_file_location('clef_metric_script',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_identical_disjoint_and_zero():
    pred=np.zeros((3,512));true=np.zeros_like(pred)
    true[:,0]=1;pred[0,0]=1;pred[1,1]=1
    for values in m.metrics(pred,true).values():
        assert np.allclose(values,[1,0,0])

def test_physical_sdp_and_invalid_reference():
    pred=np.zeros((1,512));true=np.zeros_like(pred)
    pred[0,[0,1]]=[.5,.5];true[0,[0,1]]=[.25,.75]
    left=np.array([1.,8.])*pred[0,:2]**.6
    right=np.array([1.,8.])*true[0,:2]**.6
    expected=np.dot(left,right)/(np.linalg.norm(left)*np.linalg.norm(right)+1e-10)
    assert m.metrics(pred,true)['SDP'][0]==pytest.approx(expected)
    with pytest.raises(ValueError):m.metrics(pred,np.zeros_like(true))
