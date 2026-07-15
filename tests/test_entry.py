import numpy as np
from dvgc.entry import ENTRY_FEATURE_NAMES, calibrate_radius, robust_normalization

def test_entry_feature_contract_and_physical_scale_floors():
    x=np.zeros((3,len(ENTRY_FEATURE_NAMES))); x[1,0]=.1; x[2,0]=.2
    floors=np.full(len(ENTRY_FEATURE_NAMES),.05); center,scale=robust_normalization(x,floors)
    assert len(ENTRY_FEATURE_NAMES)==20 and np.all(scale>=floors)

def test_radius_calibration_uses_safe_loo_and_rejects_dead():
    x=np.asarray([[0.,0.],[.1,0.],[.2,0.],[3.,3.]])
    result=calibrate_radius(x,['safe','safe','safe','dead'],np.zeros(2),np.ones(2),.95)
    assert result['precision']==1 and result['recall']==1 and result['radius']<1
