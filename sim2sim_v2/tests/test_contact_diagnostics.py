import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_diagnostics import aggregate

def test_sector_duplicates_do_not_double_count():
 c=dict(actor0='wheel',actor1='floor',collider0='sector1',collider1='floor',position=[0,0,0],impulse=[0,0,1],normal_impulse=1.)
 out=aggregate([dict(contacts=[c,c])],.005)
 assert out[0]['points']==1
 assert out[0]['mean_normal_force']==200

def test_empty_is_not_implicit_validity():
 assert aggregate([dict(contacts=[])],.005)==[]
