from landing_evaluation import evaluate_landing

def row(t,**kw):
 d=dict(time_s=t,contact_valid=True,support_floor_N=50.,support_step_N=0.,nonwheel_N=0.,vz=0.,min_wheel_clearance_m=0.,roll=0.,pitch=0.,roll_rate=0.,pitch_rate=0.,vy=0.,vx=1.,x=1.5);d.update(kw);return d

def test_no_jump_timeout_not_success():
 assert not evaluate_landing([row(i*.02,x=4,support_step_N=50) for i in range(100)],50)['strict_landing_1p0s']
def test_prefix_and_one_second_hold_required():
 rs=[row(0),row(.02),row(.04,support_floor_N=0,vz=1,min_wheel_clearance_m=.1),row(.06,support_floor_N=0,vz=1,min_wheel_clearance_m=.1)]
 rs += [row(.08+i*.02,x=4,support_floor_N=0,support_step_N=50,vz=-.01) for i in range(51)]
 assert evaluate_landing(rs,50)['strict_landing_1p0s']
 rs[-1]['roll']=.5
 assert not evaluate_landing(rs,50)['strict_landing_1p0s']
def test_invalid_sensor_not_flight():
 rs=[row(0)]+[row(i*.02,contact_valid=False,support_floor_N=0,vz=1,min_wheel_clearance_m=.1) for i in range(1,10)]
 assert evaluate_landing(rs,50)['takeoff_time_s'] is None

def test_gap_rejected():
 import pytest
 with pytest.raises(ValueError):evaluate_landing([row(0),row(.04)],50)
def test_no_success_from_airborne_reset():
 rs=[row(0),row(.02),row(.04,support_floor_N=0,vz=1,min_wheel_clearance_m=.1),row(.06,support_floor_N=0,vz=1,min_wheel_clearance_m=.1)]
 rs += [row(.08+i*.02,x=4,support_floor_N=0,support_step_N=50) for i in range(55)]
 assert not evaluate_landing(rs,50,reset_type='airborne_rsi')['strict_landing_1p0s']
 rs[0]['contact_valid']=False
 assert not evaluate_landing(rs,50)['strict_landing_1p0s']
