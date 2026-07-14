from dvgc.reference import ReferenceTrajectory

def test_supplied_reference():
    r=ReferenceTrajectory.load("data/reference_jump.csv")
    a=r.anchors(); report=r.report(a)
    assert report["rows"]==821
    assert abs(report["median_dt_s"]-.002)<1e-9
    assert report["angle_unit"]=="degree"
    assert a.approach_end<a.takeoff_end<a.apex<a.landing_start<a.recovery_start<a.recovery_end
