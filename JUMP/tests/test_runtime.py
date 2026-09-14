"""Host runtime contracts; these tests integrate zero simulation steps."""

from types import SimpleNamespace

import mujoco
import pytest

from jump_planning.runtime import ApproachQualification, TriggerAdapter, read_contacts


def limits():
    return dict(min_forward_speed=1.0, max_roll=.6, max_pitch=.8, max_yaw=.5,
                corridor_half_width=.3, contact_force_threshold=.01,
                contact_distance_tolerance=1e-6)


def frame(time, **kwargs):
    return dict(time=time, front_contact=True, rear_contact=True, vx=2., roll=0.,
                pitch=0., yaw=0., y=0., vz=0., front_clearance=0.,
                rear_clearance=0., finite=True, **kwargs)


def test_qualification_requires_continuous_ground_contact():
    state = ApproachQualification(dict(timing=dict(settle_seconds=.1), limits=limits()))
    assert not state.update(frame(0))
    assert not state.update(frame(.08))
    airborne = frame(.09)
    airborne['front_contact'] = False
    assert not state.update(airborne)
    assert not state.update(frame(.10))
    assert not state.update(frame(.195))
    assert state.update(frame(.20))
    assert state.qualified_time == pytest.approx(.20)


def test_initial_clearance_is_not_a_trigger_adapter_apex():
    adapter = TriggerAdapter(.01, .001)
    adapter.observe(dict(vz=-1., front_contact=False, rear_contact=False,
                         front_clearance=.1, rear_clearance=.1), qualified=False, time=.1)
    assert not adapter.apex_seen
    assert adapter.signal(False) == 0
    assert adapter.signal(True) == 1
    adapter.observe(dict(vz=-1., front_contact=False, rear_contact=False,
                         front_clearance=.1, rear_clearance=.1), qualified=True, time=.2)
    assert adapter.signal(True) == 1  # A falling initial state is not a jump apex.
    rising = dict(vz=1., front_contact=False, rear_contact=False,
                  front_clearance=.1, rear_clearance=.1)
    adapter.observe(rising, qualified=True, time=.21)
    adapter.observe(rising, qualified=True, time=.225)
    adapter.observe({**rising, 'vz': -.1}, qualified=True, time=.23)
    assert adapter.apex_seen
    assert adapter.signal(True) == 0


def test_real_contacts_do_not_turn_geometric_proximity_into_collision():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="step" type="box" pos="2 0 .1" size=".1 .5 .1"/>
      <body pos="0 0 .1005"><freejoint/><geom name="front" type="sphere" size=".1" margin=".01"/></body>
      <body pos="-.5 0 .099"><freejoint/><geom name="rear" type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = SimpleNamespace(floor_geom_id=model.geom('floor').id,
                          obstacle_geom_id=model.geom('step').id,
                          frontwheel_geom_id=model.geom('front').id,
                          rearwheel_geom_id=model.geom('rear').id)
    # The sphere's positive margin may exert force before geometric touching;
    # remove that artificial force to isolate a proximity-only contact record.
    data.efc_force[:] = 0
    contacts = read_contacts(model, data, ids, {ids.frontwheel_geom_id, ids.rearwheel_geom_id}, limits())
    assert contacts['front_contact'] is False
    assert contacts['rear_contact'] is True  # Actual penetration is retained.
    assert contacts['obstacle_contact'] is False
    assert len(contacts['contact_pairs']) >= 2
    assert contacts['rear_touch_x'] == pytest.approx(-.5)


def test_force_contacts_record_solver_force_and_touch_location():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <geom name="step" type="box" pos="2 0 .1" size=".1 .5 .1"/>
    <body pos="0 0 .099"><freejoint/><geom name="front" type="sphere" size=".1"/></body>
    <body pos="-.5 0 .099"><freejoint/><geom name="rear" type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = SimpleNamespace(floor_geom_id=model.geom('floor').id,
                          obstacle_geom_id=model.geom('step').id,
                          frontwheel_geom_id=model.geom('front').id,
                          rearwheel_geom_id=model.geom('rear').id)
    result = read_contacts(model, data, ids, {ids.frontwheel_geom_id, ids.rearwheel_geom_id}, limits())
    assert result['front_contact'] and result['rear_contact']
    assert result['front_touch_x'] == pytest.approx(0.)
    assert any(pair['normal_force'] > 0 for pair in result['contact_pairs'])
    assert not result['body_contact']


def test_obstacle_and_nonwheel_floor_collisions_use_actual_pairs():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <geom name="step" type="box" pos="2 0 .1" size=".1 .5 .1"/>
    <body pos="2 0 .25"><freejoint/><geom name="front" type="sphere" size=".1"/></body>
    <body pos="-.5 0 1"><freejoint/><geom name="rear" type="sphere" size=".1"/></body>
    <body pos="-.8 0 .08"><freejoint/><geom name="body" type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = SimpleNamespace(floor_geom_id=model.geom('floor').id,
                          obstacle_geom_id=model.geom('step').id,
                          frontwheel_geom_id=model.geom('front').id,
                          rearwheel_geom_id=model.geom('rear').id)
    robot_ids = {ids.frontwheel_geom_id, ids.rearwheel_geom_id, model.geom('body').id}
    result = read_contacts(model, data, ids, robot_ids, limits())
    assert result['obstacle_contact']
    assert result['body_contact']
    assert not result['front_contact']
    assert not result['rear_contact']


@pytest.mark.parametrize('changed', [dict(vx=.99), dict(roll=.61), dict(pitch=.81),
                                    dict(yaw=.51), dict(y=.31), dict(finite=False)])
def test_approach_constraints_are_required_before_qualification(changed):
    state = ApproachQualification(dict(timing=dict(settle_seconds=.1), limits=limits()))
    for time in (0., .1, .2):
        row = frame(time)
        row.update(changed)
        assert not state.update(row)
    assert state.qualified_time is None
