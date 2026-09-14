# JUMP — preparation and trigger planning

> Implementation plan approved in conversation 2026-09-14. Use isolated, bounded tasks and behavioral verification. First qualification and single-variable near-start diagnosis COMPLETE; engineering execution succeeded, scientific approach qualification failed. Next: bounded observation/backend compatibility diagnosis before lower-skill adaptation.

Goal: determine whether a frozen JIT Actor can maintain a genuine grounded forward approach and respond to an external trigger, before learning preparation/timing for complete narrow-obstacle crossing.

Architecture: a new host-MuJoCo qualification runtime, reusing the frozen JIT model loader, actuator mapping, observation contract and checkpoint inference. A separate per-physics-step oracle evaluates the new task. CPU execution is an engineering/transfer diagnosis, not a replay-equivalence claim or a measured GPU training result. Raw old runs and source are immutable.

## Global constraints

- New source and artifacts only within JUMP; no STTW_CONTROL access or changes, no old final TEST.
- sim_dt=0.005, control_dt=0.020, substeps=4; original robot mass, forces and friction unchanged.
- The new narrow box has declared front_x, length, height and lateral half width; box bottom at ground.
- Episode starts from the declared original keyframe and forward velocity; naturally settle while executing the frozen Actor with signal=0. No artificial qpos/qvel injection into a supposed reachable approach state.
- Qualify an approach only after both wheels contact the floor continuously for declared settle_steps and satisfy vx, pose and corridor requirements. Trigger delay is measured after that qualification. Never-trigger controls use null trigger delay. Initialization clearance alone must not be labeled a jump.
- Trigger stays on until a qualified apex event; the frozen Actor always sees real obstacle-relative distance and retained three-frame history. This is an explicitly new external-trigger adapter, not an unchanged legacy event protocol.
- Episode evidence records every physical substep, normalized actions and actual controls. New oracle uses actual MuJoCo contacts, not old first_valid_landing or AABB collision labels.
- New campaign ceiling 200M control steps, with finite stage and run budgets. A negative qualification is a completed experiment, not permission for arbitrary training or weakened success rules.

## Approved research design

First compare true-simulation no-preparation timing against legal steer/drive preparation plus timing. Once mechanism evidence exists, fit five bootstrap task-outcome MLPs (94→256→256→128, SiLU; success, actual liftoff state/delay, and validity-masked margins). Three-segment two-channel preparation plus trigger tick is seven-dimensional. This predictor is planned, not implemented in qualification. Freeze lower controllers for all high-level comparisons. Group data splits by complete parent approach, retain unknowns and all costs; final test remains closed until method freeze.

## Budget

Total 200M CONTROL steps (=800M physics substeps for fully executed controls), about 11.2× the preceding 17.8256M proposal. Allocations: qualification/engineering 2M; common lower-skill adaptation 40M; window/mechanism 20M; outcome-data collection 60M; closed-loop development 30M; independent comparisons 30M; recorded retries and contingency 18M. No automatic infinite extension. Supervised optimizer updates and wall time are separately budgeted because they are not simulator interactions. Every invocation reserves a finite share before its first step and records actual charged/physics steps afterward. The first run is a small diagnostic batch, not consumption of the full campaign.

## Implementation tasks

- [x] Task 1: independent oracle in src/jump_planning/oracle.py and tests/test_oracle.py. update(frame, triggered, qualified) sees mappings described below; physical failures before qualification count, but initial wheel clearance does not imply premature jump. Full success requires all conjunctions and holds.
- [x] Task 2: host runtime in src/jump_planning/runtime.py and tests/test_runtime.py. Constructor QualificationRuntime(config, policy_spec, source_root); exposes metadata, reset(seed), step(triggered)->list[dict], observation(), policy_action(signal). A convenience run_case(config, policy_spec, source_root, case) returns (summary, rows), with optional cached runtime. Existing checkpoint identity and payload hashes validated.
- [x] Task 3: config validation, persistent campaign ledger, first batch runner/CLI, process status/desktop watcher and failure handling. No runs overwritten, source code hash snapshot, policy/config/model provenance. Configuration/startup errors after directory creation now receive terminal error receipts too.
- [x] Task 4: per-case plots/report, actual bounded execution, independent code review, current documentation. Git delivery status is verified separately at handoff.

## Shared interface for implementation

Config JSON has scene(front_x,length,height,half_width,landing_near,landing_far), timing(sim_dt,control_dt,episode_seconds,approach_timeout_seconds,settle_seconds,airborne_seconds,recovery_seconds,hold_seconds), limits(nominal_speed,min_forward_speed,corridor_half_width,max_roll,max_pitch,max_yaw,recovery_roll,recovery_pitch,recovery_rate,recovery_yaw,recovery_lateral,recovery_speed_error,contact_force_threshold,contact_distance_tolerance,airborne_clearance), initial(keyframe,x,forward_velocity), action(base_rear_speed,rear_speed_delta,joint_target_semantics), policies list(name,checkpoint relative to source_root), cases list(name,seed,trigger_delay_seconds null/float), budget(max_control_steps,wall_seconds), data_role=ENGINEERING_DEVELOPMENT.

Runtime physical frames are dictionaries: time,x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz,front_contact,rear_contact,body_contact,obstacle_contact,front_clearance,rear_clearance,front_touch_x,rear_touch_x,robot_back_x,finite. Also include qpos/qvel as lists, action and ctrl lists, signal, control_step,substep,phase. Contact positions only meaningful if corresponding floor contact is true. Runtime must compute support bounds using actual collision shapes for clearance/back edge; contacts come from mjData.contact + mj_contactForce, named floor and step.

Oracle constructor TaskMonitor(config). update(frame, triggered: bool, qualified: bool=True)->dict returns status (ongoing/success/failure/unknown), done, reason plus event times. finish()->dict produces timeout failure for a fully executed episode, or approach_not_qualified if no approach qualified. update failures are latched. Trigger and actual airborne are distinct. Valid liftoff requires both wheels clear and no floor contact for airborne_seconds after qualification. If it occurs before trigger, fail premature_liftoff. Each wheel first floor recontact after liftoff must be inside [scene.front_x+scene.length+landing_near, ...+landing_far]; no requirement to wait for full back edge crossing before first front-wheel recontact. Full passage requires robot_back_x>obstacle back. Recovery begins only after passage plus both recorded legal landings. Hold clock resets on any tracking violation. All physical failure conditions apply before/after landing; never stop at first touchdown.

Policy input compatibility: same JIT 25-field frame/FIFO and 76D Actor. Metadata must record source HEAD/file hashes, payload+identity hashes, actual nq/nv/nu/mass/actuator ranges, scene override, backend, no additional actuator-delay queue. No observer or controller state may be silently reset on trigger.

## Progress

- Created branch agent/jump-preparation-timing from f46c94e in a separate worktree. Original workspace untouched.
- The user approved a substantially larger budget; declared 200M rather than automatically multiplying this first diagnostic batch.
- Baseline JIT action/observation: 15 passed. Final JUMP suite plus inherited notification tests: 73 passed (70 new-project tests + 3 inherited notification tests).
- Initial run `runs/qualification/initial_20260914T114808Z`: 12 declared cases, only 2 unique trajectories because all failed approach before trigger; pi0 yaw_limit at .405s, repair0000 yaw_limit at .285s. 216 charged controls / 828 physics steps, 0 qualified, 0 task successes, 0 unknown.
- Single-variable `qualification_near_start.json` moves initial x=1.8→2.5 only, keeps null trigger; `runs/qualification/near_start_20260914T115254Z`: pi0 roll_limit at .675s, repair0000 yaw_limit at .390s. 54 controls / 213 physics steps, 0 qualified. No standards relaxed.
- Combined actual: 270 charged controls / 1041 physics steps; no PPO or supervised training. Both execution pipelines completed and both desktop completion notifications delivered with zero recorded delivery errors. First 12-case plots remain immutable; later reporting adds yaw/wz and distinct-trace counts.
- Read-only trace audit: signal stays zero; both Actors produce contact-followed uncommanded airborne motion but never qualify for the task's official liftoff. FIFO masks and previous-action/velocity/distance/height history are correctly aligned. No observed successful trigger window can be drawn from these cases.
- Compatibility limit: new host runtime runs mj_forward after each substep to align contact/sensor data with state; legacy Warp RK4 observation sampling differs. Together with backend and scene/trigger changes, this prevents attributing all failure to the Actor or declaring that separate pi_drive training is already proven necessary.

## Next bounded stage

Before spending the 40M common-skill adaptation allocation, compare frozen Actor inputs/actions from the same complete state under the host and declared Warp sampling contracts. Preserve current negative runs. Then verify or adapt a grounded no-jump approach controller with a finite training configuration; freeze the resulting approach/jump/recovery bundle before window construction. The 200M budget is available stage-by-stage; the 270-step early failures do not justify consuming the rest on identical unsuccessful rollouts.
