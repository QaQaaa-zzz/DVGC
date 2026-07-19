import json
from pathlib import Path
from cli.migrate_stage_reachability import SUPERSEDED,migrate
from cli.stage_reachability_controller import ROUTE

def test_migration_preserves_atomic_outputs_and_retires_old_route(tmp_path):
 old=tmp_path/'old';new=tmp_path/'new';old.mkdir();out=old/'result.json';out.write_text('{}')
 (old/'controller_state.json').write_text(json.dumps({'active_worker_unit':None,'expected_outputs':[str(out)],'current_stage':'h1'}))
 pointer=tmp_path/'active.json';payload=migrate(old,new,pointer,check_systemd=False)
 state=json.loads((old/'controller_state.json').read_text())
 assert state['terminal_state']==SUPERSEDED and state['superseded_artifacts'][0]['path']==str(out)
 assert payload['status']=='PREPARING' and Path(payload['run_path'])==new

def test_route_matches_authorized_ral_sequence():
 assert ROUTE[:5]==('safe_pause_old_route','h4_replay_smoke','define_stage_entry_protocol','relabel_existing_data','coverage_inventory')
 assert ROUTE[-3:]==('build_stage_tubes','Tube_RSI_final_PPO','final_policy_certification')
