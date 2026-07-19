from dvgc.config import default_config
from cli.stage_reward_preflight import run_preflight

def test_complete_reward_preflight_passes():
 report=run_preflight(default_config())
 assert report['status']=='PASS' and report['finite_and_bounded']
 assert report['terminal_mutually_exclusive'] and report['landing_recovery_not_physical_failure']
 assert all(row['event_dominates'] for row in report['checks'].values())
 assert all({'mean','p95','max','positive_reward_share'}<=set(x) for x in report['term_statistics'].values())
