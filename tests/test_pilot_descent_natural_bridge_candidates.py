from cli.pilot_descent_natural_bridge_candidates_v1 import select_targeted

def test_targeted_selection_balances_regions_and_parents():
 rows=[]
 for j,region in enumerate(('early','middle','late')):
  for i in range(3):rows.append({'region':region,'target_distance':i,'candidate_id':f'{j}-{i}','proposal_id':f's{j}-{i}'})
 selected=select_targeted(rows,2)
 assert len(selected)==6 and len({r['candidate_id'] for r in selected})==6
