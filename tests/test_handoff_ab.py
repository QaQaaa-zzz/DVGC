from cli.evaluate_handoff_ab import summarize

def test_handoff_ab_summary_separates_chain_final_and_timeout():
 rows=[{'chain':False,'final':True,'truncated':False},{'chain':True,'final':False,'truncated':False},{'chain':False,'final':False,'truncated':True}]
 s=summarize(rows);assert s['handoff_missed']==1;assert s['false_progress']==1;assert s['timeout']==1;assert s['physical_failure']==1
