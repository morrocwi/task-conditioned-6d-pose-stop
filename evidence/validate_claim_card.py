#!/usr/bin/env python3
import json
from pathlib import Path
p=Path(__file__).with_name('claim_card.json'); d=json.loads(p.read_text())
assert d['readout_condition']=={'E_existence':True,'A_attribution':True,'D_disclosure':True}
q=d['five_questions']
for key in ('Q1_seen','Q2_record_separates','Q3_ai_filled','Q4_assumed','Q5_tested'):
    assert isinstance(q[key],list) and q[key], key
assert any('AI assistant' in x for x in q['Q3_ai_filled'])
assert any('I4' in x for x in q['Q5_tested'])
assert any('I5' in x and 'not yet' in x for x in q['Q5_tested'])
for forbidden in ('real RGB-D/GPU speedup established','physical robot non-inferiority established','safety certified'):
    assert forbidden.lower() not in json.dumps(d).lower()
print('glosa claim card: PASS (presence/disclosure + I4 test declarations)')
