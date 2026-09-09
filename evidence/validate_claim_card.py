#!/usr/bin/env python3
import json
from pathlib import Path

p = Path(__file__).with_name('claim_card.json')
d = json.loads(p.read_text(encoding='utf-8'))
assert d['readout_condition'] == {'E_existence': True, 'A_attribution': True, 'D_disclosure': True}
q = d['five_questions']
for key in ('Q1_seen', 'Q2_record_separates', 'Q3_ai_filled', 'Q4_assumed', 'Q5_tested'):
    assert isinstance(q[key], list) and q[key], key

blob = json.dumps(d, ensure_ascii=False).lower()
assert any('AI assistant' in x for x in q['Q3_ai_filled'])
assert any('I4' in x for x in q['Q5_tested'])
assert any('I5' in x and 'not yet' in x for x in q['Q5_tested'])
assert '+infinity' in blob
assert 'fail closed' in blob
assert 'matched comparator' in blob
assert 'reduced configuration' in blob
assert 'trajectory-prefix' in blob
assert 'direct online-policy latency advantage' in blob
assert 'universal superiority' in blob

for forbidden in (
    'real rgb-d/gpu speedup established',
    'physical robot non-inferiority established',
    'safety certified',
    'ten-cycle study repeats the first full configuration',
    'trajectory-prefix timing establishes speedup',
):
    assert forbidden not in blob

print('glosa claim card: PASS (post-review boundaries + I4 mechanical declarations)')
