"""Executable finite diagnostic. [SimulatedData]; NOT a vision/physics benchmark.
Python standard library only. Run: python benchmark.py
"""
import csv
import hashlib
import heapq
import itertools
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path
from vendor.aggregate import all_pairs, MIN_PLUS

ROOT = Path(__file__).resolve().parent
AXES = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
TASKS = {'suction': 31, 'label': 63, 'insertion': 35}
METHODS = ('one_shot', 'full6d', 'mask_greedy', 'action_greedy', 'dijkstra', 'idm_core')
SEED = 20260909
SAMPLES = 128

def widths(state):
    return tuple(2.0 if state & (1 << i) else 0.25 for i in range(6))

def physical_success(task, error):
    required = TASKS[task]
    if task == 'insertion':
        return abs(error[0]) + abs(error[1]) + abs(error[5]) <= 1.0
    return all(abs(error[i]) <= 1.0 for i in range(6) if required & (1 << i))

def gate(task, state):
    return physical_success(task, widths(state))

def actions(menu):
    a = [(f'refine_{name}', 1 << i, 4) for i, name in enumerate(AXES)]
    if menu == 'rich':
        a += [('view_xy_yaw', 35, 7), ('refine_all', 63, 18)]
    return a

def edges(state, acts):
    return [(name, state & ~mask, cost, mask) for name, mask, cost in acts if state & mask]

def dijkstra_first(start, task, acts):
    todo = [(0, start, None)]
    best = {start: 0}
    while todo:
        cost, state, first = heapq.heappop(todo)
        if cost != best[state]:
            continue
        if gate(task, state):
            return first
        for name, nxt, weight, mask in edges(state, acts):
            cand = cost + weight
            if cand < best.get(nxt, math.inf):
                best[nxt] = cand
                heapq.heappush(todo, (cand, nxt, first or name))
    return None

def idm_first(start, task, acts):
    states, seen = [start], {start}
    for s in states:
        if gate(task, s):
            continue
        for _, nxt, _, _ in edges(s, acts):
            if nxt not in seen:
                seen.add(nxt)
                states.append(nxt)
    index = {s: i for i, s in enumerate(states)}
    W = [[None] * len(states) for _ in states]
    for s in states:
        if gate(task, s):
            continue
        for _, nxt, c, _ in edges(s, acts):
            i, j = index[s], index[nxt]
            W[i][j] = c if W[i][j] is None else min(W[i][j], c)
    D = all_pairs(W, MIN_PLUS)
    goals = [index[s] for s in states if gate(task, s)]
    candidates = []
    for name, nxt, cost, _ in edges(start, acts):
        ds = [D[index[nxt]][g] for g in goals if D[index[nxt]][g] is not None]
        if ds:
            candidates.append((cost + min(ds), name))
    return min(candidates)[1] if candidates else None

def choose(method, task, state, acts):
    if method == 'full6d':
        es = edges(state, acts)
        return min(es, key=lambda e: (-int(state ^ e[1]).bit_count(), e[2], e[0]))[0] if es else None
    if method == 'mask_greedy':
        for i in range(6):
            if state & TASKS[task] & (1 << i):
                return f'refine_{AXES[i]}'
        return None
    if method == 'action_greedy':
        es = [e for e in edges(state, acts) if (state ^ e[1]) & TASKS[task]]
        return min(es, key=lambda e: (-((state ^ e[1]) & TASKS[task]).bit_count()/e[2], e[2], e[0]))[0] if es else None
    return (dijkstra_first if method == 'dijkstra' else idm_first)(state, task, acts)

def rollout(method, task, start, acts, stall=False):
    state, route, total_cost, policy_ns = start, [], 0, 0
    for step in range(13):
        tic = time.perf_counter_ns()
        done = method == 'one_shot' or (state == 0 if method == 'full6d' else gate(task, state))
        if done:
            policy_ns += time.perf_counter_ns() - tic
            return state, route, total_cost, policy_ns/1e6, 'ACT'
        if step == 12:
            policy_ns += time.perf_counter_ns() - tic
            return state, route, total_cost, policy_ns/1e6, 'HOLD'
        name = choose(method, task, state, acts)
        policy_ns += time.perf_counter_ns() - tic
        if name is None:
            return state, route, total_cost, policy_ns/1e6, 'HOLD'
        _, mask, cost = next(a for a in acts if a[0] == name)
        route.append(name)
        total_cost += cost
        if not stall:
            state &= ~mask
    raise AssertionError('unreachable')

def evaluate(task, start, route, acts, z, undercoverage=1.0):
    actual_width = [w * undercoverage for w in widths(start)]
    for name in route:
        _, mask, _ = next(a for a in acts if a[0] == name)
        for i in range(6):
            if mask & (1 << i):
                actual_width[i] = 0.25
    return sum(physical_success(task, [zz[i]*actual_width[i] for i in range(6)]) for zz in z)

def independent_optimum(start, task, acts):
    best = math.inf
    for bits in range(1 << len(acts)):
        state, cost = start, 0
        for j, (_, mask, c) in enumerate(acts):
            if bits & (1 << j):
                state &= ~mask
                cost += c
        if gate(task, state):
            best = min(best, cost)
    return best

def diagnostics():
    checks = {}
    before = [1.1, 8.0]
    after = [u * 0.5 for u in before]
    checks['bit_state_aliasing'] = {'before': before, 'threshold': 1,
        'same_fail_bit': all(u > 1 for u in before), 'after_halving': after,
        'different_next_bits': [int(u > 1) for u in after]}
    checks['coupled_false_pass'] = {'error_xy_yaw': [0.7, 0, 0.7], 'diagonal_pass': all(abs(x)<=1 for x in [0.7,0,0.7]),
                                  'actual_coupled_success': physical_success('insertion', [0.7,0,0,0,0,0.7])}
    acts = actions('rich')
    checks['task_switch'] = {t: rollout('idm_core', t, 32, acts)[1] for t in ('suction','label')}
    checks['unreachable'] = rollout('idm_core','label',32,actions('single')[:-1])[-1]
    checks['no_progress_budget'] = rollout('idm_core','label',32,acts,stall=True)[-1]
    error = [1.4,0,0,0,0,0]
    checks['hidden_bias'] = {'task_gate_pass': gate('suction',32),
                           'success_before_action': physical_success('suction',error),
                           'full_reset_success': physical_success('suction',[0.1]*6)}
    w = widths(32)
    checks['embodiment_switch'] = {'strict_tolerance':1,'compliant_tolerance':3,
        'bound_sum':w[0]+w[1]+w[5], 'strict_pass':gate('insertion',32),
        'compliant_pass':w[0]+w[1]+w[5] <= 3}
    trap = [('xy',3,3),('x_yaw',33,3),('y_yaw',34,3),('all_required',35,5)]
    checks['greedy_trap'] = {m: rollout(m,'insertion',35,trap)[2]
                             for m in ('action_greedy','dijkstra','idm_core')}
    assert checks['unreachable'] == checks['no_progress_budget'] == 'HOLD'
    assert checks['task_switch']['suction'] == [] and checks['task_switch']['label'] == ['refine_yaw']
    assert not checks['coupled_false_pass']['actual_coupled_success']
    assert checks['greedy_trap'] == {'action_greedy':6,'dijkstra':5,'idm_core':5}
    return checks

def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    samples = {s: [[rng.uniform(-1,1) for _ in range(6)] for _ in range(SAMPLES)] for s in range(64)}
    rows, oracle_checks = [], 0
    for menu, task, start in itertools.product(('single','rich'), TASKS, range(64)):
        acts = actions(menu)
        optimum = independent_optimum(start, task, acts)
        order = list(METHODS)
        rng.shuffle(order)
        for method in order:
            final, route, cost, ms, status = rollout(method,task,start,acts)
            if method in ('idm_core','dijkstra'):
                assert cost == optimum, (menu,task,start,method,cost,optimum)
                oracle_checks += 1
            for regime, factor in [('bounded',1.0),('undercoverage',6.0)]:
                succ = evaluate(task,start,route,acts,samples[start],factor) if status == 'ACT' else 0
                rows.append(dict(menu=menu,task=task,start=start,method=method,regime=regime,
                    samples=SAMPLES,successes=succ,status=status,cost_units=cost,
                    policy_ms=ms,refines=sum(not n.startswith('view') for n in route),
                    views=sum(n.startswith('view') for n in route),route=';'.join(route)))
    with (out/'raw.csv').open('w',newline='') as f:
        wr = csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    summary=[]
    for menu,task,regime,method in itertools.product(('single','rich'),TASKS,('bounded','undercoverage'),METHODS):
        rr=[r for r in rows if (r['menu'],r['task'],r['regime'],r['method'])==(menu,task,regime,method)]
        summary.append(dict(menu=menu,task=task,regime=regime,method=method,
          success_pct=100*sum(r['successes'] for r in rr)/sum(r['samples'] for r in rr),
          mean_cost_units=statistics.mean(r['cost_units'] for r in rr),
          mean_policy_ms=statistics.mean(r['policy_ms'] for r in rr),
          p95_policy_ms=sorted(r['policy_ms'] for r in rr)[60],
          mean_refines=statistics.mean(r['refines'] for r in rr),mean_views=statistics.mean(r['views'] for r in rr)))
    with (out/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    data=dict(label='[SimulatedData]',tier='finite_diagnostic',seed=SEED,samples_per_state=SAMPLES,
        start_states=64,oracle_comparisons=oracle_checks,diagnostics=diagnostics(),
        python=platform.python_version(),platform=platform.platform(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_idm_core=True,full_idm_solve_executed=False,vision_model_executed=False,
        rigid_body_physics_executed=False,summary=summary)
    (out/'results.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!='summary'},indent=2))
    for r in summary:
        if r['menu']=='rich' and r['regime']=='bounded': print(r)

if __name__ == '__main__':
    main()
