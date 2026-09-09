#!/usr/bin/env python3
"""Pinned IDM aggregate core used by the finite diagnostic.
Vendored from morrocwi/information-discrete-math at the commit recorded in provenance.json.
"""
from fractions import Fraction as Q
import math

class Semiring:
    def __init__(self, name, oplus_core, otimes_core, one):
        self.name = name; self._op = oplus_core; self._ot = otimes_core; self.one = one
    def oplus(self, a, b):
        if a is None: return b
        if b is None: return a
        return self._op(a, b)
    def otimes(self, a, b):
        if a is None or b is None: return None
        return self._ot(a, b)

MIN_PLUS   = Semiring("min-plus",   min,                    lambda a, b: a + b, 0)
MAX_PLUS   = Semiring("max-plus",   max,                    lambda a, b: a + b, 0)
BOTTLENECK = Semiring("bottleneck", max,                    min,                None)
MINIMAX    = Semiring("minimax",    min,                    max,                None)
REACH      = Semiring("reachability", (lambda a, b: a or b), (lambda a, b: a and b), True)
COUNT      = Semiring("count/prob", (lambda a, b: a + b),   (lambda a, b: a * b), 1)

def mat_mul(A, B, sr):
    n, p, m = len(A), len(B), len(B[0])
    C = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = None
            for k in range(p):
                acc = sr.oplus(acc, sr.otimes(A[i][k], B[k][j]))
            C[i][j] = acc
    return C

def walk_count(A):
    n = len(A)
    total = [[A[i][j] for j in range(n)] for i in range(n)]
    P = A
    for _ in range(n - 1):
        P = mat_mul(P, A, COUNT)
        for i in range(n):
            for j in range(n):
                total[i][j] += P[i][j]
    return total

def all_pairs(W, sr):
    n = len(W)
    D = [[W[i][j] for j in range(n)] for i in range(n)]
    one = sr.one
    if one is None:
        vals = [W[i][j] for i in range(n) for j in range(n) if W[i][j] is not None]
        if sr is BOTTLENECK: one = (max(vals) if vals else 0)
        else: one = (min(vals) if vals else 0)
    for i in range(n):
        D[i][i] = sr.oplus(D[i][i], one)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i][j] = sr.oplus(D[i][j], sr.otimes(D[i][k], D[k][j]))
    return D

def _Q(xs): return [Q(x) for x in xs]
def rmin(xs): return min(xs)
def rmax(xs): return max(xs)
def rsum(xs): return sum(_Q(xs), Q(0))
def count(xs): return len(xs)
def mean(xs): return rsum(xs) / len(xs)
def rrange(xs): return max(xs) - min(xs)
def peak(xs): return max(abs(x) for x in xs)
def peak_to_peak(xs): return max(xs) - min(xs)
def argmin(xs): return min(range(len(xs)), key=lambda i: xs[i])
def argmax(xs): return max(range(len(xs)), key=lambda i: xs[i])
def energy(xs): return sum((Q(x) * Q(x) for x in xs), Q(0))
def power(xs): return energy(xs) / len(xs)
def rms(xs): return math.sqrt(float(power(xs)))
def variance_pop(xs):
    m = mean(xs); return sum(((Q(x) - m) ** 2 for x in xs), Q(0)) / len(xs)
def variance_samp(xs):
    m = mean(xs); return sum(((Q(x) - m) ** 2 for x in xs), Q(0)) / (len(xs) - 1)
def std_pop(xs): return math.sqrt(float(variance_pop(xs)))
def std_samp(xs): return math.sqrt(float(variance_samp(xs)))
def mad(xs):
    m = mean(xs); return sum((abs(Q(x) - m) for x in xs), Q(0)) / len(xs)
def median(xs):
    s = sorted(_Q(xs)); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
def mode(xs):
    from collections import Counter
    return Counter(xs).most_common(1)[0][0]
def geometric_mean(xs): return math.exp(sum(math.log(float(x)) for x in xs) / len(xs))
def harmonic_mean(xs): return len(xs) / float(sum(Q(1) / Q(x) for x in xs))
def weighted_mean(xs, w): return sum((Q(x) * Q(wi) for x, wi in zip(xs, w)), Q(0)) / sum(_Q(w), Q(0))
def crest_factor(xs): return peak(xs) / rms(xs) if rms(xs) else float("nan")
def form_factor(xs):
    r = rms(xs); a = sum((abs(Q(x)) for x in xs), Q(0)) / len(xs); return r / float(a) if a else float("nan")
def norm_L1(xs): return sum((abs(Q(x)) for x in xs), Q(0))
def norm_L2(xs): return math.sqrt(float(energy(xs)))
def norm_Linf(xs): return max(abs(x) for x in xs)
def percentile(xs, p):
    s = sorted(xs); k = max(0, min(len(s) - 1, int(math.ceil(p / 100 * len(s))) - 1)); return s[k]
def prefix_sum(xs):
    out, acc = [], Q(0)
    for x in xs: acc += Q(x); out.append(acc)
    return out
def running_min(xs):
    out, m = [], None
    for x in xs: m = x if m is None else min(m, x); out.append(m)
    return out
def running_max(xs):
    out, m = [], None
    for x in xs: m = x if m is None else max(m, x); out.append(m)
    return out
def moving_average(xs, k): return [sum(_Q(xs[i:i + k]), Q(0)) / k for i in range(len(xs) - k + 1)]
def first_diff(xs): return [Q(xs[i + 1]) - Q(xs[i]) for i in range(len(xs) - 1)]
def second_diff(xs):
    d = first_diff(xs); return [d[i + 1] - d[i] for i in range(len(d) - 1)]
