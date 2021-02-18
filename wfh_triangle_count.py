import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

n = 2**13
p = 0.1
g = nx.random_graphs.gnp_random_graph(n=n, p=p)

M = nx.adj_matrix(g)

A = (M @ M).todense()

D0 = A.sum(0)
D1 = A.sum(1)
B = D0 + D1 - 2*A

pair_dict = dict()
for a,b in zip(np.array(A).flatten(), np.array(B).flatten()):
    if a in pair_dict:
        pair_dict[a].append(b)
    else:
        pair_dict[a] = [b]

survivors = sorted({a: max(pair_dict[a]) for a in pair_dict}.items(), reverse=True)

prev_survivor = survivors[0]
break_points = {prev_survivor: n+1}
for survivor in survivors[1:]:
    a0, b0 = prev_survivor
    a1, b1 = survivor
    intersection = 2*(a0 - a1) + b0
    if b0 <= intersection <= b1:
        break_points[prev_survivor] = intersection
        break_points[survivor] = n+1
        prev_survivor = survivor

print(survivors)
print(break_points)

break_list = sorted(break_points.items(), key=lambda _: _[1])
def first_hit(x, break_list):
    return next((y[0] for y in break_list if x <= y[1]))

def f(s, a, b, n):
    return min(a + ((s + min(s,b)) / 2.0), n-2)

def g(s, break_list):
    a, b = first_hit(s, break_list)
    return f(s, a, b, n)

%time local_sensitivities = np.array([g(s, break_list) for s in range(n+1)])

def h(s, survivors):
    max_result = 0
    for a,b in survivors:
        foo = f(s, a, b, n)
        if foo > max_result:
            max_result = foo
    return max_result

%time local_sensitivities = np.array([h(s, survivors) for s in range(n+1)])
