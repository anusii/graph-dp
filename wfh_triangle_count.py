import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

n = 2**8
p = 0.25
g = nx.random_graphs.gnp_random_graph(n=n, p=p)

M = nx.adj_matrix(g)

A = (M @ M).todense()

D0 = A.sum(0)
D1 = A.sum(1)

B = D0 + D1 - 2*A

def f(s, a, b, n):
    return min(a + ((s + min(s,b)) / 2.0), n-2)

i = np.random.randint(n)
j = np.random.randint(n)

a = A[i,j]
b = B[i,j]

xs = np.arange(0,n,0.01)
ys = [f(x, a, b, n) for x in xs]

plt.plot(xs, ys)

A_set = set(np.array(A).flatten())

pair_dict = dict()
for a,b in zip(np.array(A).flatten(), np.array(B).flatten()):
    if a in pair_dict:
        pair_dict[a].append(b)
    else:
        pair_dict[a] = [b]

survivors = sorted({a: max(pair_dict[a]) for a in pair_dict}.items(), reverse=True)
    
