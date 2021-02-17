import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import chain, combinations
def nonempty_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def f1(k, edge_weights, bound):
    if k >= len(edge_weights):
        return bound
    else:
        return edge_weights[k]

def f2(k, edge_weights, bound):
    if (k+1) >= len(edge_weights):
        temp = bound
    else:
        temp = edge_weights[k+1]
    if len(edge_weights) > 0:
        return temp - edge_weights[0]
    else:
        return 0

n = 2**8 # This needs to be pretty small if we want to enumerate all elements of the powerset!
p = 1.0
g = nx.random_graphs.gnp_random_graph(n=n, p=p)

bound = 1.0
weights = {e: bound*np.random.random() for e in g.edges()}
nx.set_edge_attributes(g, weights, "weight")

cuts = list(nonempty_powerset(g))
cut_edge_weights = []
for cut in cuts:
    S = set(cut)
    T = set(g).difference(S)
    cut_edges = []
    for e in g.edges():
        if (((e[0] in S) and (e[1] in T)) or ((e[1] in S) and (e[0] in T))):
            cut_edges.append(e)
    edge_weights = sorted([g.edges[e]["weight"] for e in cut_edges])
    cut_edge_weights.append(edge_weights)


temp1 = np.array([max([f1(k, edge_weights, bound) for edge_weights in cut_edge_weights])
                  for k in range(n+1)])

temp2 = np.array([max([f2(k, edge_weights, bound) for edge_weights in cut_edge_weights])
                  for k in range(n+1)])

temp = np.maximum(temp1, temp2)

def lightweight_graph(g, w):
    gw = nx.Graph()
    gw.add_nodes_from(g)
    gw.add_edges_from([e for e in g.edges() if g.edges[e]["weight"] <= w])
    return gw

def bin_first_hit(k, w, bound, costs, **args):
    if (len(w)-1) not in costs:
        costs[len(w)-1] = nx.edge_connectivity(lightweight_graph(g, w[len(w)-1]), **args)
    cost = costs[len(w) - 1]
    if cost <= k:
        return bound

    cost = costs.get(0, nx.edge_connectivity(lightweight_graph(g, w[0]), **args))
    if 0 not in costs:
        costs[0] = nx.edge_connectivity(lightweight_graph(g, w[0]), **args)
    cost = costs[0]
    if cost > k:
        return w[0]

    high = len(w) - 1
    low = 1
    while (high - low) > 1:
        mid = (low + high) // 2
        if mid not in costs:
            costs[mid] = nx.edge_connectivity(lightweight_graph(g, w[mid]), **args)
        cost = costs[mid]
        if cost <= k:
            low = mid
        else:
            high = mid
    return w[high]

edge_weights = sorted([g.edges[e]["weight"] for e in g.edges()])
costs = dict()
%time foo = np.array([bin_first_hit(k, edge_weights, bound, costs) for k in range(n+1)])

t = nx.minimum_spanning_tree(g)
bar = [0 for k in range(n+1)]
for e in t.edges():
    costs = dict()
    e_bar = np.array([bin_first_hit(k+1, edge_weights, bound, costs, s=e[0], t=e[1]) - t.edges[e]["weight"]
                     for k in range(n+1)])
    for k in range(n+1):
        if e_bar[k] > bar[k]:
            bar[k] = e_bar[k]

ram = np.maximum(foo, bar)

print(np.all(temp1 == foo))
print(np.all(temp2 == bar))
print(np.all(temp == ram))
