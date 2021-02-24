import numpy as np
import networkx as nx

import copy

from relm.mechanisms import LaplaceMechanism, CauchyMechanism

# =============================================================================
# Convex Optimization Approach
# =============================================================================

# =============================================================================
# Edge Counts
# =============================================================================
def flow_graph(G, D):
    V_left = list(zip(["left"] * len(G), G.nodes()))
    V_right = list(zip(["right"] * len(G), G.nodes()))
    F = nx.DiGraph()
    F.add_nodes_from(V_left)
    F.add_nodes_from(V_right)
    F.add_nodes_from("st")
    F.add_weighted_edges_from([("s", vl, D) for vl in V_left], weight="capacity")
    F.add_weighted_edges_from([(vr, "t", D) for vr in V_right], weight="capacity")
    F.add_weighted_edges_from(
        [(("left", u), ("right", v), 1) for u, v in G.edges()], weight="capacity"
    )
    F.add_weighted_edges_from(
        [(("right", u), ("left", v), 1) for u, v in G.edges()], weight="capacity"
    )
    return F


n = 2 ** 8
D = 2 ** 7
p = 0.25
G = nx.random_graphs.gnp_random_graph(n, p)
print(max(dict(G.degree()).values()))

F = flow_graph(G, D)

foo = nx.minimum_cut_value(F, _s="s", _t="t")

print(foo)
print(len(G.edges()))

# =============================================================================
# Linear functions of the degree sequence with concave base function h
# =============================================================================
class Edge(object):
    """ An undirected, capacity limited edge. """

    def __init__(self, capacity):
        self.capacity = capacity
        self.flow = cp.Variable()

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [cp.abs(self.flow) <= self.capacity]


class Node(object):
    """ A node with accumulation. """

    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        self.edge_flows = []

    # Returns the node's internal constraints.
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]


n = np.random.randint(2 ** 6, 2 ** 7)
D = 2 ** 5
p = 2 ** -2
G = nx.random_graphs.gnp_random_graph(n, p)

left_nodes = [Node() for node in G.nodes()]
right_nodes = [Node() for node in G.nodes()]
source = Node(accumulation=cp.Variable())
target = Node(accumulation=cp.Variable())
nodes = left_nodes + right_nodes + [source, target]

source_edges = []
for ln in left_nodes:
    edge = Edge(capacity=D)
    edge.connect(source, ln)
    source_edges.append(edge)

target_edges = []
for rn in right_nodes:
    edge = Edge(capacity=D)
    edge.connect(rn, target)
    target_edges.append(edge)

inter_edges = []
for u, v in G.edges():
    edge = Edge(capacity=1.0)
    edge.connect(left_nodes[u], right_nodes[v])
    inter_edges.append(edge)
    edge = Edge(capacity=1.0)
    edge.connect(left_nodes[v], right_nodes[u])
    inter_edges.append(edge)

edges = source_edges + target_edges + inter_edges

constraints = []
for obj in nodes + edges:
    constraints += obj.constraints()

x = np.arange(n)
# y = x / 2.0
y = np.ones(x.shape)


def h1(x):
    x_low = cp.floor(x)
    x_high = cp.ceil(x)
    return y[int(x_low.value)] * (x - x_low) + y[int(x_high.value)] * (x_high - x)


def h2(x):
    m = y[1:] - y[:-1]
    x0 = np.arange(n - 1)
    y0 = y[:-1]
    z = y0 + cp.multiply(m, (x - x0))
    return cp.min(z)


def h(x):
    try:
        ret = h1(x)
    except TypeError:
        ret = h2(x)
    return ret


prob = cp.Problem(cp.Maximize(cp.sum([h(e.flow) for e in source_edges])), constraints)
value = prob.solve()

print(value)
# print(len(G.edges()))
print(len(G.nodes()))

# =============================================================================
# Linear programming approach
# =============================================================================
n = np.random.randint(2 ** 4, 2 ** 5)
D = 2 ** 2
p = 2 ** -2
G = nx.random_graphs.gnp_random_graph(n, p)

k = 3
triangles = nx.triads_by_type(nx.DiGraph(G))["300"]
print(D)
print(max(dict(G.degree).values()))


print(k * D * (D - 1) ** (k - 2))
print(max(nx.triangles(G).values()))

c = -np.ones(len(triangles))
A_ub = np.zeros((n, len(triangles)))
for i, t in enumerate(triangles):
    nodes = triangles[i].nodes()
    for node in nodes:
        A_ub[node, i] = 1

b_ub = np.ones(n) * k * D * (D - 1) ** (k - 2)

bounds = (0.0, 1.0)

res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print(len(triangles))
print(-res.fun)
