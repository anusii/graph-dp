import numpy as np
import networkx as nx

import scipy.optimize
import scipy.interpolate
import scipy.special

from relm.mechanisms import LaplaceMechanism, CauchyMechanism

# =============================================================================
# Convex Optimization Approach
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
        [(("left", v), ("right", u), 1) for u, v in G.edges()], weight="capacity"
    )
    return F


# =============================================================================

n = np.random.randint(2 ** 7, 2 ** 8)
D = 2 ** 3
p = 2 ** -6
G = nx.random_graphs.gnp_random_graph(n, p)
# G = nx.star_graph(n - 1)
F = flow_graph(G, D)
nodes = list(F.nodes())
edges = list(F.edges())
adjacency = np.zeros((len(nodes), len(edges)))
for j in range(len(edges)):
    i0 = nodes.index(edges[j][0])
    i1 = nodes.index(edges[j][1])
    adjacency[i0, j] = -1
    adjacency[i1, j] = 1

capacities = np.array([F.edges[e]["capacity"] for e in F.edges()])
x0 = np.random.random(capacities.size) * capacities
mask = np.array([("s" in edge) for edge in edges])
bounds = [(0, capacity) for capacity in capacities]
constraint = scipy.optimize.LinearConstraint(adjacency[:-2], 0, 0)

# -----------------------------------------------------------------------------
# edge count
print("Exact edge count = %i" % len(G.edges()))

x = np.arange(D + 1)
h_edge = scipy.interpolate.interp1d(x, x / 2.0)
f_edge = lambda x, *args: -np.sum(h_edge(x[tuple(args[0])]))
res_edge = scipy.optimize.minimize(
    fun=f_edge, x0=x0, args=[mask], bounds=bounds, constraints=[constraint]
)
print("Bounded-degree edge count = %f" % -res_edge.fun)

epsilon = 1.0
y = h_edge(x)
sensitivity = np.max(y) + np.max(y[1:] - y[:-1])
mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
dp_edge_count = mechanism.release(np.array([-res_edge.fun]))[0]
print("Differentially private edge count = %f\n" % dp_edge_count)

# -----------------------------------------------------------------------------
# node count
print("Exact node count = %i" % len(G.nodes()))

x = np.arange(D + 1)
h_node = scipy.interpolate.interp1d(x, np.ones(x.size))
f_node = lambda x, *args: -np.sum(h_node(x[tuple(args[0])]))
res_node = scipy.optimize.minimize(
    fun=f_node, x0=x0, args=[mask], bounds=bounds, constraints=[constraint]
)
print("Bounded-degree node count = %f" % -res_node.fun)

epsilon = 1.0
y = h_node(x)
sensitivity = np.max(y) + np.max(y[1:] - y[:-1])
mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
dp_node_count = mechanism.release(np.array([-res_node.fun]))[0]
print("Differentially private node count = %f\n" % dp_node_count)

# -----------------------------------------------------------------------------
# 3-star count
x = np.arange(D + 1)
h_3star = scipy.interpolate.interp1d(x, scipy.special.comb(x, 3))
f_3star = lambda x, *args: -np.sum(h_3star(x[tuple(args[0])]))
res_3star = scipy.optimize.minimize(
    fun=f_3star, x0=x0, args=[mask], bounds=bounds, constraints=[constraint]
)
print("Bounded-degree 3-star count = %f" % -res_3star.fun)

epsilon = 1.0
y = h_3star(x)
sensitivity = np.max(y) + np.max(y[1:] - y[:-1])
mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
dp_3star_count = mechanism.release(np.array([-res_3star.fun]))[0]
print("Differentially private 3-star count = %f\n" % dp_3star_count)

# =============================================================================
# Linear programming approach
# =============================================================================
n = np.random.randint(2 ** 7, 2 ** 8)
D = 2 ** 2
p = 2 ** -4
G = nx.random_graphs.gnp_random_graph(n, p)

k = 3
triangles = nx.triads_by_type(nx.DiGraph(G))["300"]
print("Exact triangle count = %i" % len(triangles))

# Compute bounded-degree triangle count using linear programming technique
c = -np.ones(len(triangles))
A_ub = np.zeros((n, len(triangles)))
for i, t in enumerate(triangles):
    nodes = triangles[i].nodes()
    for node in nodes:
        A_ub[node, i] = 1

sensitivity = k * D * (D - 1) ** (k - 2)
b_ub = np.ones(n) * sensitivity

bounds = (0.0, 1.0)

res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print("Bounded-degree triangle count = %f" % -res.fun)

# Compute differentially private triangle count
epsilon = 1.0
mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
dp_triangle_count = mechanism.release(np.array([-res.fun]))[0]
print("Differentially private triangle count = %f\n" % dp_triangle_count)
