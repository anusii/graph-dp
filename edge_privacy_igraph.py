import numpy as np
import igraph
from relm.mechanisms import LaplaceMechanism, CauchyMechanism

# =============================================================================
# Triangle Counts
# =============================================================================
def f(s, a, b, n):
    """ Utility function used in computation of local sensitivities. """
    return np.minimum(a + np.floor((s + np.minimum(s, b)) / 2.0), n - 2)

def find_survivors(A, B, n):
    """ Find surviving pairs A[i,j], B[i,j].

        For each distinct value set(A), keep only the pair
        (A[i,j], B[i,j]) with the largest value of B[i,j].
    """
    W = np.array(A).flatten()
    X = np.array(B).flatten()
    idx = np.argsort(W)
    Y = W[idx]
    Z = X[idx]
    delta = np.concatenate((np.zeros(1), Y[1:] != Y[:-1]))
    new_val_idxs = np.concatenate((np.array([0]),
                                   np.where(delta)[0],
                                   np.array([len(Y)])))
    M = np.empty((len(new_val_idxs)-1), dtype=np.int)
    N = np.empty((len(new_val_idxs)-1), dtype=np.int)
    for i in range(len(new_val_idxs)-1):
        M[i] = Y[new_val_idxs[i]]
        N[i] = np.max(Z[new_val_idxs[i]:new_val_idxs[i+1]])
    return M[::-1], N[::-1]

def first_hit(x, break_list):
    """ Find the key for the first pair (k,v) with v > x. """
    return next((k for k,v in break_list if x <= v))

def h(s, break_list):
    """ Compute the maximal value of f(s, a, b, n) over all possible a, b. """
    a, b = first_hit(s, break_list)
    return f(s, a, b, n)

def local_sensitivity_dist(A, B, n):
    """ Compute the local sensitivity at distance s for 0 <= s <= n. """
    M, N = find_survivors(A, B, n)
    prev_survivor = (M[0], N[0])
    break_points = {prev_survivor: n+1}
    for survivor in zip(M[1:], N[1:]):
        a0, b0 = prev_survivor
        a1, b1 = survivor
        intersection = 2*(a0 - a1) + b0
        if b0 <= intersection <= b1:
            break_points[prev_survivor] = intersection
            break_points[survivor] = n+1
            prev_survivor = survivor
    break_list = sorted(break_points.items(), key=lambda _: _[1])
    return np.array([h(s, break_list) for s in range(n+1)])

# =============================================================================
# Generate a random graph
n = 2**8
p = 0.5
# g = nx.random_graphs.gnp_random_graph(n, p)
g = igraph.Graph.Erdos_Renyi(n=n, p=p)

# Compute the adjacency matrix
M = g.get_adjacency_sparse()

# -----------------------------------------------------------------------------
# Compute the partial count matrices
A = (M @ M).todense()
B = M.sum(0) + M.sum(1) - 2*A

# Zero out the main diagonal because we are
# interested only in indices (i,j) with i != j
np.fill_diagonal(A, 0)
np.fill_diagonal(B, 0)

# Compute the local sensitivity at distance s for 0 <= s <= n
lsd = local_sensitivity_dist(A, B, n)

# Compute the smooth sensitivity
epsilon = 1.0
beta = epsilon / 6.0
smooth_scaling = np.exp(-beta * np.arange(n+1))
smooth_sensitivity = np.max(lsd * smooth_scaling)

# -----------------------------------------------------------------------------
# Compute the exact triangle count
g.to_directed()
triangle_count = np.array([g.triad_census()[-1]], dtype=np.float)

# Create a differentiall private release mechanism
mechanism = CauchyMechanism(epsilon=epsilon, beta=beta)

# Compute the differentially private query response
dp_triangle_count = mechanism.release(triangle_count, smooth_sensitivity)

print(triangle_count)
print(dp_triangle_count)

# =============================================================================
# Minimum Spanning Tree
# =============================================================================
def f1(k, edge_weights, bound):
    if k >= len(edge_weights): return bound
    else: return edge_weights[k]

def f2(k, edge_weights, bound):
    if (k+1) >= len(edge_weights): temp = bound
    else: temp = edge_weights[k+1]
    if len(edge_weights) > 0: return temp - edge_weights[0]
    else: return 0

def lightweight_graph(g, w):
    heavy_edges = [e for e in g.es if e["weight"] > w]
    gg = g.copy()
    gg.delete_edges(heavy_edges)
    return gg

def min_cut(g, w, **args):
    source = args.get("source", -1)
    target = args.get("target", -1)
    return lightweight_graph(g, w).edge_connectivity(source, target)

def retrieve(costs, key, g, w, **args):
    if key not in costs:
        costs[key] = min_cut(g, w[key], **args)
    return costs[key]

def first_hit_weight(k, g, w, bound, costs, **args):
    cost = retrieve(costs, len(w)-1, g, w, **args)
    if cost <= k:
        return bound

    cost = retrieve(costs, 0, g, w, **args)
    if cost > k:
        return w[0]

    high = len(w) - 1
    low = 0
    while (high - low) > 1:
        mid = (low + high) // 2
        cost = retrieve(costs, mid, g, w, **args)
        if cost <= k:
            low = mid
        else:
            high = mid
    return w[high]

# =============================================================================
# Generate a random graph
n = 2**8
p = 0.1
g = igraph.Graph.Erdos_Renyi(n=n, p=p)
bound = 10.0 # An upper bound on the edge weights in the graph
edge_weights = bound * np.random.randint(1,11, size=len(g.es))/10.0
g.es["weight"] = edge_weights
edge_weights.sort()
edge_weights = sorted(set(edge_weights))

# -----------------------------------------------------------------------------
# Compute the local sensitivity at distance s for 0 <= s <= n
costs = dict()
lsd1 = np.array([first_hit_weight(k, g, edge_weights, bound, costs)
                 for k in range(n+1)])

mst = g.spanning_tree(weights="weight")
lsd2 = np.zeros(n+1)
for e, st in zip(mst.es, mst.get_edgelist()):
    costs = dict()
    first_hit_weights = [first_hit_weight(k+1, g, edge_weights, bound, costs, source=st[0], target=st[1])
                         for k in range(n+1)]
    lsd2_e = np.array(first_hit_weights) - e["weight"]
    lsd2 = np.maximum(lsd2, lsd2_e)

lsd = np.maximum(lsd1, lsd2)

# Compute the smooth sensitivity
epsilon = 1.0
beta = epsilon / 6.0
smooth_scaling = np.exp(-beta * np.arange(n+1))
smooth_sensitivity = np.max(lsd * smooth_scaling)
print(smooth_sensitivity)

# -----------------------------------------------------------------------------
# Compute the exact MST cost
mst = g.spanning_tree(weights="weight")
mst_cost = np.array([np.sum(mst.strength(weights="weight")) / 2.0])

# Create a differentiall private release mechanism
mechanism = CauchyMechanism(epsilon=epsilon, beta=beta)

# Compute the differentially private query response
dp_mst_cost = mechanism.release(mst_cost, smooth_sensitivity)

print(mst_cost)
print(dp_mst_cost)
