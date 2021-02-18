import numpy as np
import networkx as nx
from relm.mechanisms import LaplaceMechanism, CauchyMechanism

# =============================================================================
# Subgraph Counting
 # =============================================================================
n = 2**6
p = 0.1
%time g = nx.random_graphs.gnp_random_graph(n, p)
%time M = nx.linalg.graphmatrix.adjacency_matrix(g).astype(np.int)

%time A = (M @ M)#.todense()

%time D0 = M.sum(0)
%time D1 = M.sum(1)
%time D = D0 + D1
%time B = D - 2*A

# np.fill_diagonal(A, 0)
# np.fill_diagonal(B, 0)

A.setdiag(0)
B.setdiag(0)

# def local_sensitivity_dist(A, B, s):
#     n = A.shape[0]
#     temp = np.floor((s + np.minimum(s, B))/2.0)
#     C = np.minimum(A + temp, n - 2)
#     return np.max(C)
#
# %time temp = np.array([local_sensitivity_dist(A,B,s) for s in range(n+1)])

def local_sensitivity_dist(A, B, n):
    W = np.array(A).flatten()
    X = np.array(B).flatten()
    idx = np.argsort(W)
    Y = W[idx]
    Z = X[idx]
    delta = np.concatenate((np.zeros(1), Y[1:] != Y[:-1]))
    new_val_idxs = np.concatenate((np.array([0]), np.where(delta)[0], np.array([len(Y)])))
    M = np.empty((1, len(new_val_idxs)-1), dtype=np.int)
    N = np.empty((1, len(new_val_idxs)-1), dtype=np.int)
    for i in range(len(new_val_idxs)-1):
        M[0,i] = Y[new_val_idxs[i]]
        N[0,i] = np.max(Z[new_val_idxs[i]:new_val_idxs[i+1]])
    s = np.expand_dims(np.arange(n+1), 1)
    C = np.minimum(M + np.floor((s + np.minimum(s, N))/2.0), n - 2)
    return np.max(C, axis=1)

%time temp = local_sensitivity_dist(A, B, n)

epsilon = 0.1
beta = epsilon / 6.0
mult = np.exp(-beta * np.arange(n+1))

smooth_sensitivity = np.max(temp * mult)
print(smooth_sensitivity)
print(n-2)

# dummy_A = np.array([[np.sum(M[i,:] * M[:,j]) for j in range(n)] for i in range(n)])
# dummy_B = np.array([[np.sum(M[i,:].todense().flatten() ^ M[:,j].todense().flatten()) for j in range(n)] for i in range(n)])
# print(np.all(dummy_A == A))
# print(np.all(dummy_B == B))

triangle_count = np.array([sum(nx.triangles(g).values()) / 3.0])

mechanism = CauchyMechanism(epsilon=epsilon, beta=beta)
dp_triangle_count = mechanism.release(triangle_count, smooth_sensitivity)

print(triangle_count)
print(dp_triangle_count)
