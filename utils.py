import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from scipy.stats import binom
import matplotlib as mpl
from sklearn import datasets
import itertools

def L2_dist(a, b):
    return np.linalg.norm(a-b)

def compute_G0(X, d=L2_dist):
    W = np.zeros((len(X), len(X)))
    for i in range(len(X)-1):
        for j in range(i+1, len(X)):
            W[i, j] = d(X[i], X[j])
            W[j, i] = d(X[i], X[j])
    D_sqinv = np.diag(np.sum(W, axis=0) ** -0.5)
    return W, D_sqinv

def compute_laplacian(W, D_sqinv):
    return np.eye(len(W)) - D_sqinv @ W @ D_sqinv

def compute_AM_normalised(L, D_sqinv, M):
    V, A = np.linalg.eig(L)
    V = 1 - V
    idx = V.argsort()  
    V = V[idx]
    A = A[:,idx]
    V = np.diag(V)
    A = A[:, :M]
    return D_sqinv @ A

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
        self.hierarchy = None
        self.A = None
        self.omega = 0

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])
        self.omega += w

    # Search function, ça trouve un point fixe en fait, en partant de celui qu'on veut

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        hierarchy = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        rank = [0] * self.V
        parent = list(range(self.V))
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e += 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
                hierarchy.append(result.copy())
        self.hierarchy = hierarchy
        return hierarchy
            
        
    def compute_A(self):
        A = np.zeros((self.V, self.V))
        for e in self.graph:
            A[e[0], e[1]] = e[2]
            A[e[1], e[0]] = e[2]
        self.A = A

    def cross_sum(self, A, l1, l2):
        return np.sum((A * l1).T * l2)
    
    
    def compute_mat(self):
        A = self.A
        mat = np.zeros((2 * self.V - 1, self.V + 6))
        for i in range(self.V):
            mat[i, 5] = 1
            mat[i, i + 6] = 1
        hierarchy = self.hierarchy
        l_buf = list(range(self.V))
        for i, e in enumerate(hierarchy[-1]):
            u, v, _ = e
            j = i + self.V
            mat[l_buf[u], 0] = j
            mat[l_buf[v], 0] = j
            mat[l_buf[u], 1] = l_buf[v]
            mat[l_buf[v], 1] = l_buf[u]
            mat[j, 2] = l_buf[u]
            mat[j, 3] = l_buf[v]
            mat[j, 4:] = mat[l_buf[u], 4:] + mat[l_buf[v], 4:]
            mat[j, 4] += self.cross_sum(A, mat[l_buf[u], 6:], mat[l_buf[v], 6:])
            for k in range(self.V):
                if mat[j, k + 6]:
                    l_buf[k] = j
        return mat
    
def compute_Ge(A, d=L2_dist):
    g = Graph(len(A))
    W = np.zeros((len(A), len(A)))
    for i in range(len(A)-1):
        for j in range(i+1, len(A)):
            W[i, j] = d(A[i], A[j])
            W[j, i] = d(A[i], A[j])
            g.add_edge(i, j, d(A[i], A[j]))
    return W, g


def compute_pfa(p, V, V_):
    return binom.cdf(V_, V, p)

def compute_nfa(p, V, V_, deltas, K=10, eps=1):
    return compute_pfa(p, V, V_) * eps / deltas[hash_function(V_, K, V)]


def build_rand_mat(K, x_min, y_min, x_max, y_max):
    X_H0 = np.random.uniform(size=(K,2)) * np.array([x_max - x_min, y_max - y_min]) + np.array([x_min, y_min])
    W, D_sqinv = compute_G0(X_H0)
    L = compute_laplacian(W, D_sqinv)
    A = compute_AM_normalised(L, D_sqinv, 3) #3 peut être faible
    W, g = compute_Ge(A)
    g.compute_A()
    g.kruskal_algo()
    mat = g.compute_mat()
    return mat, g

def hash_function(V_, K, N):
    return int((V_ - 1) * K / N)


def update_deltas(deltas, eps, K, size, x_min, y_min, x_max, y_max, update_, Q=30, conf_level=0.05):
    res_q = np.zeros((K, Q), dtype=int)
    for q in range(Q):
        res = np.zeros(K, dtype=int)
        mat, g = build_rand_mat(size - 1, x_min, y_min, x_max, y_max)
        omega = g.omega
        V = g.V
        for i in range(len(mat)):
            k = hash_function(int(mat[i, 5]), K, size) 
            pfa = compute_pfa(min(mat[i, 4] / omega, 1), V , mat[i,5])
            if pfa < deltas[k]:
                res[k] += 1
        res_q[:, q] = res.copy()
    s = np.maximum(np.std(res_q, axis=1), 1e-5)
    m = np.mean(res_q, axis=1)
    upd = t.cdf((m - eps / K) / s * np.sqrt(Q - 1), Q-1) > conf_level
    deltas -= (2 * upd - 1) * update_
    deltas = np.minimum(1, deltas)
    return deltas, res_q

def compute_deltas(size, Q=30, conf_level=0.05, eps=1, K=10):
    deltas = 0.99 * np.ones(K)
    update_ = 1e-3
    for i in tqdm(range(100)):
        if i == 10:
            update_ = 1e-4
        if i == 20:
            update_ = 1e-5
        if i == 30:
            update_ = 1e-6
        deltas, res_q = update_deltas(deltas, eps, K, size, x_min, y_min, x_max, y_max, update_, Q, conf_level)
    return deltas


