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

def compute_nfa(p, V, V_):
    return compute_pfa(p, V, V_)

