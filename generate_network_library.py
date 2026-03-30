"""
Generate and save a reusable BA network library.

This script separates network generation from downstream dynamics so the same
base graphs can be reused for LV / SIS / Kuramoto / etc.
"""

import os
import pickle
import numpy as np
import networkx as nx


# ============================================================
# Configuration
# ============================================================
SAVE_DIR = './percolation_data/network_library'
os.makedirs(SAVE_DIR, exist_ok=True)

N_LIST = [50, 100, 200, 400]
M_BA = 7
N_INSTANCES = 20
BASE_SEED = 500
WEIGHT_LOW = 0.5
WEIGHT_HIGH = 1.5


def generate_ba_weighted_directed(N, m, seed, w_low=0.5, w_high=1.5):
    """
    Generate a weighted directed BA network by:
      1. sampling an undirected BA graph,
      2. duplicating each edge into both directions,
      3. assigning independent directional weights.

    This produces a reciprocated directed topology with asymmetric weights.
    """
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    G = G.to_directed()
    A = nx.to_numpy_array(G).astype(float)
    rng = np.random.RandomState(seed + 1000)
    A *= rng.uniform(w_low, w_high, size=A.shape)
    return A


def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def edge_count(A, tol=1e-9):
    return int(np.sum(np.abs(A) > tol))


for N in N_LIST:
    graphs = []
    for i in range(N_INSTANCES):
        seed_i = BASE_SEED + N * 100 + i
        graph_id = f'BA_N{N}_seed{seed_i}'
        A_true = generate_ba_weighted_directed(
            N, M_BA, seed_i, w_low=WEIGHT_LOW, w_high=WEIGHT_HIGH
        )
        graphs.append({
            'graph_id': graph_id,
            'seed': int(seed_i),
            'A_true': A_true,
            'n_edges': edge_count(A_true),
            'rho_true': spectral_radius(A_true),
        })
        print(f'[{N}] {i+1:>2d}/{N_INSTANCES}: edges={graphs[-1]["n_edges"]}, rho={graphs[-1]["rho_true"]:.4f}')

    save_data = {
        'net_type': 'BA',
        'directed_style': 'reciprocal_topology_asymmetric_weights',
        'N': int(N),
        'n_instances': int(N_INSTANCES),
        'graphs': graphs,
        'params': {
            'M_BA': int(M_BA),
            'weight_low': float(WEIGHT_LOW),
            'weight_high': float(WEIGHT_HIGH),
            'base_seed': int(BASE_SEED),
        },
    }

    save_path = os.path.join(SAVE_DIR, f'BA_N{N}_networks_n{N_INSTANCES}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'Saved: {save_path}\n')
