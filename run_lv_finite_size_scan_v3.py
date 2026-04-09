"""
Run LV finite-size scaling scans on a pre-generated network library.

v3 changes relative to v2
-------------------------
- Keep the OLD proxy-generation mechanism unchanged:
      sparsify_network_v2(...) in sparsification_v2.py
- Replace dense proxy-level saves with a compact replay schema:
      A_true_edge_index + deleted_edge_ids + added_edges
- Save the finalized variable set described in README_saved_variables_lv_scaling.md
- Store both the universal and the LV-specialized shielding objects
- Store K_leak_norm at graph-level and DeltaA_norm_F at proxy-level
- Drop dense matrices / vectors that were only needed transiently during runtime

This script is designed so that later analyses can:
1. verify the decomposition of K_eff in formula (38),
2. verify the LV response bound in formulas (32)/(35), and
3. reconstruct each final A_proxy without storing dense adjacency matrices.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np

from netdyn import LVDynamics
from sparsification_v2 import (
    compute_effective_matrix_red,
    compute_projector,
    extract_dynamical_basis,
    sparsify_network_v2,
)

# ============================================================
# Configuration
# ============================================================
NETWORK_DIR = './percolation_data/network_library'
SAVE_DIR = './percolation_data/scaling_lv_v3'
os.makedirs(SAVE_DIR, exist_ok=True)

N_LIST = [50, 100, 200, 400]
N_INSTANCES = 20
ENERGY_THRESH = 0.95
MIN_K = 3
T_TOTAL = 300
T_SIM = 500
T_DISCARD = 200
BASE_SEED = 500

SUPPORT_TOL = 1e-8
EDGE_TOL = 1e-9

EPSILON_RANGE = np.sort(np.concatenate([
    np.linspace(0.01, 0.10, 10),
    np.linspace(0.11, 0.30, 8),
    np.linspace(0.31, 0.50, 7),
]))
EPSILON_RANGE = np.unique(np.round(EPSILON_RANGE, 4))


# ============================================================
# LV helpers
# ============================================================

def make_lv_model(N: int, seed: int) -> Tuple[LVDynamics, np.ndarray]:
    rng = np.random.RandomState(seed)
    alpha = rng.uniform(0.5, 1.5, N)
    theta = rng.uniform(0.5, 1.5, N)
    model = LVDynamics(N, alpha=alpha, theta=theta)
    x0 = rng.uniform(0.1, 1.0, N)
    return model, x0


def simulate_steady_state(
    model: LVDynamics,
    A: np.ndarray,
    x0: np.ndarray,
    T_sim: int = 500,
    T_discard: int = 200,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Simulate and return the empirical steady state and macro order parameter."""
    X = model.simulate(A, T_steps=T_sim, init_state=x0, seed=seed)
    x_star = np.mean(X[T_discard:], axis=0)
    phi = float(np.mean(x_star))
    return x_star, phi


def jacobian_lv(model: LVDynamics, A: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    """Full Jacobian J = dF/dx evaluated at x_star."""
    Ax = A @ x_star
    diag_part = model.alpha - 2.0 * model.theta * x_star - Ax
    return np.diag(diag_part) - np.diag(x_star) @ A


def hessian_skeleton_lv(model: LVDynamics, A: np.ndarray) -> np.ndarray:
    """Convenience matrix H = -(Theta + A), used by the LV-specialized response."""
    return -(np.diag(model.theta) + A)


def macro_gradient_mean_abundance(N: int) -> np.ndarray:
    """Gradient of Phi(x) = mean(x) used in the current LV experiments."""
    return np.ones(N, dtype=float) / float(N)


def compute_universal_response_row(
    J_true: np.ndarray,
    x_star_true: np.ndarray,
    grad_phi: np.ndarray,
) -> np.ndarray:
    """
    Compute the universal node-level response row

        w^T = grad_phi^T J^{-1} diag(x_star)

    returned as a 1D array of length N.
    """
    y = np.linalg.solve(J_true.T, grad_phi)
    return y * x_star_true


def compute_lv_response_row(theta: np.ndarray, A_true: np.ndarray) -> np.ndarray:
    """
    Compute the LV-specialized node-level response row

        v^T = (1/N) 1^T (Theta + A)^{-1}

    returned as a 1D array of length N.
    """
    N = A_true.shape[0]
    M = np.diag(theta) + A_true
    grad_phi = macro_gradient_mean_abundance(N)
    return np.linalg.solve(M.T, grad_phi)


def compute_part_shielding(shield_vec: np.ndarray) -> float:
    return float(np.linalg.norm(shield_vec, 2))


def compute_consistency_error(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Universal and LV shielding vectors are theoretically equal up to sign under
    the LV specialization, so use the smaller of same-sign / flipped-sign errors.
    """
    denom = max(np.linalg.norm(vec_a, 2), np.linalg.norm(vec_b, 2), eps)
    err_same = np.linalg.norm(vec_a - vec_b, 2) / denom
    err_flip = np.linalg.norm(vec_a + vec_b, 2) / denom
    return float(min(err_same, err_flip))


def adjacency_to_edge_index(A: np.ndarray, edge_tol: float = EDGE_TOL) -> np.ndarray:
    """Return directed edge list (src, dst) for all nonzero entries in row-major order."""
    return np.argwhere(np.abs(A) > edge_tol).astype(np.int32)


def adjacency_hash(A: np.ndarray) -> str:
    """Hash the dense adjacency to guard against accidental library drift."""
    arr = np.ascontiguousarray(A.astype(np.float64, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def extract_edge_edits(
    A_true: np.ndarray,
    A_proxy: np.ndarray,
    edge_tol: float = EDGE_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a compact replay schema for A_proxy relative to A_true.

    deleted_edge_ids:
        integer indices into adjacency_to_edge_index(A_true)
    added_edges:
        directed edge list (src, dst) present in A_proxy but absent in A_true
    """
    mask_true = np.abs(A_true) > edge_tol
    mask_proxy = np.abs(A_proxy) > edge_tol

    deleted_flags = (~mask_proxy)[mask_true]
    deleted_edge_ids = np.flatnonzero(deleted_flags).astype(np.int32)

    added_edges = np.argwhere((~mask_true) & mask_proxy).astype(np.int32)
    if added_edges.size == 0:
        added_edges = np.empty((0, 2), dtype=np.int32)

    return deleted_edge_ids, added_edges


def compute_base_lv_quantities(
    A_true: np.ndarray,
    model: LVDynamics,
    x_star_true: np.ndarray,
    phi_true: float,
    U_dyn: np.ndarray,
) -> Dict[str, Any]:
    """Compute and return graph-level quantities shared across all proxies."""
    N = A_true.shape[0]
    grad_phi = macro_gradient_mean_abundance(N)

    P = compute_projector(U_dyn)
    Q = np.eye(N) - P

    A_eff_true = compute_effective_matrix_red(A_true, U_dyn)
    part_topology = float(np.linalg.norm(A_eff_true, 'fro'))

    c_true = U_dyn.T @ x_star_true
    part_forcing = float(np.linalg.norm(c_true, 2))
    r_true = x_star_true - U_dyn @ c_true

    J_true = jacobian_lv(model, A_true, x_star_true)
    H_true = hessian_skeleton_lv(model, A_true)

    w_row_univ = compute_universal_response_row(J_true, x_star_true, grad_phi)
    shield_vec_univ = w_row_univ @ U_dyn
    part_shielding_univ = compute_part_shielding(shield_vec_univ)

    v_row_lv = compute_lv_response_row(model.theta, A_true)
    shield_vec_lv = v_row_lv @ U_dyn
    part_shielding_lv = compute_part_shielding(shield_vec_lv)

    shield_consistency_error = compute_consistency_error(shield_vec_univ, shield_vec_lv)

    part_macro = float(1.0 / max(abs(phi_true), 1e-12))
    K_leak_norm = float(part_macro * np.linalg.norm(v_row_lv, 2))

    sigvals_J = np.linalg.svd(J_true, compute_uv=False)
    sigvals_H = np.linalg.svd(H_true, compute_uv=False)

    return {
        'A_eff_true': A_eff_true,
        'part_topology': part_topology,
        'c_true': c_true,
        'part_forcing': part_forcing,
        'r_true': r_true,
        'P': P,
        'Q': Q,
        'J_true': J_true,
        'sigma_min_J': float(sigvals_J[-1]),
        'sigma_min_H': float(sigvals_H[-1]),
        'w_row_univ': w_row_univ,
        'shield_vec_univ': shield_vec_univ,
        'part_shielding_univ': part_shielding_univ,
        'v_row_lv': v_row_lv,
        'shield_vec_lv': shield_vec_lv,
        'part_shielding_lv': part_shielding_lv,
        'shield_consistency_error': shield_consistency_error,
        'part_macro': part_macro,
        'K_leak_norm': K_leak_norm,
    }


def compute_proxy_diagnostics(
    A_true: np.ndarray,
    A_proxy: np.ndarray,
    U_dyn: np.ndarray,
    x_star_true: np.ndarray,
    x_star_proxy: np.ndarray,
    phi_true: float,
    phi_proxy: float,
    base: Dict[str, Any],
    support_tol: float = SUPPORT_TOL,
) -> Dict[str, Any]:
    """Compute the compact proxy-level quantities required by the v3 schema."""
    DeltaA = A_proxy - A_true
    DeltaA_norm_F = float(np.linalg.norm(DeltaA, 'fro'))

    A_eff_proxy = compute_effective_matrix_red(A_proxy, U_dyn)
    DeltaA_eff = A_eff_proxy - base['A_eff_true']
    delta = float(np.linalg.norm(DeltaA_eff, 'fro') / max(base['part_topology'], 1e-15))

    P = base['P']
    Q = base['Q']
    c_true = base['c_true']
    r_true = base['r_true']
    Uc = U_dyn @ c_true

    xi = P @ DeltaA @ Q @ r_true + Q @ DeltaA @ P @ Uc + Q @ DeltaA @ Q @ r_true
    xi_norm = float(np.linalg.norm(xi, 2))

    support_true = x_star_true > support_tol
    support_proxy = x_star_proxy > support_tol
    same_support = bool(np.all(support_true == support_proxy))
    support_hamming = float(np.mean(support_true != support_proxy))

    eps_macro = float(abs(phi_proxy - phi_true) / max(abs(phi_true), 1e-12))

    deleted_edge_ids, added_edges = extract_edge_edits(A_true, A_proxy, edge_tol=EDGE_TOL)

    return {
        'epsilon': None,  # filled by caller
        'eps_macro': eps_macro,
        'DeltaA_eff': DeltaA_eff,
        'delta': delta,
        'xi_norm': xi_norm,
        'DeltaA_norm_F': DeltaA_norm_F,
        'same_support': same_support,
        'support_hamming': support_hamming,
        'deleted_edge_ids': deleted_edge_ids,
        'added_edges': added_edges,
    }


# ============================================================
# Main scan
# ============================================================

def load_network_file(N: int) -> Tuple[Dict[str, Any], str]:
    path = os.path.join(NETWORK_DIR, f'BA_N{N}_networks_n{N_INSTANCES}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f), path


for N in N_LIST:
    network_data, network_path = load_network_file(N)
    results = []

    print('\n' + '=' * 76)
    print(f'LV scaling scan v3 | N={N} | {len(network_data["graphs"])} graphs')
    print('=' * 76)

    for idx, graph_rec in enumerate(network_data['graphs']):
        graph_id = graph_rec['graph_id']
        seed_i = int(graph_rec['seed'])
        A_true = graph_rec['A_true']

        model_i, x0_i = make_lv_model(N, seed_i)

        X_obs_i = model_i.simulate(A_true, T_steps=T_TOTAL, init_state=x0_i, seed=seed_i)
        U_dyn_i = extract_dynamical_basis(X_obs_i, energy_threshold=ENERGY_THRESH, min_k=MIN_K)
        k_i = int(U_dyn_i.shape[1])

        x_star_true, phi_true = simulate_steady_state(
            model_i, A_true, x0_i, T_sim=T_SIM, T_discard=T_DISCARD
        )
        base = compute_base_lv_quantities(A_true, model_i, x_star_true, phi_true, U_dyn_i)

        A_true_edge_index = adjacency_to_edge_index(A_true)
        A_true_hash = adjacency_hash(A_true)

        records_i = []
        for eps in EPSILON_RANGE:
            A_proxy, _deletion_history = sparsify_network_v2(
                A_true, U_dyn_i, epsilon=float(eps), verbose=False
            )

            x_star_proxy, phi_proxy = simulate_steady_state(
                model_i, A_proxy, x0_i, T_sim=T_SIM, T_discard=T_DISCARD
            )

            proxy = compute_proxy_diagnostics(
                A_true=A_true,
                A_proxy=A_proxy,
                U_dyn=U_dyn_i,
                x_star_true=x_star_true,
                x_star_proxy=x_star_proxy,
                phi_true=phi_true,
                phi_proxy=phi_proxy,
                base=base,
                support_tol=SUPPORT_TOL,
            )
            proxy['epsilon'] = float(eps)
            records_i.append(proxy)

        results.append({
            'graph_id': graph_id,
            'seed': seed_i,
            'lv_params': {
                'alpha': model_i.alpha,
                'theta': model_i.theta,
                'dt': float(model_i.dt),
            },
            'x0': x0_i,
            'k': k_i,
            'A_true_edge_index': A_true_edge_index,
            'A_true_hash': A_true_hash,
            'U_dyn': U_dyn_i,
            'x_star_true': x_star_true,
            'phi_true': float(phi_true),
            'part_macro': base['part_macro'],
            'A_eff_true': base['A_eff_true'],
            'part_topology': base['part_topology'],
            'w_row_univ': base['w_row_univ'],
            'shield_vec_univ': base['shield_vec_univ'],
            'part_shielding_univ': base['part_shielding_univ'],
            'v_row_lv': base['v_row_lv'],
            'shield_vec_lv': base['shield_vec_lv'],
            'part_shielding_lv': base['part_shielding_lv'],
            'shield_consistency_error': base['shield_consistency_error'],
            'c_true': base['c_true'],
            'part_forcing': base['part_forcing'],
            'K_leak_norm': base['K_leak_norm'],
            # Optional diagnostics kept at graph-level only.
            'sigma_min_J': base['sigma_min_J'],
            'sigma_min_H': base['sigma_min_H'],
            'records': records_i,
        })

        print(
            f'[{N}] {idx+1:>2d}/{len(network_data["graphs"])} done '
            f'(k={k_i}, phi={phi_true:.4f}, '
            f'shield_lv={base["part_shielding_lv"]:.5e}, '
            f'cons_err={base["shield_consistency_error"]:.3e})'
        )

    save_data = {
        'net_type': network_data['net_type'],
        'N': int(N),
        'dynamics': 'LV',
        'n_instances': int(len(results)),
        'source_network_pkl': network_path,
        'epsilon_range': EPSILON_RANGE,
        'results': results,
        'params': {
            'energy_thresh': float(ENERGY_THRESH),
            'min_k': int(MIN_K),
            'T_total': int(T_TOTAL),
            'T_sim': int(T_SIM),
            'T_discard': int(T_DISCARD),
            'support_tol': float(SUPPORT_TOL),
            'edge_tol': float(EDGE_TOL),
            'base_seed': int(BASE_SEED),
        },
    }

    save_path = os.path.join(SAVE_DIR, f'BA_N{N}_LV_scan_v3_n{len(results)}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'Saved: {save_path}')

print('\nAll LV scans completed.')
