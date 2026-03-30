"""
Run LV finite-size scaling scans on a pre-generated network library.

Design choice
-------------
- Proxy generation keeps the OLD sparsification algorithm unchanged:
      sparsify_network_v2(...) in sparsification_v2.py
  This uses projected-space quantities (suffix _proj) internally.

- Scaling / theory diagnostics are computed in parallel using reduced-space
  quantities (suffix _red), e.g. A_eff_red = U.T @ A @ U.

This keeps historical comparability while exposing the reduced-space objects
needed by the current theoretical analysis.
"""

import os
import pickle
import numpy as np

from netdyn import LVDynamics
from sparsification_v2 import (
    extract_dynamical_basis,
    compute_projector,
    compute_effective_matrix_proj,
    compute_effective_matrix_red,
    delta_proj,
    delta_red,
    sparsify_network_v2,
)

# ============================================================
# Configuration
# ============================================================
NETWORK_DIR = './percolation_data/network_library'
SAVE_DIR = './percolation_data/scaling_lv_v2'
os.makedirs(SAVE_DIR, exist_ok=True)

N_LIST = [50, 100, 200, 400]
N_INSTANCES = 20
ENERGY_THRESH = 0.95
MIN_K = 3
T_TOTAL = 300
T_SIM = 500
T_DISCARD = 200
BASE_SEED = 500

STORE_PROXY_MATRICES = True
STORE_DELETION_HISTORY = False
SUPPORT_TOL = 1e-8

EPSILON_RANGE = np.sort(np.concatenate([
    np.linspace(0.01, 0.10, 10),
    np.linspace(0.11, 0.30, 8),
    np.linspace(0.31, 0.50, 7),
]))
EPSILON_RANGE = np.unique(np.round(EPSILON_RANGE, 4))


# ============================================================
# LV helpers
# ============================================================

def make_lv_model(N, seed):
    rng = np.random.RandomState(seed)
    alpha = rng.uniform(0.5, 1.5, N)
    theta = rng.uniform(0.5, 1.5, N)
    model = LVDynamics(N, alpha=alpha, theta=theta)
    x0 = rng.uniform(0.1, 1.0, N)
    return model, x0


def simulate_steady_state(model, A, x0, T_sim=500, T_discard=200, seed=None):
    X = model.simulate(A, T_steps=T_sim, init_state=x0, seed=seed)
    x_star = np.mean(X[T_discard:], axis=0)
    phi = float(np.mean(x_star))
    return X, x_star, phi


def jacobian_lv_F(model, A, x_star):
    Ax = A @ x_star
    diag_part = model.alpha - 2 * model.theta * x_star - Ax
    return np.diag(diag_part) - np.diag(x_star) @ A


def jacobian_lv_H(model, A):
    return -(np.diag(model.theta) + A)


def compute_z_row(theta, A, U_dyn):
    N = A.shape[0]
    M = np.diag(theta) + A
    ones_scaled = np.ones(N) / N
    y = np.linalg.solve(M.T, ones_scaled)
    return y @ U_dyn


def compute_base_lv_quantities(A_true, model, x_star_true, phi_true, U_dyn):
    N = A_true.shape[0]
    P = compute_projector(U_dyn)
    Q = np.eye(N) - P

    A_eff_proj_true = compute_effective_matrix_proj(A_true, U_dyn)
    A_eff_red_true = compute_effective_matrix_red(A_true, U_dyn)
    A_eff_proj_norm_F = float(np.linalg.norm(A_eff_proj_true, 'fro'))
    A_eff_red_norm_F = float(np.linalg.norm(A_eff_red_true, 'fro'))

    c_true = U_dyn.T @ x_star_true
    r_true = x_star_true - U_dyn @ c_true

    J_F_true = jacobian_lv_F(model, A_true, x_star_true)
    J_H_true = jacobian_lv_H(model, A_true)

    sigvals_F = np.linalg.svd(J_F_true, compute_uv=False)
    sigvals_H = np.linalg.svd(J_H_true, compute_uv=False)

    z_row = compute_z_row(model.theta, A_true, U_dyn)
    G_obs = float(np.linalg.norm(z_row, 2))
    K_eff_base_red = float((A_eff_red_norm_F / max(abs(phi_true), 1e-12)) * G_obs * np.linalg.norm(c_true, 2))

    return {
        'A_eff_proj_true': A_eff_proj_true,
        'A_eff_red_true': A_eff_red_true,
        'A_eff_proj_norm_F': A_eff_proj_norm_F,
        'A_eff_red_norm_F': A_eff_red_norm_F,
        'c_true': c_true,
        'r_true': r_true,
        'P': P,
        'Q': Q,
        'J_F_true': J_F_true,
        'J_H_true': J_H_true,
        'sigma_min_F': float(sigvals_F[-1]),
        'sigma_min_H': float(sigvals_H[-1]),
        'z_row': z_row,
        'G_obs': G_obs,
        'K_eff_base_red': K_eff_base_red,
    }


def compute_proxy_diagnostics(A_true, A_proxy, U_dyn, x_star_true, x_star_proxy,
                              phi_true, phi_proxy, base, support_tol=1e-8):
    DeltaA = A_proxy - A_true

    A_eff_proj_true = base['A_eff_proj_true']
    A_eff_red_true = base['A_eff_red_true']
    A_eff_proj_proxy = compute_effective_matrix_proj(A_proxy, U_dyn)
    A_eff_red_proxy = compute_effective_matrix_red(A_proxy, U_dyn)
    DeltaA_eff_proj = A_eff_proj_proxy - A_eff_proj_true
    DeltaA_eff_red = A_eff_red_proxy - A_eff_red_true

    delta_val_proj = float(np.linalg.norm(DeltaA_eff_proj, 'fro') /
                           max(np.linalg.norm(A_eff_proj_true, 'fro'), 1e-15))
    delta_val_red = float(np.linalg.norm(DeltaA_eff_red, 'fro') /
                          max(np.linalg.norm(A_eff_red_true, 'fro'), 1e-15))

    P = base['P']
    Q = base['Q']
    c_true = base['c_true']
    r_true = base['r_true']
    z_row = base['z_row']

    Uc = U_dyn @ c_true
    xi = P @ DeltaA @ Q @ r_true + Q @ DeltaA @ P @ Uc + Q @ DeltaA @ Q @ r_true
    xi_norm = float(np.linalg.norm(xi, 2))

    c_proxy = U_dyn.T @ x_star_proxy
    eps_sub = float(np.linalg.norm(c_proxy - c_true, 2) /
                    max(np.linalg.norm(c_true, 2), 1e-12))

    support_true = x_star_true > support_tol
    support_proxy = x_star_proxy > support_tol
    support_hamming = float(np.mean(support_true != support_proxy))
    same_support = bool(np.all(support_true == support_proxy))

    lin_response_vec_red = DeltaA_eff_red @ c_true
    lin_macro_pred = float(abs(z_row @ lin_response_vec_red) /
                           max(abs(phi_true), 1e-12))

    eps_macro = float(abs(phi_proxy - phi_true) / max(abs(phi_true), 1e-12))
    sparsity = float(np.sum(np.abs(A_proxy) > 1e-9) / max(np.sum(np.abs(A_true) > 1e-9), 1))

    return {
        'DeltaA': DeltaA,
        'A_eff_proj_proxy': A_eff_proj_proxy,
        'A_eff_red_proxy': A_eff_red_proxy,
        'DeltaA_eff_proj': DeltaA_eff_proj,
        'DeltaA_eff_red': DeltaA_eff_red,
        'delta_proj': delta_val_proj,
        'delta_red': delta_val_red,
        'eps_macro': eps_macro,
        'phi_proxy': float(phi_proxy),
        'x_star_proxy': x_star_proxy,
        'eps_sub': eps_sub,
        'support_hamming': support_hamming,
        'same_support': same_support,
        'xi': xi,
        'xi_norm': xi_norm,
        'lin_response_vec_red': lin_response_vec_red,
        'lin_macro_pred': lin_macro_pred,
        'sparsity': sparsity,
    }


# ============================================================
# Main scan
# ============================================================

def load_network_file(N):
    path = os.path.join(NETWORK_DIR, f'BA_N{N}_networks_n{N_INSTANCES}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f), path


for N in N_LIST:
    network_data, network_path = load_network_file(N)
    results = []

    print('\n' + '=' * 76)
    print(f'LV scaling scan v2 | N={N} | {len(network_data["graphs"])} graphs')
    print('=' * 76)

    for idx, graph_rec in enumerate(network_data['graphs']):
        graph_id = graph_rec['graph_id']
        seed_i = int(graph_rec['seed'])
        A_true = graph_rec['A_true']

        model_i, x0_i = make_lv_model(N, seed_i)

        X_obs_i = model_i.simulate(A_true, T_steps=T_TOTAL, init_state=x0_i, seed=seed_i)
        U_dyn_i = extract_dynamical_basis(X_obs_i, energy_threshold=ENERGY_THRESH, min_k=MIN_K)
        k_i = int(U_dyn_i.shape[1])

        _, x_star_true, phi_true = simulate_steady_state(
            model_i, A_true, x0_i, T_sim=T_SIM, T_discard=T_DISCARD
        )
        base = compute_base_lv_quantities(A_true, model_i, x_star_true, phi_true, U_dyn_i)

        records_i = []
        for eps in EPSILON_RANGE:
            # Keep old proxy-generation mechanism unchanged.
            A_proxy, deletion_history = sparsify_network_v2(
                A_true, U_dyn_i, epsilon=float(eps), verbose=False
            )

            _, x_star_proxy, phi_proxy = simulate_steady_state(
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

            rec = {
                'epsilon': float(eps),
                'delta_proj': proxy['delta_proj'],
                'delta_red': proxy['delta_red'],
                # Convenience alias: the current scaling analysis uses reduced-space delta.
                'delta': proxy['delta_red'],
                'eps_macro': proxy['eps_macro'],
                'phi_proxy': proxy['phi_proxy'],
                'sparsity': proxy['sparsity'],
                'eps_sub': proxy['eps_sub'],
                'support_hamming': proxy['support_hamming'],
                'same_support': proxy['same_support'],
                'xi_norm': proxy['xi_norm'],
                'lin_macro_pred': proxy['lin_macro_pred'],
                'DeltaA_eff_proj': proxy['DeltaA_eff_proj'],
                'DeltaA_eff_red': proxy['DeltaA_eff_red'],
                'A_eff_proj_proxy': proxy['A_eff_proj_proxy'],
                'A_eff_red_proxy': proxy['A_eff_red_proxy'],
                'x_star_proxy': proxy['x_star_proxy'],
                'xi': proxy['xi'],
            }
            if STORE_PROXY_MATRICES:
                rec['A_proxy'] = A_proxy
                rec['DeltaA'] = proxy['DeltaA']
            if STORE_DELETION_HISTORY:
                rec['deletion_history'] = deletion_history

            records_i.append(rec)

        results.append({
            'graph_id': graph_id,
            'seed': seed_i,
            'lv_params': {
                'alpha': model_i.alpha,
                'theta': model_i.theta,
                'dt': float(model_i.dt),
            },
            'x0': x0_i,
            'X_obs': X_obs_i,
            'U_dyn': U_dyn_i,
            'k': k_i,
            'x_star_true': x_star_true,
            'phi_true': float(phi_true),
            'A_eff_proj_true': base['A_eff_proj_true'],
            'A_eff_red_true': base['A_eff_red_true'],
            'A_eff_proj_norm_F': base['A_eff_proj_norm_F'],
            'A_eff_red_norm_F': base['A_eff_red_norm_F'],
            'c_true': base['c_true'],
            'r_true': base['r_true'],
            'z_row': base['z_row'],
            'G_obs': base['G_obs'],
            'K_eff_base_red': base['K_eff_base_red'],
            'sigma_min_F': base['sigma_min_F'],
            'sigma_min_H': base['sigma_min_H'],
            'records': records_i,
        })

        print(
            f'[{N}] {idx+1:>2d}/{len(network_data["graphs"])} done '
            f'(k={k_i}, phi={phi_true:.4f}, '
            f'sigmaF={base["sigma_min_F"]:.5f}, sigmaH={base["sigma_min_H"]:.5f})'
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
            'store_proxy_matrices': bool(STORE_PROXY_MATRICES),
            'store_deletion_history': bool(STORE_DELETION_HISTORY),
            'base_seed': int(BASE_SEED),
        },
    }

    save_path = os.path.join(SAVE_DIR, f'BA_N{N}_LV_scan_v2_n{len(results)}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'Saved: {save_path}')

print('\nAll LV scans completed.')
