import numpy as np
from numpy.linalg import svd

# ==============================================================
# 稀疏化算法 v2（保留旧算法 + 新增 reduced-space 量）
# ==============================================================

"""
sparsification_v2.py
====================

设计原则
--------
1. 保留旧版 proxy 生成算法的核心逻辑不变：
   - 旧版 functional reference 仍使用
         A_eff_proj = P @ A @ P,  P = U_dyn @ U_dyn.T
   - 稀疏化排序、接受/拒绝判据、谱半径归一化流程保持兼容。

2. 新增 reduced-space 量用于理论 / finite-size scaling：
         A_eff_red = U_dyn.T @ A @ U_dyn
   这与论文当前的 reduced dynamical subspace 叙事保持一致。

命名约定
--------
- *_proj : N x N 的 full-space projected quantity
- *_red  : k x k 的 reduced-space quantity

兼容性
------
- 为兼容旧脚本，compute_effective_matrix(A, U_dyn) 仍返回 A_eff_proj。
- 新脚本应优先显式使用 compute_effective_matrix_proj / _red。
"""


# ==============================================================
# Basis / projection helpers
# ==============================================================

def extract_dynamical_basis(X_obs, energy_threshold=0.95, min_k=3):
    """
    Extract the dynamical subspace basis via SVD of the trajectory matrix.

    Returns
    -------
    U_dyn : (N, k) orthonormal columns
    """
    U, S, _ = svd(X_obs.T, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    k = max(np.searchsorted(energy, energy_threshold) + 1, min_k)
    return U[:, :k]


def compute_projector(U_dyn):
    """Return P = U_dyn @ U_dyn.T."""
    return U_dyn @ U_dyn.T


def compute_effective_matrix_proj(A, U_dyn):
    """Old full-space projected effective matrix: A_eff_proj = P @ A @ P."""
    P = compute_projector(U_dyn)
    return P @ A @ P


def compute_effective_matrix_red(A, U_dyn):
    """Reduced effective matrix: A_eff_red = U_dyn.T @ A @ U_dyn."""
    return U_dyn.T @ A @ U_dyn


# Backward-compatible alias: old code expects compute_effective_matrix -> N x N projected object.
def compute_effective_matrix(A, U_dyn):
    return compute_effective_matrix_proj(A, U_dyn)


def compute_projected_matrix(A, U_dyn):
    """Alias for the old projected effective matrix."""
    return compute_effective_matrix_proj(A, U_dyn)


def delta_proj(A_candidate, A_eff_proj_ref, U_dyn):
    """Relative projected-space deviation in N x N ambient space."""
    A_eff_proj_cand = compute_effective_matrix_proj(A_candidate, U_dyn)
    ref_norm = np.linalg.norm(A_eff_proj_ref, 'fro')
    if ref_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(A_eff_proj_cand - A_eff_proj_ref, 'fro') / ref_norm)


def delta_red(A_candidate, A_eff_red_ref, U_dyn):
    """Relative reduced-space deviation in k x k space."""
    A_eff_red_cand = compute_effective_matrix_red(A_candidate, U_dyn)
    ref_norm = np.linalg.norm(A_eff_red_ref, 'fro')
    if ref_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(A_eff_red_cand - A_eff_red_ref, 'fro') / ref_norm)


# ==============================================================
# Old greedy sparsification algorithm (kept compatible)
# ==============================================================

def _aeff_deviation(A_candidate, A_eff_ref, U_dyn):
    """
    Backward-compatible helper used by the original algorithm.
    Here A_eff_ref is the projected N x N object, and the deviation is delta_proj.
    """
    return delta_proj(A_candidate, A_eff_ref, U_dyn)


def sparsify_network_v2(A_true, U_dyn, epsilon=0.05, verbose=True):
    """
    Greedy edge-deletion sparsification constrained by projected-space equivalence.

    IMPORTANT
    ---------
    This function intentionally preserves the original proxy-generation logic.
    It uses:
        A_eff_proj = P @ A @ P
    both as the functional reference and for the acceptance criterion.

    Returns
    -------
    A_sparse : (N, N)
    deletion_history : list of dict
        step / edge / delta_proj / accepted / n_edges
    """
    rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))
    A_eff_ref = compute_effective_matrix_proj(A_true, U_dyn)

    rows, cols = np.where(np.abs(A_true) > 1e-9)
    energies = np.abs(A_eff_ref[rows, cols])
    order = np.argsort(energies)
    edge_queue = [(int(rows[o]), int(cols[o])) for o in order]

    A_current = A_true.copy()
    deletion_history = []
    n_deleted = 0
    n_protected = 0
    total = len(edge_queue)

    if verbose:
        print(f"Sparsification v2: {total} edges, epsilon={epsilon}")
        print(f"{'step':>5}  {'edge':>10}  {'delta_proj':>11}  {'decision':>8}  {'remaining':>9}")
        print("-" * 56)

    for step, (i, j) in enumerate(edge_queue):
        A_tmp = A_current.copy()
        A_tmp[i, j] = 0.0
        delta_val_proj = _aeff_deviation(A_tmp, A_eff_ref, U_dyn)
        accepted = delta_val_proj < epsilon

        deletion_history.append({
            'step': int(step),
            'edge': (i, j),
            'delta_proj': float(delta_val_proj),
            'accepted': bool(accepted),
            'n_edges': int(np.sum(np.abs(A_current) > 1e-9)),
        })

        if accepted:
            A_current = A_tmp
            n_deleted += 1
        else:
            n_protected += 1

        if verbose and (step % max(1, total // 20) == 0 or not accepted):
            status = 'delete' if accepted else 'keep'
            n_rem = int(np.sum(np.abs(A_current) > 1e-9))
            print(f"{step:>5}  ({i:>3},{j:>3})  {delta_val_proj:>11.4f}  {status:>8}  {n_rem:>9}")

    rho_curr = float(np.max(np.abs(np.linalg.eigvals(A_current))))
    if rho_curr > 1e-9:
        A_sparse = A_current * (rho_true / rho_curr)
    else:
        print("Warning: spectral radius = 0 after sparsification. Rolling back...")
        A_rebuild = A_true.copy()
        for rec in deletion_history:
            if not rec['accepted']:
                continue
            ri, ci = rec['edge']
            A_rebuild[ri, ci] = 0.0
            if np.max(np.abs(np.linalg.eigvals(A_rebuild))) < 1e-9:
                A_rebuild[ri, ci] = A_true[ri, ci]
                break
        rho_r = float(np.max(np.abs(np.linalg.eigvals(A_rebuild))))
        A_sparse = A_rebuild * (rho_true / rho_r) if rho_r > 1e-9 else A_rebuild

    n_orig = int(np.sum(np.abs(A_true) > 1e-9))
    n_final = int(np.sum(np.abs(A_sparse) > 1e-9))
    if verbose:
        print("-" * 56)
        print(f"Done: {n_orig} -> {n_final} edges "
              f"(retained {n_final / max(n_orig, 1):.1%}, deleted {n_deleted}, kept {n_protected})")
    return A_sparse, deletion_history


# ==============================================================
# Additional reduced-space utilities for theory / scaling
# ==============================================================

def compute_edge_reduced_contribution_norms(A_true, U_dyn):
    """
    Frobenius norm of each edge contribution in reduced space:
        || U^T (a_ij E_ij) U ||_F = |a_ij| * ||U[i,:]|| * ||U[j,:]||
    """
    row_norms = np.linalg.norm(U_dyn, axis=1)
    rows, cols = np.where(np.abs(A_true) > 1e-9)
    weights = np.abs(A_true[rows, cols])
    scores = weights * row_norms[rows] * row_norms[cols]
    return {(int(i), int(j)): float(s) for i, j, s in zip(rows, cols, scores)}


def compute_edge_projected_contribution_norms(A_true, U_dyn):
    """
    Projected-space edge score used only for diagnostics, not for changing the old algorithm.
    """
    P = compute_projector(U_dyn)
    col_norms = np.linalg.norm(P, axis=0)
    row_norms = np.linalg.norm(P, axis=1)
    edges = list(zip(*np.where(np.abs(A_true) > 1e-9)))
    return {(int(i), int(j)): float(row_norms[i] * col_norms[j]) for (i, j) in edges}


def compute_all_effective_matrices(A, U_dyn):
    """Convenience helper returning both projected and reduced effective matrices."""
    return {
        'A_eff_proj': compute_effective_matrix_proj(A, U_dyn),
        'A_eff_red': compute_effective_matrix_red(A, U_dyn),
    }


# ==============================================================
# Legacy helpers retained (compatible with old workflows)
# ==============================================================

def find_kernel_edges(A_true, U_dyn, rel_tol=1e-3):
    """
    Identify edges that lie near the kernel of the OLD projected-space metric.
    This function is kept for compatibility with earlier workflows.
    """
    P = compute_projector(U_dyn)
    col_norms = np.linalg.norm(P, axis=0)
    row_norms = np.linalg.norm(P, axis=1)

    edges = list(zip(*np.where(np.abs(A_true) > 1e-9)))
    proj_norms = {(int(i), int(j)): float(row_norms[i] * col_norms[j])
                  for (i, j) in edges}

    max_norm = max(proj_norms.values()) if proj_norms else 1.0
    threshold = rel_tol * max_norm

    kernel_edges = [(i, j) for (i, j), v in proj_norms.items() if v < threshold]
    non_kernel_edges = [(i, j) for (i, j), v in proj_norms.items() if v >= threshold]

    print(f"Kernel edges     (projected-space weak) : {len(kernel_edges)}")
    print(f"Non-kernel edges (projected-space relevant): {len(non_kernel_edges)}")
    print(f"Projection norm range: [{min(proj_norms.values()):.4e}, "
          f"{max(proj_norms.values()):.4e}]  threshold: {threshold:.4e}")

    return kernel_edges, non_kernel_edges, proj_norms


def compute_order_parameter_LV(A, model, T_steady=150, init_state=None, seed=42):
    if init_state is None:
        np.random.seed(seed)
        init_state = np.random.rand(model.N)

    X = model.simulate(A, T_steps=T_steady, init_state=init_state)
    t_ss = max(1, int(0.8 * T_steady))
    x_inf = np.mean(X[t_ss:], axis=0)
    phi = float(np.mean(x_inf))
    return phi, x_inf


def sparsify_by_projection(A_true, U_dyn, model, eps_macro_tol=0.05,
                           T_steady=150, init_state=None, seed=42):
    """
    Legacy macro-constrained routine retained unchanged in spirit.
    Ranking still uses the old projected-space score to remain compatible.
    """
    if init_state is None:
        np.random.seed(seed)
        init_state = np.random.rand(model.N)

    rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))
    P = compute_projector(U_dyn)
    col_norms = np.linalg.norm(P, axis=0)
    row_norms = np.linalg.norm(P, axis=1)

    phi_true, _ = compute_order_parameter_LV(
        A_true, model, T_steady=T_steady, init_state=init_state
    )

    edges = list(zip(*np.where(np.abs(A_true) > 1e-9)))
    edges_sorted = sorted(edges, key=lambda e: row_norms[e[0]] * col_norms[e[1]])

    A_current = A_true.copy()
    n_deleted = 0

    for (i, j) in edges_sorted:
        A_tmp = A_current.copy()
        A_tmp[i, j] = 0.0

        rho_tmp = float(np.max(np.abs(np.linalg.eigvals(A_tmp))))
        if rho_tmp > 1e-9:
            A_tmp_norm = A_tmp * (rho_true / rho_tmp)
        else:
            break

        phi_tmp, _ = compute_order_parameter_LV(
            A_tmp_norm, model, T_steady=T_steady, init_state=init_state
        )
        eps_macro = abs(phi_tmp - phi_true) / max(abs(phi_true), 1e-9)

        if eps_macro < eps_macro_tol:
            A_current = A_tmp_norm
            n_deleted += 1

    n_orig = len(edges)
    n_final = int(np.sum(np.abs(A_current) > 1e-9))
    sparsity = n_final / max(n_orig, 1)
    print(f"Projection sparsification: {n_orig} -> {n_final} edges "
          f"(sparsity {sparsity:.1%}, deleted {n_deleted})")
    return A_current, sparsity
