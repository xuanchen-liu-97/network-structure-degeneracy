import numpy as np
from numpy.linalg import svd

# ==============================================================
# 稀疏化算法 v2（贪心逐边删除）
# ==============================================================

"""
sparsification.py
=================
Greedy edge-deletion sparsification constrained by A_eff equivalence.

Main function
-------------
sparsify_network_v2(A_true, U_dyn, epsilon=0.05, verbose=True)
    -> A_sparse, deletion_history

Helper functions
----------------
extract_dynamical_basis(X_obs, energy_threshold=0.95, min_k=3)
compute_effective_matrix(A, U_dyn)

Usage
-----
    from sparsification import sparsify_network_v2, extract_dynamical_basis
    U_dyn = extract_dynamical_basis(X_obs)
    A_sparse, history = sparsify_network_v2(A_true, U_dyn, epsilon=0.05)
"""

'''
对NRMSE约束
'''
def extract_dynamical_basis(X_obs, energy_threshold=0.95, min_k=3):
    """
    Extract the dynamical subspace basis via SVD of the trajectory matrix.

    Returns
    -------
    U_dyn : (N, k) orthonormal columns
    """
    U, S, _ = svd(X_obs.T, full_matrices=False)
    energy  = np.cumsum(S**2) / np.sum(S**2)
    k       = max(np.searchsorted(energy, energy_threshold) + 1, min_k)
    return U[:, :k]


def compute_effective_matrix(A, U_dyn):
    """A_eff = P @ A @ P,  P = U_dyn @ U_dyn.T"""
    P = U_dyn @ U_dyn.T
    return P @ A @ P


def _aeff_deviation(A_candidate, A_eff_ref, U_dyn):
    """delta = ||A_eff(A_candidate) - A_eff_ref||_F / ||A_eff_ref||_F"""
    A_eff_cand = compute_effective_matrix(A_candidate, U_dyn)
    ref_norm   = np.linalg.norm(A_eff_ref, 'fro')
    if ref_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(A_eff_cand - A_eff_ref, 'fro') / ref_norm)


def sparsify_network_v2(A_true, U_dyn, epsilon=0.05, verbose=True):
    """
    Greedy edge-deletion sparsification constrained by A_eff equivalence.

    Algorithm:
    1. Compute A_eff_ref = P @ A_true @ P as functional reference.
    2. Sort all edges by A_eff energy (ascending — weakest first).
    3. Attempt to delete each edge; accept if delta < epsilon, else keep.
    4. Normalise spectral radius of the result to match A_true.

    Parameters
    ----------
    A_true  : (N, N)
    U_dyn   : (N, k) orthonormal basis
    epsilon : float  A_eff relative deviation tolerance (e.g. 0.01-0.10)
    verbose : bool

    Returns
    -------
    A_sparse         : (N, N) sparsified network
    deletion_history : list of dict
                       step / edge / delta / accepted / n_edges
    """
    rho_true  = float(np.max(np.abs(np.linalg.eigvals(A_true))))
    A_eff_ref = compute_effective_matrix(A_true, U_dyn)

    rows, cols = np.where(np.abs(A_true) > 1e-9)
    energies   = np.abs(A_eff_ref[rows, cols])
    order      = np.argsort(energies)
    edge_queue = [(rows[o], cols[o]) for o in order]

    A_current        = A_true.copy()
    deletion_history = []
    n_deleted        = 0
    n_protected      = 0
    total            = len(edge_queue)

    if verbose:
        print(f"Sparsification v2: {total} edges, epsilon={epsilon}")
        print(f"{'step':>5}  {'edge':>10}  {'delta':>8}  {'decision':>8}  {'remaining':>9}")
        print("-" * 50)

    for step, (i, j) in enumerate(edge_queue):
        A_tmp    = A_current.copy()
        A_tmp[i, j] = 0.0
        delta    = _aeff_deviation(A_tmp, A_eff_ref, U_dyn)
        accepted = delta < epsilon

        deletion_history.append({
            "step": step, "edge": (i, j), "delta": delta,
            "accepted": accepted,
            "n_edges": int(np.sum(np.abs(A_current) > 1e-9)),
        })

        if accepted:
            A_current = A_tmp
            n_deleted += 1
        else:
            n_protected += 1

        if verbose and (step % max(1, total // 20) == 0 or not accepted):
            status = "delete" if accepted else "keep"
            n_rem  = int(np.sum(np.abs(A_current) > 1e-9))
            print(f"{step:>5}  ({i:>3},{j:>3})  {delta:>8.4f}  {status:>8}  {n_rem:>9}")

    rho_curr = float(np.max(np.abs(np.linalg.eigvals(A_current))))
    if rho_curr > 1e-9:
        A_sparse = A_current * (rho_true / rho_curr)
    else:
        print("Warning: spectral radius = 0 after sparsification. Rolling back...")
        A_rebuild = A_true.copy()
        for rec in deletion_history:
            if not rec["accepted"]:
                continue
            ri, ci = rec["edge"]
            A_rebuild[ri, ci] = 0.0
            if np.max(np.abs(np.linalg.eigvals(A_rebuild))) < 1e-9:
                A_rebuild[ri, ci] = A_true[ri, ci]
                break
        rho_r    = float(np.max(np.abs(np.linalg.eigvals(A_rebuild))))
        A_sparse = A_rebuild * (rho_true / rho_r) if rho_r > 1e-9 else A_rebuild

    n_orig  = int(np.sum(np.abs(A_true)   > 1e-9))
    n_final = int(np.sum(np.abs(A_sparse) > 1e-9))
    if verbose:
        print("-" * 50)
        print(f"Done: {n_orig} -> {n_final} edges "
              f"(retained {n_final/n_orig:.1%}, deleted {n_deleted}, kept {n_protected})")
    return A_sparse, deletion_history



'''
对eps_macro约束
'''

def find_kernel_edges(A_true, U_dyn, rel_tol=1e-3):
    """
    Identify edges that lie entirely in the kernel of the A_eff projection.

    An edge (i, j) is "kernel-redundant" if removing it does not change
    A_eff = P @ A @ P at all.  This happens when either node i or node j
    has negligible projection onto the dynamical subspace U_dyn, i.e.
    ||P e_j||_2 * ||e_i^T P||_2 is near zero.

    The projection norm for edge (i,j) is computed analytically as:
        proj_norm(i,j) = ||P[:,j]||_2 * ||P[i,:]||_2
    which equals ||U_dyn.T @ E_ij @ U_dyn||_F  (vectorised, no per-edge loop).

    Parameters
    ----------
    A_true  : (N, N)
    U_dyn   : (N, k)  orthonormal dynamical basis
    rel_tol : float   threshold relative to the maximum projection norm.
                      Edges with proj_norm < rel_tol * max_proj_norm
                      are classified as kernel edges. Default 1e-3.

    Returns
    -------
    kernel_edges     : list of (i, j)  absolutely redundant edges
    non_kernel_edges : list of (i, j)  functionally relevant edges
    proj_norms       : dict {(i,j): float}  projection norm for every edge
                       (useful for soft-deletion ranking)
    """
    P = U_dyn @ U_dyn.T                           # (N, N)

    # Row and column projection norms of P
    col_norms = np.linalg.norm(P, axis=0)          # ||P[:,j]||  shape (N,)
    row_norms = np.linalg.norm(P, axis=1)          # ||P[i,:]||  shape (N,)

    edges = list(zip(*np.where(np.abs(A_true) > 1e-9)))

    # Vectorised projection norm for every edge
    proj_norms = {(i, j): float(row_norms[i] * col_norms[j])
                  for (i, j) in edges}

    max_norm = max(proj_norms.values()) if proj_norms else 1.0
    threshold = rel_tol * max_norm

    kernel_edges     = [(i, j) for (i, j), v in proj_norms.items() if v < threshold]
    non_kernel_edges = [(i, j) for (i, j), v in proj_norms.items() if v >= threshold]

    print(f"Kernel edges     (absolutely redundant) : {len(kernel_edges)}")
    print(f"Non-kernel edges (functionally relevant): {len(non_kernel_edges)}")
    print(f"Projection norm range: [{min(proj_norms.values()):.4e}, "
          f"{max(proj_norms.values()):.4e}]  threshold: {threshold:.4e}")

    return kernel_edges, non_kernel_edges, proj_norms


def compute_order_parameter_LV(A, model, T_steady=150, init_state=None, seed=42):
    """
    Compute the order parameter for Lotka-Volterra (or generic mutualistic)
    dynamics: steady-state mean abundance averaged over all nodes.

    Phi(A) = (1/N) * mean_{t in steady window} sum_i x_i(t)

    The last 20% of the T_steady steps are used as the steady-state window.

    Parameters
    ----------
    A          : (N, N)
    model      : NetworkDynamics  (must expose .simulate and .N)
    T_steady   : int   total simulation steps (should be long enough to reach
                       steady state; default 150)
    init_state : (N,) or None   initial condition; None -> uniform random
    seed       : int

    Returns
    -------
    phi      : float   scalar order parameter
    x_inf    : (N,)    node-level steady-state mean (for diagnostics)
    """
    if init_state is None:
        np.random.seed(seed)
        init_state = np.random.rand(model.N)

    X = model.simulate(A, T_steps=T_steady, init_state=init_state)

    # Use last 20% as steady-state window
    t_ss  = max(1, int(0.8 * T_steady))
    x_inf = np.mean(X[t_ss:], axis=0)   # (N,)
    phi   = float(np.mean(x_inf))
    return phi, x_inf


def sparsify_by_projection(A_true, U_dyn, model, eps_macro_tol=0.05,
                            T_steady=150, init_state=None, seed=42):
    """
    Greedy edge deletion ranked by projection norm (weakest A_eff contribution
    first), stopping when eps_macro exceeds the tolerance.
    
    This is the kernel-guided analogue of sparsify_network_v2, but uses
    the order parameter distance as the stopping criterion instead of
    the A_eff Frobenius deviation.
    """
    if init_state is None:
        np.random.seed(seed)
        init_state = np.random.rand(model.N)

    rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))
    P        = U_dyn @ U_dyn.T
    col_norms = np.linalg.norm(P, axis=0)
    row_norms = np.linalg.norm(P, axis=1)

    # Reference order parameter
    phi_true, _ = compute_order_parameter_LV(
        A_true, model, T_steady=T_steady, init_state=init_state
    )

    # Sort edges by projection norm ascending (weakest first)
    edges = list(zip(*np.where(np.abs(A_true) > 1e-9)))
    edges_sorted = sorted(edges,
                          key=lambda e: row_norms[e[0]] * col_norms[e[1]])

    A_current = A_true.copy()
    n_deleted = 0

    for (i, j) in edges_sorted:
        A_tmp      = A_current.copy()
        A_tmp[i, j] = 0.0

        # Normalise spectral radius before testing
        rho_tmp = float(np.max(np.abs(np.linalg.eigvals(A_tmp))))
        if rho_tmp > 1e-9:
            A_tmp_norm = A_tmp * (rho_true / rho_tmp)
        else:
            break   # network collapsed, stop

        phi_tmp, _ = compute_order_parameter_LV(
            A_tmp_norm, model, T_steady=T_steady, init_state=init_state
        )
        eps_macro = abs(phi_tmp - phi_true) / max(abs(phi_true), 1e-9)

        if eps_macro < eps_macro_tol:
            A_current = A_tmp_norm
            n_deleted += 1
        # if rejected, skip this edge and continue trying weaker ones

    n_orig   = len(edges)
    n_final  = int(np.sum(np.abs(A_current) > 1e-9))
    sparsity = n_final / max(n_orig, 1)
    print(f"Projection sparsification: {n_orig} -> {n_final} edges "
          f"(sparsity {sparsity:.1%}, deleted {n_deleted})")
    return A_current, sparsity



'''
双重约束
'''

def sparsify_by_projection_v2(A_true, U_dyn, model, X_obs, T_train_steps,
                               eps_macro_tol=0.05, nrmse_tol=0.10,
                               T_steady=150, init_state=None, seed=42):
    """
    Greedy edge deletion with dual stopping criterion:
      1. eps_macro < eps_macro_tol  (steady-state functional equivalence)
      2. NRMSE     < nrmse_tol      (trajectory prediction accuracy)
    An edge is deleted only if BOTH conditions are satisfied after deletion.
    """
    if init_state is None:
        np.random.seed(seed)
        init_state = np.random.rand(model.N)

    rho_true   = float(np.max(np.abs(np.linalg.eigvals(A_true))))
    P          = U_dyn @ U_dyn.T
    col_norms  = np.linalg.norm(P, axis=0)
    row_norms  = np.linalg.norm(P, axis=1)

    steps_pred = len(X_obs) - T_train_steps
    X_train    = X_obs[:T_train_steps]

    # Reference quantities
    phi_true, _ = compute_order_parameter_LV(
        A_true, model, T_steady=T_steady, init_state=init_state
    )

    # Sort edges by projection norm ascending (weakest A_eff contribution first)
    edges        = list(zip(*np.where(np.abs(A_true) > 1e-9)))
    edges_sorted = sorted(edges,
                          key=lambda e: row_norms[e[0]] * col_norms[e[1]])

    A_current = A_true.copy()
    n_deleted = 0
    n_rejected_macro = 0
    n_rejected_nrmse = 0

    print(f"Dual-criterion sparsification: "
          f"eps_macro_tol={eps_macro_tol}, nrmse_tol={nrmse_tol}")
    print(f"Total edges to attempt: {len(edges_sorted)}")

    for (i, j) in edges_sorted:
        A_tmp      = A_current.copy()
        A_tmp[i, j] = 0.0

        rho_tmp = float(np.max(np.abs(np.linalg.eigvals(A_tmp))))
        if rho_tmp < 1e-9:
            break
        A_tmp_norm = A_tmp * (rho_true / rho_tmp)

        # Check eps_macro
        phi_tmp, _ = compute_order_parameter_LV(
            A_tmp_norm, model, T_steady=T_steady, init_state=init_state
        )
        eps_macro = abs(phi_tmp - phi_true) / max(abs(phi_true), 1e-9)
        if eps_macro >= eps_macro_tol:
            n_rejected_macro += 1
            continue

        # Check NRMSE
        try:
            X_pred = model.simulate(
                A_tmp_norm, T_steps=steps_pred,
                init_state=X_obs[T_train_steps]
            )
            X_full = np.vstack([X_train, X_pred])
            _, nrmse = evaluate_prediction_nrmse(X_obs, X_full, T_train_steps)
        except Exception:
            n_rejected_nrmse += 1
            continue

        if nrmse >= nrmse_tol:
            n_rejected_nrmse += 1
            continue

        # Both criteria satisfied: accept deletion
        A_current = A_tmp_norm
        n_deleted += 1

    n_orig   = len(edges)
    n_final  = int(np.sum(np.abs(A_current) > 1e-9))
    sparsity = n_final / max(n_orig, 1)

    print(f"Done: {n_orig} -> {n_final} edges (sparsity {sparsity:.1%})")
    print(f"  Deleted: {n_deleted}  "
          f"Rejected by eps_macro: {n_rejected_macro}  "
          f"Rejected by NRMSE: {n_rejected_nrmse}")
    return A_current, sparsity