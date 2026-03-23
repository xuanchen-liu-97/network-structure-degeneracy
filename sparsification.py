import numpy as np
from numpy.linalg import svd

# ==============================================================
# 稀疏化算法 v2（贪心逐边删除）
# ==============================================================

def extract_dynamical_basis(X_obs, energy_threshold=0.95, min_k=3):
    """
    对观测轨迹做 SVD，提取能解释指定能量比例的动力学子空间基底。

    返回
    ----
    U_dyn : np.ndarray (N, k)，列正交归一
    """
    U, S, _ = svd(X_obs.T, full_matrices=False)
    energy  = np.cumsum(S**2) / np.sum(S**2)
    k       = max(np.searchsorted(energy, energy_threshold) + 1, min_k)
    return U[:, :k]


def compute_effective_matrix(A, U_dyn):
    """A_eff = P @ A @ P，P = U_dyn @ U_dyn.T"""
    P = U_dyn @ U_dyn.T
    return P @ A @ P


def _aeff_deviation(A_candidate, A_eff_ref, U_dyn):
    """δ = ‖A_eff(A_candidate) - A_eff_ref‖_F / ‖A_eff_ref‖_F"""
    A_eff_cand = compute_effective_matrix(A_candidate, U_dyn)
    ref_norm   = np.linalg.norm(A_eff_ref, 'fro')
    if ref_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(A_eff_cand - A_eff_ref, 'fro') / ref_norm)


def sparsify_network_v2(A_true, U_dyn, epsilon=0.05, verbose=True):
    """
    基于 A_eff 等价约束的贪心逐边删除稀疏化算法。

    流程：
    1. 计算 A_eff_ref = P @ A_true @ P 作为功能基准；
    2. 将所有边按 A_eff 能量从小到大排队（低能量边优先删除）；
    3. 逐边尝试删除：若删后偏差 δ < epsilon 则接受，否则保留；
    4. 遍历结束后对谱半径归一化。

    参数
    ----
    A_true  : np.ndarray (N, N)
    U_dyn   : np.ndarray (N, k)   动力学子空间基底（列正交归一）
    epsilon : float               A_eff 相对偏差容忍上限（建议 0.01~0.10）
    verbose : bool

    返回
    ----
    A_sparse         : np.ndarray (N, N)   稀疏化后的网络（谱半径已归一化）
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
        print(f"稀疏化 v2 开始：共 {total} 条边，epsilon={epsilon}")
        print(f"{'步骤':>5}  {'边':>10}  {'δ':>8}  {'决策':>6}  {'剩余边数':>8}")
        print("─" * 50)

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
            status = "删除✓" if accepted else "保护✗"
            n_rem  = int(np.sum(np.abs(A_current) > 1e-9))
            print(f"{step:>5}  ({i:>3},{j:>3})  {delta:>8.4f}  {status:>6}  {n_rem:>8}")

    # 谱半径归一化
    rho_curr = float(np.max(np.abs(np.linalg.eigvals(A_current))))
    if rho_curr > 1e-9:
        A_sparse = A_current * (rho_true / rho_curr)
    else:
        print("警告：稀疏化后谱半径为 0，尝试回退到最后有效状态…")
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
        print("─" * 50)
        print(f"稀疏化 v2 完成：{n_orig} → {n_final} 条边"
              f"（保留 {n_final/n_orig:.1%}，删除 {n_deleted}，保护 {n_protected}）")
    return A_sparse, deletion_history