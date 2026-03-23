import numpy as np
import warnings
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# ==============================================================
# LASSO 网络重构
# ==============================================================

def LASSO_reconstruct_network(X, model, rho=None, val_ratio=0.2,
                         n_alphas=20, threshold=0.01):
    """
    基于 LASSO 从时间序列重构网络邻接矩阵。

    参数
    ----
    X         : np.ndarray (T, N)
    model     : NetworkDynamics 对象
    rho       : None | float | array(N,)
                None  → 对每个节点自动 CV 搜索
                float → 所有节点使用同一固定值
                array → 每个节点使用对应的 rho[i]
    val_ratio : float  验证集比例（rho=None 时有效）
    n_alphas  : int    CV 候选数（rho=None 时有效）
    threshold : float  硬阈值，绝对值低于此的权重置零

    返回
    ----
    A_hat    : np.ndarray (N, N)
    rho_used : np.ndarray (N,)
    """
    T, N = X.shape
    dt   = model.dt
    if T <= 1:
        raise ValueError("时间序列至少需要 2 步。")

    if rho is None:
        rho_mode = 'auto'
        print(f"重构模式：Auto（n_alphas={n_alphas}）")
    elif np.isscalar(rho):
        rho_mode = 'scalar'
        print(f"重构模式：Scalar（rho={rho}）")
    else:
        rho_vec  = np.array(rho)
        if rho_vec.shape != (N,):
            raise ValueError(f"rho 数组长度 {len(rho_vec)} 与 N={N} 不符。")
        rho_mode = 'vector'

    X_t, X_tp1 = X[:-1], X[1:]
    t_len = len(X_t)
    Y   = (X_tp1 - X_t) / dt - model.f(X_t)
    Phi = model.get_interaction_matrix(X_t)

    n_train = int(t_len * (1 - val_ratio))
    Y_tr,   Y_val   = Y[:n_train],   Y[n_train:]
    Phi_tr, Phi_val = Phi[:n_train], Phi[n_train:]

    A_hat    = np.zeros((N, N))
    rho_used = np.zeros(N)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N):
            Phi_tr_i  = Phi_tr[:, i, :]
            y_tr_i    = Y_tr[:, i]
            Phi_val_i = Phi_val[:, i, :]
            y_val_i   = Y_val[:, i]

            if rho_mode == 'auto':
                dot     = Phi_tr_i.T @ y_tr_i
                rho_max = max(np.max(np.abs(dot)) / max(n_train, 1), 1e-3)
                alphas  = np.logspace(np.log10(1e-4 * rho_max),
                                      np.log10(rho_max), n_alphas)
                best_mse, best_alpha = np.inf, alphas[0]
                for a in alphas:
                    lasso = Lasso(alpha=a, fit_intercept=False, positive=True,
                                  max_iter=2000, tol=1e-4)
                    lasso.fit(Phi_tr_i, y_tr_i)
                    coef = lasso.coef_.copy()
                    coef[np.abs(coef) < threshold] = 0
                    mse = (np.mean(y_val_i**2) if np.sum(np.abs(coef)) == 0
                           else mean_squared_error(y_val_i, Phi_val_i @ coef))
                    if mse < best_mse:
                        best_mse, best_alpha = mse, a
                final_rho = best_alpha
            elif rho_mode == 'scalar':
                final_rho = rho
            else:
                final_rho = rho_vec[i]

            rho_used[i] = final_rho
            lasso_f = Lasso(alpha=final_rho, fit_intercept=False, positive=True,
                            max_iter=10000, tol=1e-4)
            lasso_f.fit(Phi[:, i, :], Y[:, i])
            coef = lasso_f.coef_.copy()
            coef[np.abs(coef) < threshold] = 0
            A_hat[i, :] = coef

    if rho_mode == 'auto':
        print(f"重构完成，自动选择 rho 均值：{np.mean(rho_used):.4e}")
    return A_hat, rho_used