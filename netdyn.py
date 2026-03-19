"""
netdyn.py
==========================
Network Dynamics Simulation Toolkit (Heterogeneous Parameters Version)
-----------------------------------
版本特点:
1. 物理参数 (alpha, theta, beta 等) 无默认值，必须手动传入。
2. 支持异质性参数：所有参数均可传入标量或长度为 N 的列表/数组。
"""

import numpy as np
from scipy.integrate import solve_ivp

class NetworkDynamics:
    """所有动力学模型的基类"""
    def __init__(self, N, dt=0.1):
        self.N = N
        self.dt = dt

    def _process_param(self, param, name):
        """
        辅助函数：处理传入的参数。
        如果传入的是列表或数组，确保其形状为 (N,)。
        如果传入的是标量，保持原样（利用广播机制）。
        """
        # 如果是列表或元组，先转为数组
        if isinstance(param, (list, tuple)):
            param = np.array(param)
        
        # 如果是数组，检查形状
        if isinstance(param, np.ndarray):
            if param.ndim == 0: # 0-d array (scalar)
                return float(param)
            if param.shape != (self.N,):
                raise ValueError(f"参数 '{name}' 的形状必须为 ({self.N},)，但得到了 {param.shape}")
            return param
        
        # 如果是标量 (float/int)，直接返回
        if np.isscalar(param):
            return float(param)
            
        raise ValueError(f"参数 '{name}' 类型不支持: {type(param)}")

    def f(self, x):
        raise NotImplementedError

    def get_interaction_matrix(self, x_t):
        raise NotImplementedError

    def _ode_func(self, t, x, A):
        raise NotImplementedError

    def simulate(self, A, T_steps=1000, init_state=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        if init_state is None:
            x0 = np.random.rand(self.N)
        else:
            x0 = np.array(init_state)
            if x0.shape != (self.N,):
                 raise ValueError(f"初始状态 shape 必须为 ({self.N},)")

        t_end = T_steps * self.dt
        t_eval = np.linspace(0.0, t_end, T_steps)

        # 使用 lambda 包装 ode_func 以传入 A
        fun = lambda t, x: self._ode_func(t, x, A)

        sol = solve_ivp(
            fun,
            (0, t_end),
            x0,
            t_eval=t_eval,
            method='LSODA'
        )
        
        if not sol.success:
            print(f"Warning: Solver failed - {sol.message}")
            
        return sol.y.T


# ==============================================================
# 1. Lotka–Volterra (LV)
# ==============================================================
class LVDynamics(NetworkDynamics):
    def __init__(self, N, alpha, theta, dt=0.1):
        """
        alpha, theta: 标量 或 shape (N,) 的数组/列表
        """
        super().__init__(N, dt)
        self.alpha = self._process_param(alpha, "alpha")
        self.theta = self._process_param(theta, "theta")

    def f(self, x):
        # Numpy 广播机制会自动处理：
        # 如果 self.alpha 是 (N,)，x 是 (T, N)，则结果正确对应
        return x * (self.alpha - self.theta * x)

    def get_interaction_matrix(self, x):
        if x.ndim == 1:
            return -x[:, None] * x[None, :]
        return -x[:, :, None] * x[:, None, :]

    def _ode_func(self, t, x, A):
        return self.f(x) - x * (A @ x)


# ==============================================================
# 2. Mutualistic Population (MP)
# ==============================================================
class MPDynamics(NetworkDynamics):
    def __init__(self, N, alpha, theta, dt=0.1):
        super().__init__(N, dt)
        self.alpha = self._process_param(alpha, "alpha")
        self.theta = self._process_param(theta, "theta")

    def f(self, x):
        return x * (self.alpha - self.theta * x)

    def get_interaction_matrix(self, x):
        term_j = (x**2) / (1 + x**2)
        if x.ndim == 1:
            return x[:, None] * term_j[None, :]
        else:
            return x[:, :, None] * term_j[:, None, :]

    def _ode_func(self, t, x, A):
        term_j = (x**2) / (1 + x**2)
        return self.f(x) + x * (A @ term_j)


# ==============================================================
# 3. Michaelis–Menten (MM)
# ==============================================================
class MMDynamics(NetworkDynamics):
    def __init__(self, N, h, dt=0.1):
        super().__init__(N, dt)
        # h 通常是生化反应系数，通常为标量，但也支持向量化
        self.h = self._process_param(h, "h")

    def f(self, x):
        return -x

    def get_interaction_matrix(self, x):
        term_j = (x**self.h) / (1 + x**self.h)
        if x.ndim == 1:
            return np.tile(term_j[None, :], (self.N, 1))
        else:
            return np.tile(term_j[:, None, :], (1, self.N, 1))

    def _ode_func(self, t, x, A):
        term_j = (x**self.h) / (1 + x**self.h)
        return -x + (A @ term_j)


# ==============================================================
# 4. SIS Model
# ==============================================================
class SISDynamics(NetworkDynamics):
    def __init__(self, N, beta, delta, dt=0.1):
        """
        beta: 感染率 (可以是节点特有的易感性)
        delta: 治愈率 (可以是节点特有的恢复力)
        """
        super().__init__(N, dt)
        self.beta = self._process_param(beta, "beta")
        self.delta = self._process_param(delta, "delta")

    def f(self, x):
        return -self.delta * x

    def get_interaction_matrix(self, x):
        # g(x_i, x_j) = beta_i * (1 - x_i) * x_j
        # 注意：如果 beta 是向量 beta_i，它在这里作用于 i
        
        # 扩展 beta 以匹配维度
        if np.ndim(self.beta) > 0: # 向量
            if x.ndim == 1:
                beta_expand = self.beta[:, None]
            else:
                beta_expand = self.beta[None, :, None] # (1, N, 1) 适配 (T, N, 1)
        else: # 标量
            beta_expand = self.beta

        if x.ndim == 1:
            term_i = beta_expand * (1 - x[:, None]) # (N, 1)
            term_j = x[None, :]                     # (1, N)
            return term_i * term_j                  # (N, N)
        else:
            # x: (T, N)
            # term_i: (T, N, 1)
            term_i = beta_expand * (1 - x[:, :, None]) 
            term_j = x[:, None, :]
            return term_i * term_j

    def _ode_func(self, t, x, A):
        # dx = -delta*x + beta*(1-x) * (A @ x)
        return -self.delta * x + self.beta * (1 - x) * (A @ x)


# ==============================================================
# 5. Kuramoto
# ==============================================================
class KuramotoDynamics(NetworkDynamics):
    def __init__(self, N, omega, dt=0.05):
        """
        omega: 自然频率，必须传入，支持列表。
        """
        super().__init__(N, dt)
        self.omega = self._process_param(omega, "omega")

    def f(self, x):
        # f_i = omega_i
        if x.ndim == 2: # (T, N)
            # 如果 omega 是 (N,)，广播到 (T, N)
            return np.tile(self.omega, (x.shape[0], 1))
        return self.omega

    def get_interaction_matrix(self, x):
        if x.ndim == 1:
            return np.sin(x[None, :] - x[:, None])
        else:
            return np.sin(x[:, None, :] - x[:, :, None])

    def _ode_func(self, t, x, A):
        diff = x[None, :] - x[:, None]
        return self.omega + np.sum(A * np.sin(diff), axis=1)
    
    def simulate(self, A, T_steps=1000, init_state=None, seed=None):
        if seed is not None: np.random.seed(seed)
        if init_state is None:
            init_state = np.random.uniform(-np.pi, np.pi, self.N)
        
        raw_data = super().simulate(A, T_steps, init_state)
        return raw_data


# ==============================================================
# 6. Wilson-Cowan (WC)
# ==============================================================
class WCDynamics(NetworkDynamics):
    def __init__(self, N, tau, mu, dt=0.1):
        super().__init__(N, dt)
        self.tau = self._process_param(tau, "tau")
        self.mu = self._process_param(mu, "mu")

    def f(self, x):
        return -x

    def get_interaction_matrix(self, x):
        # g(x_i, x_j) = 1 / (1 + exp(-tau * (x_j - mu)))
        # 这里 tau 和 mu 是 sigmoid 的参数
        # 如果 tau/mu 是向量，通常这里的物理含义是突触后神经元(i)的响应属性还是突触前(j)的？
        # Source 2 Table 1 显示这些是节点 i 的属性。但在 g(x_i, x_j) 中通常只有 x_j。
        # 标准 WC 模型中，Response function 是 S(input_to_i)。
        # 在 Source 2 中，g(xi, xj) = S(xj)。这意味着 Sigmoid 参数通常是 j 的属性。
        # 但如果是 heterogeneous，我们需要确保维度匹配。
        
        # 假设 tau, mu 是节点 j 的属性（因为它们在 g(xj) 内部）
        if np.ndim(self.tau) > 0:
            tau_j = self.tau
            mu_j = self.mu
        else:
            tau_j = self.tau
            mu_j = self.mu
            
        sigmoid = 1 / (1 + np.exp(-tau_j * (x - mu_j)))
        
        if x.ndim == 1:
            return np.tile(sigmoid[None, :], (self.N, 1))
        else:
            return np.tile(sigmoid[:, None, :], (1, self.N, 1))

    def _ode_func(self, t, x, A):
        sigmoid = 1 / (1 + np.exp(-self.tau * (x - self.mu)))
        return -x + (A @ sigmoid)