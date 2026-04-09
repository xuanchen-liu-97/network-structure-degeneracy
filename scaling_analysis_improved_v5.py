
"""
scaling_analysis.py
===================

Utilities for finite-size scaling analysis of scan_v2 datasets
(BA/LV now; usable for SIS as long as the saved field names are consistent).

Core workflow
-------------
1. load scan pickles
2. flatten to true-level / proxy-level DataFrames
3. fit empirical K_eff in a small-delta window
4. optionally enrich proxy rows with reduced-space direction metrics
5. optionally enrich true rows with LV z-side decomposition using network-library pickles
6. subset by delta / eps_sub / support_hamming
7. aggregate by graph, then by N
8. fit and plot power laws

Designed for notebook use.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, t


PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_scan_pickles(paths_by_N: Mapping[int, PathLike]) -> Dict[int, dict]:
    """Load scan_v2 pickle files keyed by N."""
    out: Dict[int, dict] = {}
    for N, path in sorted(paths_by_N.items()):
        with open(path, "rb") as f:
            out[int(N)] = pickle.load(f)
    return out


def load_network_pickles(paths_by_N: Mapping[int, PathLike]) -> Dict[int, dict]:
    """Load network-library pickle files keyed by N."""
    out: Dict[int, dict] = {}
    for N, path in sorted(paths_by_N.items()):
        with open(path, "rb") as f:
            out[int(N)] = pickle.load(f)
    return out


def build_A_map(network_data_by_N: Mapping[int, dict]) -> Dict[str, np.ndarray]:
    """Build graph_id -> A_true map from network-library pickles."""
    A_map: Dict[str, np.ndarray] = {}
    for _, d in network_data_by_N.items():
        graphs = d.get("graphs", [])
        for g in graphs:
            A_map[g["graph_id"]] = np.asarray(g["A_true"], dtype=float)
    return A_map


# ---------------------------------------------------------------------
# Flatten scan data
# ---------------------------------------------------------------------

def _norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def flatten_scan_data(
    data_by_N: Mapping[int, dict],
    keep_arrays: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return
    ------
    df_true  : one row per graph instance
    df_proxy : one row per proxy / epsilon record
    """
    true_rows: List[dict] = []
    proxy_rows: List[dict] = []

    for N, d in sorted(data_by_N.items()):
        for inst in d["results"]:
            graph_id = inst["graph_id"]

            true_row = {
                "N": int(N),
                "graph_id": graph_id,
                "seed": inst.get("seed"),
                "k": inst.get("k"),
                "phi_true": float(inst["phi_true"]),
                "A_eff_proj_norm_F": float(inst["A_eff_proj_norm_F"]),
                "A_eff_red_norm_F": float(inst["A_eff_red_norm_F"]),
                "G_obs": float(inst.get("G_obs", np.nan)),
                "K_eff_base_red": float(inst.get("K_eff_base_red", np.nan)),
                "sigma_min_F": float(inst.get("sigma_min_F", np.nan)),
                "sigma_min_H": float(inst.get("sigma_min_H", np.nan)),
                "c_norm": _norm(inst["c_true"]),
                "r_norm": _norm(inst["r_true"]),
                "z_norm": _norm(inst["z_row"]),
            }

            lv_params = inst.get("lv_params")
            sis_params = inst.get("sis_params")
            if lv_params is not None:
                true_row["dynamics"] = "LV"
            elif sis_params is not None:
                true_row["dynamics"] = "SIS"
            else:
                true_row["dynamics"] = d.get("dynamics", None)

            if keep_arrays:
                for key in [
                    "U_dyn", "x_star_true", "c_true", "r_true", "z_row",
                    "A_eff_proj_true", "A_eff_red_true", "X_obs", "x0",
                    "lv_params", "sis_params",
                ]:
                    if key in inst:
                        true_row[key] = inst[key]

            true_rows.append(true_row)

            for rec in inst["records"]:
                proxy_row = {
                    "N": int(N),
                    "graph_id": graph_id,
                    "epsilon": float(rec["epsilon"]),
                    "delta_proj": float(rec["delta_proj"]),
                    "delta_red": float(rec["delta_red"]),
                    "delta": float(rec.get("delta", rec["delta_red"])),
                    "eps_macro": float(rec["eps_macro"]),
                    "phi_proxy": float(rec["phi_proxy"]),
                    "sparsity": float(rec["sparsity"]),
                    "eps_sub": float(rec["eps_sub"]),
                    "support_hamming": float(rec["support_hamming"]),
                    "same_support": bool(rec["same_support"]),
                    "xi_norm": float(rec["xi_norm"]),
                    "lin_macro_pred": float(rec["lin_macro_pred"]),
                }
                if keep_arrays:
                    for key in [
                        "DeltaA_eff_proj", "DeltaA_eff_red",
                        "A_eff_proj_proxy", "A_eff_red_proxy",
                        "x_star_proxy", "xi", "A_proxy", "DeltaA",
                    ]:
                        if key in rec:
                            proxy_row[key] = rec[key]
                proxy_rows.append(proxy_row)

    df_true = pd.DataFrame(true_rows)
    df_proxy = pd.DataFrame(proxy_rows)
    return df_true, df_proxy


# ---------------------------------------------------------------------
# Empirical K_eff from small-delta slope
# ---------------------------------------------------------------------

def fit_slope_through_origin(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.dot(x, x)
    if denom < 1e-15:
        return np.nan
    return float(np.dot(x, y) / denom)


def empirical_keff_by_window(
    df_proxy: pd.DataFrame,
    delta_col: str = "delta",
    response_col: str = "eps_macro",
    delta_max: float = 0.5,
    min_points_per_graph: int = 3,
    agg: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    df_slope  : one slope per graph
    summary   : one aggregated slope per N
    fit_df    : one-row power-law fit summary for aggregated slopes
    """
    rows = []
    for (N, graph_id), sub in df_proxy.groupby(["N", "graph_id"]):
        sub2 = sub[sub[delta_col] <= delta_max].sort_values(delta_col)
        if len(sub2) < min_points_per_graph:
            continue
        slope = fit_slope_through_origin(sub2[delta_col], sub2[response_col])
        rows.append({"N": N, "graph_id": graph_id, "K_eff_emp": slope, "n_points": len(sub2)})

    df_slope = pd.DataFrame(rows)
    if len(df_slope) == 0:
        empty_slope = pd.DataFrame(columns=['N','graph_id','K_eff_emp','n_points','delta_lo','delta_hi'])
        return empty_slope, pd.DataFrame(), pd.DataFrame()

    summary = df_slope.groupby("N")["K_eff_emp"].agg(
        median="median",
        mean="mean",
        std="std",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    )

    col = agg
    fit = fit_power_law_from_summary(summary, col)
    fit_df = pd.DataFrame([fit])
    return df_slope, summary, fit_df


def empirical_keff_window_scan(
    df_proxy: pd.DataFrame,
    delta_max_list: Sequence[float],
    delta_col: str = "delta",
    response_col: str = "eps_macro",
    min_points_per_graph: int = 3,
    agg: str = "median",
) -> pd.DataFrame:
    """Scan multiple delta_max windows and fit a power law for each."""
    rows = []
    for delta_max in delta_max_list:
        df_slope, summary, fit_df = empirical_keff_by_window(
            df_proxy,
            delta_col=delta_col,
            response_col=response_col,
            delta_max=float(delta_max),
            min_points_per_graph=min_points_per_graph,
            agg=agg,
        )
        if len(fit_df) == 0:
            continue
        fit = fit_df.iloc[0].to_dict()
        fit["delta_max"] = float(delta_max)
        fit["n_graph_slopes"] = len(df_slope)
        rows.append(fit)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Power-law fit utilities
# ---------------------------------------------------------------------

def fit_power_law_xy(Ns: Sequence[float], Ys: Sequence[float], quantity: str = "") -> dict:
    Ns = np.asarray(Ns, dtype=float)
    Ys = np.asarray(Ys, dtype=float)

    mask = (Ns > 0) & (Ys > 0) & np.isfinite(Ns) & np.isfinite(Ys)
    Ns = Ns[mask]
    Ys = Ys[mask]

    if len(Ns) < 3:
        return {
            "quantity": quantity,
            "exponent": np.nan,
            "C": np.nan,
            "R2": np.nan,
            "pvalue": np.nan,
            "RMSE_log": np.nan,
            "slope_ci_low": np.nan,
            "slope_ci_high": np.nan,
            "error": "need at least 3 positive points",
        }

    x = np.log(Ns)
    y = np.log(Ys)

    fit = linregress(x, y)
    slope = fit.slope
    intercept = fit.intercept
    y_pred = intercept + slope * x
    resid = y - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_log = float(np.sqrt(np.mean(resid ** 2)))

    n = len(x)
    dfree = n - 2
    tcrit = t.ppf(0.975, df=dfree) if dfree > 0 else np.nan
    slope_ci_low = slope - tcrit * fit.stderr if np.isfinite(tcrit) else np.nan
    slope_ci_high = slope + tcrit * fit.stderr if np.isfinite(tcrit) else np.nan

    return {
        "quantity": quantity,
        "exponent": float(slope),   # Y ~ N^(exponent)
        "C": float(np.exp(intercept)),
        "R2": float(r2),
        "pvalue": float(fit.pvalue),
        "RMSE_log": rmse_log,
        "slope_ci_low": float(slope_ci_low) if np.isfinite(slope_ci_low) else np.nan,
        "slope_ci_high": float(slope_ci_high) if np.isfinite(slope_ci_high) else np.nan,
    }


def fit_power_law_from_summary(summary_df: pd.DataFrame, col: str) -> dict:
    Ns = summary_df.index.to_numpy(dtype=float)
    Ys = summary_df[col].to_numpy(dtype=float)
    return fit_power_law_xy(Ns, Ys, quantity=col)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_power_law_from_summary(
    summary_df: pd.DataFrame,
    col: str,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    show_fit: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
    Ns = summary_df.index.to_numpy(dtype=float)
    Ys = summary_df[col].to_numpy(dtype=float)
    ax.loglog(Ns, Ys, "o-", label=label or col)
    if show_fit:
        fit = fit_power_law_from_summary(summary_df, col)
        if np.isfinite(fit["exponent"]):
            Nfit = np.linspace(Ns.min(), Ns.max(), 200)
            ax.loglog(
                Nfit,
                fit["C"] * Nfit ** fit["exponent"],
                "--",
                label=f'fit ~ N^({fit["exponent"]:.2f})',
            )
    ax.set_xlabel("N")
    ax.set_ylabel(col)
    ax.legend()
    return ax


# ---------------------------------------------------------------------
# Proxy-level reduced-space direction metrics
# ---------------------------------------------------------------------

def attach_proxy_direction_metrics(
    df_true: pd.DataFrame,
    df_proxy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge true-level fields into proxy rows and compute:
      kappa_emp, kappa_lin, kappa_recon,
      S_dir, eta_align, M_factor, amp_per_delta
    Requires:
      df_true: graph_id, N, phi_true, A_eff_red_norm_F, c_true, z_row
      df_proxy: delta[_red], eps_macro, lin_macro_pred, DeltaA_eff_red
    """
    true_use = df_true[[
        "N", "graph_id", "phi_true", "A_eff_red_norm_F", "c_true", "z_row",
        "c_norm", "z_norm"
    ]].copy()

    df = df_proxy.merge(true_use, on=["N", "graph_id"], how="left")
    rows = []
    for _, row in df.iterrows():
        delta = float(row["delta_red"] if "delta_red" in row else row["delta"])
        phi = float(row["phi_true"])
        Aeff_norm = float(row["A_eff_red_norm_F"])
        c = np.asarray(row["c_true"], dtype=float).reshape(-1)
        z = np.asarray(row["z_row"], dtype=float).reshape(-1)
        DeltaA_eff_red = np.asarray(row["DeltaA_eff_red"], dtype=float)

        c_norm = np.linalg.norm(c)
        z_norm = np.linalg.norm(z)
        c_hat = c / (c_norm + 1e-15)

        denom = delta * Aeff_norm
        if denom < 1e-15:
            M = np.full_like(DeltaA_eff_red, np.nan, dtype=float)
            w = np.full_like(c_hat, np.nan, dtype=float)
            S_dir = np.nan
            eta_align = np.nan
            M_factor = np.nan
            amp_per_delta = np.nan
            kappa_recon = np.nan
        else:
            M = DeltaA_eff_red / denom
            w = M @ c_hat
            S_dir = float(np.linalg.norm(w))
            eta_align = float(abs(np.dot(z, w)) / (z_norm * S_dir + 1e-15))
            M_factor = S_dir * eta_align
            amp_per_delta = float(np.linalg.norm(DeltaA_eff_red @ c) / (delta + 1e-15))
            kappa_recon = float((Aeff_norm / (abs(phi) + 1e-15)) * c_norm * z_norm * M_factor)

        row = row.to_dict()
        row["kappa_emp"] = float(row["eps_macro"]) / (delta + 1e-15)
        row["kappa_lin"] = float(row["lin_macro_pred"]) / (delta + 1e-15)
        row["kappa_recon"] = kappa_recon
        row["S_dir"] = S_dir
        row["eta_align"] = eta_align
        row["M_factor"] = M_factor
        row["amp_per_delta"] = amp_per_delta
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# LV true-level z decomposition (needs network-library A_true)
# ---------------------------------------------------------------------

def attach_lv_z_decomposition(
    df_true: pd.DataFrame,
    A_map: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """
    For LV only, compute:
      X = (Theta + A)^(-1) U
      X_norm, z_norm, z_vis, z_vis_rescaled,
      per-mode weighted summaries:
        cancel_wrms, spread_wrms, effsupport_wmean
    Requires df_true columns:
      graph_id, N, U_dyn, lv_params
    """
    rows = []
    for _, row in df_true.iterrows():
        graph_id = row["graph_id"]
        if graph_id not in A_map:
            raise KeyError(f"graph_id {graph_id} not in A_map")

        if "lv_params" not in row or row["lv_params"] is None:
            # skip non-LV rows
            newrow = row.to_dict()
            newrow.update({
                "X_norm": np.nan, "z_norm_recalc": np.nan, "z_vis": np.nan,
                "z_vis_rescaled": np.nan, "mode_vis_wrms": np.nan,
                "cancel_wrms": np.nan, "spread_wrms": np.nan,
                "effsupport_wmean": np.nan, "sanity_ratio": np.nan,
            })
            rows.append(newrow)
            continue

        A = np.asarray(A_map[graph_id], dtype=float)
        U = np.asarray(row["U_dyn"], dtype=float)
        theta = np.asarray(row["lv_params"]["theta"], dtype=float)

        X = np.linalg.solve(np.diag(theta) + A, U)
        X_norm = np.linalg.norm(X, ord="fro")

        z = np.mean(X, axis=0)
        z_norm = np.linalg.norm(z)
        z_vis = z_norm / (X_norm + 1e-15)
        z_vis_rescaled = math.sqrt(int(row["N"])) * z_vis

        col_norms = np.linalg.norm(X, axis=0) + 1e-15
        weights = (col_norms ** 2) / (np.sum(col_norms ** 2) + 1e-15)

        mode_vis = []
        cancel_list = []
        spread_list = []
        effsupport_list = []

        N = int(row["N"])
        for j in range(X.shape[1]):
            xj = X[:, j]
            rms_j = np.linalg.norm(xj) / math.sqrt(N)
            mean_j = float(np.mean(xj))
            mean_abs_j = float(np.mean(np.abs(xj))) + 1e-15

            vis_j = abs(mean_j) / (rms_j + 1e-15)
            cancel_j = abs(mean_j) / mean_abs_j
            spread_j = mean_abs_j / (rms_j + 1e-15)

            p = (xj ** 2) / (np.sum(xj ** 2) + 1e-15)
            ipr_j = float(np.sum(p ** 2))
            effsupport_frac_j = 1.0 / (N * ipr_j + 1e-15)

            mode_vis.append(vis_j)
            cancel_list.append(cancel_j)
            spread_list.append(spread_j)
            effsupport_list.append(effsupport_frac_j)

        mode_vis = np.asarray(mode_vis)
        cancel_list = np.asarray(cancel_list)
        spread_list = np.asarray(spread_list)
        effsupport_list = np.asarray(effsupport_list)

        mode_vis_wrms = float(np.sqrt(np.sum(weights * mode_vis ** 2)))
        cancel_wrms = float(np.sqrt(np.sum(weights * cancel_list ** 2)))
        spread_wrms = float(np.sqrt(np.sum(weights * spread_list ** 2)))
        effsupport_wmean = float(np.sum(weights * effsupport_list))
        sanity_ratio = float(mode_vis_wrms / (z_vis_rescaled + 1e-15))

        newrow = row.to_dict()
        newrow.update({
            "X_norm": float(X_norm),
            "z_norm_recalc": float(z_norm),
            "z_vis": float(z_vis),
            "z_vis_rescaled": float(z_vis_rescaled),
            "mode_vis_wrms": mode_vis_wrms,
            "cancel_wrms": cancel_wrms,
            "spread_wrms": spread_wrms,
            "effsupport_wmean": effsupport_wmean,
            "sanity_ratio": sanity_ratio,
        })
        rows.append(newrow)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Filtering / summarising
# ---------------------------------------------------------------------

def subset_proxy(
    df: pd.DataFrame,
    delta_lo: Optional[float] = None,
    delta_hi: Optional[float] = None,
    eps_sub_max: Optional[float] = None,
    eps_sub_quantile: Optional[float] = None,
    support_hamming_max: Optional[float] = None,
    same_support: Optional[bool] = None,
    fp_rel_max: Optional[float] = None,
    fp_rel_quantile: Optional[float] = None,
    fp_sub_rel_max: Optional[float] = None,
    fp_sub_rel_quantile: Optional[float] = None,
    fp_orth_rel_max: Optional[float] = None,
    fp_orth_rel_quantile: Optional[float] = None,
    fp_residual_ratio_max: Optional[float] = None,
    delta_col: str = "delta",
) -> pd.DataFrame:
    """Flexible proxy-row subsetting.

    Notes
    -----
    same_support / support_hamming are retained only for backward compatibility.
    For current fixed-point branch filtering, prefer the continuous metrics
    fp_rel, fp_sub_rel, fp_orth_rel, fp_residual_ratio.
    """
    sub = df.copy()
    if delta_lo is not None:
        sub = sub[sub[delta_col] >= float(delta_lo)]
    if delta_hi is not None:
        sub = sub[sub[delta_col] <= float(delta_hi)]
    if eps_sub_max is not None:
        sub = sub[sub["eps_sub"] <= float(eps_sub_max)]
    if eps_sub_quantile is not None:
        qcut = sub.groupby("N")["eps_sub"].transform(lambda s: s.quantile(float(eps_sub_quantile)))
        sub = sub[sub["eps_sub"] <= qcut]
    if support_hamming_max is not None:
        sub = sub[sub["support_hamming"] <= float(support_hamming_max)]
    if same_support is not None:
        sub = sub[sub["same_support"] == bool(same_support)]
    if fp_rel_max is not None:
        sub = sub[sub['fp_rel'] <= float(fp_rel_max)]
    if fp_rel_quantile is not None:
        qcut = sub.groupby('N')['fp_rel'].transform(lambda s: s.quantile(float(fp_rel_quantile)))
        sub = sub[sub['fp_rel'] <= qcut]
    if fp_sub_rel_max is not None:
        sub = sub[sub['fp_sub_rel'] <= float(fp_sub_rel_max)]
    if fp_sub_rel_quantile is not None:
        qcut = sub.groupby('N')['fp_sub_rel'].transform(lambda s: s.quantile(float(fp_sub_rel_quantile)))
        sub = sub[sub['fp_sub_rel'] <= qcut]
    if fp_orth_rel_max is not None:
        sub = sub[sub['fp_orth_rel'] <= float(fp_orth_rel_max)]
    if fp_orth_rel_quantile is not None:
        qcut = sub.groupby('N')['fp_orth_rel'].transform(lambda s: s.quantile(float(fp_orth_rel_quantile)))
        sub = sub[sub['fp_orth_rel'] <= qcut]
    if fp_residual_ratio_max is not None:
        sub = sub[sub['fp_residual_ratio'] <= float(fp_residual_ratio_max)]
    return sub.copy()

def aggregate_proxy_by_graph(
    df: pd.DataFrame,
    metrics: Sequence[str],
    agg: str = "median",
) -> pd.DataFrame:
    """Aggregate proxy rows to one row per graph."""
    grouped = df.groupby(["N", "graph_id"])[list(metrics)]
    if agg == "median":
        return grouped.median().reset_index()
    if agg == "mean":
        return grouped.mean().reset_index()
    raise ValueError("agg must be 'median' or 'mean'")


def aggregate_by_N(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Aggregate graph-level rows to one row per N using median + IQR."""
    out = df.groupby("N")[list(metrics)].agg(
        ["median", "mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
    )
    # flatten multi-index columns
    out.columns = [
        f"{a}_{b if isinstance(b, str) else 'q'}"
        for a, b in out.columns.to_flat_index()
    ]
    return out


def aggregate_graph_medians_by_N(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Simple N-summary using graph-level medians (one value per N per metric)."""
    return df.groupby("N")[list(metrics)].median()


def fit_multiple_from_summary(summary_df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = [fit_power_law_from_summary(summary_df, m) for m in metrics]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Pretty helpers
# ---------------------------------------------------------------------

def available_metrics(df: pd.DataFrame) -> List[str]:
    exclude = {"N", "graph_id", "seed", "dynamics"}
    return [c for c in df.columns if c not in exclude]


def describe_by_N(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df.groupby("N")[list(cols)].describe().round(4)


# ---------------------------------------------------------------------
# Added helpers for theory-facing finite-size scaling analysis
# ---------------------------------------------------------------------


def attach_proxy_theory_metrics(
    df_true: pd.DataFrame,
    df_proxy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge true-level theory quantities into proxy rows and add normalized
    leakage / nonlinearity diagnostics aligned with the current manuscript.

    Added columns
    -------------
    phi_true, K_eff_base_red, sigma_min_F, sigma_min_H,
    xi_over_phi, xi_over_phi_per_delta,
    lin_gap_abs, lin_gap_rel, lin_ratio,
    eps_over_delta, eps_over_linpred,
    fp_rel, fp_sub_rel, fp_orth_rel, fp_residual_ratio,
    r_proxy_norm, c_proxy_norm

    Notes
    -----
    The previous support-based filters are intentionally not used here as the
    primary fixed-point branch screen. For finite-size scaling, continuous
    fixed-point proximity metrics are much more stable than binary support
    equality under a hard tolerance.
    """
    true_use_cols = [
        'N', 'graph_id', 'phi_true', 'K_eff_base_red',
        'sigma_min_F', 'sigma_min_H', 'U_dyn', 'x_star_true', 'r_true'
    ]
    true_use = df_true[true_use_cols].copy()
    df = df_proxy.merge(true_use, on=['N', 'graph_id'], how='left')

    phi_abs = df['phi_true'].abs() + 1e-15
    delta_use = df['delta_red'] if 'delta_red' in df.columns else df['delta']
    lin_abs = df['lin_macro_pred'].abs() + 1e-15

    df['xi_over_phi'] = df['xi_norm'] / phi_abs
    df['xi_over_phi_per_delta'] = df['xi_norm'] / (phi_abs * (delta_use + 1e-15))
    df['lin_gap_abs'] = (df['eps_macro'] - df['lin_macro_pred']).abs()
    df['lin_gap_rel'] = df['lin_gap_abs'] / lin_abs
    df['lin_ratio'] = df['eps_macro'] / lin_abs
    df['eps_over_delta'] = df['eps_macro'] / (delta_use + 1e-15)
    df['eps_over_linpred'] = df['eps_macro'] / lin_abs

    fp_rel_list = []
    fp_sub_rel_list = []
    fp_orth_rel_list = []
    fp_residual_ratio_list = []
    r_proxy_norm_list = []
    c_proxy_norm_list = []

    for _, row in df.iterrows():
        U = np.asarray(row['U_dyn'], dtype=float)
        x_true = np.asarray(row['x_star_true'], dtype=float).reshape(-1)
        x_proxy = np.asarray(row['x_star_proxy'], dtype=float).reshape(-1)
        r_true = np.asarray(row['r_true'], dtype=float).reshape(-1)

        c_true = U.T @ x_true
        c_proxy = U.T @ x_proxy
        r_proxy = x_proxy - U @ c_proxy

        fp_rel = float(np.linalg.norm(x_proxy - x_true) / (np.linalg.norm(x_true) + 1e-15))
        fp_sub_rel = float(np.linalg.norm(c_proxy - c_true) / (np.linalg.norm(c_true) + 1e-15))
        fp_orth_rel = float(np.linalg.norm(r_proxy - r_true) / (np.linalg.norm(r_true) + 1e-15))
        fp_resid = float(np.linalg.norm(r_proxy) / (np.linalg.norm(x_proxy) + 1e-15))

        fp_rel_list.append(fp_rel)
        fp_sub_rel_list.append(fp_sub_rel)
        fp_orth_rel_list.append(fp_orth_rel)
        fp_residual_ratio_list.append(fp_resid)
        r_proxy_norm_list.append(float(np.linalg.norm(r_proxy)))
        c_proxy_norm_list.append(float(np.linalg.norm(c_proxy)))

    df['fp_rel'] = fp_rel_list
    df['fp_sub_rel'] = fp_sub_rel_list
    df['fp_orth_rel'] = fp_orth_rel_list
    df['fp_residual_ratio'] = fp_residual_ratio_list
    df['r_proxy_norm'] = r_proxy_norm_list
    df['c_proxy_norm'] = c_proxy_norm_list
    return df

def empirical_keff_by_window_filtered(
    df_proxy: pd.DataFrame,
    delta_lo: float = 0.0,
    delta_hi: float = 0.10,
    delta_col: str = 'delta',
    response_col: str = 'eps_macro',
    min_points_per_graph: int = 3,
    agg: str = 'median',
    same_support: Optional[bool] = None,
    support_hamming_max: Optional[float] = None,
    eps_sub_max: Optional[float] = None,
    eps_sub_quantile: Optional[float] = None,
    fp_rel_max: Optional[float] = None,
    fp_rel_quantile: Optional[float] = None,
    fp_sub_rel_max: Optional[float] = None,
    fp_sub_rel_quantile: Optional[float] = None,
    fp_orth_rel_max: Optional[float] = None,
    fp_orth_rel_quantile: Optional[float] = None,
    fp_residual_ratio_max: Optional[float] = None,
    xi_over_phi_max: Optional[float] = None,
    xi_over_phi_per_delta_max: Optional[float] = None,
    lin_gap_rel_max: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Theory-facing empirical K_eff fit with continuous fixed-point filters.
    """
    sub = subset_proxy(
        df_proxy,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        eps_sub_max=eps_sub_max,
        eps_sub_quantile=eps_sub_quantile,
        support_hamming_max=support_hamming_max,
        same_support=same_support,
        fp_rel_max=fp_rel_max,
        fp_rel_quantile=fp_rel_quantile,
        fp_sub_rel_max=fp_sub_rel_max,
        fp_sub_rel_quantile=fp_sub_rel_quantile,
        fp_orth_rel_max=fp_orth_rel_max,
        fp_orth_rel_quantile=fp_orth_rel_quantile,
        fp_residual_ratio_max=fp_residual_ratio_max,
        delta_col=delta_col,
    )

    if xi_over_phi_max is not None:
        if 'xi_over_phi' not in sub.columns:
            raise KeyError('xi_over_phi not found; run attach_proxy_theory_metrics first')
        sub = sub[sub['xi_over_phi'] <= float(xi_over_phi_max)]
    if xi_over_phi_per_delta_max is not None:
        if 'xi_over_phi_per_delta' not in sub.columns:
            raise KeyError('xi_over_phi_per_delta not found; run attach_proxy_theory_metrics first')
        sub = sub[sub['xi_over_phi_per_delta'] <= float(xi_over_phi_per_delta_max)]
    if lin_gap_rel_max is not None:
        if 'lin_gap_rel' not in sub.columns:
            raise KeyError('lin_gap_rel not found; run attach_proxy_theory_metrics first')
        sub = sub[sub['lin_gap_rel'] <= float(lin_gap_rel_max)]

    rows = []
    for (N, graph_id), subg in sub.groupby(['N', 'graph_id']):
        sub2 = subg.sort_values(delta_col)
        if len(sub2) < min_points_per_graph:
            continue
        slope = fit_slope_through_origin(sub2[delta_col], sub2[response_col])
        rows.append({'N': N, 'graph_id': graph_id, 'K_eff_emp': slope, 'n_points': len(sub2)})

    if len(rows) == 0:
        empty = pd.DataFrame(columns=['N', 'graph_id', 'K_eff_emp', 'n_points'])
        return empty, pd.DataFrame(), pd.DataFrame()

    df_slope = pd.DataFrame(rows)
    summary = df_slope.groupby('N')['K_eff_emp'].agg(
        median='median',
        mean='mean',
        std='std',
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    )
    fit = fit_power_law_from_summary(summary, agg)
    fit_df = pd.DataFrame([fit])
    return df_slope, summary, fit_df

def diagnose_keff_filter_funnel(
    df_proxy: pd.DataFrame,
    delta_lo: float = 0.0,
    delta_hi: float = 0.10,
    delta_col: str = 'delta',
    same_support: Optional[bool] = None,
    support_hamming_max: Optional[float] = None,
    eps_sub_max: Optional[float] = None,
    eps_sub_quantile: Optional[float] = None,
    fp_rel_max: Optional[float] = None,
    fp_rel_quantile: Optional[float] = None,
    fp_sub_rel_max: Optional[float] = None,
    fp_sub_rel_quantile: Optional[float] = None,
    fp_orth_rel_max: Optional[float] = None,
    fp_orth_rel_quantile: Optional[float] = None,
    fp_residual_ratio_max: Optional[float] = None,
    xi_over_phi_max: Optional[float] = None,
    xi_over_phi_per_delta_max: Optional[float] = None,
    lin_gap_rel_max: Optional[float] = None,
    min_points_per_graph: int = 3,
) -> pd.DataFrame:
    """Diagnose which filtering stage removes the empirical-K_eff candidates."""
    stages = []
    sub = df_proxy.copy()

    def _add_stage(name, frame):
        stages.append({
            'stage': name,
            'n_rows': int(len(frame)),
            'n_graphs': int(frame[['N','graph_id']].drop_duplicates().shape[0]) if len(frame) else 0,
            'n_N': int(frame['N'].nunique()) if ('N' in frame.columns and len(frame)) else 0,
        })

    _add_stage('start', sub)
    sub = sub[(sub[delta_col] >= float(delta_lo)) & (sub[delta_col] <= float(delta_hi))]
    _add_stage('delta_window', sub)
    if same_support is not None:
        sub = sub[sub['same_support'] == bool(same_support)]
        _add_stage('same_support', sub)
    if support_hamming_max is not None:
        sub = sub[sub['support_hamming'] <= float(support_hamming_max)]
        _add_stage('support_hamming_max', sub)
    if eps_sub_max is not None:
        sub = sub[sub['eps_sub'] <= float(eps_sub_max)]
        _add_stage('eps_sub_max', sub)
    if eps_sub_quantile is not None:
        qcut = sub.groupby('N')['eps_sub'].transform(lambda s: s.quantile(float(eps_sub_quantile)))
        sub = sub[sub['eps_sub'] <= qcut]
        _add_stage('eps_sub_quantile', sub)
    if fp_rel_max is not None:
        sub = sub[sub['fp_rel'] <= float(fp_rel_max)]
        _add_stage('fp_rel_max', sub)
    if fp_rel_quantile is not None:
        qcut = sub.groupby('N')['fp_rel'].transform(lambda s: s.quantile(float(fp_rel_quantile)))
        sub = sub[sub['fp_rel'] <= qcut]
        _add_stage('fp_rel_quantile', sub)
    if fp_sub_rel_max is not None:
        sub = sub[sub['fp_sub_rel'] <= float(fp_sub_rel_max)]
        _add_stage('fp_sub_rel_max', sub)
    if fp_sub_rel_quantile is not None:
        qcut = sub.groupby('N')['fp_sub_rel'].transform(lambda s: s.quantile(float(fp_sub_rel_quantile)))
        sub = sub[sub['fp_sub_rel'] <= qcut]
        _add_stage('fp_sub_rel_quantile', sub)
    if fp_orth_rel_max is not None:
        sub = sub[sub['fp_orth_rel'] <= float(fp_orth_rel_max)]
        _add_stage('fp_orth_rel_max', sub)
    if fp_orth_rel_quantile is not None:
        qcut = sub.groupby('N')['fp_orth_rel'].transform(lambda s: s.quantile(float(fp_orth_rel_quantile)))
        sub = sub[sub['fp_orth_rel'] <= qcut]
        _add_stage('fp_orth_rel_quantile', sub)
    if fp_residual_ratio_max is not None:
        sub = sub[sub['fp_residual_ratio'] <= float(fp_residual_ratio_max)]
        _add_stage('fp_residual_ratio_max', sub)
    if xi_over_phi_max is not None:
        sub = sub[sub['xi_over_phi'] <= float(xi_over_phi_max)]
        _add_stage('xi_over_phi_max', sub)
    if xi_over_phi_per_delta_max is not None:
        sub = sub[sub['xi_over_phi_per_delta'] <= float(xi_over_phi_per_delta_max)]
        _add_stage('xi_over_phi_per_delta_max', sub)
    if lin_gap_rel_max is not None:
        sub = sub[sub['lin_gap_rel'] <= float(lin_gap_rel_max)]
        _add_stage('lin_gap_rel_max', sub)

    counts = sub.groupby(['N', 'graph_id']).size().reset_index(name='n_points') if len(sub) else pd.DataFrame(columns=['N','graph_id','n_points'])
    counts = counts[counts['n_points'] >= int(min_points_per_graph)]
    _add_stage('graphs_with_enough_points', counts)
    return pd.DataFrame(stages)

def empirical_keff_grid_scan(
    df_proxy: pd.DataFrame,
    delta_windows: Sequence[Tuple[float, float]],
    delta_col: str = 'delta',
    response_col: str = 'eps_macro',
    min_points_per_graph: int = 3,
    agg: str = 'median',
    same_support: Optional[bool] = None,
    support_hamming_max: Optional[float] = None,
    eps_sub_quantile: Optional[float] = None,
    xi_over_phi_max: Optional[float] = None,
    xi_over_phi_per_delta_max: Optional[float] = None,
    lin_gap_rel_max: Optional[float] = None,
) -> pd.DataFrame:
    """Scan multiple (delta_lo, delta_hi) windows and collect fit summaries."""
    rows = []
    for delta_lo, delta_hi in delta_windows:
        _, _, fit_df = empirical_keff_by_window_filtered(
            df_proxy,
            delta_lo=delta_lo,
            delta_hi=delta_hi,
            delta_col=delta_col,
            response_col=response_col,
            min_points_per_graph=min_points_per_graph,
            agg=agg,
            same_support=same_support,
            support_hamming_max=support_hamming_max,
            eps_sub_quantile=eps_sub_quantile,
            xi_over_phi_max=xi_over_phi_max,
            xi_over_phi_per_delta_max=xi_over_phi_per_delta_max,
            lin_gap_rel_max=lin_gap_rel_max,
        )
        if len(fit_df) > 0:
            rows.append(fit_df.iloc[0].to_dict())
    return pd.DataFrame(rows)



def build_true_candidate_metrics(df_true: pd.DataFrame) -> pd.DataFrame:
    """
    Add manuscript-aligned component metrics at the true-graph level.
    """
    df = df_true.copy()
    phi_abs = df['phi_true'].abs() + 1e-15
    df['inv_phi_true'] = 1.0 / phi_abs
    df['Aeff_over_phi'] = df['A_eff_red_norm_F'] / phi_abs
    df['shielding_obs'] = df['G_obs']
    df['forcing_amp'] = df['c_norm']
    df['K_base_rebuilt'] = df['Aeff_over_phi'] * df['shielding_obs'] * df['forcing_amp']

    if 'sigma_min_H' in df.columns:
        df['inv_sigma_min_H'] = 1.0 / (df['sigma_min_H'].abs() + 1e-15)
        df['c_over_sigmaH'] = df['c_norm'] * df['inv_sigma_min_H']
        df['Aeff_c_over_sigmaH_phi'] = df['A_eff_red_norm_F'] * df['c_norm'] * df['inv_sigma_min_H'] / phi_abs
    if 'sigma_min_F' in df.columns:
        df['inv_sigma_min_F'] = 1.0 / (df['sigma_min_F'].abs() + 1e-15)
        df['c_over_sigmaF'] = df['c_norm'] * df['inv_sigma_min_F']
        df['Aeff_c_over_sigmaF_phi'] = df['A_eff_red_norm_F'] * df['c_norm'] * df['inv_sigma_min_F'] / phi_abs
    if 'r_norm' in df.columns and 'c_norm' in df.columns:
        df['r_over_c'] = df['r_norm'] / (df['c_norm'] + 1e-15)
    return df



def summarize_true_metric_scalings(
    df_true: pd.DataFrame,
    metrics: Sequence[str],
    agg: str = 'median',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate selected true-level metrics by N and fit power laws.
    """
    graph_df = df_true.groupby(['N', 'graph_id'])[list(metrics)].median().reset_index()
    if agg == 'median':
        summary = graph_df.groupby('N')[list(metrics)].median()
    elif agg == 'mean':
        summary = graph_df.groupby('N')[list(metrics)].mean()
    else:
        raise ValueError("agg must be 'median' or 'mean'")
    fits = fit_multiple_from_summary(summary, metrics)
    return summary, fits



def bootstrap_power_law_exponent(
    df_graph: pd.DataFrame,
    metric: str,
    n_boot: int = 1000,
    agg: str = 'median',
    random_state: int = 0,
) -> dict:
    """
    Bootstrap graph instances within each N to quantify exponent uncertainty.
    Expects one row per graph.
    """
    rng = np.random.RandomState(random_state)
    exponents = []
    Ns = sorted(df_graph['N'].unique())
    for _ in range(int(n_boot)):
        vals = []
        keep_N = []
        for N in Ns:
            sub = df_graph[df_graph['N'] == N][metric].dropna().to_numpy(dtype=float)
            if len(sub) == 0:
                continue
            sample = rng.choice(sub, size=len(sub), replace=True)
            val = np.median(sample) if agg == 'median' else np.mean(sample)
            if np.isfinite(val) and val > 0:
                keep_N.append(float(N))
                vals.append(float(val))
        if len(keep_N) >= 3:
            fit = fit_power_law_xy(keep_N, vals, quantity=metric)
            if np.isfinite(fit['exponent']):
                exponents.append(float(fit['exponent']))

    exponents = np.asarray(exponents, dtype=float)
    if len(exponents) == 0:
        return {
            'metric': metric,
            'n_boot': 0,
            'exp_median': np.nan,
            'exp_q025': np.nan,
            'exp_q975': np.nan,
        }
    return {
        'metric': metric,
        'n_boot': int(len(exponents)),
        'exp_median': float(np.median(exponents)),
        'exp_q025': float(np.quantile(exponents, 0.025)),
        'exp_q975': float(np.quantile(exponents, 0.975)),
    }



def compare_empirical_vs_true_metrics(
    df_keff_graph: pd.DataFrame,
    df_true: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """
    Correlate graph-level empirical K_eff against true-level candidate metrics.
    Returns Pearson correlations on graph medians pooled across N.
    """
    true_graph = df_true.groupby(['N', 'graph_id'])[list(metrics)].median().reset_index()
    merged = df_keff_graph.merge(true_graph, on=['N', 'graph_id'], how='left')

    rows = []
    for metric in metrics:
        sub = merged[['K_eff_emp', metric]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            rows.append({'metric': metric, 'pearson_r': np.nan, 'n': int(len(sub))})
            continue
        r = float(np.corrcoef(sub['K_eff_emp'], sub[metric])[0, 1])
        rows.append({'metric': metric, 'pearson_r': r, 'n': int(len(sub))})
    return pd.DataFrame(rows).sort_values('pearson_r', ascending=False)



# ---------------------------------------------------------------------
# Section 5.1 decomposition helpers
# ---------------------------------------------------------------------

SECTION51_PARTS = [
    'inv_phi_true',         # macro baseline
    'A_eff_red_norm_F',     # subspace topology
    'G_obs',                # topological shielding
    'c_norm',               # forcing amplitude
]


def build_section51_parts(df_true: pd.DataFrame) -> pd.DataFrame:
    """
    Add the four manuscript Sec. 5.1 part metrics and reconstruction checks.

    Added columns
    -------------
    inv_phi_true      = 1 / |phi_true|
    part_macro        = inv_phi_true
    part_topology     = A_eff_red_norm_F
    part_shielding    = G_obs
    part_forcing      = c_norm
    K_parts_product   = inv_phi_true * A_eff_red_norm_F * G_obs * c_norm
    K_parts_ratio_to_saved = K_parts_product / K_eff_base_red

    Notes
    -----
    `K_parts_product` should numerically reconstruct `K_eff_base_red`
    because the saved scan quantity was defined as
        (||A_eff_red||_F / |phi_true|) * G_obs * ||c_true||_2.
    """
    df = build_true_candidate_metrics(df_true).copy()

    df['part_macro'] = df['inv_phi_true']
    df['part_topology'] = df['A_eff_red_norm_F']
    df['part_shielding'] = df['G_obs']
    df['part_forcing'] = df['c_norm']
    df['K_parts_product'] = (
        df['part_macro'] *
        df['part_topology'] *
        df['part_shielding'] *
        df['part_forcing']
    )
    if 'K_eff_base_red' in df.columns:
        df['K_parts_ratio_to_saved'] = df['K_parts_product'] / (df['K_eff_base_red'] + 1e-15)
    return df


def fit_section51_part_scalings(
    df_true: pd.DataFrame,
    agg: str = 'median',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit finite-size power laws for the four Sec. 5.1 parts and their products.

    Returns
    -------
    summary : one row per N, aggregated graph-level values
    fits    : one row per metric with exponent fit
    compare : concise comparison table including the sum of part exponents
    """
    df = build_section51_parts(df_true)

    metrics = [
        'part_macro',
        'part_topology',
        'part_shielding',
        'part_forcing',
        'K_parts_product',
        'K_eff_base_red',
    ]
    summary, fits = summarize_true_metric_scalings(df, metrics=metrics, agg=agg)

    exp_map = dict(zip(fits['quantity'], fits['exponent']))
    compare = pd.DataFrame([
        {
            'target': 'sum_of_part_exponents',
            'exponent': (
                exp_map.get('part_macro', np.nan) +
                exp_map.get('part_topology', np.nan) +
                exp_map.get('part_shielding', np.nan) +
                exp_map.get('part_forcing', np.nan)
            ),
            'note': 'sum of separately fitted exponents after N-level aggregation',
        },
        {
            'target': 'K_parts_product',
            'exponent': exp_map.get('K_parts_product', np.nan),
            'note': 'fit of aggregated product-of-parts',
        },
        {
            'target': 'K_eff_base_red',
            'exponent': exp_map.get('K_eff_base_red', np.nan),
            'note': 'fit of saved theory-side K_eff base',
        },
    ])
    return summary, fits, compare


def merge_true_with_keff_emp(
    df_true: pd.DataFrame,
    df_keff_graph: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge true-level graph quantities with graph-level empirical K_eff slopes.
    Returns an empty but well-formed DataFrame when no graph slopes survived.
    """
    true_use = build_section51_parts(df_true).copy()
    if df_keff_graph is None or len(df_keff_graph) == 0:
        cols = list(true_use.columns) + ['K_eff_emp', 'n_points', 'delta_lo', 'delta_hi']
        return pd.DataFrame(columns=cols)
    need_cols = {'N', 'graph_id'}
    if not need_cols.issubset(df_keff_graph.columns):
        cols = list(true_use.columns) + [c for c in ['K_eff_emp', 'n_points', 'delta_lo', 'delta_hi'] if c not in true_use.columns]
        return pd.DataFrame(columns=cols)
    return true_use.merge(df_keff_graph, on=['N', 'graph_id'], how='inner')


def section51_empirical_vs_parts(
    df_true: pd.DataFrame,
    df_proxy: pd.DataFrame,
    delta_lo: float = 0.0,
    delta_hi: float = 0.10,
    delta_col: str = 'delta',
    response_col: str = 'eps_macro',
    min_points_per_graph: int = 3,
    agg: str = 'median',
    same_support: Optional[bool] = None,
    support_hamming_max: Optional[float] = None,
    eps_sub_max: Optional[float] = None,
    eps_sub_quantile: Optional[float] = None,
    fp_rel_max: Optional[float] = None,
    fp_rel_quantile: Optional[float] = None,
    fp_sub_rel_max: Optional[float] = None,
    fp_sub_rel_quantile: Optional[float] = None,
    fp_orth_rel_max: Optional[float] = None,
    fp_orth_rel_quantile: Optional[float] = None,
    fp_residual_ratio_max: Optional[float] = None,
    xi_over_phi_max: Optional[float] = None,
    xi_over_phi_per_delta_max: Optional[float] = None,
    lin_gap_rel_max: Optional[float] = None,
) -> dict:
    """
    Compare the empirical finite-size exponent of K_eff with the Sec. 5.1
    part exponents.

    Important statistical note
    --------------------------
    There are *three* distinct composition checks here:

    1) sum_parts_exp:
       Sum of exponents fitted to the N-aggregated summaries of the four parts.
       This can differ from the exponent of an aggregated product because
       aggregation (especially median) does not commute with multiplication.

    2) K_parts_product_exp:
       Exponent of the graph-level product
           (1/|phi|) * ||A_eff||_F * G_obs * ||c||
       aggregated by N and then fit.
       This should track K_eff_base_red closely.

    3) K_eff_emp_exp:
       Exponent of the empirical small-delta slope from proxy data.

    If (1) fails but (2) matches the theory-side K term, the issue may be
    aggregation non-commutativity rather than missing physics.
    """
    df_true2 = build_section51_parts(df_true)
    df_proxy2 = attach_proxy_theory_metrics(df_true2, df_proxy)

    df_keff_graph, keff_summary, keff_fit = empirical_keff_by_window_filtered(
        df_proxy2,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        delta_col=delta_col,
        response_col=response_col,
        min_points_per_graph=min_points_per_graph,
        agg=agg,
        same_support=same_support,
        support_hamming_max=support_hamming_max,
        eps_sub_max=eps_sub_max,
        eps_sub_quantile=eps_sub_quantile,
        fp_rel_max=fp_rel_max,
        fp_rel_quantile=fp_rel_quantile,
        fp_sub_rel_max=fp_sub_rel_max,
        fp_sub_rel_quantile=fp_sub_rel_quantile,
        fp_orth_rel_max=fp_orth_rel_max,
        fp_orth_rel_quantile=fp_orth_rel_quantile,
        fp_residual_ratio_max=fp_residual_ratio_max,
        xi_over_phi_max=xi_over_phi_max,
        xi_over_phi_per_delta_max=xi_over_phi_per_delta_max,
        lin_gap_rel_max=lin_gap_rel_max,
    )

    part_summary, part_fits, part_compare = fit_section51_part_scalings(df_true2, agg=agg)
    merged = merge_true_with_keff_emp(df_true2, df_keff_graph)
    funnel = diagnose_keff_filter_funnel(
        df_proxy2,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        delta_col=delta_col,
        same_support=same_support,
        support_hamming_max=support_hamming_max,
        eps_sub_max=eps_sub_max,
        eps_sub_quantile=eps_sub_quantile,
        fp_rel_max=fp_rel_max,
        fp_rel_quantile=fp_rel_quantile,
        fp_sub_rel_max=fp_sub_rel_max,
        fp_sub_rel_quantile=fp_sub_rel_quantile,
        fp_orth_rel_max=fp_orth_rel_max,
        fp_orth_rel_quantile=fp_orth_rel_quantile,
        fp_residual_ratio_max=fp_residual_ratio_max,
        xi_over_phi_max=xi_over_phi_max,
        xi_over_phi_per_delta_max=xi_over_phi_per_delta_max,
        lin_gap_rel_max=lin_gap_rel_max,
        min_points_per_graph=min_points_per_graph,
    )

    comparison_rows = []
    exp_map = dict(zip(part_fits['quantity'], part_fits['exponent']))
    exp_sum = (
        exp_map.get('part_macro', np.nan) +
        exp_map.get('part_topology', np.nan) +
        exp_map.get('part_shielding', np.nan) +
        exp_map.get('part_forcing', np.nan)
    )

    comparison_rows.append({
        'quantity': 'K_eff_emp',
        'exponent': float(keff_fit.iloc[0]['exponent']) if len(keff_fit) else np.nan,
        'source': 'proxy small-delta slope',
    })
    comparison_rows.append({
        'quantity': 'sum_parts_exp',
        'exponent': float(exp_sum),
        'source': 'sum of separately fitted part exponents',
    })
    comparison_rows.append({
        'quantity': 'K_parts_product',
        'exponent': float(exp_map.get('K_parts_product', np.nan)),
        'source': 'fit of aggregated part product',
    })
    comparison_rows.append({
        'quantity': 'K_eff_base_red',
        'exponent': float(exp_map.get('K_eff_base_red', np.nan)),
        'source': 'fit of saved theory-side base K',
    })

    comparison = pd.DataFrame(comparison_rows)

    return {
        'df_true': df_true2,
        'df_proxy': df_proxy2,
        'df_keff_graph': df_keff_graph,
        'keff_summary': keff_summary,
        'keff_fit': keff_fit,
        'part_summary': part_summary,
        'part_fits': part_fits,
        'part_compare': part_compare,
        'comparison': comparison,
        'merged_graph_df': merged,
        'filter_funnel': funnel,
    }


def _bootstrap_joint_N_summary(
    df_graph: pd.DataFrame,
    metrics: Sequence[str],
    agg: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    Jointly resample graph rows within each N so cross-metric covariance is preserved.
    Returns one row per N with aggregated metrics.
    """
    rows = []
    for N in sorted(df_graph['N'].unique()):
        sub = df_graph[df_graph['N'] == N].copy()
        if len(sub) == 0:
            continue
        idx = rng.choice(sub.index.to_numpy(), size=len(sub), replace=True)
        boot = sub.loc[idx, list(metrics)].copy()

        if agg == 'median':
            vals = boot.median(axis=0)
        elif agg == 'mean':
            vals = boot.mean(axis=0)
        else:
            raise ValueError("agg must be 'median' or 'mean'")

        row = {'N': float(N)}
        for m in metrics:
            row[m] = float(vals[m])
        rows.append(row)

    return pd.DataFrame(rows).sort_values('N')


def bootstrap_section51_composition(
    df_true: pd.DataFrame,
    df_proxy: pd.DataFrame,
    delta_lo: float = 0.0,
    delta_hi: float = 0.10,
    delta_col: str = 'delta',
    response_col: str = 'eps_macro',
    min_points_per_graph: int = 3,
    agg: str = 'median',
    same_support: Optional[bool] = None,
    support_hamming_max: Optional[float] = None,
    eps_sub_max: Optional[float] = None,
    eps_sub_quantile: Optional[float] = None,
    xi_over_phi_max: Optional[float] = None,
    xi_over_phi_per_delta_max: Optional[float] = None,
    lin_gap_rel_max: Optional[float] = None,
    n_boot: int = 1000,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap comparison between:
      - empirical K_eff exponent
      - exponent of aggregated part product
      - exponent of saved K_eff_base_red
      - sum of the four part exponents

    The bootstrap resamples graph instances within each N jointly, so the
    covariance among parts is preserved.

    Returns
    -------
    boot_df      : one row per bootstrap replicate with exponent estimates
    boot_summary : median / 95% interval summary
    """
    result = section51_empirical_vs_parts(
        df_true=df_true,
        df_proxy=df_proxy,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        delta_col=delta_col,
        response_col=response_col,
        min_points_per_graph=min_points_per_graph,
        agg=agg,
        same_support=same_support,
        support_hamming_max=support_hamming_max,
        eps_sub_max=eps_sub_max,
        eps_sub_quantile=eps_sub_quantile,
        fp_rel_max=fp_rel_max,
        fp_rel_quantile=fp_rel_quantile,
        fp_sub_rel_max=fp_sub_rel_max,
        fp_sub_rel_quantile=fp_sub_rel_quantile,
        fp_orth_rel_max=fp_orth_rel_max,
        fp_orth_rel_quantile=fp_orth_rel_quantile,
        fp_residual_ratio_max=fp_residual_ratio_max,
        xi_over_phi_max=xi_over_phi_max,
        xi_over_phi_per_delta_max=xi_over_phi_per_delta_max,
        lin_gap_rel_max=lin_gap_rel_max,
    )

    merged = result['merged_graph_df'].copy()
    metrics = [
        'K_eff_emp',
        'part_macro',
        'part_topology',
        'part_shielding',
        'part_forcing',
        'K_parts_product',
        'K_eff_base_red',
    ]
    if len(merged) == 0:
        empty_boot = pd.DataFrame(columns=['K_eff_emp_exp','sum_parts_exp','K_parts_product_exp','K_eff_base_red_exp'])
        empty_summary = pd.DataFrame(columns=['quantity','median','q025','q975'])
        return empty_boot, empty_summary
    merged = merged[['N', 'graph_id'] + metrics].dropna()
    if len(merged) == 0:
        empty_boot = pd.DataFrame(columns=['K_eff_emp_exp','sum_parts_exp','K_parts_product_exp','K_eff_base_red_exp'])
        empty_summary = pd.DataFrame(columns=['quantity','median','q025','q975'])
        return empty_boot, empty_summary

    rng = np.random.RandomState(random_state)
    rows = []
    for _ in range(int(n_boot)):
        boot_summary = _bootstrap_joint_N_summary(
            merged[['N'] + metrics],
            metrics=metrics,
            agg=agg,
            rng=rng,
        )

        if len(boot_summary) < 3:
            continue

        exp_map = {}
        for m in metrics:
            fit = fit_power_law_xy(boot_summary['N'].to_numpy(), boot_summary[m].to_numpy(), quantity=m)
            exp_map[m] = fit['exponent']

        rows.append({
            'K_eff_emp_exp': exp_map.get('K_eff_emp', np.nan),
            'part_macro_exp': exp_map.get('part_macro', np.nan),
            'part_topology_exp': exp_map.get('part_topology', np.nan),
            'part_shielding_exp': exp_map.get('part_shielding', np.nan),
            'part_forcing_exp': exp_map.get('part_forcing', np.nan),
            'sum_parts_exp': (
                exp_map.get('part_macro', np.nan) +
                exp_map.get('part_topology', np.nan) +
                exp_map.get('part_shielding', np.nan) +
                exp_map.get('part_forcing', np.nan)
            ),
            'K_parts_product_exp': exp_map.get('K_parts_product', np.nan),
            'K_eff_base_red_exp': exp_map.get('K_eff_base_red', np.nan),
        })

    boot_df = pd.DataFrame(rows)

    summary_rows = []
    for col in boot_df.columns:
        vals = boot_df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            summary_rows.append({
                'quantity': col,
                'n_boot': 0,
                'median': np.nan,
                'q025': np.nan,
                'q975': np.nan,
            })
            continue
        summary_rows.append({
            'quantity': col,
            'n_boot': int(len(vals)),
            'median': float(np.median(vals)),
            'q025': float(np.quantile(vals, 0.025)),
            'q975': float(np.quantile(vals, 0.975)),
        })

    boot_summary = pd.DataFrame(summary_rows)
    return boot_df, boot_summary


def section51_notebook_recipe() -> str:
    """
    Return a compact notebook recipe string for the Sec. 5.1 analysis.
    """
    return """
from scaling_analysis_improved_v4 import *

paths_by_N = {
    50: "BA_N50_LV_scan_v2_n20.pkl",
    100: "BA_N100_LV_scan_v2_n20.pkl",
    200: "BA_N200_LV_scan_v2_n20.pkl",
    400: "BA_N400_LV_scan_v2_n20.pkl",
}

data_by_N = load_scan_pickles(paths_by_N)
df_true, df_proxy = flatten_scan_data(data_by_N, keep_arrays=True)

result = section51_empirical_vs_parts(
    df_true, df_proxy,
    delta_lo=0.00,
    delta_hi=0.10,
    same_support=True,
    support_hamming_max=0.0,
    eps_sub_quantile=0.5,
    xi_over_phi_per_delta_max=1.0,
    lin_gap_rel_max=0.5,
)

print(result['comparison'])
print(result['part_fits'][['quantity', 'exponent', 'R2', 'slope_ci_low', 'slope_ci_high']])
print(result['keff_fit'][['exponent', 'R2', 'slope_ci_low', 'slope_ci_high']])

boot_df, boot_summary = bootstrap_section51_composition(
    df_true, df_proxy,
    delta_lo=0.00,
    delta_hi=0.10,
    same_support=True,
    support_hamming_max=0.0,
    eps_sub_quantile=0.5,
    xi_over_phi_per_delta_max=1.0,
    lin_gap_rel_max=0.5,
    n_boot=2000,
    random_state=0,
)
print(boot_summary)
"""


def recommend_fixedpoint_filters(
    df_true: pd.DataFrame,
    df_proxy: pd.DataFrame,
    delta_lo: float = 0.0,
    delta_hi: float = 0.10,
    delta_col: str = 'delta',
) -> pd.DataFrame:
    """
    Suggest reasonable continuous fixed-point filters from within the chosen
    small-delta window by reporting by-N quantiles of the key metrics.
    """
    dfp = attach_proxy_theory_metrics(df_true, df_proxy)
    sub = dfp[(dfp[delta_col] >= float(delta_lo)) & (dfp[delta_col] <= float(delta_hi))].copy()
    metrics = ['fp_rel', 'fp_sub_rel', 'fp_orth_rel', 'fp_residual_ratio', 'xi_over_phi_per_delta', 'lin_gap_rel']
    rows = []
    for N, sg in sub.groupby('N'):
        for m in metrics:
            vals = sg[m].astype(float)
            rows.append({
                'N': N,
                'metric': m,
                'q25': float(vals.quantile(0.25)),
                'q50': float(vals.quantile(0.50)),
                'q75': float(vals.quantile(0.75)),
                'q90': float(vals.quantile(0.90)),
            })
    return pd.DataFrame(rows)
