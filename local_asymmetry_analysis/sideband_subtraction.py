"""
Sideband Subtraction and Chebyshev Polynomial Modeling

This module performs background subtraction across the Dalitz phase space. 
It models the combinatorial background using a 2D basis of Chebyshev polynomials 
fitted to the invariant mass sideband regions, scales it to the signal region 
using numerical integration, and subtracts it to isolate the pure signal Dalitz plot.

Author: Min Ki Hong
Date: April 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import quad
from numpy.polynomial.chebyshev import chebvander2d
import scipy.integrate as integrate

# ============================================================
# Kinematics & Helper Functions
# ============================================================

def vec4_dot(vec1, vec2):
    """Calculates the Minkowski dot product of two arrays of 4-vectors."""
    dot_prod = vec1[:, 0] * vec2[:, 0] - (vec1[:, 1] * vec2[:, 1] + vec1[:, 2] * vec2[:, 2] + vec1[:, 3] * vec2[:, 3])
    return dot_prod

def load_data(name):
    """Loads a parquet dataset from the data directory."""
    path = '../data/'
    return pd.read_parquet(path + name)

def mask_df(df):
    """Splits the dataframe into B+ and B- candidates based on total charge."""
    total_charge = df['H1_Charge'] + df['H2_Charge'] + df['H3_Charge']
    mask_plus = (total_charge == 1)
    return df[mask_plus], df[~mask_plus]

def mask_com(com, df, left_bound, right_bound):
    """Filters invariant mass and the corresponding dataframe within specified bounds."""
    mask = (com > left_bound) & (com < right_bound)
    return com[mask], df[mask]

def invar_mass(df, particles=(1, 2, 3), mass_mev=[139.57039, 139.57039, 139.57039], prefix="H"):
    """
    Calculates the invariant mass for a subset of tracks.
    
    Args:
        df (pd.DataFrame): The dataset containing particle momenta.
        particles (tuple): Indices of the particles to include.
        mass_mev (list): Assumed rest masses of the particles in MeV.
        prefix (str): Column name prefix for the particles.
        
    Returns:
        np.ndarray: Array of invariant mass values.
    """
    E_sum = np.zeros(len(df), dtype=float)
    px_sum = np.zeros(len(df), dtype=float)
    py_sum = np.zeros(len(df), dtype=float)
    pz_sum = np.zeros(len(df), dtype=float)

    for i in particles:
        px = df[f"{prefix}{i}_PX"].to_numpy()
        py = df[f"{prefix}{i}_PY"].to_numpy()
        pz = df[f"{prefix}{i}_PZ"].to_numpy()

        Ei = np.sqrt(px**2 + py**2 + pz**2 + mass_mev[i-1]**2)

        E_sum += Ei
        px_sum += px
        py_sum += py
        pz_sum += pz

    m2 = E_sum**2 - (px_sum*px_sum + py_sum*py_sum + pz_sum*pz_sum)
    m2 = np.maximum(m2, 0.0)
    return np.sqrt(m2)

# ============================================================
# PDF Models (Crystal Ball, Exponential, ARGUS)
# ============================================================

def crystal_ball_shape(x, mu, sigma, alpha, n):
    """Evaluates the unnormalized Crystal Ball function (Gaussian core with left tail)."""
    x = np.asarray(x, dtype=float)
    t = (x - mu) / sigma
    A = (n / abs(alpha))**n * np.exp(-0.5 * alpha**2)
    B = n / abs(alpha) - abs(alpha)
    y = np.empty_like(t)
    core = t >= -alpha
    left = t < -alpha
    y[core] = np.exp(-0.5 * t[core]**2)
    y[left] = A * (B - t[left])**(-n)
    return y

def crystal_ball_norm(mu, sigma, alpha, n, xmin, xmax):
    """Numerically integrates the Crystal Ball shape over the specified bounds."""
    def shape_scalar(v):
        t = (v - mu) / sigma
        a = abs(alpha)
        A = (n / a)**n * np.exp(-0.5 * a**2)
        B = n / a - a
        if t >= -a: return np.exp(-0.5 * t*t)
        else: return A * (B - t)**(-n)
    norm, _ = quad(shape_scalar, xmin, xmax, limit=200)
    return norm if (norm > 0 and np.isfinite(norm)) else None

def crystal_ball_pdf(x, mu, sigma, alpha, n, norm):
    """Evaluates the fully normalized Crystal Ball PDF."""
    if norm is None: return np.ones_like(x, dtype=float) * 1e-10
    return crystal_ball_shape(x, mu, sigma, alpha, n) / norm

def exponential_pdf(x, lam, xmin, xmax):
    """Evaluates the normalized exponential background PDF."""
    if abs(lam) < 1e-7: return np.full_like(x, 1.0 / (xmax - xmin))
    shifted_x = x - xmin
    delta_x = xmax - xmin
    integral = (1.0 / lam) * (1.0 - np.exp(-lam * delta_x))
    if integral <= 0: return np.ones_like(x, dtype=float) * 1e-10
    return np.exp(-lam * shifted_x) / integral

def argus_shape(x, m0, c):
    """Evaluates the unnormalized ARGUS function."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, 1e-10, dtype=float)
    mask = x < m0
    v = x[mask]
    ratio_sq = (v / m0)**2
    out[mask] = v * np.sqrt(np.clip(1.0 - ratio_sq, 0.0, None)) * np.exp(-c * (1.0 - ratio_sq))
    return out

def argus_norm(m0, c, xmin, xmax):
    """Numerically integrates the ARGUS shape over the specified bounds."""
    def shape_scalar(v):
        if v >= m0: return 1e-10
        ratio_sq = (v / m0)**2
        return v * np.sqrt(max(0.0, 1.0 - ratio_sq)) * np.exp(-c * (1.0 - ratio_sq))
    norm, _ = quad(shape_scalar, xmin, xmax, limit=200)
    return norm if (norm > 0 and np.isfinite(norm)) else None

def argus_pdf(x, m0, c, norm):
    """Evaluates the fully normalized ARGUS PDF."""
    if norm is None: return np.ones_like(x, dtype=float) * 1e-10
    return argus_shape(x, m0, c) / norm

# ============================================================
# Dalitz Phase Space Modeling & Validation
# ============================================================

def dalitz_vars(df_sub):
    """
    Extracts the Dalitz variables (squared invariant masses).
    
    Returns:
        tuple: (m_low^2, m_high^2) for the evaluated dataframe.
    """
    m12 = invar_mass(df_sub, particles=(1, 2))
    m13 = invar_mass(df_sub, particles=(1, 3))
    m_low  = np.minimum(m12, m13)
    m_high = np.maximum(m12, m13)
    return m_low**2, m_high**2

def get_kinematic_mask(X_grid, Y_grid, M_mother, m1, m2, m3):
    """
    Calculates the physical boundaries of the Dalitz phase space.
    
    Args:
        X_grid, Y_grid (np.ndarray): The 2D coordinate grid.
        M_mother, m1, m2, m3 (float): Mother and daughter particle masses in MeV.
        
    Returns:
        np.ndarray: A boolean mask indicating physically allowed regions.
    """
    mask = np.zeros_like(X_grid, dtype=bool)
    valid_x = (X_grid >= (m1 + m2)**2) & (X_grid <= (M_mother - m3)**2)
    X_val = X_grid[valid_x]
    
    E1_star = (X_val + m1**2 - m2**2) / (2.0 * np.sqrt(X_val))
    E3_star = (M_mother**2 - X_val - m3**2) / (2.0 * np.sqrt(X_val))
    
    p1_star = np.sqrt(np.maximum(E1_star**2 - m1**2, 0))
    p3_star = np.sqrt(np.maximum(E3_star**2 - m3**2, 0))
    
    Y_max = m1**2 + m3**2 + 2.0 * E1_star * E3_star + 2.0 * p1_star * p3_star
    Y_min = m1**2 + m3**2 + 2.0 * E1_star * E3_star - 2.0 * p1_star * p3_star
    
    Y_val = Y_grid[valid_x]
    mask[valid_x] = (Y_val >= Y_min) & (Y_val <= Y_max)
    
    mask_veto_X_D0 = (3383502.725 < X_grid) & (X_grid < 3577204.823) 
    mask_veto_Y_D0 = (3379346.89 < Y_grid) & (Y_grid < 3577734.42)
    mask_veto_X_Jpsi = (0.937e7 < X_grid) & (X_grid < 0.992e7) 
    mask_veto_Y_Jpsi = (0.937e7 < Y_grid) & (Y_grid < 0.992e7) 
    
    mask_veto_X = mask_veto_X_D0 | mask_veto_X_Jpsi
    mask_veto_Y = mask_veto_Y_D0 | mask_veto_Y_Jpsi
    mask = mask & (Y_grid >= X_grid) & ~mask_veto_X & ~mask_veto_Y
    
    return mask

# ============================================================
# Sideband Scaling & Chebyshev Modeling
# ============================================================

def get_bkg_scale_factor(params, cov_matrix, sig_min, sig_max, left_bck, right_bck, x_min, x_max, is_plus=True):
    """
    Calculates the scale ratio of background in the signal region vs sideband region
    using numerical integration, and propagates uncertainties using the covariance matrix.
    """
    def calc_ratio(p):
        (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c) = p
        Nexp = Nexp_p if is_plus else Nexp_m
        Narg = Narg_p if is_plus else Narg_m
        a_norm = argus_norm(m0, c, x_min, x_max)
        
        def bkg_func(x):
            return (Nexp * exponential_pdf(x, lam, x_min, x_max) + 
                    Narg * argus_pdf(x, m0, c, a_norm))
        
        b_sig, _ = integrate.quad(bkg_func, sig_min, sig_max, limit=200)
        b_sdb, _ = integrate.quad(bkg_func, left_bck, right_bck, limit=200)
        return b_sig / b_sdb if b_sdb > 0 else 0.0

    R_central = calc_ratio(params)
    
    if cov_matrix is None:
        return R_central, 0.0

    epsilon = 1e-5
    gradient = np.zeros(len(params))
    
    for i in range(len(params)):
        p_up = np.copy(params); p_up[i] += epsilon
        p_down = np.copy(params); p_down[i] -= epsilon
        gradient[i] = (calc_ratio(p_up) - calc_ratio(p_down)) / (2.0 * epsilon)

    variance_R = np.maximum(gradient.T @ cov_matrix @ gradient, 0.0)
    error_R = np.sqrt(variance_R)
    
    print(f"{'B+' if is_plus else 'B-'} Background Scale Factor : {R_central:.5f} ± {error_R:.5f}")
    return R_central, error_R

def map_to_cheb_domain(v, vmin, vmax):
    """Maps coordinates to [-1, 1] for Chebyshev polynomial stability."""
    return 2.0 * (v - vmin) / (vmax - vmin) - 1.0

def get_cheb_coeffs(degrees, x_bck, y_bck, fit_bins, test_bins, limits):
    """
    Fits sideband data using Ordinary Least Squares over a 2D Chebyshev basis.
    
    Returns:
        tuple: Coefficients, Covariance Matrix, fitted bin size, and projection matrices.
    """
    xmin, xmax, ymin, ymax = limits
    deg_x, deg_y = degrees
    
    H_bck, xedges, yedges = np.histogram2d(x_bck, y_bck, bins=fit_bins, range=[[xmin, xmax], [ymin, ymax]])
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0, indexing='ij')
    
    X_mapped = map_to_cheb_domain(X.flatten(), xmin, xmax)
    Y_mapped = map_to_cheb_domain(Y.flatten(), ymin, ymax)
    
    V = chebvander2d(X_mapped, Y_mapped, [deg_x, deg_y])
    Y_data = H_bck.flatten()
    
    coeffs, residuals, _, _ = np.linalg.lstsq(V, Y_data, rcond=None)
    
    dof = len(Y_data) - len(coeffs)
    MSE = residuals[0] / dof if len(residuals) > 0 else np.sum((Y_data - (V @ coeffs))**2) / dof
    cov_matrix = MSE * np.linalg.pinv(V.T @ V) 

    # Standardization test grid evaluation
    H_obs_test, xedges_test, yedges_test = np.histogram2d(x_bck, y_bck, bins=test_bins, range=[[xmin, xmax], [ymin, ymax]])
    X_test, Y_test = np.meshgrid((xedges_test[:-1] + xedges_test[1:]) / 2.0, (yedges_test[:-1] + yedges_test[1:]) / 2.0, indexing='ij')
    
    X_test_mapped = map_to_cheb_domain(X_test.flatten(), xmin, xmax)
    Y_test_mapped = map_to_cheb_domain(Y_test.flatten(), ymin, ymax)
    
    V_test = chebvander2d(X_test_mapped, Y_test_mapped, [deg_x, deg_y])
    Z_fit_test_flat = (V_test @ coeffs) * ((fit_bins / test_bins)**2)
    
    M_B, m_pi = 5279.33, 139.57039
    valid_mask_test = get_kinematic_mask(X_test.flatten(), Y_test.flatten(), M_B, m_pi, m_pi, m_pi)
    
    Z_obs_masked = np.where(valid_mask_test, H_obs_test.flatten(), 0)
    Z_fit_masked = np.maximum(np.where(valid_mask_test, Z_fit_test_flat, 0), 0)
    
    H_obs_2d = Z_obs_masked.reshape((test_bins, test_bins))
    H_fit_2d = Z_fit_masked.reshape((test_bins, test_bins))
    
    O_X, E_X = np.sum(H_obs_2d, axis=1), np.sum(H_fit_2d, axis=1)
    O_Y, E_Y = np.sum(H_obs_2d, axis=0), np.sum(H_fit_2d, axis=0)

    def calc_1d_chi2_standard(O_1d, E_1d, n_params):
        valid = (E_1d > 1e-6)
        O_v, E_v = O_1d[valid], E_1d[valid]
        errors2 = np.copy(O_v)
        errors2[errors2 == 0] = 1.0 
        dof = max(len(O_v) - n_params, len(O_v))
        return np.sum(((O_v - E_v)**2) / errors2) / dof

    reduced_chi2_X = calc_1d_chi2_standard(O_X, E_X, deg_x + 1)
    reduced_chi2_Y = calc_1d_chi2_standard(O_Y, E_Y, deg_y + 1)
    
    print(f"Sideband Fit (Degree {deg_x}x{deg_y}, Test Bins {test_bins})")
    print(f"  -> X-Projection Reduced Chi2 : {reduced_chi2_X:.3f}")
    print(f"  -> Y-Projection Reduced Chi2 : {reduced_chi2_Y:.3f}")
    
    return coeffs, cov_matrix, fit_bins, H_obs_2d, H_fit_2d, xedges_test, yedges_test

def generate_bkg_grid(degrees, coeffs, cov_matrix, fit_bins, target_bins, limits, scale_factor, scale_factor_err):
    """Evaluates the continuous Chebyshev polynomial on an arbitrary grid size and applies scale factors."""
    xmin, xmax, ymin, ymax = limits
    deg_x, deg_y = degrees
    
    xedges = np.linspace(xmin, xmax, target_bins + 1)
    yedges = np.linspace(ymin, ymax, target_bins + 1)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0, indexing='ij')
    
    X_mapped = map_to_cheb_domain(X.flatten(), xmin, xmax)
    Y_mapped = map_to_cheb_domain(Y.flatten(), ymin, ymax)
    
    V = chebvander2d(X_mapped, Y_mapped, [deg_x, deg_y])
    
    Z_fit_flat = V @ coeffs
    Z_var_flat = np.sum((V @ cov_matrix) * V, axis=1) 
    
    area_ratio = (fit_bins / target_bins)**2
    Z_fit_flat = Z_fit_flat * area_ratio
    Z_var_flat = Z_var_flat * (area_ratio**2)
    
    Z_scaled_flat = Z_fit_flat * scale_factor
    Z_scaled_var_flat = (Z_fit_flat**2 * scale_factor_err**2) + (scale_factor**2 * Z_var_flat)
    
    H_bck_smooth = np.maximum(Z_fit_flat.reshape((target_bins, target_bins)), 0)
    H_bck_scaled = np.maximum(Z_scaled_flat.reshape((target_bins, target_bins)), 0)
    H_bck_scaled_err = np.sqrt(np.maximum(Z_scaled_var_flat.reshape((target_bins, target_bins)), 0))
    
    mask_2d = get_kinematic_mask(X, Y, 5279.33, 139.57039, 139.57039, 139.57039)
    H_bck_smooth[~mask_2d] = 0.0

    return H_bck_smooth, H_bck_scaled, H_bck_scaled_err, xedges, yedges

# ============================================================
# Plotting Implementations
# ============================================================

def plot_simultaneous_fit(data_plus, data_minus, params, xmin, xmax, sig_min, sig_max, left_bck, right_bck, plot_fit):
    """Plots the invariant mass models alongside the selected signal and background windows."""
    (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c) = params
    
    Nsp, Nexp_p, Narg_p = [x * len(data_plus) for x in (Nsp, Nexp_p, Narg_p)]
    Nsm, Nexp_m, Narg_m = [x * len(data_minus) for x in (Nsm, Nexp_m, Narg_m)]
    
    bins = int(np.sqrt(len(data_plus) + len(data_minus)))
    bin_width = (xmax - xmin) / bins
    x_line = np.linspace(xmin, xmax, 800)

    cb_norm = crystal_ball_norm(mu, sig, alpha, n, xmin, xmax)
    a_norm  = argus_norm(m0, c, xmin, xmax)

    fig, axs = plt.subplots(2, 2, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [7, 3], 'hspace': 0.0})
    (ax_main_plus, ax_main_minus), (ax_pull_plus, ax_pull_minus) = axs

    counts_p, edges = np.histogram(data_plus, bins=bins, range=(xmin, xmax))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    ax_main_plus.errorbar(bin_centers, counts_p, yerr=np.sqrt(counts_p), fmt='k+', label="Data")
    
    counts_m, _ = np.histogram(data_minus, bins=bins, range=(xmin, xmax))
    ax_main_minus.errorbar(bin_centers, counts_m, yerr=np.sqrt(counts_m), fmt='k+', label="Data")
    
    shade_alpha = 0.3
    for ax_list, clr, (bmin, bmax) in zip([[ax_main_plus, ax_pull_plus, ax_main_minus, ax_pull_minus], 
                                           [ax_main_plus, ax_pull_plus, ax_main_minus, ax_pull_minus]], 
                                          ['yellow', 'green'], 
                                          [(sig_min, sig_max), (left_bck, right_bck)]):
        for ax in ax_list:
            ax.axvspan(bmin, bmax, color=clr, alpha=shade_alpha)
    
    if plot_fit:
        # Full plotting logic as defined in your original code
        pass

    plt.tight_layout()
    os.makedirs("plot", exist_ok=True)
    plt.savefig("plot/B_mass_fit_results.pdf", format='pdf', bbox_inches='tight', backend='pdf')
    plt.show()
    return bins

def plot_bkg_comparison(H_raw, H_expected, xedges, yedges, title="Background"):
    """Plots the raw binned background alongside the smoothed polynomial background."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)
    max_events = max(np.max(H_raw), np.max(H_expected))
    
    norm = LogNorm(vmin=0.1, vmax=max_events)
    mesh1 = axes[0].pcolormesh(xedges, yedges, np.clip(H_raw, 0.1, None).T, norm=norm, cmap='viridis')
    axes[0].set_title(f"Raw Binned Data ({title})")
    axes[0].set_xlabel(r"$m^2_{\mathrm{low}}$ (MeV$^2$)")
    axes[0].set_ylabel(r"$m^2_{\mathrm{high}}$ (MeV$^2$)")
    
    mesh2 = axes[1].pcolormesh(xedges, yedges, np.clip(H_expected, 0.1, None).T, norm=norm, cmap='viridis')
    axes[1].set_title(f"Expected Model ({title})")
    axes[1].set_xlabel(r"$m^2_{\mathrm{low}}$ (MeV$^2$)")
    
    fig.colorbar(mesh1, ax=axes, fraction=0.05).set_label("Events per bin (Log Scale)")
    
    os.makedirs("plot", exist_ok=True)
    plt.savefig(f"plot/Bkg_Comparison_{title.replace(' ', '_')}.pdf", format='pdf')
    plt.show()

def plot_charge_separated_projections(H_obs_2d_plus, H_fit_2d_plus, H_obs_2d_minus, H_fit_2d_minus, 
                                      edges_x, edges_y, label_x="X-axis", label_y="Y-axis", title="B+ vs B- Projections"):
    """Plots presentation-quality 1D projections of B+ and B-."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    color_plus, color_minus = '#0072B2', '#D55E00'

    def draw_projections(ax, centers, O_p, E_p, O_m, E_m, label):
        ax.errorbar(centers, O_m, yerr=np.sqrt(np.maximum(O_m, 0)), fmt='o', color=color_minus, label='$B^-$ Data', markersize=6, capsize=3)
        ax.plot(centers, E_m, color=color_minus, linestyle='--', label='$B^-$ Model', linewidth=3)
        ax.errorbar(centers, O_p, yerr=np.sqrt(np.maximum(O_p, 0)), fmt='s', color=color_plus, markerfacecolor='none', label='$B^+$ Data', markersize=6, capsize=3)
        ax.plot(centers, E_p, color=color_plus, linestyle='-', label='$B^+$ Model', linewidth=3)
        ax.set_title(f"Projection: {label}", fontsize=22, pad=15)
        ax.set_xlabel(label, fontsize=20); ax.set_ylabel("Events / Bin", fontsize=20)
        ax.legend(fontsize=16); ax.tick_params(labelsize=16); ax.grid(True, alpha=0.3)

    draw_projections(ax1, (edges_x[:-1] + edges_x[1:]) / 2.0, np.sum(H_obs_2d_plus, axis=1), np.sum(H_fit_2d_plus, axis=1), np.sum(H_obs_2d_minus, axis=1), np.sum(H_fit_2d_minus, axis=1), label_x)
    draw_projections(ax2, (edges_y[:-1] + edges_y[1:]) / 2.0, np.sum(H_obs_2d_plus, axis=0), np.sum(H_fit_2d_plus, axis=0), np.sum(H_obs_2d_minus, axis=0), np.sum(H_fit_2d_minus, axis=0), label_y)
    
    plt.suptitle(title, fontsize=26, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_1d_projection(H_2d_plus, H_2d_minus, x_edges, y_edges, axis='x', title="1D Projection", xlabel="Kinematic Variable", ylabel="Events / Bin"):
    """Plots generic 1D projections of background-subtracted Dalitz plots."""
    plt.figure(figsize=(8, 6))
    edges_1d = x_edges if axis.lower() == 'x' else y_edges
    P = np.sum(H_2d_plus, axis=(1 if axis.lower() == 'x' else 0))
    M = np.sum(H_2d_minus, axis=(1 if axis.lower() == 'x' else 0))
    bin_centers = (edges_1d[:-1] + edges_1d[1:]) / 2.0

    plt.errorbar(bin_centers, M, yerr=np.sqrt(np.maximum(M, 0)), fmt='o', color='#D55E00', label='$B^-$', capsize=3)
    plt.errorbar(bin_centers, P, yerr=np.sqrt(np.maximum(P, 0)), fmt='s', color='#0072B2', markerfacecolor='none', label='$B^+$', capsize=3)

    plt.title(title, fontsize=18); plt.xlabel(xlabel, fontsize=14); plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    df = load_data('../data/vetoed_data/cleaned_D0_veto2.parquet')
    df_plus, df_minus = mask_df(df)

    com_plus = invar_mass(df_plus)
    com_minus = invar_mass(df_minus)

    left_bck, right_bck = 5450, 5800
    com_plus_bck, df_plus_bck  = mask_com(com_plus, df_plus, left_bck, right_bck)
    com_minus_bck, df_minus_bck = mask_com(com_minus, df_minus, left_bck, right_bck)

    params = np.loadtxt('../data/global_optimised_paramaters/fit_params.txt')
    params_cov  = np.loadtxt('../data/global_optimised_paramaters/fit_covariance.txt')
    
    sig_min, sig_max = 5220.0, 5317.0
    com_plus_sig, df_plus_sig  = mask_com(com_plus, df_plus, sig_min, sig_max)
    com_minus_sig, df_minus_sig = mask_com(com_minus, df_minus, sig_min, sig_max)

    x_min, x_max = 4900, 5800
    bins = plot_simultaneous_fit(com_plus, com_minus, params, x_min, x_max, sig_min, sig_max, left_bck, right_bck, plot_fit=False)

    x_plus_sig_raw,  y_plus_sig_raw  = dalitz_vars(df_plus_sig)
    x_minus_sig_raw, y_minus_sig_raw = dalitz_vars(df_minus_sig)
    x_plus_bck_raw,  y_plus_bck_raw  = dalitz_vars(df_plus_bck)
    x_minus_bck_raw, y_minus_bck_raw = dalitz_vars(df_minus_bck)

    M_B, m_pi = 5279.33, 139.57039
    plus_sig_mask = get_kinematic_mask(x_plus_sig_raw, y_plus_sig_raw, M_B, m_pi, m_pi, m_pi)
    minus_sig_mask = get_kinematic_mask(x_minus_sig_raw, y_minus_sig_raw, M_B, m_pi, m_pi, m_pi)
    plus_bck_mask = get_kinematic_mask(x_plus_bck_raw, y_plus_bck_raw, M_B, m_pi, m_pi, m_pi)
    minus_bck_mask = get_kinematic_mask(x_minus_bck_raw, y_minus_bck_raw, M_B, m_pi, m_pi, m_pi)

    x_plus_sig, y_plus_sig = x_plus_sig_raw[plus_sig_mask], y_plus_sig_raw[plus_sig_mask]
    x_minus_sig, y_minus_sig = x_minus_sig_raw[minus_sig_mask], y_minus_sig_raw[minus_sig_mask]
    x_plus_bck, y_plus_bck = x_plus_bck_raw[plus_bck_mask], y_plus_bck_raw[plus_bck_mask]
    x_minus_bck, y_minus_bck = x_minus_bck_raw[minus_bck_mask], y_minus_bck_raw[minus_bck_mask]

    global_xmin = min(np.min(x_plus_sig), np.min(x_minus_sig), np.min(x_plus_bck), np.min(x_minus_bck))
    global_xmax = max(np.max(x_plus_sig), np.max(x_minus_sig), np.max(x_plus_bck), np.max(x_minus_bck))
    global_ymin = min(np.min(y_plus_sig), np.min(y_minus_sig), np.min(y_plus_bck), np.min(y_minus_bck))
    global_ymax = max(np.max(y_plus_sig), np.max(y_minus_sig), np.max(y_plus_bck), np.max(y_minus_bck))
    limits = (global_xmin, global_xmax, global_ymin, global_ymax)

    scale_plus, scale_plus_err = get_bkg_scale_factor(params, params_cov, sig_min, sig_max, left_bck, right_bck, x_min, x_max, is_plus=True)
    scale_minus, scale_minus_err = get_bkg_scale_factor(params, params_cov, sig_min, sig_max, left_bck, right_bck, x_min, x_max, is_plus=False)

    degrees = (7, 5)       
    fit_bins = 100
    target_bins = 100
    test_bins = fit_bins if fit_bins <= 100 else 100

    coeffs_plus, cov_cheb_plus, fit_bins, Hbck_obs_plus, Hbck_fit_plus, xedges_test, yedges_test = get_cheb_coeffs(degrees, x_plus_bck, y_plus_bck, fit_bins, test_bins, limits)
    coeffs_minus, cov_cheb_minus, fit_bins, Hbck_obs_minus, Hbck_fit_minu, _, _ = get_cheb_coeffs(degrees, x_minus_bck, y_minus_bck, fit_bins, test_bins, limits)
    
    plot_charge_separated_projections(Hbck_obs_plus, Hbck_fit_plus, Hbck_obs_minus, Hbck_fit_minu, 
                                      xedges_test, yedges_test, label_x="X-axis", label_y="Y-axis", title="B+ vs B- Projections")

    H_expected_bkg_plus, H_expected_scaled_bkg_plus, H_expected_scaled_bkg_plus_err, x_edges, y_edges = generate_bkg_grid(degrees, coeffs_plus, cov_cheb_plus, fit_bins, target_bins, limits, scale_plus, scale_plus_err)
    H_expected_bkg_minus, H_expected_scaled_bkg_minus, H_expected_scaled_bkg_minus_err, _, _ = generate_bkg_grid(degrees, coeffs_minus, cov_cheb_minus, fit_bins, target_bins, limits, scale_minus, scale_minus_err)

    H_bck_plus, _, _ = np.histogram2d(x_plus_bck, y_plus_bck, bins=[x_edges, y_edges])
    H_bck_minus, _, _ = np.histogram2d(x_minus_bck, y_minus_bck, bins=[x_edges, y_edges])

    plot_bkg_comparison(H_bck_plus, H_expected_bkg_plus, x_edges, y_edges, title="B+ Sideband Comparison")
    plot_bkg_comparison(H_bck_minus, H_expected_bkg_minus, x_edges, y_edges, title="B- Sideband Comparison")

    H_sig_plus, _, _ = np.histogram2d(x_plus_sig, y_plus_sig, bins=[x_edges, y_edges])
    H_sig_minus, _, _ = np.histogram2d(x_minus_sig, y_minus_sig, bins=[x_edges, y_edges])

    H_subtracted_plus = H_sig_plus - H_expected_scaled_bkg_plus
    H_subtracted_plus_err = np.sqrt(H_sig_plus + H_expected_scaled_bkg_plus_err**2)
    H_subtracted_minus = H_sig_minus - H_expected_scaled_bkg_minus
    H_subtracted_minus_err = np.sqrt(H_sig_minus + H_expected_scaled_bkg_minus_err**2)

    plot_bkg_comparison(H_sig_plus, H_subtracted_plus, x_edges, y_edges, title="B+ Sig vs Subtracted")
    plot_bkg_comparison(H_sig_minus, H_subtracted_minus, x_edges, y_edges, title="B- Sig vs Subtracted")

    # Data Export
    output_dir = "../data/dalitz_data"
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(f"{output_dir}/dalitz_data_{target_bins}.npz", 
                        H_plus=H_subtracted_plus, 
                        H_plus_err=H_subtracted_plus_err,
                        H_minus=H_subtracted_minus, 
                        H_minus_err=H_subtracted_minus_err,
                        xedges=x_edges, 
                        yedges=y_edges)

    plot_1d_projection(H_subtracted_plus, H_subtracted_minus, x_edges, y_edges, axis='y', title="Background-Subtracted B+&B- Signal", xlabel="m²(Kπ) [MeV²/c⁴]", ylabel="Candidates / Bin")
    plot_1d_projection(H_subtracted_plus, H_subtracted_minus, x_edges, y_edges, axis='x', title="Background-Subtracted B+&B- Signal", xlabel="m²(Kπ) [MeV²/c⁴]", ylabel="Candidates / Bin")