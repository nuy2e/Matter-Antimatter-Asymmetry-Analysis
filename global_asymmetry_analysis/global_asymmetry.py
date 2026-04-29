"""
Global CP Asymmetry Fitting and Analysis

This module performs a simultaneous extended unbinned maximum likelihood fit 
to the invariant mass distributions of B+ and B- candidates. It uses a Crystal 
Ball function to model the signal, an Exponential function for combinatorial 
background, and an ARGUS function for partially reconstructed background.

Authors: Min Ki Hong and A. Knight (Collaboration)
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from iminuit import Minuit

# ============================================================
# Kinematics & Mass Calculations
# ============================================================

def vec4_dot(vec1, vec2):
    """
    Calculates the Minkowski dot product of two arrays of 4-vectors.

    Args:
        vec1 (np.ndarray): First array of 4-vectors, shape (N, 4).
        vec2 (np.ndarray): Second array of 4-vectors, shape (N, 4).

    Returns:
        np.ndarray: Evaluated dot products, shape (N,).
    """
    return vec1[:, 0] * vec2[:, 0] - (vec1[:, 1] * vec2[:, 1] + vec1[:, 2] * vec2[:, 2] + vec1[:, 3] * vec2[:, 3])

def invar_mass(df):
    """
    Calculates the invariant mass of the B meson candidates assuming pion daughters.

    Args:
        df (pd.DataFrame): DataFrame containing kinematic data (PX, PY, PZ) for three daughters.

    Returns:
        np.ndarray: Array of invariant mass values for each candidate.
    """
    m_pi = 139.57039  # MeV
    E1 = np.sqrt(df['H1_PX']**2 + df['H1_PY']**2 + df['H1_PZ']**2 + m_pi**2)
    E2 = np.sqrt(df['H2_PX']**2 + df['H2_PY']**2 + df['H2_PZ']**2 + m_pi**2)
    E3 = np.sqrt(df['H3_PX']**2 + df['H3_PY']**2 + df['H3_PZ']**2 + m_pi**2)

    p1_vec = df[['H1_PX', 'H1_PY', 'H1_PZ']].to_numpy()
    p2_vec = df[['H2_PX', 'H2_PY', 'H2_PZ']].to_numpy()
    p3_vec = df[['H3_PX', 'H3_PY', 'H3_PZ']].to_numpy()

    p1_4vec = np.column_stack((E1, p1_vec))
    p2_4vec = np.column_stack((E2, p2_vec))
    p3_4vec = np.column_stack((E3, p3_vec))

    s = 3*(m_pi)**2 + 2*(vec4_dot(p1_4vec, p2_4vec) + vec4_dot(p1_4vec, p3_4vec) + vec4_dot(p2_4vec, p3_4vec))
    return np.sqrt(s)

# ============================================================
# Data Loading & Preparation
# ============================================================

def load_data(path_name):
    """
    Loads dataset from the pre-processed parquet files.

    Args:
        path_name (str): The specific parquet file name to load.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    path = '../data/vetoed_data/'
    return pd.read_parquet(path + path_name)

def mask_df(df):
    """
    Splits the dataframe into B+ and B- candidates based on total charge.

    Args:
        df (pd.DataFrame): The full dataset.

    Returns:
        tuple: (df_plus, df_minus) where df_plus contains B+ events and df_minus contains B- events.
    """
    total_charge = df['H1_Charge'] + df['H2_Charge'] + df['H3_Charge']
    mask_plus = (total_charge == 1)
    return df[mask_plus], df[~mask_plus]

def mask_com(com, left_bound, right_bound):
    """
    Filters an invariant mass array within specified bounds.

    Args:
        com (np.ndarray): Array of invariant mass values.
        left_bound (float): Lower mass limit.
        right_bound (float): Upper mass limit.

    Returns:
        np.ndarray: Filtered array of mass values.
    """
    mask = (com > left_bound) & (com < right_bound)
    return com[mask]

# ============================================================
# Probability Density Functions (PDFs)
# ============================================================

def crystal_ball_shape(x, mu, sigma, alpha, n):
    """
    Evaluates the unnormalized Crystal Ball function (Gaussian core with left-side power-law tail).

    Args:
        x (np.ndarray): Independent variable (invariant mass).
        mu (float): Mean of the Gaussian core.
        sigma (float): Standard deviation of the Gaussian core.
        alpha (float): Point where the Gaussian transitions to the power-law tail.
        n (float): Exponent of the power-law tail.

    Returns:
        np.ndarray: Evaluated Crystal Ball shape values.
    """
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
    """Integrates the Crystal Ball shape over the specified bounds for normalization."""
    def shape_scalar(v):
        t = (v - mu) / sigma
        a = abs(alpha)
        A = (n / a)**n * np.exp(-0.5 * a**2)
        B = n / a - a
        
        if t >= -a:
            return np.exp(-0.5 * t*t)
        else:
            return A * (B - t)**(-n)
    
    norm, _ = quad(shape_scalar, xmin, xmax, limit=200)
    if norm <= 0 or not np.isfinite(norm):
        return None
    return norm

def crystal_ball_pdf(x, mu, sigma, alpha, n, norm):
    """Evaluates the fully normalized Crystal Ball PDF."""
    if norm is None:
        return np.ones_like(x, dtype=float) * 1e-10
    return crystal_ball_shape(x, mu, sigma, alpha, n) / norm

def exponential_pdf(x, lam, xmin, xmax):
    """
    Evaluates the normalized exponential background PDF.

    Args:
        x (np.ndarray): Independent variable (invariant mass).
        lam (float): Decay constant.
        xmin (float): Lower bound of the fit region.
        xmax (float): Upper bound of the fit region.

    Returns:
        np.ndarray: Evaluated PDF values.
    """
    if abs(lam) < 1e-7:
        return np.full_like(x, 1.0 / (xmax - xmin))
    
    shifted_x = x - xmin
    delta_x = xmax - xmin
    
    integral = (1.0 / lam) * (1.0 - np.exp(-lam * delta_x))
    
    if integral <= 0:
        return np.ones_like(x, dtype=float) * 1e-10
        
    return np.exp(-lam * shifted_x) / integral

def argus_shape(x, m0, c):
    """
    Evaluates the unnormalized ARGUS function for partially reconstructed backgrounds.

    Args:
        x (np.ndarray): Independent variable (invariant mass).
        m0 (float): Kinematic threshold/cutoff parameter.
        c (float): Curvature parameter defining the shape of the tail.

    Returns:
        np.ndarray: Evaluated ARGUS shape values.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, 1e-10, dtype=float)
    mask = x < m0
    v = x[mask]
    ratio_sq = (v / m0)**2
    out[mask] = v * np.sqrt(np.clip(1.0 - ratio_sq, 0.0, None)) * np.exp(-c * (1.0 - ratio_sq))
    return out

def argus_norm(m0, c, xmin, xmax):
    """Integrates the ARGUS shape over the specified bounds for normalization."""
    def shape_scalar(v):
        if v >= m0:
            return 1e-10
        ratio_sq = (v / m0)**2
        return v * np.sqrt(max(0.0, 1.0 - ratio_sq)) * np.exp(-c * (1.0 - ratio_sq))

    norm, _ = quad(shape_scalar, xmin, xmax, limit=200)
    if norm <= 0 or not np.isfinite(norm):
        return None
    return norm

def argus_pdf(x, m0, c, norm):
    """Evaluates the fully normalized ARGUS PDF."""
    if norm is None:
        return np.ones_like(x, dtype=float) * 1e-10
    return argus_shape(x, m0, c) / norm

# ============================================================
# Likelihood Function & Minimization
# ============================================================

def negative_log_likelihood(params, data_plus, data_minus, xmin, xmax):
    """
    Computes the negative log-likelihood for the simultaneous B+ and B- fit.

    Args:
        params (list): Contains [Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c].
        data_plus (np.ndarray): Mass array for B+ candidates.
        data_minus (np.ndarray): Mass array for B- candidates.
        xmin (float): Lower fit bound.
        xmax (float): Upper fit bound.

    Returns:
        float: The calculated negative log-likelihood scalar.
    """
    (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c) = params

    cb_norm = crystal_ball_norm(mu, sig, alpha, n, xmin, xmax)
    a_norm  = argus_norm(m0, c, xmin, xmax)

    pdf_sig_p = crystal_ball_pdf(data_plus,  mu, sig, alpha, n, cb_norm)
    pdf_sig_m = crystal_ball_pdf(data_minus, mu, sig, alpha, n, cb_norm)

    pdf_exp_p = exponential_pdf(data_plus, lam, xmin, xmax)
    pdf_exp_m = exponential_pdf(data_minus, lam, xmin, xmax)

    pdf_arg_p = argus_pdf(data_plus,  m0, c, a_norm)
    pdf_arg_m = argus_pdf(data_minus, m0, c, a_norm)

    total_p = Nsp * pdf_sig_p + Nexp_p * pdf_exp_p + Narg_p * pdf_arg_p
    total_m = Nsm * pdf_sig_m + Nexp_m * pdf_exp_m + Narg_m * pdf_arg_m

    total_p = np.clip(total_p, 1e-10, None)
    total_m = np.clip(total_m, 1e-10, None)

    logL = -(Nsp + Nsm + Nexp_p + Nexp_m + Narg_p + Narg_m) + np.sum(np.log(total_p)) + np.sum(np.log(total_m))
    
    return -logL

def perform_unbinned_fit(data_plus, data_minus, xmin, xmax):
    """
    Executes the iminuit optimization routine for the extended maximum likelihood.

    Args:
        data_plus (np.ndarray): Mass array for B+ candidates.
        data_minus (np.ndarray): Mass array for B- candidates.
        xmin (float): Lower fit bound.
        xmax (float): Upper fit bound.

    Returns:
        tuple: (params, errors, cov_matrix) containing the optimized parameters, 
               their uncertainties, and the covariance matrix.
    """
    print(f"Starting minimization on {len(data_plus) + len(data_minus)} events using iminuit...")

    def nll(Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c):
        params = (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c)
        return negative_log_likelihood(params, data_plus, data_minus, xmin, xmax)

    m = Minuit(
        nll,
        Nsp=len(data_plus) * 0.35,
        Nsm=len(data_minus) * 0.35,
        Nexp_p=len(data_plus) * 0.45,
        Nexp_m=len(data_minus) * 0.45,
        Narg_p=len(data_plus) * 0.20,
        Narg_m=len(data_minus) * 0.20,
        mu=5279.0,
        sig=30.0,
        alpha=1.5,
        n=10.0,
        lam=0.001,
        m0=5140.0,
        c=5.0
    )

    m.errordef = 0.5 

    m.limits["Nsp"] = (0, None)
    m.limits["Nsm"] = (0, None)
    m.limits["Nexp_p"] = (0, None)
    m.limits["Nexp_m"] = (0, None)
    m.limits["Narg_p"] = (0, None)
    m.limits["Narg_m"] = (0, None)
    m.limits["mu"] = (5250, 5310)
    m.limits["sig"] = (1, 100)
    m.limits["alpha"] = (0.5, 5.0) 
    m.limits["n"] = (1.01, 20.0)    
    m.limits["lam"] = (0.0, 5.0)
    m.fixed["m0"] = True
    m.limits["c"] = (0.1, 100.0)

    m.migrad()
    m.hesse()

    if m.valid:
        print("\nFit Succeeded!")
        for key in m.parameters:
            print(f"{key:10} = {m.values[key]:.3f} ± {m.errors[key]:.3f}")
        cov_matrix = np.array(m.covariance)
    else:
        print("\nFit Failed!")
        print(m.fmin)

    params = [m.values[k] for k in m.parameters]
    errors = [m.errors[k] for k in m.parameters]
    
    return params, errors, cov_matrix

# ============================================================
# Plotting Functions
# ============================================================

def plot_simultaneous_fit(data_plus, data_minus, params, xmin, xmax, sig_min, sig_max):
    """Generates standard dual-panel diagnostic plots with pull distributions."""
    (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c) = params
    
    bin_width = 5.0 
    bins = int((xmax - xmin) / bin_width)
    x_line = np.linspace(xmin, xmax, 800)

    cb_norm = crystal_ball_norm(mu, sig, alpha, n, xmin, xmax)
    a_norm  = argus_norm(m0, c, xmin, xmax)

    sig_line = crystal_ball_pdf(x_line, mu, sig, alpha, n, cb_norm)
    exp_line = exponential_pdf(x_line, lam, xmin, xmax)
    arg_line = argus_pdf(x_line, m0, c, a_norm)

    y_sig_plus = Nsp * sig_line * bin_width
    y_exp_plus = Nexp_p * exp_line * bin_width
    y_arg_plus = Narg_p * arg_line * bin_width
    y_tot_plus = y_sig_plus + y_exp_plus + y_arg_plus

    y_sig_minus = Nsm * sig_line * bin_width
    y_exp_minus = Nexp_m * exp_line * bin_width
    y_arg_minus = Narg_m * arg_line * bin_width
    y_tot_minus = y_sig_minus + y_exp_minus + y_arg_minus

    report_fonts = {
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    }

    with plt.rc_context(report_fonts):
        fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True, 
                                gridspec_kw={'height_ratios': [7, 3], 'hspace': 0.05})
        
        (ax_main_plus, ax_main_minus), (ax_pull_plus, ax_pull_minus) = axs

        counts_p, edges = np.histogram(data_plus, bins=bins, range=(xmin, xmax))
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        
        ax_main_plus.errorbar(bin_centers, counts_p, yerr=np.sqrt(counts_p), fmt='k+', label="Data")
        ax_main_plus.plot(x_line, y_tot_plus, linewidth=2.5, color='blue', label="Total Fit")
        ax_main_plus.plot(x_line, y_sig_plus, linestyle="--", linewidth=2, color='red', label="Signal")
        ax_main_plus.plot(x_line, y_exp_plus, linestyle=":", linewidth=2, color='green', label="Bkg (Exp)")
        ax_main_plus.plot(x_line, y_arg_plus, linestyle="-.", linewidth=2, color='purple', label="Bkg (ARGUS)")
        ax_main_plus.set_title(r"$B^+$ Candidates")
        ax_main_plus.set_ylabel(f"Events / ({bin_width:.1f} MeV/$c^2$)")
        ax_main_plus.set_ylim(bottom=0)
        ax_main_plus.legend(loc='upper right')
        
        counts_m, _ = np.histogram(data_minus, bins=bins, range=(xmin, xmax))
        
        ax_main_minus.errorbar(bin_centers, counts_m, yerr=np.sqrt(counts_m), fmt='k+', label="Data")
        ax_main_minus.plot(x_line, y_tot_minus, linewidth=2.5, color='blue', label="Total Fit")
        ax_main_minus.plot(x_line, y_sig_minus, linestyle="--", linewidth=2, color='red', label="Signal")
        ax_main_minus.plot(x_line, y_exp_minus, linestyle=":", linewidth=2, color='green', label="Bkg (Exp)")
        ax_main_minus.plot(x_line, y_arg_minus, linestyle="-.", linewidth=2, color='purple', label="Bkg (ARGUS)")
        ax_main_minus.set_title(r"$B^-$ Candidates")
        ax_main_minus.set_ylim(bottom=0)
        ax_main_minus.legend(loc='upper right')
        
        exp_plus = (Nsp * crystal_ball_pdf(bin_centers, mu, sig, alpha, n, cb_norm) + 
                    Nexp_p * exponential_pdf(bin_centers, lam, xmin, xmax) + 
                    Narg_p * argus_pdf(bin_centers, m0, c, a_norm)) * bin_width
        err_p = np.sqrt(counts_p)
        err_p[err_p == 0] = 1.0
        pull_plus = (counts_p - exp_plus) / err_p
        
        ax_pull_plus.bar(bin_centers, pull_plus, width=bin_width, color='gray', alpha=0.7)
        ax_pull_plus.axhline(0, color='black', linestyle='-', linewidth=1.5)
        ax_pull_plus.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_pull_plus.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_pull_plus.set_ylabel("Pull")
        ax_pull_plus.set_xlabel(r"Invariant Mass $[MeV/c^2]$")
        ax_pull_plus.set_ylim(-5, 5)

        exp_minus = (Nsm * crystal_ball_pdf(bin_centers, mu, sig, alpha, n, cb_norm) + 
                     Nexp_m * exponential_pdf(bin_centers, lam, xmin, xmax) + 
                     Narg_m * argus_pdf(bin_centers, m0, c, a_norm)) * bin_width
        err_m = np.sqrt(counts_m)
        err_m[err_m == 0] = 1.0 
        pull_minus = (counts_m - exp_minus) / err_m
        
        ax_pull_minus.bar(bin_centers, pull_minus, width=bin_width, color='gray', alpha=0.7)
        ax_pull_minus.axhline(0, color='black', linestyle='-', linewidth=1.5)
        ax_pull_minus.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_pull_minus.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_pull_minus.set_xlabel(r"Invariant Mass $[MeV/c^2]$")
        ax_pull_minus.set_ylim(-5, 5)
        
        plt.tight_layout()
        plt.savefig("/plots/B_mass_fit_results_pull.pdf", format='pdf', bbox_inches='tight', backend='pdf')
        plt.savefig("/plots/B_mass_fit_results_pull.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

    return bins

def plot_simultaneous_fit2(data_plus, data_minus, params, xmin, xmax, sig_min, sig_max):
    """Generates a high-quality presentation plot without pull distributions."""
    (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c) = params
    
    bin_width = 5.0 
    bins = int((xmax - xmin) / bin_width)
    x_line = np.linspace(xmin, xmax, 800)

    cb_norm = crystal_ball_norm(mu, sig, alpha, n, xmin, xmax)
    a_norm  = argus_norm(m0, c, xmin, xmax)

    sig_line = crystal_ball_pdf(x_line, mu, sig, alpha, n, cb_norm)
    exp_line = exponential_pdf(x_line, lam, xmin, xmax)
    arg_line = argus_pdf(x_line, m0, c, a_norm)

    y_sig_plus = Nsp * sig_line * bin_width
    y_exp_plus = Nexp_p * exp_line * bin_width
    y_arg_plus = Narg_p * arg_line * bin_width
    y_tot_plus = y_sig_plus + y_exp_plus + y_arg_plus

    y_sig_minus = Nsm * sig_line * bin_width
    y_exp_minus = Nexp_m * exp_line * bin_width
    y_arg_minus = Narg_m * arg_line * bin_width
    y_tot_minus = y_sig_minus + y_exp_minus + y_arg_minus

    report_fonts = {
        'font.size': 12,          
        'axes.labelsize': 32,     
        'xtick.labelsize': 26,    
        'ytick.labelsize': 26,    
        'legend.fontsize': 24,    
        'figure.titlesize': 37      
    }

    with plt.rc_context(report_fonts):
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        ax_minus, ax_plus = axs
        
        counts_m, edges = np.histogram(data_minus, bins=bins, range=(xmin, xmax))
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        
        # Convert to GeV
        bin_centers = bin_centers / 10**3
        x_line = x_line / 10**3
        
        ax_minus.errorbar(bin_centers, counts_m, yerr=np.sqrt(counts_m), fmt='k+', label="Data", markersize=12)
        ax_minus.plot(x_line, y_tot_minus, linewidth=2.5, color='blue', label="Total Fit")
        ax_minus.plot(x_line, y_sig_minus, linestyle="--", linewidth=2.5, color='red', label="Signal")
        ax_minus.plot(x_line, y_exp_minus, linestyle=":", linewidth=2.5, color='green', label="Bkg (Exp)")
        ax_minus.plot(x_line, y_arg_minus, linestyle="-.", linewidth=2.5, color='purple', label="Bkg (ARGUS)")
        
        ax_minus.set_xlabel(r"$m(\pi^-\pi^-\pi^+)$ [GeV/$c^2$]")
        ax_minus.set_ylabel(f"Events / ({bin_width/10**3:.3f} GeV/$c^2$)")
        ax_minus.set_ylim(bottom=0)

        counts_p, _ = np.histogram(data_plus, bins=bins, range=(xmin, xmax))
        
        ax_plus.errorbar(bin_centers, counts_p, yerr=np.sqrt(counts_p), fmt='k+', label="Data", markersize=8)
        ax_plus.plot(x_line, y_tot_plus, linewidth=2.5, color='blue', label="Total Fit")
        ax_plus.plot(x_line, y_sig_plus, linestyle="--", linewidth=2.5, color='red', label="Signal")
        ax_plus.plot(x_line, y_exp_plus, linestyle=":", linewidth=2.5, color='green', label="Bkg (Exp)")
        ax_plus.plot(x_line, y_arg_plus, linestyle="-.", linewidth=2.5, color='purple', label="Bkg (ARGUS)")
        
        ax_plus.set_xlabel(r"$m(\pi^+\pi^+\pi^-)$ [GeV/$c^2$]")

        plt.tight_layout()
        
        handles, labels = ax_minus.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=5, frameon=False)
        fig.suptitle(r"Simultaneous Mass Fit of $B^\pm \to \pi^\pm\pi^\pm\pi^\mp$ Candidates", y=1.15)
        
        plt.savefig("plots/B_mass_fit_results2.pdf", format='pdf', bbox_inches='tight', backend='pdf')
        plt.savefig("plots/B_mass_fit_results2.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

    return bins

# ============================================================
# Diagnostic & Output Tools
# ============================================================

def calculate_global_chi2(params, data_plus, data_minus, xmin, xmax, sig_min, sig_max, bins):
    """
    Evaluates the goodness-of-fit using Pearson's chi-squared test.

    Args:
        params (list): Fitted parameters.
        data_plus (np.ndarray): Mass array for B+ candidates.
        data_minus (np.ndarray): Mass array for B- candidates.
        xmin (float): Lower fit bound.
        xmax (float): Upper fit bound.
        sig_min (float): Lower signal bound.
        sig_max (float): Upper signal bound.
        bins (int): Number of histogram bins.

    Returns:
        tuple: (total_chi2, ndf_total, red_chi2) metrics.
    """
    (Nsp, Nsm, Nexp_p, Nexp_m, Narg_p, Narg_m, mu, sig, alpha, n, lam, m0, c) = params

    counts_plus, bin_edges = np.histogram(data_plus, bins=bins, range=(xmin, xmax))
    counts_minus, _ = np.histogram(data_minus, bins=bins, range=(xmin, xmax))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]
    
    cb_norm = crystal_ball_norm(mu, sig, alpha, n, xmin, xmax)
    a_norm  = argus_norm(m0, c, xmin, xmax)

    sig_pdf = crystal_ball_pdf(bin_centers, mu, sig, alpha, n, cb_norm)
    exp_pdf = exponential_pdf(bin_centers, lam, xmin, xmax)
    arg_pdf = argus_pdf(bin_centers, m0, c, a_norm)

    exp_plus = (Nsp * sig_pdf + Nexp_p * exp_pdf + Narg_p * arg_pdf) * bin_width
    exp_minus = (Nsm * sig_pdf + Nexp_m * exp_pdf + Narg_m * arg_pdf) * bin_width

    valid_plus = exp_plus > 0
    valid_minus = exp_minus > 0

    chi2_plus = np.sum((counts_plus[valid_plus] - exp_plus[valid_plus])**2 / exp_plus[valid_plus])
    chi2_minus = np.sum((counts_minus[valid_minus] - exp_minus[valid_minus])**2 / exp_minus[valid_minus])
    total_chi2 = chi2_plus + chi2_minus
    
    plus_valid_bins = np.sum(valid_plus)
    minus_valid_bins = np.sum(valid_minus)
    total_valid_bins = plus_valid_bins + minus_valid_bins
    
    ndf_plus = plus_valid_bins - len(params)
    ndf_minus = minus_valid_bins - len(params)
    ndf_total = total_valid_bins - len(params)
    
    red_chi2_plus = chi2_plus / ndf_plus if ndf_plus > 0 else 0.0
    red_chi2_minus = chi2_minus / ndf_minus if ndf_minus > 0 else 0.0
    red_chi2 = total_chi2 / ndf_total if ndf_total > 0 else 0.0

    mask_sig_region = (bin_centers >= sig_min) & (bin_centers <= sig_max)
    valid_sig_plus = mask_sig_region & valid_plus
    valid_sig_minus = mask_sig_region & valid_minus
    
    chi2_plus_sig = np.sum((counts_plus[valid_sig_plus] - exp_plus[valid_sig_plus])**2 / exp_plus[valid_sig_plus])
    chi2_minus_sig = np.sum((counts_minus[valid_sig_minus] - exp_minus[valid_sig_minus])**2 / exp_minus[valid_sig_minus])
    total_sig_chi2 = chi2_plus_sig + chi2_minus_sig
    
    bins_in_sig_plus = np.sum(valid_sig_plus)
    bins_in_sig_minus = np.sum(valid_sig_minus)
    bins_in_sig_total =  bins_in_sig_plus + bins_in_sig_minus
    
    red_chi2_sig_plus = chi2_plus_sig / bins_in_sig_plus if bins_in_sig_plus > 0 else 0.0
    red_chi2_sig_minus = chi2_minus_sig / bins_in_sig_minus if bins_in_sig_minus > 0 else 0.0
    red_chi2_sig_total = total_sig_chi2 / bins_in_sig_total if bins_in_sig_total > 0 else 0.0

    print("\n--- Global Goodness of Fit ---")
    print(f"Plus Chi2: {chi2_plus:.2f} / {ndf_plus} NDF = {red_chi2_plus:.3f}")
    print(f"Minus Chi2: {chi2_minus:.2f} / {ndf_minus} NDF = {red_chi2_minus:.3f}")
    print(f"Global Chi2: {total_chi2:.2f} / {ndf_total} NDF = {red_chi2:.3f}")
    print("\n--- Signal Goodness of Fit ---")
    print(f"Signal Region ({sig_min}-{sig_max}) Plus Chi2: {chi2_plus_sig:.2f} / {bins_in_sig_plus} Bins = {red_chi2_sig_plus:.3f}")
    print(f"Signal Region ({sig_min}-{sig_max}) Minus Chi2: {chi2_minus_sig:.2f} / {bins_in_sig_minus} Bins = {red_chi2_sig_minus:.3f}")
    print(f"Signal Region ({sig_min}-{sig_max}) Chi2: {total_sig_chi2:.2f} / {bins_in_sig_total} Bins = {red_chi2_sig_total:.3f}")

    return total_chi2, ndf_total, red_chi2

def calculate_asymmetry(params, errors, cov_minus_plus=0.0):
    """
    Computes the raw CP asymmetry based on the fitted signal yields.

    Args:
        params (list): Fitted parameters [Nsp, Nsm, ...].
        errors (list): Uncertainties for the parameters.
        cov_minus_plus (float, optional): Covariance between Nsm and Nsp. Defaults to 0.0.

    Returns:
        tuple: (A_cp, err_A_cp) representing the asymmetry and its statistical error.
    """
    N_plus = params[0]
    N_minus = params[1]
    err_N_plus = errors[0]
    err_N_minus = errors[1]
    
    total_N = N_minus + N_plus
    
    if total_N <= 0:
        print("Error: Total yield is zero or negative. Cannot calculate A_CP.")
        return 0.0, 0.0

    A_cp = (N_minus - N_plus) / total_N
    dA_dN_minus = (2.0 * N_plus) / (total_N**2)
    dA_dN_plus  = (-2.0 * N_minus) / (total_N**2)

    variance_A = (dA_dN_minus**2) * (err_N_minus**2) + \
                 (dA_dN_plus**2) * (err_N_plus**2) + \
                 (2.0 * dA_dN_minus * dA_dN_plus * cov_minus_plus)

    variance_A = max(variance_A, 0.0)
    err_A_cp = np.sqrt(variance_A)

    print("\n--- CP Asymmetry ---")
    print(f"(A_CP = {A_cp:.5f} ± {err_A_cp:.5f}), statistical significance: {A_cp/err_A_cp}")

    return A_cp, err_A_cp

def save_normalized_params(params, filename='fit_params.txt'):
    """
    Normalizes the yield parameters for the plus and minus datasets 
    separately, and saves the full parameter list to a text file.

    Args:
        params (list): Full list of raw parameters from the fit.
        filename (str): Target save location.
    """
    params = np.array(params)
    
    total_N_p = params[0] + params[2] + params[4] 
    total_N_m = params[1] + params[3] + params[5] 
    
    norm_params = np.copy(params)
    
    norm_params[0] = params[0] / total_N_p 
    norm_params[2] = params[2] / total_N_p  
    norm_params[4] = params[4] / total_N_p  
    
    norm_params[1] = params[1] / total_N_m  
    norm_params[3] = params[3] / total_N_m  
    norm_params[5] = params[5] / total_N_m  
    
    np.savetxt(filename, norm_params)
    
    print(f"Parameters saved to {filename}")
    print(f"Normalized by -> Total N+: {total_N_p:.2f} | Total N-: {total_N_m:.2f}")
    
def calculate_signal_fraction(mu, sigma, alpha, n, xmin, xmax, sig_min, sig_max):
    """
    Calculates the fraction of the Crystal Ball signal distribution 
    that falls within [sig_min, sig_max], relative to the full fit range.

    Args:
        mu, sigma, alpha, n (float): Shape parameters of the fitted signal.
        xmin, xmax (float): Full integration boundaries.
        sig_min, sig_max (float): Specific signal window boundaries.

    Returns:
        float: Calculated fractional area.
    """
    total_area = crystal_ball_norm(mu, sigma, alpha, n, xmin, xmax)
    region_area = crystal_ball_norm(mu, sigma, alpha, n, sig_min, sig_max)
    
    if total_area is None or total_area <= 0:
        print("Error: Invalid total area for the Crystal Ball distribution.")
        return 0.0
        
    if region_area is None or region_area <= 0:
        print(f"Error: Invalid region area for the Crystal Ball distribution in [{sig_min}, {sig_max}].")
        return 0.0
        
    fraction = region_area / total_area
    
    print("\n--- Signal Fraction ---")
    print(f"Fraction of signal in [{sig_min}, {sig_max}]: {fraction:.2%} (relative to [{xmin}, {xmax}])")
    
    return fraction

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    df = load_data(path_name = 'cleaned_D0_veto2.parquet')
    df_plus, df_minus = mask_df(df)

    com_plus = invar_mass(df_plus)
    com_minus = invar_mass(df_minus)

    xmin, xmax = 4900, 5375
    sig_min, sig_max = 5220.0, 5317.0
    com_plus_mask  = mask_com(com_plus, xmin, xmax)
    com_minus_mask = mask_com(com_minus, xmin, xmax)

    params, errors, params_cov = perform_unbinned_fit(com_plus_mask, com_minus_mask, xmin, xmax)
    #bins = plot_simultaneous_fit(com_plus_mask, com_minus_mask, params, xmin, xmax, sig_min, sig_max)
    bins = plot_simultaneous_fit2(com_plus_mask, com_minus_mask, params, xmin, xmax, sig_min, sig_max)
    
    chi2, ndf, red_chi2 = calculate_global_chi2(params, com_plus_mask, com_minus_mask, xmin, xmax, 
                                                sig_min, sig_max, bins)
    calculate_asymmetry(params, errors, cov_minus_plus=0.0)
    
    mu_fit, sig_fit, alpha_fit, n_fit = params[6], params[7], params[8], params[9]
    fraction = calculate_signal_fraction(mu_fit, sig_fit, alpha_fit, n_fit, xmin, xmax, sig_min, sig_max)
    
    if params_cov is not None:
        np.savetxt('../data/global_optimised_paramaters/fit_params.txt', params)
        np.savetxt('../data/global_optimised_paramaters/fit_errors.txt', errors)
        np.savetxt('../data/global_optimised_paramaters/fit_covariance.txt', params_cov)