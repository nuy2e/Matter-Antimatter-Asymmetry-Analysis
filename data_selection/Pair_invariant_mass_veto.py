"""
Charm Resonance and J/psi Vetoes for B -> pi pi pi Analysis

This module searches for and removes intermediate D0 and J/psi resonances 
that can contaminate the charmless three-body B meson decay signal. It computes 
the invariant mass of 2-body subsets (m_low and m_high), fits the resonance peaks 
with a Gaussian plus linear background, and applies a dynamic veto window.

Author: A. Knight (Collaboration)
Date: April 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# Constants
# ============================================================

MASS = {
    "pi": 139.57039,      # MeV
    "K":  493.677,        # MeV
    "mu": 105.6583745,    # MeV
}

M_D0 = 1864.84   # MeV/c^2
M_JPSI = 3096.9  # MeV/c^2

HYPOTHESES = {
    "pi_pi_K": ("pi", "pi", "K"),
    "pi_K_pi": ("pi", "K", "pi"),
    "K_pi_pi": ("K", "pi", "pi"),
    "pi_K_K":  ("pi", "K", "K"),
    "K_pi_K":  ("K", "pi", "K"),
    "K_K_pi":  ("K", "K", "pi"),
}

JPSI_HYPOTHESES = {
    "mu_pi_mu": ("mu", "pi", "mu"),
    "mu_mu_pi": ("mu", "mu", "pi"),
}

FIT_WINDOWS = {
    "pi_pi_K": {"low": (1840, 1900), "high": (1840, 1900)},
    "pi_K_pi": {"low": (1840, 1900), "high": (1840, 1900)},
    "K_pi_pi": {"low": (1840, 1900), "high": (1840, 1900)},
    "pi_K_K":  {"low": (1840, 1910), "high": (1840, 1910)},
    "K_pi_K":  {"low": (1840, 1910), "high": (1840, 1910)},
    "K_K_pi":  {"low": (1840, 1910), "high": (1840, 1910)},
}

JPSI_WINDOWS = {
    "pi_mu_mu": {"low": (3000, 3200), "high": (3000, 3200)},
    "mu_pi_mu": {"low": (3000, 3200), "high": (3000, 3200)},
    "mu_mu_pi": {"low": (3000, 3200), "high": (3000, 3200)},
}

BINS = 120
N_SIGMA = 3.0
OUTDIR = "veto_plots"

# ============================================================
# Data Loading
# ============================================================

def load_data(path="../data/SNR_optimised_data/best_selection.parquet"):
    """
    Loads the selected dataframe from a parquet file.

    Args:
        path (str): Filepath to the parquet data.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    return pd.read_parquet(path)

# ============================================================
# Kinematics & Mass Calculations
# ============================================================

def invariant_mass(df, particles=(1, 2), mass_mev=(139.57039, 139.57039, 139.57039), prefix="H"):
    """
    Computes the invariant mass for a subset of tracks under specific mass hypotheses.

    Args:
        df (pd.DataFrame): The dataset containing particle momenta.
        particles (tuple): Tuple of integers indicating which tracks to combine (e.g., (1, 2)).
        mass_mev (tuple): Assumed masses for the daughter particles in MeV.
        prefix (str): Column prefix for the particles. Defaults to "H".

    Returns:
        np.ndarray: Calculated invariant mass array for the given track combinations.
    """
    E_sum = np.zeros(len(df), dtype=float)
    px_sum = np.zeros(len(df), dtype=float)
    py_sum = np.zeros(len(df), dtype=float)
    pz_sum = np.zeros(len(df), dtype=float)

    for i in particles:
        px = df[f"{prefix}{i}_PX"].to_numpy()
        py = df[f"{prefix}{i}_PY"].to_numpy()
        pz = df[f"{prefix}{i}_PZ"].to_numpy()

        Ei = np.sqrt(px**2 + py**2 + pz**2 + mass_mev[i - 1]**2)

        E_sum += Ei
        px_sum += px
        py_sum += py
        pz_sum += pz

    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
    m2 = np.maximum(m2, 0.0)
    return np.sqrt(m2)


def hypothesis_masses(labels):
    """
    Converts a tuple of particle labels into their corresponding mass values.

    Args:
        labels (tuple): String labels for the particle hypothesis (e.g., ('pi', 'K', 'pi')).

    Returns:
        tuple: Corresponding mass values in MeV.
    """
    return tuple(MASS[x] for x in labels)


def compute_mlow_mhigh(df, labels):
    """
    Computes the lower and higher invariant masses of the possible opposite-sign pairs.

    Args:
        df (pd.DataFrame): The event dataset.
        labels (tuple): The assumed particle identities.

    Returns:
        tuple: A tuple containing four np.ndarrays (m12, m13, m_low, m_high).
    """
    masses = hypothesis_masses(labels)
    m12 = invariant_mass(df, particles=(1, 2), mass_mev=masses)
    m13 = invariant_mass(df, particles=(1, 3), mass_mev=masses)
    m_low = np.minimum(m12, m13)
    m_high = np.maximum(m12, m13)
    return m12, m13, m_low, m_high

# ============================================================
# Fit Models
# ============================================================

def gauss_plus_linear(x, A, mu, sigma, c0, c1):
    """
    Evaluates a combined Gaussian and linear background model.

    Args:
        x (np.ndarray): Independent variable array.
        A (float): Amplitude of the Gaussian peak.
        mu (float): Mean of the Gaussian.
        sigma (float): Width (standard deviation) of the Gaussian.
        c0 (float): Linear intercept.
        c1 (float): Linear slope.

    Returns:
        np.ndarray: The evaluated model values.
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + (c0 + c1 * x)


def fit_gaussian_linear_to_hist(data, low, high, bins=120, mu_guess=M_D0):
    """
    Fits a Gaussian + linear background to histogrammed data within a specified range.

    Args:
        data (np.ndarray): The 1D array of mass values to fit.
        low (float): Lower bound of the fit window.
        high (float): Upper bound of the fit window.
        bins (int): Number of histogram bins. Defaults to 120.
        mu_guess (float): Initial guess for the resonance peak mean. Defaults to M_D0.

    Returns:
        tuple: Contains optimized parameters (popt), covariance matrix (pcov), 
               bin centers (centers), bin counts (counts), and Poisson errors (yerr).
    """
    counts, edges = np.histogram(data, bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    yerr = np.sqrt(np.maximum(counts, 1.0))

    A0 = max(counts.max(), 1.0)
    sigma0 = 8.0

    n_edge = max(3, bins // 10)
    y_left = counts[:n_edge].mean()
    y_right = counts[-n_edge:].mean()
    x_left = centers[:n_edge].mean()
    x_right = centers[-n_edge:].mean()

    c1_0 = 0.0 if (x_right == x_left) else (y_right - y_left) / (x_right - x_left)
    c0_0 = y_left - c1_0 * x_left

    p0 = [A0, mu_guess, sigma0, c0_0, c1_0]
    bounds = (
        [0.0, low, 0.5, -np.inf, -np.inf],
        [np.inf, high, 100.0, np.inf, np.inf]
    )

    popt, pcov = curve_fit(
        gauss_plus_linear,
        centers,
        counts,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000
    )

    return popt, pcov, centers, counts, yerr

# ============================================================
# Veto Execution & Plotting
# ============================================================

def safe_name(text):
    """Formats a text string to be safe for file saving."""
    return (
        text.replace(" ", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
    )


def fit_and_build_veto(inv_mass, fit_range, bins, label, mu_guess=M_D0, n_sigma=3.0, make_plot=True):
    """
    Fits the resonance peak and constructs a boolean mask to veto events 
    within the n_sigma window around the fitted mean.

    Args:
        inv_mass (np.ndarray): Array of invariant masses.
        fit_range (tuple): Range (low, high) to perform the fit.
        bins (int): Number of bins for the histogram.
        label (str): Title/label for plotting and logging.
        mu_guess (float): Initial guess for the mean. Defaults to M_D0.
        n_sigma (float): Number of standard deviations to veto. Defaults to 3.0.
        make_plot (bool): If True, generates and saves a visualization. Defaults to True.

    Returns:
        tuple: Boolean mask indicating vetoed events (True means vetoed) and a 
               dictionary of fit results and statistics.
    """
    low, high = fit_range
    fit_mask = (inv_mass > low) & (inv_mass < high)
    fit_data = inv_mass[fit_mask]

    if fit_data.size == 0:
        print(f"{label}: no entries in fit window [{low}, {high}]")
        return np.zeros_like(inv_mass, dtype=bool), None

    popt, pcov, x, y, yerr = fit_gaussian_linear_to_hist(
        fit_data, low=low, high=high, bins=bins, mu_guess=mu_guess
    )

    A, mu, sigma, c0, c1 = popt
    perr = np.sqrt(np.diag(pcov))

    veto_low = mu - n_sigma * sigma
    veto_high = mu + n_sigma * sigma
    veto_mask = (inv_mass > veto_low) & (inv_mass < veto_high)

    print(f"{label}")
    print(f"  A     = {A:.2f} ± {perr[0]:.2f}")
    print(f"  mu    = {mu:.3f} ± {perr[1]:.3f} MeV")
    print(f"  sigma = {sigma:.3f} ± {perr[2]:.3f} MeV")
    print(f"  c0    = {c0:.2f} ± {perr[3]:.2f}")
    print(f"  c1    = {c1:.6f} ± {perr[4]:.6f}")
    print(f"  veto window = [{veto_low:.2f}, {veto_high:.2f}] MeV")
    print(f"  removed = {veto_mask.sum()} ({100.0 * veto_mask.mean():.2f}%)")
    print()

    result = {
        "A": A, "mu": mu, "sigma": sigma, "c0": c0, "c1": c1,
        "mu_err": perr[1], "sigma_err": perr[2],
        "veto_low": veto_low, "veto_high": veto_high,
        "removed": int(veto_mask.sum())
    }

    if make_plot:
        xx = np.linspace(low, high, 1000)
        yy = gauss_plus_linear(xx, *popt)
        yy_sig = A * np.exp(-0.5 * ((xx - mu) / sigma)**2)
        yy_bkg = c0 + c1 * xx

        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, yerr=yerr, fmt='.', capsize=1, label="Data")
        plt.plot(xx, yy, label="Gaussian + linear fit")
        plt.plot(xx, yy_sig, "--", label="Gaussian")
        plt.plot(xx, yy_bkg, "--", label="Linear background")
        plt.axvline(M_D0, color="black", linestyle=":", label=r"$m_{D^0}$")
        plt.axvline(veto_low, color="red", linestyle="--", alpha=0.8, label="Veto window")
        plt.axvline(veto_high, color="red", linestyle="--", alpha=0.8)

        plt.xlabel(r"Invariant Mass [MeV/$c^2$]")
        plt.ylabel("Counts per bin")
        plt.title(label)
        plt.legend()
        plt.tight_layout()

        os.makedirs(OUTDIR, exist_ok=True)
        plt.savefig(os.path.join(OUTDIR, f"{safe_name(label)}.pdf"))
        plt.show()
        plt.close()

    return veto_mask, result


def overlay_before_after(raw, kept, title, bins=120, mass_line=M_D0, filename=None):
    """
    Plots the mass distribution before and after the veto is applied.

    Args:
        raw (np.ndarray): Original mass array before veto.
        kept (np.ndarray): Mass array after veto.
        title (str): Plot title.
        bins (int): Number of bins. Defaults to 120.
        mass_line (float): Reference mass for the vertical line. Defaults to M_D0.
        filename (str, optional): Destination filename. Defaults to None.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(raw, bins=bins, histtype="step", linewidth=1.5, label="Raw")
    plt.hist(kept, bins=bins, histtype="step", linewidth=1.5, label="After veto")
    plt.axvline(mass_line, color="black", linestyle="--", linewidth=1.2, label=r"Reference Mass")
    plt.xlabel(r"Invariant Mass [MeV/$c^2$]")
    plt.ylabel("Number of Events")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        os.makedirs(OUTDIR, exist_ok=True)
        plt.savefig(os.path.join(OUTDIR, filename))

    plt.show()
    plt.close()


def fit_peak_and_get_window(inv_mass, fit_range, bins, label, mu_guess, n_sigma=3.0, make_plot=True):
    """
    Fits the resonance in a control sample to determine the veto boundaries 
    to be applied to the primary analysis sample.

    Args:
        inv_mass (np.ndarray): Mass array from the control sample.
        fit_range (tuple): Range (low, high) to perform the fit.
        bins (int): Number of bins.
        label (str): Title/label for logging and plotting.
        mu_guess (float): Initial guess for the mean.
        n_sigma (float): Number of standard deviations to veto. Defaults to 3.0.
        make_plot (bool): Flag to generate a plot. Defaults to True.

    Returns:
        dict: Fitted parameters and the calculated veto window boundaries.
    """
    low, high = fit_range
    fit_mask = (inv_mass > low) & (inv_mass < high)
    fit_data = inv_mass[fit_mask]

    if fit_data.size == 0:
        raise ValueError(f"{label}: no entries in fit window [{low}, {high}]")

    popt, pcov, x, y, yerr = fit_gaussian_linear_to_hist(
        fit_data, low=low, high=high, bins=bins, mu_guess=mu_guess
    )

    A, mu, sigma, c0, c1 = popt
    perr = np.sqrt(np.diag(pcov))

    veto_low = mu - n_sigma * sigma
    veto_high = mu + n_sigma * sigma

    print(f"{label} (control fit)")
    print(f"  mu    = {mu:.3f} ± {perr[1]:.3f} MeV")
    print(f"  sigma = {sigma:.3f} ± {perr[2]:.3f} MeV")
    print(f"  veto window = [{veto_low:.2f}, {veto_high:.2f}] MeV")
    print()

    if make_plot:
        xx = np.linspace(low, high, 1000)
        yy = gauss_plus_linear(xx, *popt)
        yy_sig = A * np.exp(-0.5 * ((xx - mu) / sigma)**2)
        yy_bkg = c0 + c1 * xx

        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, yerr=yerr, fmt='.', capsize=1, label="Control sample")
        plt.plot(xx, yy, label="Gaussian + linear fit")
        plt.plot(xx, yy_sig, "--", label="Gaussian")
        plt.plot(xx, yy_bkg, "--", label="Linear background")
        plt.axvline(mu, color="black", linestyle=":", label="Fitted mean")
        plt.axvline(veto_low, color="red", linestyle="--", label="Transferred veto")
        plt.axvline(veto_high, color="red", linestyle="--")
        plt.xlabel(r"Invariant Mass [MeV/$c^2$]")
        plt.ylabel("Counts per bin")
        plt.title(label)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return {
        "mu": mu,
        "mu_err": perr[1],
        "sigma": sigma,
        "sigma_err": perr[2],
        "veto_low": veto_low,
        "veto_high": veto_high,
        "popt": popt,
        "pcov": pcov,
    }


def apply_fixed_veto(inv_mass, low, high, label):
    """
    Applies a pre-calculated fixed veto window to an invariant mass array.

    Args:
        inv_mass (np.ndarray): Array of invariant masses.
        low (float): Lower bound of the veto window.
        high (float): Upper bound of the veto window.
        label (str): Logging label.

    Returns:
        np.ndarray: Boolean mask indicating vetoed events.
    """
    mask = (inv_mass > low) & (inv_mass < high)

    print(f"{label} (applied to analysis sample)")
    print(f"  veto window = [{low:.2f}, {high:.2f}] MeV")
    print(f"  removed = {mask.sum()} ({100.0 * mask.mean():.2f}%)")
    print()

    return mask

# ============================================================
# Main Execution
# ============================================================

def main():
    """
    Primary execution loop. Iterates through defined particle hypotheses, 
    calculates m_low and m_high, identifies D0 and J/psi backgrounds, 
    applies the vetoes, and saves the cleaned dataset.
    """
    df = load_data()
    df_jpsi = load_data("../data/SNR_optimised_data/best_selection_wmuons.parquet")
    n_events = len(df)

    total_remove = np.zeros(n_events, dtype=bool)
    summary_rows = []

    cached_masses = {}

    for hypo_name, labels in HYPOTHESES.items():
        print("=" * 70)
        print(f"Hypothesis: {hypo_name} = {labels}")
        print("=" * 70)

        m12, m13, m_low, m_high = compute_mlow_mhigh(df, labels)
        cached_masses[hypo_name] = {
            "m12": m12,
            "m13": m13,
            "m_low": m_low,
            "m_high": m_high
        }

        low_window = FIT_WINDOWS[hypo_name]["low"]
        high_window = FIT_WINDOWS[hypo_name]["high"]

        mask_low, res_low = fit_and_build_veto(
            m_low,
            fit_range=low_window,
            bins=BINS,
            label=f"{hypo_name}: m_low",
            mu_guess=M_D0,
            n_sigma=N_SIGMA,
            make_plot=True
        )

        mask_high, res_high = fit_and_build_veto(
            m_high,
            fit_range=high_window,
            bins=BINS,
            label=f"{hypo_name}: m_high",
            mu_guess=M_D0,
            n_sigma=N_SIGMA,
            make_plot=True
        )

        hypo_remove = mask_low | mask_high
        total_remove |= hypo_remove

        summary_rows.append({
            "hypothesis": hypo_name,
            "labels": "-".join(labels),
            "m_low_removed": int(mask_low.sum()),
            "m_high_removed": int(mask_high.sum()),
            "union_removed": int(hypo_remove.sum())
        })

        overlay_before_after(
            m_low,
            m_low[~hypo_remove],
            title=f"{hypo_name}: m_low before/after veto",
            bins=BINS,
            filename=f"{hypo_name}_mlow_before_after.pdf"
        )

        overlay_before_after(
            m_high,
            m_high[~hypo_remove],
            title=f"{hypo_name}: m_high before/after veto",
            bins=BINS,
            filename=f"{hypo_name}_mhigh_before_after.pdf"
        )
        
    for hypo_name, labels in JPSI_HYPOTHESES.items():
        print("=" * 70)
        print(f"J/psi hypothesis: {hypo_name} = {labels}")
        print("=" * 70)
    
        m12, m13, m_low_ctrl, m_high_ctrl = compute_mlow_mhigh(df_jpsi, labels)
    
        jpsi_low_fit = fit_peak_and_get_window(
            m_low_ctrl,
            fit_range=(3050, 3150),
            bins=BINS,
            label=rf"J/$\psi$ control fit: m_low ({labels}$)",
            mu_guess=M_JPSI,
            n_sigma=N_SIGMA,
            make_plot=True
        )
        
        jpsi_high_fit = fit_peak_and_get_window(
            m_high_ctrl,
            fit_range=(3050, 3150),
            bins=BINS,
            label=rf"J/$\psi$ control fit: m_high ({labels}$)",
            mu_guess=M_JPSI,
            n_sigma=N_SIGMA,
            make_plot=True
        )
    
        _, _, m_low_main, m_high_main = compute_mlow_mhigh(df, labels)
        
        mask_jpsi_low = apply_fixed_veto(
            m_low_main,
            jpsi_low_fit["veto_low"],
            jpsi_low_fit["veto_high"],
            label=fr"J/$\psi$ veto on analysis sample: m_low ({labels}$)"
        )
        
        mask_jpsi_high = apply_fixed_veto(
            m_high_main,
            jpsi_high_fit["veto_low"],
            jpsi_high_fit["veto_high"],
            label=fr"J/$\psi$ veto on analysis sample: m_high ({labels}$)"
        )

        total_remove |= (mask_jpsi_low | mask_jpsi_high)

        jpsi_remove = mask_jpsi_low | mask_jpsi_high
        
        summary_rows.append({
            "hypothesis": hypo_name,
            "labels": "-".join(labels),
            "m_low_removed": int(mask_jpsi_low.sum()),
            "m_high_removed": int(mask_jpsi_high.sum()),
            "union_removed": int(jpsi_remove.sum())
        })
        
        overlay_before_after(
            m_low_main,
            m_low_main[~jpsi_remove],
            title=f"{hypo_name}: m_low before/after J/psi veto",
            bins=BINS,
            mass_line=M_JPSI,
            filename=f"{hypo_name}_mlow_before_after_jpsi.pdf"
        )
        
        overlay_before_after(
            m_high_main,
            m_high_main[~jpsi_remove],
            title=f"{hypo_name}: m_high before/after J/psi veto",
            bins=BINS,
            mass_line=M_JPSI,
            filename=f"{hypo_name}_mhigh_before_after_jpsi.pdf"
        )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()
    print(f"Total removed across all hypotheses: {total_remove.sum()} / {n_events} "
          f"({100.0 * total_remove.mean():.2f}%)")

    cleaned_df = df.loc[~total_remove].copy()
    removed_df = df.loc[total_remove].copy()

    cleaned_df.to_parquet("../data/vetoed_data/cleaned_D0_veto2.parquet", compression="snappy")
    removed_df.to_parquet("../data/vetoed_data/removed_D0_veto2.parquet", compression="snappy")


if __name__ == "__main__":
    main()