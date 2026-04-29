"""
Event Selection and Optimization for B -> pi pi pi Analysis

This script processes LHCb data to optimize the signal-to-noise ratio (SNR) 
for the charmless three-body decay of B mesons. It applies pre-selection masks, 
constructs invariant mass distributions, and performs a grid search over 
PID and IP variables using an extended maximum likelihood fit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools

# --- Kinematics & Calculations ---

def vec4_dot(vec1, vec2):
    """
    Calculates the Minkowski dot product of two 4-vectors.

    Args:
        vec1 (np.ndarray): Array of 4-vectors, shape (N, 4) where columns are (E, px, py, pz).
        vec2 (np.ndarray): Array of 4-vectors, shape (N, 4) where columns are (E, px, py, pz).

    Returns:
        np.ndarray: Array of shape (N,) containing the dot products.
    """
    dot_prod = vec1[:, 0] * vec2[:, 0] - (vec1[:, 1] * vec2[:, 1] + vec1[:, 2] * vec2[:, 2] + vec1[:, 3] * vec2[:, 3])
    return dot_prod

def invar_mass(df):
    """
    Calculates the invariant mass of the B meson candidates assuming all daughters are pions.

    Args:
        df (pd.DataFrame): DataFrame containing the kinematic data (PX, PY, PZ) 
            for the three daughter particles (H1, H2, H3).

    Returns:
        np.ndarray: Array containing the center-of-mass energy (invariant mass) for each event.
    """
    m_pi = 139.57039 # MeV
    
    E1 = np.sqrt(df['H1_PX']**2 + df['H1_PY']**2 + df['H1_PZ']**2 + m_pi**2)
    E2 = np.sqrt(df['H2_PX']**2 + df['H2_PY']**2 + df['H2_PZ']**2 + m_pi**2)
    E3 = np.sqrt(df['H3_PX']**2 + df['H3_PY']**2 + df['H3_PZ']**2 + m_pi**2)

    columns1 = ['H1_PX', 'H1_PY', 'H1_PZ']
    columns2 = ['H2_PX', 'H2_PY', 'H2_PZ']
    columns3 = ['H3_PX', 'H3_PY', 'H3_PZ']
    
    p1_vec = df[columns1].to_numpy()
    p2_vec = df[columns2].to_numpy()
    p3_vec = df[columns3].to_numpy()

    p1_4vec = np.column_stack((E1, p1_vec))
    p2_4vec = np.column_stack((E2, p2_vec))
    p3_4vec = np.column_stack((E3, p3_vec))

    s = 3*(m_pi)**2 + 2*(vec4_dot(p1_4vec, p2_4vec) + vec4_dot(p1_4vec, p3_4vec) + vec4_dot(p2_4vec, p3_4vec))
    com = np.sqrt(s)

    return com

# --- Data Handling ---

def load_data(path):
    """
    Loads and concatenates the magnet polarity data (up and down) from parquet files.
    
    Args:
        path (string): path do the raw data
    
    Returns:
        pd.DataFrame: A combined DataFrame of the up and down polarity datasets.
    """
    
    df_up = pd.read_parquet(path + "trees_up.parquet")
    df_down = pd.read_parquet(path + "trees_up.parquet")
    df_combined = pd.concat([df_down, df_up], ignore_index=True)
    return df_combined

def mask_data(df_combined):
    """
    Applies basic quality cuts, PID boundaries, and muon vetoes to the raw dataset.

    Args:
        df_combined (pd.DataFrame): The raw, combined polarity DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame with invalid or background-heavy events removed.
    """
    full_mask = (
        (df_combined['H1_isMuon'] == 0) & 
        (df_combined['H2_isMuon'] == 0) & 
        (df_combined['H3_isMuon'] == 0) & 
        (df_combined['H1_ProbPi'] > 0) & (df_combined['H1_ProbPi'] < 1) & 
        (df_combined['H2_ProbPi'] > 0) & (df_combined['H2_ProbPi'] < 1) & 
        (df_combined['H3_ProbPi'] > 0) & (df_combined['H3_ProbPi'] < 1) & 
        (df_combined['H1_ProbK'] > 0) & (df_combined['H1_ProbK'] < 1) & 
        (df_combined['H2_ProbK'] > 0) & (df_combined['H2_ProbK'] < 1) & 
        (df_combined['H3_ProbK'] > 0) & (df_combined['H3_ProbK'] < 1) 
    )
    
    df_clean = df_combined[full_mask]
    return df_clean

# --- Physics Models ---

def gaussian(x, N, mu, sigma):
    """
    Normalized Gaussian model for signal distribution.

    Args:
        x (np.ndarray): Array of invariant mass values.
        N (float): Estimated yield (number of signal events).
        mu (float): Mean of the Gaussian peak.
        sigma (float): Standard deviation of the Gaussian peak.

    Returns:
        np.ndarray: Evaluated Gaussian values corresponding to the input array.
    """
    return N * (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

def exponential(x, N, tau, shift=5150):
    """
    Exponential model for combinatorial background.

    Args:
        x (np.ndarray): Array of invariant mass values.
        N (float): Estimated background yield.
        tau (float): Decay constant of the exponential distribution.
        shift (float, optional): Shift parameter to prevent computational overflow. Defaults to 5150.

    Returns:
        np.ndarray: Evaluated Exponential values corresponding to the input array.
    """
    return N * np.exp(-tau * (x - shift))

def fit_model(x, ns, mu, sigma, nb, tau):
    """
    Combined Signal + Background model for curve fitting.

    Args:
        x (np.ndarray): Array of invariant mass values.
        ns (float): Estimated number of signal events.
        mu (float): Mean of the Gaussian signal peak.
        sigma (float): Standard deviation of the signal peak.
        nb (float): Estimated number of background events.
        tau (float): Decay constant for the exponential background.

    Returns:
        np.ndarray: The evaluated combined probability distribution.
    """
    return gaussian(x, ns, mu, sigma) + exponential(x, nb, tau)

# --- Selection & Optimization ---

def selection_fit(df, cut_value):       
    """
    Computes PID probability differences and isolates required topological 
    variables prior to performing the optimization grid search.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        cut_value (float): Initial baseline cut value applied to all pions.

    Returns:
        tuple: A tuple containing:
            - df_cut (pd.DataFrame): DataFrame filtered by the initial cut.
            - diff_array_cut (np.ndarray): Array of shape (N, 3) containing PID differences.
            - B_vertex_chi2 (pd.Series): Series containing B vertex chi-squared values.
            - IP_chi2_array (np.ndarray): Array of shape (N, 3) containing IP chi-squared values.
    """
    diff_h1 = df['H1_ProbPi'] - df['H1_ProbK']
    diff_h2 = df['H2_ProbPi'] - df['H2_ProbK']
    diff_h3 = df['H3_ProbPi'] - df['H3_ProbK']
    
    diff_array = np.column_stack((diff_h1, diff_h2, diff_h3))
    print("Shape of difference array:", diff_array.shape)

    mask_all_pions = np.all(diff_array > cut_value, axis=1)
    df_cut = df[mask_all_pions]
    diff_array_cut = diff_array[mask_all_pions]
    
    B_vertex_chi2 = df_cut['B_VertexChi2']
    H1_IPChi2 = df_cut['H1_IPChi2']
    H2_IPChi2 = df_cut['H2_IPChi2']
    H3_IPChi2 = df_cut['H3_IPChi2']
    
    IP_chi2_array = np.column_stack((H1_IPChi2, H2_IPChi2, H3_IPChi2,))
    
    return df_cut, diff_array_cut, B_vertex_chi2, IP_chi2_array

def optimize_cuts_with_fitting(mass_array, pid_matrix, ip_matrix, pid_range, ip_range):
    """
    Scans a grid of PID and IP cuts to find the optimal combination that 
    maximizes signal significance using a Gaussian + Exponential fit model.

    Args:
        mass_array (np.ndarray): 1D array of invariant masses.
        pid_matrix (np.ndarray): 2D array of PID differences for the three daughters.
        ip_matrix (np.ndarray): 2D array of IP chi-squared values for the three daughters.
        pid_range (np.ndarray): 1D array defining the grid of PID cut values to test.
        ip_range (np.ndarray): 1D array defining the grid of IP cut values to test.

    Returns:
        tuple: A tuple containing:
            - best_cuts (tuple): The optimal (PID_cut, IP_cut) combination.
            - best_score (float): The maximum calculated significance (S / sqrt(S+B)).
            - best_popt (list): The optimal parameters from the curve_fit [ns, mu, sigma, nb, tau].
    """
    fit_min, fit_max = 5150, 5600
    n_bins = 100
    sig_region_min, sig_region_max = 5200, 5358.5
    
    bin_edges = np.linspace(fit_min, fit_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    best_score = -1
    best_cuts = (None, None)
    best_popt = None
    
    param_grid = list(itertools.product(pid_range, ip_range))
    print(f"Scanning {len(param_grid)} combinations using Full Fits...")

    for i, (pid_cut, ip_cut) in enumerate(param_grid):
        print(f"\rProcessing iteration: {i}", end="", flush=True)
        
        mask_pid = np.all(pid_matrix > pid_cut, axis=1)
        mask_ip  = np.all(ip_matrix > ip_cut, axis=1)
        
        selected_masses = mass_array[mask_pid & mask_ip]
        
        if len(selected_masses) < 50: 
            continue

        counts, _ = np.histogram(selected_masses, bins=bin_edges)
        
        try:
            p0 = [len(selected_masses)*0.1 * bin_width, 5279.6, 15.0, len(selected_masses)*0.9, 0.005]
            bounds = ([0, 5250, 5, 0, -0.1], [np.inf, 5310, 30, np.inf, 0.1])

            popt, pcov = curve_fit(fit_model, bin_centers, counts, p0=p0, bounds=bounds, maxfev=2000)
            ns_fit, mu_fit, sigma_fit, nb_fit, tau_fit = popt
            
            x_fine = np.linspace(sig_region_min, sig_region_max, 1000)
            dx = x_fine[1] - x_fine[0]
            
            sig_curve = gaussian(x_fine, ns_fit, mu_fit, sigma_fit)
            bkg_curve = exponential(x_fine, nb_fit, tau_fit)
            
            S_est = np.sum(sig_curve) / bin_width * dx
            B_est = np.sum(bkg_curve) / bin_width * dx
            
            significance = S_est / np.sqrt(S_est + B_est) if (S_est > 0 and B_est >= 0) else 0
            if S_est < 10: 
                significance = 0

            if significance > best_score:
                best_score = significance
                best_cuts = (pid_cut, ip_cut)
                best_popt = popt
                
        except (RuntimeError, ValueError):
            continue 

    print(f"\nOptimization Complete.")
    print(f"Best Cuts: PID > {best_cuts[0]:.3f} | IPChi2 > {best_cuts[1]:.3f}")
    print(f"Max Significance: {best_score:.3f}")
    
    return best_cuts, best_score, best_popt

def apply_and_save_selection(df, best_cuts, pid_array, vtx_array, ip_array, output_filename="best_selection.parquet"):
    """
    Applies the optimal cut parameters to the dataframe and saves the processed data.

    Args:
        df (pd.DataFrame): The original dataframe.
        best_cuts (tuple): Tuple containing the optimal (PID_cut, IP_cut).
        pid_array (np.ndarray): The PID difference array used in optimization.
        vtx_array (pd.Series): The Vertex Chi2 array. (Note: vertex cuts omitted as per analysis).
        ip_array (np.ndarray): The IP Chi2 array used in optimization.
        output_filename (str, optional): Destination filepath. Defaults to "best_selection.parquet".

    Returns:
        pd.DataFrame: The final selected dataframe after cuts are applied.
    """
    cut_pid, cut_ip = best_cuts
    
    print(f"--- Applying Cuts ---")
    print(f"PID > {cut_pid:.3f} | IP > {cut_ip:.3f}")
    
    mask_pid = np.all(pid_array > cut_pid, axis=1)
    mask_ip = np.all(ip_array > cut_ip, axis=1)
    
    total_mask = mask_pid & mask_ip
    df_selected = df[total_mask].copy()
    
    n_before = len(df)
    n_after = len(df_selected)
    eff = (n_after / n_before) * 100 if n_before > 0 else 0
    
    print(f"Original Events: {n_before}")
    print(f"Selected Events: {n_after}")
    print(f"Efficiency:      {eff:.2f}%")
    
    print(f"Saving to {output_filename}...")
    df_selected.to_parquet(output_filename, compression='snappy')
    print("Done.")
    
    return df_selected

# --- Plotting ---

def plot_fit_result(mass_array, pid_matrix, ip_matrix, best_cuts, best_popt):
    """
    Plots the invariant mass histogram overlaid with the optimal fitted signal and background.

    Args:
        mass_array (np.ndarray): 1D array of invariant masses.
        pid_matrix (np.ndarray): 2D array of PID differences.
        ip_matrix (np.ndarray): 2D array of IP chi-squared values.
        best_cuts (tuple): Tuple of (PID_cut, IP_cut) used to filter the plot data.
        best_popt (list): Fitted parameters [ns, mu, sigma, nb, tau].
    """
    pid_cut, ip_cut = best_cuts
    ns, mu, sigma, nb, tau = best_popt
    
    mask_pid = np.all(pid_matrix > pid_cut, axis=1)
    mask_ip  = np.all(ip_matrix > ip_cut, axis=1)
    selected_data = mass_array[mask_pid & mask_ip]
    
    fit_min, fit_max = 5150, 5600
    n_bins = 100
    bin_edges = np.linspace(fit_min, fit_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    x_plot = np.linspace(fit_min, fit_max, 1000)
    y_signal = gaussian(x_plot, ns, mu, sigma)
    y_bkg    = exponential(x_plot, nb, tau)
    y_total  = y_signal + y_bkg
    
    plt.figure(figsize=(10, 7))
    counts, _ = np.histogram(selected_data, bins=bin_edges)
    y_err = np.sqrt(counts) 
    
    plt.errorbar(bin_centers, counts, yerr=y_err, fmt='ko', label='Data', capsize=2)
    plt.plot(x_plot, y_total, 'b-', linewidth=2, label='Total Fit')
    plt.plot(x_plot, y_signal, 'r--', linewidth=2, label=f'Signal (Mass={mu:.1f})')
    plt.plot(x_plot, y_bkg, 'g:', linewidth=2, label='Background')
    
    plt.xlabel('Mass [MeV/$c^2$]', fontsize=14)
    plt.ylabel(f'Candidates / {(fit_max-fit_min)/n_bins:.1f} MeV', fontsize=14)
    plt.title(f'Fit Result: PID > {pid_cut:.1f}, IP $\chi^2$ > {ip_cut:.1f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    #plt.axvline(5239.25, color='gray', linestyle='--', alpha=0.5)
    #plt.axvline(5319.25, color='gray', linestyle='--', alpha=0.5)
    
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    
    df_raw = load_data('../data/raw_data/')
    df_clean = mask_data(df_raw)

    cut_value = -1
    df_precut, diff_array_precut, B_vertex_chi2, IP_chi2_array = selection_fit(df_clean, cut_value)

    com_precut = invar_mass(df_precut)

    # PID and IP grid search configuration
    pid_range = np.linspace(-0.5, -0.3, 10) 
    vtx_range = np.linspace(13, 13, 1) 
    ip_range = np.linspace(27, 28, 10)

    best_cuts, max_sig, params = optimize_cuts_with_fitting(
        com_precut, diff_array_precut, IP_chi2_array, pid_range, ip_range
    )

    plot_fit_result(com_precut, diff_array_precut, IP_chi2_array, best_cuts, params)

    print("\n--- Final Results ---")
    print(f"Max Significance: {max_sig:.3f}")
    print(f"Optimal PID Cut:  > {best_cuts[0]}")
    print(f"Optimal IP Cut:   > {best_cuts[1]}")

    apply_and_save_selection(
        df_precut, best_cuts, diff_array_precut, B_vertex_chi2, 
        IP_chi2_array, output_filename="../data/SNR_optimised_data/best_selection.parquet"
    )