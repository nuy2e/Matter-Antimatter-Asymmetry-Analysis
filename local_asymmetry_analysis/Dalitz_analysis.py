"""
Dalitz Plot Asymmetry Analysis

This module loads background-subtracted Dalitz grids to evaluate and visualize 
local matter-antimatter (CP) asymmetry across the phase space. It calculates 
the fractional asymmetry (A_CP) and its statistical significance bin-by-bin.

Author: Min Ki Hong
Date: April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from pathlib import Path

# ============================================================
# Plotting Functions
# ============================================================

def plot_comparison_pm(H_plus, H_minus, xedges, yedges, title):
    """
    Plots the B+ and B- subtracted backgrounds side-by-side using a shared color scale.
    
    Args:
        H_plus (np.ndarray): 2D array of B+ events.
        H_minus (np.ndarray): 2D array of B- events.
        xedges (np.ndarray): Array of bin edges for the x-axis.
        yedges (np.ndarray): Array of bin edges for the y-axis.
        title (str): Title prefix for the subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)
    
    max_events = max(np.max(H_plus), np.max(H_minus))
    
    H_plus_plot = np.clip(H_plus, 0.1, None)
    H_minus_plot = np.clip(H_minus, 0.1, None)
    
    norm = LogNorm(vmin=0.1, vmax=max_events)
    
    mesh1 = axes[0].pcolormesh(xedges, yedges, H_plus_plot.T, norm=norm, cmap='viridis')
    axes[0].set_title(f"Plus ({title})")
    axes[0].set_xlabel(r"$m^2_{\mathrm{low}}$ (MeV$^2$)")
    axes[0].set_ylabel(r"$m^2_{\mathrm{high}}$ (MeV$^2$)")
    
    mesh2 = axes[1].pcolormesh(xedges, yedges, H_minus_plot.T, norm=norm, cmap='viridis')
    axes[1].set_title(f"Minus ({title})")
    axes[1].set_xlabel(r"$m^2_{\mathrm{low}}$ (MeV$^2$)")
    
    cbar = fig.colorbar(mesh1, ax=axes, fraction=0.05)
    cbar.set_label("Events per bin (Log Scale)")
    
    safe_title = title.replace(" ", "_")
    plt.savefig(f"plot/Bkg_Comparison_{safe_title}.pdf", format='pdf')
    plt.show()

def plot_single_dalitz_linear(H, xedges, yedges, title, usage=None, threshold=None):
    """
    Plots a single Dalitz plot utilizing large text sizes suitable for presentations.
    Adjusts colormap handling depending on the type of data being visualized 
    (e.g., error vs significance).
    
    Args:
        H (np.ndarray): 2D array of Dalitz plot data.
        xedges (np.ndarray): Array of bin edges for the x-axis.
        yedges (np.ndarray): Array of bin edges for the y-axis.
        title (str): Title for the plot and output file.
        usage (str, optional): Modifies plot behavior ('err' or 'significance'). Defaults to None.
        threshold (float, optional): Z-score threshold used when usage is 'significance'. Defaults to None.
    """
    plt.rcParams.update({'font.size': 16}) 
    
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.set_facecolor('lightgray')
    
    H_plot = np.where(H == 0.0, np.nan, H)
    max_abs = np.nanmax(np.abs(H_plot))
    
    if usage == "err":
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='none') 
        mesh = ax.pcolormesh(xedges, yedges, H_plot.T, vmin=0, vmax=max_abs, cmap=cmap)
    else:
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad(color='none') 
        mesh = ax.pcolormesh(xedges, yedges, H_plot.T, vmin=-max_abs, vmax=max_abs, cmap=cmap)
        
    if usage == "significance" and threshold is not None:
        sig_mask = np.where(np.logical_or(H_plot > threshold, H_plot < -threshold), 1, np.nan) 
        transparent_cmap = mcolors.ListedColormap(['none'])
        ax.pcolor(xedges, yedges, sig_mask.T, hatch='////', cmap=transparent_cmap, edgecolor='black', lw=0)
        
        projection = np.nansum(H, axis=1)
        fig1d, ax1d = plt.subplots(figsize=(10, 5))
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        
        ax1d.step(x_centers, projection, where='mid', color='black', lw=2.5) 
        ax1d.fill_between(x_centers, projection, step="mid", alpha=0.2, color='gray')
        
        ax1d.set_title(f"X-Projection: {title}", fontsize=24, pad=20)
        ax1d.set_xlabel(r"$m^2_{\mathrm{low}}$ (MeV$^2$)", fontsize=22)
        ax1d.set_ylabel("Significance (Summed)", fontsize=22)
        ax1d.tick_params(axis='both', which='major', labelsize=18)
        ax1d.grid(axis='y', alpha=0.3)
        
        output_dir = Path("plot")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = title.replace(" ", "_")
        fig1d.savefig(output_dir / f"Projection_1D_{safe_title}.pdf", format='pdf')
        
    ax.set_title(title, fontsize=26, pad=20, fontweight='bold')
    ax.set_xlabel(r"$m^2_{\mathrm{low}}$ (MeV$^2$)", fontsize=24, labelpad=10)
    ax.set_ylabel(r"$m^2_{\mathrm{high}}$ (MeV$^2$)", fontsize=24, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Events per bin (Linear Scale)", fontsize=22, labelpad=15)
    cbar.ax.tick_params(labelsize=18)
    
    output_dir = Path("plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = title.replace(" ", "_")
    
    fig.savefig(output_dir / f"Dalitz_Single_Visible_{safe_title}.pdf", format='pdf')
    print(f"Saved plots with presentation-sized text for: {title}")
    
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

# ============================================================
# Statistical Analysis
# ============================================================

def calculate_asymmetry(H_plus, H_minus, H_plus_err, H_minus_err):
    """
    Calculates the localized CP asymmetry and its statistical significance.
    
    Args:
        H_plus (np.ndarray): 2D array of subtracted B+ signal counts.
        H_minus (np.ndarray): 2D array of subtracted B- signal counts.
        H_plus_err (np.ndarray): 2D array of errors for B+ signal counts.
        H_minus_err (np.ndarray): 2D array of errors for B- signal counts.
        
    Returns:
        tuple: (A, err_A, significance, diff_H, total_H) arrays evaluated per-bin.
    """
    total_H = H_plus + H_minus
    diff_H = H_minus - H_plus

    min_events = 6
    A = np.divide(diff_H, total_H, out=np.zeros_like(diff_H), where=(total_H > min_events))
    
    dA_dH_minus = np.divide(2.0 * H_plus, total_H**2, out=np.zeros_like(total_H), where=(total_H > min_events))
    dA_dH_plus  = np.divide(-2.0 * H_minus, total_H**2, out=np.zeros_like(total_H), where=(total_H > min_events))

    err_A = np.sqrt((dA_dH_minus**2) * (H_minus_err**2) + (dA_dH_plus**2) * (H_plus_err**2))
    significance = np.divide(A, err_A, out=np.zeros_like(A), where=(err_A != 0))
    
    return A, err_A, significance, diff_H, total_H

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    
    data = np.load("../data/dalitz_data/dalitz_data_20.npz")
    H_signal_plus = data['H_plus']
    H_signal_plus_err = data['H_plus_err']
    H_signal_minus = data['H_minus']
    H_signal_minus_err = data['H_minus_err']
    xedges, yedges = data['xedges'], data['yedges']

    H_subtracted = H_signal_minus - H_signal_plus
    A, err_A, significance, diff_H, total_H = calculate_asymmetry(H_signal_plus, H_signal_minus, H_signal_plus_err, H_signal_minus_err)

    plot_comparison_pm(H_signal_plus, H_signal_minus, xedges, yedges, title="B signal")
    plot_single_dalitz_linear(H_subtracted, xedges, yedges, title='B subtracted')
    plot_single_dalitz_linear(A, xedges, yedges, title='Asymmetry')
    plot_single_dalitz_linear(err_A, xedges, yedges, title='err A', usage="err")
    plot_single_dalitz_linear(significance, xedges, yedges, title='significance', usage='significance', threshold=3)