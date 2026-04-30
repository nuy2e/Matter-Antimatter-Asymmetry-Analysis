# Analysis of CP Asymmetry in B± → π±π+π- Decays

## Overview
This repository contains the data analysis pipeline for investigating global and localized CP violation in the charmless three-body decay of B mesons ($B^{\pm} \rightarrow \pi^{\pm} \pi^{\pm} \pi^{\mp}$). The analysis is based on data from the LHCb experiment. 

The project aims to isolate the target signal, extract the global raw asymmetry ($A_{raw}$) via a simultaneous extended maximum likelihood fit, and evaluate localized CP asymmetries across the Dalitz phase space. 

**Note on Methodology:** While the full experimental study compared multiple background subtraction techniques, this repository specifically showcases the **Sideband Subtraction method** (modelled with 2D Chebyshev polynomials) for evaluating local CP asymmetry. 

## Authorship & Collaboration
* **Report Author:** Min Ki Hong
* **Experimental Workflow:** The experiment and presentation were performed in collaboration with **Ashwin Knight**.
* **Specific Contribution:** The **vetoing process** and associated code for charm resonance removal (Module 2) were developed solely by **Ashwin Knight**.

## Data Availability
* **No data is provided in this repository.** The raw datasets (e.g., `trees_up.parquet`, `trees_down.parquet`) originating from the LHCb detector are omitted due to file size constraints and access restrictions. The code provided here serves to demonstrate the analytical workflow and methodology.

## Repository Structure & Pipeline

The analysis pipeline is broken down into five distinct modules, meant to be executed sequentially:

### 1. Data Selection & Optimization
* **File:** `data_selection_SNR_optimisation.py`
* **Description:** Applies initial quality cuts and optimizes the Signal-to-Noise Ratio (SNR). It performs a grid search over Particle Identification (PID) and Impact Parameter (IP) variables, utilizing a Gaussian + Exponential fit to find the optimal cut thresholds.

### 2. Charm Resonance Vetoes
* **File:** `Pair_invariant_mass_veto.py`
* **Description:** Identifies and removes intermediate charm resonances ($D^0$ and $J/\psi$) that contaminate the charmless signal. It computes 2-body invariant masses, fits the resonance peaks, and applies a dynamic veto window based on the fitted standard deviation.

### 3. Global CP Asymmetry Fit
* **File:** `global_asymmetry.py`
* **Description:** Performs a simultaneous extended unbinned maximum likelihood fit to the $B^+$ and $B^-$ invariant mass distributions. It models the signal with a Crystal Ball function, the combinatorial background with an Exponential function, and partially reconstructed backgrounds with an ARGUS function to extract global asymmetry.

### 4. Sideband Subtraction Modeling
* **File:** `sideband_subtraction.py`
* **Description:** Models the combinatorial background across the Dalitz phase space. It uses a 2D basis of Chebyshev polynomials fitted to the invariant mass sideband regions, scales the expected background to the signal region, and subtracts it from signal in the Dalitz plot.

### 5. Dalitz Plot Analysis
* **File:** `Dalitz_analysis.py`
* **Description:** Takes the background-subtracted grids and visualizes the local matter-antimatter (CP) asymmetry across the Dalitz plot. It calculates the fractional asymmetry and its statistical significance bin-by-bin, highlighting regions of interest such as the $\rho^0(770)$ resonance and $\pi\pi \leftrightarrow KK$ rescattering bands.

## Requirements
To run the scripts (assuming data is present), the following standard Python scientific libraries are required:
* `numpy`
* `pandas`
* `matplotlib`
* `scipy`
* `iminuit` (for the extended maximum likelihood fits)
* `pyarrow` or `fastparquet` (for reading `.parquet` files)
