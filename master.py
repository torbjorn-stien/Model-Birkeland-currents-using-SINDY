#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:26:41 2026

@author: tos
"""

from sympy import symbols, Eq, solve
from scipy.integrate import solve_ivp
import pysindy as ps

import numpy as np
import matplotlib.pyplot as plt
from read_ampere_ncdf import read_Jpar
from pathlib import Path

import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from read_ace_files import read_dat
from sklearn.metrics import r2_score
from matplotlib.animation import FuncAnimation
import math
from itertools import groupby
import glob
import pyomnidata as pod
from nan_handling import find_nans, interpolate_nans, find_overlapping_clean_data, subset_data

import warnings
from pysindy.utils._axes import AxesWarning

warnings.filterwarnings("ignore", category=AxesWarning)

"""
Look at solar zenith angle to see if I can put that in as a driver aswell.
May or may not be easy for DMDc, but at least remember for SINDY
Will have to use GEO to calculate solar zenith.
-> Have to look at the grids/points and if they "exist" at the same point for 
both the AAGCM and GEO arrays. (e.g. plot them on/adjacent to each other)
They more than likely do
"""

pod.UpdateLocalData()

AMPERE_PATH = "/nfs/revontuli/data/bjorn/Ampere"

fontsize = 20

# Breaks for 2020:


def read_files(directory, start_year=None, end_year=None):
    """
    Reads .dat files from a directory. If a range of years is specified, only reads files within that range.

    Args:
        directory (str): Path to the directory containing the .dat files.
        start_year (int, optional): Start year to read files from. If None, reads all files.
        end_year (int, optional): End year to read files up to. If None, reads all files.

    Returns:
        pd.DataFrame: Combined DataFrame of all read files.
    """
    # Get all .dat files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith(".dat")])

    # Filter files by year range if start_year and/or end_year are specified
    if start_year is not None or end_year is not None:
        filtered_files = []
        for f in files:
            # Extract year from the filename
            for year in range(start_year or 0, (end_year or 9999) + 1):
                if str(year) in f:
                    filtered_files.append(f)
                    break
        files = filtered_files

    # List to store the dataframes
    dataframes = []

    # Loop through the filtered files and read them
    for filename in files:
        if filename == "mag_B_4min_2020.dat":
            print("Tried to read 2020, but 2020 is broken!")
            continue
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        print(f"Reading file: {filename}")  # Print the file being read
        # Read the file using function in separate file
        df = read_dat(file_path)
        dataframes.append(df)  # Append the dataframe to the list

    # Combine all dataframes into one
    if dataframes:
        combined_data = pd.concat(dataframes, ignore_index=True)
        print("Done reading!")
        return combined_data
    else:
        print("No files found for the specified year range.")
        return pd.DataFrame()


def Milan_coupling(By, Bz, Vx):
    """
    From Milan et al in JGR, https://doi.org/10.1029/2011JA017082
    ONLY TO BE USED FOR NON-SUBSTORM PERIODS
    Assumes negligent night-side reconnection during non-substorm intervals


    Params:
        Bx, By, Bz: 
            type: ndarray
            GSM coordinates

    Variables
    ---------
    B_yz : B_yz**2 = By**2 + Bz**2
    """
    R_E = 6357 * 1000
    Lambda = 3.3 * 10**5    # m**(2/3) s**(1/3)
    phi_d = np.zeros_like(Vx)
    c = 3e8
    theta = np.arctan2(By, Bz)
    Byz = np.sqrt(By**2 + Bz**2)
    # Have to force each Vx to float for calulation to work
    # DO NOT TOUCH
    for i in range(len(Vx)):
        L_eff = (3.8 * R_E * (float(Vx[i])/(4 * 10**5))**(1/3)).real

        phi_d[i] = L_eff * float(Vx[i]) * Byz[i] * \
            np.sin(0.5 * theta[i])**(9/2)  # eq 15

    # phi_d = Lambda * np.abs(Vx)**(4/3) * Byz * np.sin(1/2 * theta)**(9/2) # eq 14

    F_max = phi_d/c

    return F_max


def delay_control_data(control_dat, nr_of_delays, delay_indexes):
    """

    Parameters
    ----------
    control_dat : numpy array
        Control data that should be delayed
    nr_delays : int
        How many times the control data should be delayed.
    delay_indexes : int
        How many indexes each delay should be.
        For ACE data dt = 4min, -> 1 delay 1 index = delay 4 minutes


    Returns
    -------
    delayed_input : numpy array
        Array with each delay stacked below eachother on axis 0

    """
    u = control_dat
    delays = np.arange(0, nr_of_delays, delay_indexes)
    # 1 index delay = 4 mins, = approx spacing between field lines of 7.5 Earth radii
    # Number of delays = number of fieldlines having an appreciable input, 7 worked best for DMDc

    if nr_of_delays == 0:
        return u

    if u.ndim == 1:
        u = u[:, np.newaxis]  # Reshape to (num_rows, 1)

    num_rows, num_features = u.shape

    num_delays = len(delays)
    # Initialize final matrix
    delayed_input = np.zeros((num_rows, num_features * num_delays))

    for i, delay in enumerate(delays):
        start_col = i * num_features  # Start column for this delay
        end_col = start_col + num_features  # End column for this delay

        if delay == 0:
            # No delay; copy of original input
            delayed_input[:, start_col:end_col] = u
        else:
            # Shifts each block of inputs by the number of indexes given above.
            # Values before input = 0
            delayed = np.zeros_like(u)
            delayed[delay:] = u[:-delay]
            delayed_input[:, start_col:end_col] = delayed

    return delayed_input


def integrate_FACs(FACs, co_latitudes, longitudes):
    """
    Integrates field-aligned currents (FACs) over a polar cap region.
    Parameters:
        FACs (ndarray): 2D array of FAC values with shape (n_timesteps, n_points).
        co_latitudes (ndarray): 2D array of co-latitudes (in degrees) with shape (n_timesteps, n_points).
        longitudes (ndarray): 2D array of longitudes (in degrees) with shape (n_timesteps, n_points).
    Returns:
        tuple: Integrated FAC values for each timestep (out_currents, in_currents), 
               each with shape (n_timesteps, 1).
    """
    n_timesteps, n_points = FACs.shape
    print(f"FACs shape: {FACs.shape}")
    print(f"co_latitudes shape: {co_latitudes.shape}")
    print(f"longitudes shape: {longitudes.shape}")

    # Mask invalid (nan) values
    valid_mask = ~np.isnan(FACs) & ~np.isnan(
        co_latitudes) & ~np.isnan(longitudes)
    print(f"valid_mask shape: {valid_mask.shape}")

    # Replace invalid values with 0 for integration (they won't contribute due to the mask)
    FACs = np.where(valid_mask, FACs, 0)
    co_latitudes = np.where(valid_mask, co_latitudes, 0)
    longitudes = np.where(valid_mask, longitudes, 0)

    # Separate out and in currents
    out_currents = np.where(FACs > 0, FACs, 0)
    in_currents = np.where(FACs < 0, FACs, 0)
    print(f"out_currents shape: {out_currents.shape}")
    print(f"in_currents shape: {in_currents.shape}")

    # Convert co-latitudes and longitudes to radians
    theta = np.radians(co_latitudes)
    phi = np.radians(longitudes)
    print(f"theta shape: {theta.shape}")
    print(f"phi shape: {phi.shape}")

    # Compute weights based on spherical coordinates
    # Spacing in theta (along spatial points)
    dtheta = np.abs(np.gradient(theta, axis=1))
    # Spacing in phi (along spatial points)
    dphi = np.abs(np.gradient(phi, axis=1))
    # Area element in spherical coordinates
    weights = np.sin(theta) * dtheta * dphi
    print(f"weights shape: {weights.shape}")

    # Mask weights for invalid points
    weights = np.where(valid_mask, weights, 0)
    print(f"weights after masking shape: {weights.shape}")

    # Normalize weights for each timestep
    # Total polar cap area per timestep
    cap_area = 2 * np.pi * \
        (1 - np.cos(np.nanmax(theta, axis=1, keepdims=True)))
    # Sum of weights per timestep
    weights_sum = np.nansum(weights, axis=1, keepdims=True)
    weights /= weights_sum / cap_area  # Normalize weights
    print(f"weights after normalization shape: {weights.shape}")

    # Perform the weighted integration over the spatial dimension (axis=1)
    integrated_out_currents = np.nansum(
        out_currents * weights, axis=1, keepdims=True)
    integrated_in_currents = np.nansum(
        in_currents * weights, axis=1, keepdims=True)
    print(f"integrated_out_currents shape: {integrated_out_currents.shape}")
    print(f"integrated_in_currents shape: {integrated_in_currents.shape}")

    return integrated_out_currents, integrated_in_currents


def mean_norm(data):
    norm = (data - np.nanmean(data))/np.nanmean(data)

    return norm


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# %%
"""
2012, 2016 & 2020 were leap-years
"""
# dt = 2 min
# 2012 is broken, file 20120125
# 2019 aswell, file 20190520
# year_index = 0 is 2009, 4 is 2013
# nr_of_days = 365 * 5 gives (1314000, 1200), covers 2013, through 2018
Jpar, cLat_deg, lon_deg, missing_days, missing_indices = read_Jpar(from_year_index=4,
                                                                   nr_days=365 * 5,
                                                                   directory_path=AMPERE_PATH)
# %%
Jpar_downsampled = Jpar
cLat_deg_downsampled = cLat_deg
lon_deg_downsampled = lon_deg

interp_length = 10
want_coordinates = True
# If 1 column contains a nan, every column contains nan at the same place
# find_nans() is bottlenecking, vectorize it at some point
for column in range(Jpar_downsampled.shape[1]):
    Jpar_downsampled[:, column], nan_start_Jpar, nan_end_Jpar, nan_lengths_Jpar, no_nan_lengths_Jpar = interpolate_nans(Jpar_downsampled[:, column],
                                                                                                                        max_nan_length=interp_length)

    if want_coordinates:
        cLat_deg_downsampled[:, column], _, _, _, _ = interpolate_nans(cLat_deg_downsampled[:, column],
                                                                       max_nan_length=interp_length)
        lon_deg_downsampled[:, column],  _, _, _, _ = interpolate_nans(lon_deg_downsampled[:, column],
                                                                       max_nan_length=interp_length)

# %%

fin = -1
integrated_out_Jpar, integrated_in_Jpar = integrate_FACs(Jpar_downsampled[:fin],
                                                         cLat_deg_downsampled[:fin], lon_deg_downsampled[:fin])

integrated_out_Jpar = integrated_out_Jpar.flatten()
integrated_in_Jpar = integrated_in_Jpar.flatten()

integrated_out_Jpar[integrated_out_Jpar > 1] = np.nan
integrated_in_Jpar[integrated_in_Jpar < -1] = np.nan

plt.plot(integrated_out_Jpar)
plt.plot(integrated_in_Jpar)
plt.show()

norm_out = mean_norm(integrated_out_Jpar)
norm_in = mean_norm(integrated_in_Jpar)

plt.plot(norm_out)
plt.plot(norm_in, alpha=0.5)
plt.show()

#%%

window_size = 15 * 4

smoothed_out = moving_average(norm_out.flatten(), window_size)
smoothed_in = moving_average(norm_in.flatten(), window_size)

plt.plot(smoothed_out)
plt.plot(smoothed_in, alpha=0.5)
plt.show()

# %%
"""
pandas.resample() treats NaN as 0.
e.g mean([1, NaN, 3]) = 2, where mean([NaN, NaN, 

NaN]) = NaN
"""

# %%

# Reads in OMNI data with a time resolution of 1 minute
# DOES NOT HAVE TO BE READ IN MORE THAN ONCE PER KERNEL
omni_data = pod.GetOMNI([2013, 2020], Res=1)


AE = omni_data.AE
AL = omni_data.AL
AU = omni_data.AU
B = omni_data.B
Bx = omni_data.BxGSE
By = omni_data.ByGSE
Bz = omni_data.BzGSE
Vx = omni_data.Vx
Vy = omni_data.Vy
Vz = omni_data.Vz


interp_len = 15

AE_interp = interpolate_nans(AE, interp_len)
AL_interp = interpolate_nans(AL, interp_len)
AU_interp = interpolate_nans(AU, interp_len)

B_interp = interpolate_nans(B, interp_len)
Bx_interp = interpolate_nans(Bx, interp_len)
By_interp = interpolate_nans(By, interp_len)
Bz_interp = interpolate_nans(Bz, interp_len)
Vx_interp = interpolate_nans(Vx, interp_len)
Vy_interp = interpolate_nans(Vy, interp_len)
Vz_interp = interpolate_nans(Vz, interp_len)

# %%

train_start = 0
train_end = len(Jpar) - 1

# AU Does not contain NaNs
AE_train = AE_interp[0][::2][train_start:train_end]
AL_train = AL_interp[0][::2][train_start:train_end]
AU_train = AU_interp[0][::2][train_start:train_end]

B_train = B_interp[0][::2][train_start:train_end]
Bx_train = Bx_interp[0][::2][train_start:train_end]
By_train = By_interp[0][::2][train_start:train_end]
Bz_train = Bz_interp[0][::2][train_start:train_end]
Vx_train = Vx_interp[0][::2][train_start:train_end]
Vy_train = Vy_interp[0][::2][train_start:train_end]
Vz_train = Vz_interp[0][::2][train_start:train_end]

AE_train = mean_norm(AE_train)
AL_train = mean_norm(AL_train)
AU_train = mean_norm(AU_train)

B_train = mean_norm(B_train)
Bx_train = mean_norm(Bx_train)
By_train = mean_norm(By_train)
Bz_train = mean_norm(Bz_train)
Vx_train = mean_norm(Vx_train)
Vy_train = mean_norm(Vy_train)
Vz_train = mean_norm(Vz_train)

AE_smoothed = moving_average(AE_train, window_size)
AL_smoothed = moving_average(AL_train, window_size)
AU_smoothed = moving_average(AU_train, window_size)

B_smoothed = moving_average(B_train, window_size)
Bx_smoothed = moving_average(Bx_train, window_size)
By_smoothed = moving_average(By_train, window_size)
Bz_smoothed = moving_average(Bz_train, window_size)
Vx_smoothed = moving_average(Vx_train, window_size)
Vy_smoothed = moving_average(Vy_train, window_size)
Vz_smoothed = moving_average(Vz_train, window_size)

# %%
print(omni_data.dtype.names)

# %%


# n = np.array(n_interp, dtype=complex)[:Jpar_downsampled.shape[0]]

# Should redefine coordinates so Vx is always positive for calculating funcs like E_WAV_sqrt
v = -Vx_smoothed

"""
Fit some sort of polynomial to the smoothed n points before and after a nan interval
removes nan intervals, however can not be implemented for active periods as the 
Jpar varies too much.
"""

# %%

# Commented out coupling functions are commented out because I am missing
# density data
reconnection_voltage = Milan_coupling(By_smoothed, Bz_smoothed, Vx_smoothed)

theta_c = np.arctan2(By_smoothed, Bz_smoothed)

sin_squared = np.sin(theta_c/2)**2
sin_4th = np.sin(theta_c/2)**4

Bs = Bz_smoothed.copy()
Bs[Bs > 0] = 0

HWR = v * Bs
HWR[HWR == -0] = 0

Bs_delayed = delay_control_data(Bs, 7, 1)

# Check B_T vs B_mag
B_T = np.sqrt(Bx_smoothed.real**2 + By_smoothed**2 + Bz_smoothed**2)
B = Bz_smoothed

# p = n * v**2/2

epsilon_1 = v * B**2 * sin_4th
epsilon_2 = v * B_T**2 * sin_4th
epsilon_3 = v * B * sin_4th

sw_e_field = v * B_T

E_KL = v * B_T * sin_squared
E_KL_sqrt = np.sqrt(E_KL)

# E_KLV = v**(4/3) * B_T * sin_squared * p**(1/6)

E_WAV = v * B_T * sin_4th
E_WAV_squared = E_WAV**2
E_WAV_sqrt = np.sqrt(E_WAV)

# E_WV = v**(4/3) * B_T * sin_4th * p**(1/6)

# E_SR = v * B_T * sin_4th * p**(1/2)

# E_TL = n**(1/2) * v**2 * B_T * np.sin(theta_c/2)**6

"""
Clip the coupling functions, histogram and clip as many sigmas as necessary

Also try smoothing coupling functions
"""

# %%


def m_m_scaling(data):
    minimum = np.nanmin(data)
    maximum = np.nanmax(data)

    norm = (data - minimum)/(maximum - minimum)
    return norm


plt.plot(Vx)
plt.show()
plt.plot(mean_norm(Vx))
plt.show()
plt.plot(m_m_scaling(Vx))
plt.show()

plt.plot(Vz)
plt.show()
plt.plot(mean_norm(Vz))
plt.show()
plt.plot(m_m_scaling(Vz))
plt.show()

# %%
"""
Old SINDy definitions, for raw current points without integration

Keep just in case

DO NOT RUN

"""
# Define SINDY model parameters
dt = 4

my_library = ps.CustomLibrary([lambda x: np.sin(x),  # lambda x, y: np.sin(x + y),
                               lambda x: np.exp(x), lambda x: 1/(1e-6 + x)])

"""
3 timesteps of the full system needs more than 1.92 TiB (1.1TB) RAM to model.
"""
training_start = train_start
training_end = train_end

pos_index = 1130  # Which position to attempt to model
# (time, features) MUST BE (m, n), n > 0 NOT (m, )
J_11 = Jpar_downsampled[training_start:training_end, pos_index]
# t = np.arange(training_start * 10, training_end * 10,  10)

# The closest measured currents to the "main" (attempted) modelled current, main current = J_11
J_21 = Jpar_downsampled[training_start:training_end, pos_index + 1]  # 1 down
J_01 = Jpar_downsampled[training_start:training_end, pos_index - 1]  # 1 up
J_12 = Jpar_downsampled[training_start:training_end,
                        pos_index + 50]  # 1 to the right
J_10 = Jpar_downsampled[training_start:training_end,
                        pos_index - 50]  # 1 to the left
# Diagonals:
J_20 = Jpar_downsampled[training_start:training_end, pos_index - 50 - 1]
J_00 = Jpar_downsampled[training_start:training_end, pos_index - 50 + 1]
J_22 = Jpar_downsampled[training_start:training_end, pos_index + 50 - 1]
J_02 = Jpar_downsampled[training_start:training_end, pos_index + 50 + 1]


min_length = 200
# clean_start, clean_end, clean_len = find_overlapping_clean_data([Jpar_downsampled[:, pos_index], Bs, v],
#                                                                min_length = min_length, print_=False)


cleaned_subset_data = subset_data([J_01, J_11, J_21, E_SR.real, E_WAV.real], min_length=min_length,
                                  overlap=1)

J_01_clean = cleaned_subset_data[0]
J_11_clean = cleaned_subset_data[1]
J_21_clean = cleaned_subset_data[2]
Bs_clean = cleaned_subset_data[3]
v_clean = cleaned_subset_data[4]

X = []

for i in range(len(J_11_clean)):
    features = np.vstack((J_01_clean[i], J_11_clean[i], J_21_clean[i])).T

    X.append(features)

# +50 seems to have the greatest positive effect on model
# +50 = one step "to the right", +1 = one step "down"

u = []
for i in range(len(Bs_clean)):
    inputs = np.vstack((Bs_clean[i], v_clean[i])).T

    u.append(inputs)

# Delay the input data
u_delayed = u  # delay_control_data(u, nr_of_delays = 1, delay_indexes = 1)
# %%

# IS NOT SMOOTHED OR NORMED:
training_start = train_start
training_end = train_end - 14
pos_index = 1130  # Which position to attempt to model
# (time, features) MUST BE (m, n), n > 0 NOT (m, )
J_11 = Jpar_downsampled[training_start:training_end, pos_index]

# The closest measured currents to the "main" (attempted) modelled current, main current = J_11
J_21 = Jpar_downsampled[training_start:training_end, pos_index + 1]  # 1 down
J_01 = Jpar_downsampled[training_start:training_end, pos_index - 1]  # 1 up
J_12 = Jpar_downsampled[training_start:training_end,
                        pos_index + 50]  # 1 to the right
J_10 = Jpar_downsampled[training_start:training_end,
                        pos_index - 50]  # 1 to the left
# Diagonals:
J_20 = Jpar_downsampled[training_start:training_end, pos_index - 50 - 1]
J_00 = Jpar_downsampled[training_start:training_end, pos_index - 50 + 1]
J_22 = Jpar_downsampled[training_start:training_end, pos_index + 50 - 1]
J_02 = Jpar_downsampled[training_start:training_end, pos_index + 50 + 1]
#####


def replace_nans(arrays, replace_with):
    """
    Replace NaNs in all arrays with a specified value. If a NaN exists in any array,
    the corresponding index in all arrays will be replaced with the specified value.

    Parameters:
    -----------
    arrays : list of numpy arrays
        List of arrays to process (all should have the same length).
    replace_with : scalar
        Value to replace NaNs with.

    Returns:
    --------
    new_arrays_list : list of numpy arrays
        List of arrays with NaNs replaced.
    """
    # Check that all arrays have the same length
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All arrays must have the same length")

    array_length = lengths[0]

    # Create a combined mask where True means there is a NaN in any array at that position
    combined_nan_mask = np.zeros(array_length, dtype=bool)
    for arr in arrays:
        combined_nan_mask |= np.isnan(arr)

    # Replace values in all arrays based on the combined mask
    new_arrays_list = []
    for arr in arrays:
        new_array = arr.copy()
        new_array[combined_nan_mask] = replace_with
        # Append the modified array to the list
        new_arrays_list.append(new_array)

    return new_arrays_list


candidate_arrays = [AE_smoothed, AL_smoothed,
                    Bz_smoothed,
                    
                   
                    ]

"""
candidate_arrays = []

for pos_index in range(Jpar_downsampled.shape[1]):
    candidate_arrays.append(Jpar_downsampled[:training_end, pos_index])

candidate_arrays.append(By_smoothed)
candidate_arrays.append(Vx_smoothed)
candidate_arrays.append(Vz_smoothed)
"""
nr_of_X = 1

multiple_trajectories = True
if multiple_trajectories:

    traj_len = 10000

    split_data = subset_data(candidate_arrays,
                             traj_len, 1, variable_length_allowed=False)
    """
                              Best score so far:
                              Vx_smoothed,
                              B_smoothed, Bz_smoothed,
                               Vz_smoothed,
                              ],
                             traj_len, 1, variable_length_allowed=False)
    """
    
    print(f"The number of trajectories are: {len(split_data[0])}")

    for i in range(len(split_data)):
        plt.plot(split_data[i][0], label=f"Input {i + 1}")
    plt.title("First trajectory of all inputs")
    plt.ylim(-1, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

if not multiple_trajectories:
    
    split_data = replace_nans(candidate_arrays, replace_with=0)
    
    print(f"The length of the trajectory is: {len(split_data[0])}")

    for i in range(len(split_data)):
        plt.plot(split_data[i][0:traj_len], label=f"Input {i + 1}")
    plt.title("First trajectory of all inputs")
    plt.ylim(-1, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
X, T = np.meshgrid(x, t, indexing = "ij")
XT = np.transpose([X, T])

spatiotemporal_grid = XT

pySINDy does not currently support different spatiotemporal grids for trajectories
2 solutions (from github):
    1. Interpolate trajectories over to a common temporal grid
    2. Cut the trajectories down to the same length and define them as being on 
    the same temporal grid
"""

# SINDyCP uses ParametrizedLibrary, to create Theta(X, U) = Theta_feat(X) x Theta_par(U)
# Can be combined with weak formalized SINDy. Weak formulation can use WeakPDELibrary,
# Otherwise I must construct the system rows by projecting data onto weak samples.
# w_ik^v = \int_Omega_k theta(x;t) X^v(x;t) d^D x dt eq. 5 in SINDyCP paper

if nr_of_X > 1:
    X = []
    u = []
    for i in range(len(split_data[0])):  # For each trajectory
        # Stacks the control inputs for each trajectory
        Xs = np.vstack([split_data[j][i] for j in range(0, nr_of_X)]).T

        inputs = np.vstack([split_data[j][i]
                           for j in range(nr_of_X, len(split_data))]).T

        # Appends the control inputs for each trajectory to a list containing all trajectories
        X.append(Xs)
        u.append(inputs)

else:
    X = split_data[0]
    u = []
    if multiple_trajectories:
        for i in range(len(split_data[0])):  # For each trajectory
            # Stacks the control inputs for each trajectory
            inputs = np.vstack([split_data[j][i]
                               for j in range(1, len(split_data))]).T

            # Appends the control inputs for each trajectory to a list containing all trajectories
            u.append(inputs)

    else:

        # Stacks the control inputs
        u = np.vstack([split_data[j] for j in range(1, len(split_data))]).T


u_delayed = u  # Legacy, keep incase want to add delays to the inputs again

dt = 2

# To to do a north-south line of 3 points
spatial_grid = np.arange(0, 2, step=1)

if multiple_trajectories:
    temporal_grid = np.arange(0, len(X[0]) * dt, step=1 * dt)
else:
    temporal_grid = np.arange(0, len(X) * dt, step=1 * dt)

spatial, temporal = np.meshgrid(spatial_grid, temporal_grid, indexing="ij")

spatio_temporal_grid = np.stack((spatial, temporal), axis=-1)

print("Shape of spatio_temporal_grid:", spatio_temporal_grid.shape)

optimizer = ps.EnsembleOptimizer(opt=ps.SR3(  # reg_weight_lam= 0.03,relax_coeff_nu=5,
    regularizer="L2"),
    bagging=True, library_ensemble=True,
    n_models=10000)  # Default aggregator is median

feature_names = None

# Finite difference amplifies noise in data.
# smoother_kws={'window_length': 20})
differentiation_method = ps.SmoothedFiniteDifference()

# H_xt = np.array([1, 2])  # Adjust these values if necessary
# Only temporal grid gives dx/dt = 1/2 dx/dt + 1/2 dx/dt, R² = 1
lib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=3),
                        spatiotemporal_grid=temporal_grid,
                        derivative_order=0,
                        include_bias=True, include_interaction=True)

# Various attempted libraries
# input_lib = ps.CustomLibrary()

combined_lib = ps.GeneralizedLibrary(libraries=[ps.PolynomialLibrary(), ps.FourierLibrary(), lib],
                                     )  # tensor_array = [[1, 1]])

param_lib = ps.ParameterizedLibrary(feature_library=lib, parameter_library=lib,
                                    num_features=1, num_parameters=3)

# Initialize SINDy model
mod = ps.SINDy(optimizer=optimizer,
               feature_library= lib,
               differentiation_method=differentiation_method)

train_trajectories = 14
# Fit SINDy model
mod.fit(x=X[:train_trajectories], t=dt, u=u_delayed[:train_trajectories])

# Print the best fit approximation
mod.print()

print(f"R^2 score on training data : {mod.score(X[:train_trajectories], dt, u=u_delayed[:train_trajectories])}")  # R² score of model
print(f"R^2 score on validation data : {mod.score(X[train_trajectories:], dt, u=u_delayed[train_trajectories:])}")


# %%

def read_and_int_validation_Jpar():
    Jpar, cLat_deg, lon_deg, missing_days, missing_indices = read_Jpar(from_year_index=11,
                                                                       nr_days=365 * 3,
                                                                       directory_path=AMPERE_PATH)
    Jpar_downsampled = Jpar
    cLat_deg_downsampled = cLat_deg
    lon_deg_downsampled = lon_deg

    interp_length = 10
    want_coordinates = True
    # If 1 column contains a nan, every column contains nan at the same place
    # find_nans() is bottlenecking, vectorize it at some point
    for column in range(Jpar_downsampled.shape[1]):
        Jpar_downsampled[:, column], nan_start_Jpar, nan_end_Jpar, nan_lengths_Jpar, no_nan_lengths_Jpar = interpolate_nans(Jpar_downsampled[:, column],
                                                                                                                            max_nan_length=interp_length)

        if want_coordinates:
            cLat_deg_downsampled[:, column], _, _, _, _ = interpolate_nans(cLat_deg_downsampled[:, column],
                                                                           max_nan_length=interp_length)
            lon_deg_downsampled[:, column],  _, _, _, _ = interpolate_nans(lon_deg_downsampled[:, column],
                                                                           max_nan_length=interp_length)
    fin = -1
    integrated_out_Jpar, integrated_in_Jpar = integrate_FACs(Jpar_downsampled[:fin],
                                                             cLat_deg_downsampled[:fin], lon_deg_downsampled[:fin])

    integrated_out_Jpar = integrated_out_Jpar.flatten()
    integrated_in_Jpar = integrated_in_Jpar.flatten()

    norm_out = mean_norm(integrated_out_Jpar)
    norm_in = mean_norm(integrated_in_Jpar)

    window_size = 15

    smoothed_out = moving_average(norm_out.flatten(), window_size)
    smoothed_in = moving_average(norm_in.flatten(), window_size)

    # Reads in OMNI data with a time resolution of 1 minute
    # DOES NOT HAVE TO BE READ IN MORE THAN ONCE PER KERNEL
    omni_data = pod.GetOMNI([2020, 2023], Res=1)

    AE = omni_data.AE
    AL = omni_data.AL
    AU = omni_data.AU
    B = omni_data.B
    Bx = omni_data.BxGSE
    By = omni_data.ByGSE
    Bz = omni_data.BzGSE
    Vx = omni_data.Vx
    Vy = omni_data.Vy
    Vz = omni_data.Vz

    interp_len = 15

    AE_interp = interpolate_nans(AE, interp_len)
    AL_interp = interpolate_nans(AL, interp_len)
    AU_interp = interpolate_nans(AU, interp_len)

    B_interp = interpolate_nans(B, interp_len)
    Bx_interp = interpolate_nans(Bx, interp_len)
    By_interp = interpolate_nans(By, interp_len)
    Bz_interp = interpolate_nans(Bz, interp_len)
    Vx_interp = interpolate_nans(Vx, interp_len)
    Vy_interp = interpolate_nans(Vy, interp_len)
    Vz_interp = interpolate_nans(Vz, interp_len)

    train_start = 0
    train_end = len(Jpar) - 1

    # AU Does not contain NaNs
    AE_train = AE_interp[0][::2][train_start:train_end]
    AL_train = AL_interp[0][::2][train_start:train_end]
    AU_train = AU_interp[0][::2][train_start:train_end]

    B_train = B_interp[0][::2][train_start:train_end]
    Bx_train = Bx_interp[0][::2][train_start:train_end]
    By_train = By_interp[0][::2][train_start:train_end]
    Bz_train = Bz_interp[0][::2][train_start:train_end]
    Vx_train = Vx_interp[0][::2][train_start:train_end]
    Vy_train = Vy_interp[0][::2][train_start:train_end]
    Vz_train = Vz_interp[0][::2][train_start:train_end]

    AE_train = mean_norm(AE_train)
    AL_train = mean_norm(AL_train)
    AU_train = mean_norm(AU_train)

    B_train = mean_norm(B_train)
    Bx_train = mean_norm(Bx_train)
    By_train = mean_norm(By_train)
    Bz_train = mean_norm(Bz_train)
    Vx_train = mean_norm(Vx_train)
    Vy_train = mean_norm(Vy_train)
    Vz_train = mean_norm(Vz_train)

    AE_smoothed = moving_average(AE_train, window_size)
    AL_smoothed = moving_average(AL_train, window_size)
    AU_smoothed = moving_average(AU_train, window_size)

    B_smoothed = moving_average(B_train, window_size)
    Bx_smoothed = moving_average(Bx_train, window_size)
    By_smoothed = moving_average(By_train, window_size)
    Bz_smoothed = moving_average(Bz_train, window_size)
    Vx_smoothed = moving_average(Vx_train, window_size)
    Vy_smoothed = moving_average(Vy_train, window_size)
    Vz_smoothed = moving_average(Vz_train, window_size)

    return smoothed_out, smoothed_in, B_smoothed, Bx_smoothed, By_smoothed, Bz_smoothed, Vx_smoothed, Vy_smoothed, Vz_smoothed


data_validation = read_and_int_validation_Jpar()

# %%

test = read_Jpar(from_year_index=11,
                 nr_days=365,
                 directory_path=AMPERE_PATH)

# %%
Jpar_validation = test[0]
cLat_deg_validation = test[1]
lon_deg_validation = test[2]
Jpar_validation[Jpar_validation > 1e2] = np.nan
cLat_deg_validation[Jpar_validation > 1e2] = np.nan
lon_deg_validation[Jpar_validation > 1e2] = np.nan
Jpar_validation[Jpar_validation < -1e2] = np.nan
cLat_deg_validation[Jpar_validation < -1e2] = np.nan
lon_deg_validation[Jpar_validation < -1e2] = np.nan


fin = -1
integrated_out_Jpar_validation, integrated_in_Jpar_validation = integrate_FACs(Jpar_validation[:fin],
                                                                               cLat_deg_validation[:fin], lon_deg_validation[:fin])

integrated_out_Jpar_validation = integrated_out_Jpar_validation.flatten()
integrated_in_Jpar_validation = integrated_in_Jpar_validation.flatten()

plt.plot(integrated_out_Jpar_validation)
plt.plot(integrated_in_Jpar_validation)
plt.show()


norm_out_validation = mean_norm(integrated_out_Jpar_validation)
norm_in_validation = mean_norm(integrated_in_Jpar_validation)

plt.plot(norm_out_validation)
plt.plot(norm_in_validation, alpha=0.5)
plt.show()


window_size = 15

smoothed_out_validation = moving_average(
    norm_out_validation.flatten(), window_size)
smoothed_in_validation = moving_average(
    norm_in_validation.flatten(), window_size)

# %%
valid_start = 0
valid_end = len(smoothed_out_validation)
# %%
"""
Should maybe change code to interpolate, then find overlapping data and then smooth? Maybe

"""
validation = subset_data([test[0][valid_start:valid_end, 0], data_validation[6][valid_start:valid_end],
                          data_validation[8][valid_start:valid_end], data_validation[3][valid_start:valid_end]], traj_len, 1)


# %%

candidate_validation_arrays = [smoothed_out_validation[valid_start:valid_end], data_validation[6][valid_start:valid_end],
                               data_validation[8][valid_start:valid_end], data_validation[3][valid_start:valid_end]]


if multiple_trajectories:

    traj_len_validation = traj_len  # int(traj_len/2)

    split_data_validation = subset_data(candidate_validation_arrays,
                                        traj_len_validation, 1, variable_length_allowed=False)


if nr_of_X > 1:
    X_validation = []
    u_validation = []
    for i in range(len(split_data_validation[0])):  # For each trajectory
        # Stacks the control inputs for each trajectory
        Xs_validation = np.vstack([split_data_validation[j][i]
                                  for j in range(0, nr_of_X)]).T

        inputs_validation = np.vstack([split_data_validation[j][i] for j in range(
            nr_of_X, len(split_data_validation))]).T

        # Appends the control inputs for each trajectory to a list containing all trajectories
        X_validation.append(Xs_validation)
        u_validation.append(inputs_validation)

else:
    X_validation = split_data_validation[0]
    u_validation = []
    if multiple_trajectories:
        for i in range(len(split_data_validation[0])):  # For each trajectory
            # Stacks the control inputs for each trajectory
            inputs_validation = np.vstack(
                [split_data_validation[j][i] for j in range(1, len(split_data_validation))]).T

            # Appends the control inputs for each trajectory to a list containing all trajectories
            u_validation.append(inputs_validation)

    else:

        # Stacks the control inputs
        u_validation = np.vstack([split_data_validation[j]
                                 for j in range(1, len(split_data_validation))]).T

print(mod.score(X_validation, dt, u=u_validation))  # R² score of model

# Create the second figure for original data
plt.figure(figsize=(12, 8))
# Subplot for X[0]
plt.subplot(2, 1, 1)
plt.plot(X[0], label='X[0]')
plt.title('Original Data: X')
plt.legend()
# Subplot for u_delayed[0]
plt.subplot(2, 1, 2)
plt.plot(u_delayed[0], label='u_delayed[0]')
plt.title('Original Data: u')
plt.legend()
plt.tight_layout()
plt.show()
# Create the first figure for validation data
plt.figure(figsize=(12, 8))
# Subplot for X_validation[0]
plt.subplot(2, 1, 1)
plt.plot(X_validation[0], label='X_validation[0]')
plt.title('Validation Data: X')
plt.legend()
# Subplot for u_validation[0]
plt.subplot(2, 1, 2)
plt.plot(u_validation[0], label='u_validation[0]')
plt.title('Validation Data: u')
plt.legend()
plt.tight_layout()
plt.show()

# %%

pred_train = mod.predict(X, u_delayed)
pred_validation = mod.predict(X_validation, u_validation)

# Create the first figure for validation data
plt.figure(figsize=(12, 8))
# Subplot for X_validation[0]
plt.subplot(2, 1, 1)
plt.plot(X[0], label='X')
plt.plot(pred_train[0], alpha=0.5, ls="--")
plt.title('Real Data: X')
plt.xlim(0, 100)
plt.legend()
# Subplot for u_validation[0]
plt.subplot(2, 1, 2)
plt.plot(X_validation[0], label='X validation')
plt.plot(pred_validation[0], alpha=0.5, ls="--")
plt.title('Validation Data: X')
plt.legend()
plt.tight_layout()
plt.xlim(0, 100)
plt.show()

# %%

# %%
t_sim = np.arange(0, 100, 1)
u_sim = u_delayed[0][:len(t_sim)]
x0_sim = [X[0][0]]
sim_train = mod.simulate(x0_sim, t=t_sim, u=u_sim)


plt.figure(figsize=(12, 8))
# Subplot for X_validation[0]
plt.subplot(2, 1, 1)
plt.plot(X[0], label='X')
plt.plot(sim_train[0], alpha=0.5, ls="--")
plt.title('Real Data: X')
plt.xlim(0, 100)
plt.legend()

# %%

print(u_sim.shape)
print(t_sim.shape)

# %%


def calc_rhs(arrays):
    x0 = arrays[0]
    u0 = arrays[1]
    u1 = arrays[2]
    u2 = arrays[3]
    u3 = arrays[4]

    rhs = -0.005 * x0 + 0.001 * u2 - 0.002 * x0**2 - 0.001 * x0 * u2 + 0.001 * x0 * u3 - 0.007 * u2**2 + 0.004 * x0**3 + \
        0.005 * x0**2 * u2 - 0.001 * x0**2 * u3 + 0.001 * x0 * u0 * \
        u2 + 0.019 * x0 * u2**2 + 0.009 * u2**3 - 0.001 * u2**2 * u3

    return rhs


begin = 0
to = 1000
rhs = calc_rhs([split_data[0][0][begin:to], split_data[1][0][begin:to],
               split_data[2][0][begin:to], split_data[3][0][begin:to],
               split_data[4][0][begin:to]])

sol = solve_ivp(rhs, )

plt.plot(rhs)
plt.plot(mod.fit.x_dot)
plt.show()
# %%


def test():

    # Define the variables
    x0, u0, u2, u3 = symbols('x0 u0 u2 u3')
    # Define the right-hand side (rhs) equation
    rhs = (-0.005 * x0 + 0.001 * u2 - 0.002 * x0**2 - 0.001 * x0 * u2 + 0.001 * x0 * u3
           - 0.007 * u2**2 + 0.004 * x0**3 + 0.005 * x0**2 * u2 - 0.001 * x0**2 * u3
           + 0.001 * x0 * u0 * u2 + 0.019 * x0 * u2**2 + 0.009 * u2**3 - 0.001 * u2**2 * u3)
    # Set the equation to 0 (assuming you want to solve rhs = 0)
    equation = Eq(rhs, 0)
    # Solve for one variable (e.g., x0)
    solution = solve(equation, x0)
    # Print the solution
    print("Solution for x0:")
    print(solution)
    return


test()
# %%

"""
model.simulate() is EXTREMELY slow. 17 minutes to predict 100 timesteps.
Only run if necessary/for a good fit.

predict() is apparently quick, however only returns x_dot.
"""

sim_length = 100
t_sim = np.arange(0, sim_length, step=1 * dt)
u_sim = u_delayed[0][:len(t_sim), :]
X_initial = np.array([X[0][0]])

# print(f"X_inital's length : {X.shape}")
print(f"t_sim shape : {t_sim.shape}")
# print(f"u_sim shape : {u_sim.shape}")


pred = mod.simulate(X_initial, t_sim, u=u_sim)

# %%
print(f"Indices in WeakPDELibrary: {lib.inds_k}")

# %%
saved_mod = mod
saved_mod.print()
# %%

mod = saved_mod
mod.print()
# %%

plt.plot(x, "b", label="True x")
plt.plot(pred, "r--", label="SINDy x")
plt.ylim(-1, 1)
# plt.xlim(3900, 4100)
plt.legend()
plt.show()

print(f" R² score: {mod.score(x, t, u=u[:, :len(t)])}")

# %%
print(mod.n_features_in_)
print(mod.n_output_features_)
print(mod.n_control_features_)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1st measurement
axes[0, 0].plot(t, x[:, 0], 'b', label='True x')
axes[0, 0].plot(t[:-1], pred[:, 0], 'r--', label='SINDy x')
axes[0, 0].set_ylabel('Jpar')
# axes[0, 0].set_xlim([-2.5, 2.5])
axes[0, 0].legend()
plt.show()

if x.shape[1] > 1:
    # Plot 2nd measurement
    axes[1, 0].plot(t, x[:, 1], 'b', label='True y')
    axes[1, 0].plot(t[:-1], pred[:, 1], 'r--', label='SINDy y')
    axes[1, 0].set_ylabel('Jpar')
    axes[1, 0].set_xlabel('Minutes')
    # axes[1, 0].set_ylim([-2.5, 2.5])
    axes[1, 0].legend()

# Plot error (publicity)
axes[0, 1].plot(t[:-1], x[:-1, 0] - pred[:, 0], 'b', label='Meas 1 error')
if x.shape[1] > 1:
    axes[0, 1].plot(t[:-1], x[:-1, 1] - pred[:, 1], 'r', label='Meas 2 error')
axes[0, 1].set_xlabel('Minutes')
axes[0, 1].set_ylabel('Error')
axes[0, 1].set_ylim([-1, 1])
axes[0, 1].legend()

# Plot error (true)
axes[1, 1].plot(t[:-1], x[:-1, 0] - pred[:, 0], 'b', label='Meas 1 error')
if x.shape[1] > 1:
    axes[1, 1].plot(t[:-1], x[:-1, 1] - pred[:, 1], 'r', label='Meas 2 error')
axes[1, 1].set_xlabel('Minutes')
axes[1, 1].set_ylabel('Error')
# axes[1, 1].set_ylim([-2.5, 2.5])
axes[1, 1].legend()

plt.tight_layout()
plt.show()
print("Plotted!")


# %%

time_index = 100
pos_index1 = 52
pos_index2 = 54

longitudes = lon_deg
latitudes = 90 - cLat_deg
central_longitude = 0

longitudes = longitudes[time_index, pos_index1:pos_index2]
latitudes = latitudes[time_index, pos_index1:pos_index2]

# Create a scatter plot of Jpar over the northern hemisphere
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.NorthPolarStereo(
    central_longitude=central_longitude))
# Northern Hemisphere
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
# Add features to the map
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.gridlines(draw_labels=True)
# Scatter plot of Jpar
sc = ax.scatter(longitudes, latitudes, c=Jpar[time_index, pos_index1:pos_index2], cmap='coolwarm', s=60,
                transform=ccrs.PlateCarree())
cbar = plt.colorbar(sc, orientation='vertical')
cbar.ax.tick_params(labelsize=fontsize*1)
cbar.set_label('Radial Current Density (μA/m²)', fontsize=fontsize*1.5)
"""
# Add Magnetic Local Time (MLT) labels
mlt_labels = {6: (mlt_longitudes[6], 65), 12: (mlt_longitudes[12], 65), 
              18: (mlt_longitudes[18], 65), 24: (mlt_longitudes[24], 65)}
for mlt, (lon, lat) in mlt_labels.items():
    ax.text(lon, lat, f'{mlt} MLT', transform=ccrs.PlateCarree(),
            fontsize=15, color='black', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
"""
# Title and show plot
plt.title('Average error in reconstructed data', fontsize=fontsize*1.5)
plt.show()
