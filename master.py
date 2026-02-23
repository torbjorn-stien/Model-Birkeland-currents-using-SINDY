#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:26:41 2026

@author: tos
"""

import pysindy as ps

import numpy as np
import matplotlib.pyplot as plt
from read_ampere_ncdf import read_ampere_ncdf
from pathlib import Path
import re
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import pandas as pd
from read_ace_files import read_dat
from sklearn.metrics import r2_score
from matplotlib.animation import FuncAnimation
import math
from itertools import groupby


"""
Look at solar zenith angle to see if I can put that in as a driver aswell.
May or may not be easy for DMDc, but at least remember for SINDY
Will have to use GEO to calculate solar zenith.
-> Have to look at the grids/points and if they "exist" at the same point for 
both the AAGCM and GEO arrays. (e.g. plot them on/adjacent to each other)
They more than likely do
"""

AMPERE_PATH = "/nfs/revontuli/data/bjorn/Ampere"
IMF_PATH = "/nfs/revontuli/data/bjorn/ACE/B_IMF"

amp_root = Path(AMPERE_PATH)
imf_root = Path(IMF_PATH)

fontsize = 20

# 2009 has incomplete data
def read_Jpar(from_year_index, nr_days):
    """
    Reads Ampere data, handles missing data and returns the Birkeland current
    data with nan where there is missing data.

    Parameters
    ----------
    from_year_index : int
        index of the year to start reading from
    nr_days : int
        number of days of data to read 

    Returns
    -------
    Jpar : np.array
        Array of Birkeland currents for 1200 positions each dt
    geo_clat_deg : np.array
        DESCRIPTION.
    geo_lon_deg : TYPE
        DESCRIPTION.
    missing_full_days : TYPE
        DESCRIPTION.
    missing_indices : TYPE
        DESCRIPTION.

    """
    # Reading in Ampere data:
    year_dirs = [d for d in amp_root.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}", d.name)]
    
    #dB_Naagcm_list = []
    #dB_Eaagcm_list = []
    Jpar_list = []
    geo_cLat_list = []
    geo_lon_deg_list = []
    missing_dates = []
    missing_points = []
    missing_indices = []
    
    Jpar = np.zeros((nr_days * 720, 1200))
    
    
    """
    read_ampere function breaks for day 22 in 2012. Seems to work for all of 2010-2011
    aswell as 2013+
    
    2009 is missing a lot of days
    """
    i = 0 # Counts how many days have passed
    stop_val =  nr_days # Number of days to process
    prev_date = None
    for year in sorted(year_dirs, key=lambda p: int(p.name))[from_year_index:]: # Filter out 2009, by using [1:]
        print(f"Year: {year.name}")
        
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            
            for zip_path in sorted(year.glob("*north.grd.zip")):
                #print(f"  Zip: {zip_path.name}")
                
                with ZipFile(zip_path) as zf:
                    members = [m for m in zf.namelist() if m.lower().endswith((".nc", ".ncdf"))]
                    
                    for member in sorted(members):
                        extracted_path = tmpdir / member
                        extracted_path.parent.mkdir(parents=True, exist_ok=True)
                        zf.extract(member, path=tmpdir)
                        print(f"    Extracted: {member} -> {extracted_path}")
                        
                        date = re.search(r"(\d{4})(\d{2})(\d{2})", member)
                        
                        # Some days are missing within the dataset(s), this returns the missing dates
                        if prev_date:
                            # Checks wether there are any days missing
                            day = int(date.group(3))
                            prev_day = int(prev_date.group(3))
                            
                            diff = day - prev_day
                            
                            # If there are days missing, returns the dates and indices
                            while diff > 1:
                                diff = diff - 1
                                missing_day = day - diff
                                
                                month = int(date.group(2))
                                year = int(date.group(1))
                                
                                missing_date = int(f"{year:04d}{month:02d}{missing_day:02d}")
                                missing_dates.append(missing_date)
                                
                                # returns the indices the missing days would have occupied if they were not missing
                                start_indice = missing_day * 720
                                end_indice = start_indice + 720 
                                
                                indices_for_missing_day = np.arange(start_indice, end_indice, 1) 
                                missing_indices.extend(indices_for_missing_day)
                                
                                Jpar[start_indice:end_indice] = np.nan
                                
                                i += 1
                            
                        # Pass the full path as a string
                        data = read_ampere_ncdf(str(extracted_path), OutVars="J")
                        
                        # Checks if each day contains the correct amount of time-points
                        # (720 pr. day for full dataset, 360 for downsampled dataset)
                        if len(data["Jpar"]) != 720:
                            expected_points = np.arange(start = 0.0, stop = 24, step = 24/720)
                            actual_points = np.array(data["time"])
                            missing_indices_this_day = []
                            
                            for idx, point in enumerate(expected_points):
                                if not np.any(np.isclose(point, actual_points, atol=1e-4)):
                                    # Shifts the indices forward to the day they are missing from,
                                    # also shifts them forward by the missing full days
                                    missing_indice = idx + i * 720
                                    missing_points.append(point)
                                    missing_indices.append(missing_indice)
                                    missing_indices_this_day.append(idx)
                                    
                            
                            start_indice = i * 720
                            end_indice = start_indice + 720
                            
                            actual_indices = np.delete(np.arange(0, 720, 1), 
                                                       np.array(missing_indices_this_day))
                            
                            full_day_data = np.full((720, 1200), np.nan)
                            full_day_data[actual_indices] = np.array(data["Jpar"])
                        
                        else:
                            full_day_data = np.array(data["Jpar"])

                        start_indice = i * 720
                        end_indice = start_indice + 720
                        
                        Jpar[start_indice:end_indice] = full_day_data
                        geo_cLat_list.append(data["geo_cLat_deg"])
                        geo_lon_deg_list.append(data["geo_lon_deg"])
                        #dB_Naagcm_list.append(data["dB_Ngeo"])
                        #dB_Eaagcm_list.append(data["dB_Egeo"])
                        
                        prev_date = date
                        i += 1
                        
                        if i == stop_val:
                            break
                    if i == stop_val:
                        break
                if i == stop_val:
                    break
            if i == stop_val:
                break
        if i == stop_val:
            break
    
    #Jpar = np.concatenate(Jpar_list, axis=0)
    geo_clat_deg = np.concatenate(geo_cLat_list, axis = 0)
    geo_lon_deg = np.concatenate(geo_lon_deg_list, axis = 0)               
    #dB_Naagcm_all = np.concatenate(dB_Naagcm_list, axis=0)
    #dB_Eaagcm_all = np.concatenate(dB_Eaagcm_list, axis=0)
    #print("Final shapes:", dB_Naagcm_all.shape, dB_Eaagcm_all.shape
    
    missing_full_days = np.array(missing_dates)
    missing_indices = np.array(missing_indices)
    
    print("Final shapes:", Jpar.shape, geo_clat_deg.shape, geo_lon_deg.shape)#%%
    print(f"There are {len(missing_dates)} full days missing and an additional {len(missing_indices)-(len(missing_dates) * 720)} missing indices")
    
    return Jpar, geo_clat_deg, geo_lon_deg, missing_full_days, missing_indices

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
        file_path = os.path.join(directory, filename)  # Construct the full file path
        print(f"Reading file: {filename}")  # Print the file being read
        df = read_dat(file_path)  # Read the file using function in separate file
        dataframes.append(df)  # Append the dataframe to the list
    
    # Combine all dataframes into one
    if dataframes:
        combined_data = pd.concat(dataframes, ignore_index=True)
        print("Done reading!")
        return combined_data
    else:
        print("No files found for the specified year range.")
        return pd.DataFrame()


def train_SINDY(input_dat, dt, training_start, training_end,
                feature_library, optimizer, feature_names, differentiation_method):
    """
    Parameters
    ----------
    input_dat : nparray
        measurements of system states.
    control_dat : TYPE
        DESCRIPTION.
    feature_library : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.

    Returns
    -------
    model : 

    """
    try:
        X = input_dat[training_start:training_end, :]
    except IndexError:
        X = input_dat[training_start:training_end]
    
    """
    model = ps.SINDy(
        differentiation_method = differentiation_method,
        feature_library = feature_library, #feature_names = feature_names
        optimizer = optimizer
        )
    """
    
    model = ps.SINDy(optimizer = optimizer, feature_library=feature_library,
                     differentiation_method=differentiation_method)
    model.fit(X, t = dt, feature_names = feature_names)
    
    model.print()
    
    return model

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
        L_eff = (3.8 * R_E * (float(Vx[i])/(4 * 10**5 ))**(1/3)).real
        
        phi_d[i] = L_eff * float(Vx[i]) * Byz[i] * np.sin(0.5 * theta[i])**(9/2) # eq 15
    
    
    #phi_d = Lambda * np.abs(Vx)**(4/3) * Byz * np.sin(1/2 * theta)**(9/2) # eq 14
    
    
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
    delayed_input = np.zeros((num_rows, num_features * num_delays)) # Initialize final matrix
    
    for i, delay in enumerate(delays):
        start_col = i * num_features # Start column for this delay
        end_col = start_col + num_features # End column for this delay
    
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


#%%
Jpar, cLat_deg, lon_deg, missing_days, missing_indices = read_Jpar(from_year_index = 5, nr_days = 365)
#%%
Jpar_downsampled = Jpar[::2]

# Finds the lengths of nan intervals and non-nan intervals
start_nan_indices = []
end_nan_indices = []

for i in range(len(Jpar_downsampled)):
    if np.isnan(Jpar_downsampled[i, 0]) and not np.isnan(Jpar_downsampled[i - 1, 0]):
        start_nan_indices.append(i)
    
    if not np.isnan(Jpar_downsampled[i, 0]) and np.isnan(Jpar_downsampled[i - 1, 0]):
        end_nan_indices.append(i)

start_nan_indices = np.array(start_nan_indices)
end_nan_indices = np.array(end_nan_indices)   

nan_lengths = end_nan_indices - start_nan_indices
no_nan_lengths = np.zeros_like(nan_lengths)

for i in range(len(nan_lengths)):
    try:
        no_nan_lengths[i] = start_nan_indices[i + 1] - start_nan_indices[i] - nan_lengths[i]
    except IndexError:
        break

print(np.sum(nan_lengths))
#%%
year_data = read_files(IMF_PATH, start_year=2013, end_year=2013)
year_data_interp = year_data.interpolate(method = "linear") #dt = 4 min 
#year_data = np.array(year_data_interp)

"""
2012, 2016 & 2020 were leap-years
"""

# Reading in alternate Solar Wind parameters (Vx, y, z) Hard coded for 2010
SW_data = pd.read_csv("ASC8YJ061", skiprows = 31, sep = "\s+", 
                      names=["Year", "day", "hour", "min", "sec", 
                             "VGSM_X", "VGSM_Y", "VGSM_Z"])
SW_data[SW_data == -9999.9] = np.nan

SW_data_interp = SW_data.interpolate(method = "linear")

SW_data_interp["datetime"] = pd.to_datetime(SW_data_interp["Year"].astype(str)) + pd.to_timedelta(SW_data_interp["day"] - 1, unit="D") \
    + pd.to_timedelta(SW_data_interp["hour"], unit="h") + pd.to_timedelta(SW_data_interp["min"], unit="m") \
        + pd.to_timedelta(SW_data_interp["sec"], unit="s")

SW_data_interp = SW_data_interp.drop(columns=["Year", "day", "hour", "min", "sec"])
SW_data_interp.set_index("datetime", inplace=True)

SW_dat_dow = SW_data_interp.resample("4min")
SW_dat_dow = SW_dat_dow.mean()


#%%


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
        L_eff = (3.8 * R_E * (float(Vx[i])/(4 * 10**5 ))**(1/3)).real
        
        phi_d[i] = L_eff * float(Vx[i]) * Byz[i] * np.sin(0.5 * theta[i])**(9/2) # eq 15
    
    
    #phi_d = Lambda * np.abs(Vx)**(4/3) * Byz * np.sin(1/2 * theta)**(9/2) # eq 14
    
    
    F_max = phi_d/c
    
    return F_max

nan_pos = np.where(np.isnan(Jpar_downsampled[:, 0]))[0]

rows_with_nan = np.where(np.isnan(Jpar_downsampled).any(axis=1))[0]


# Read in control data
Bx = np.array(year_data_interp["Bgsm_x"][:Jpar_downsampled.shape[0]])
By = np.array(year_data_interp["Bgsm_y"][:Jpar_downsampled.shape[0]])
Bz = np.array(year_data_interp["Bgsm_z"][:Jpar_downsampled.shape[0]])
Bz[Bz<-200] = 0

"""
Bx = np.delete(Bx, rows_with_nan)
By = np.delete(By, rows_with_nan)
Bz = np.delete(Bz, rows_with_nan)
"""

Vx = np.array(SW_dat_dow["VGSM_X"])[:Jpar_downsampled.shape[0]]

Vx = np.delete(Vx, nan_pos)

v = Vx

Jpar_downsampled = np.delete(Jpar_downsampled, rows_with_nan, axis = 0)

"""
Fit some sort of polynomial to the smoothed n points before and after a nan interval
removes nan intervals, however can not be implemented for active periods as the 
Jpar varies too much.
"""

#%%
reconnection_voltage = Milan_coupling(By, Bz, Vx)

theta_c = np.arctan(By, Bz)

sin_squared = np.sin(theta_c/2)**2
sin_4th = np.sin(theta_c/2)**4
#%%

Bs = Bz
Bs[Bs>0] = 0

HWR = v * Bs
HWR[HWR == -0] = 0

Bs_delayed = delay_control_data(Bs, 7, 1)

mult_ = 1
for i in range(7):
    mult_ *= Bs_delayed[:, i]

for i in range(len(mult_)):
    if mult_[i] >= 0:
        mult_[i - 7:i] = 0
    
plt.plot(mult_)
plt.show()

#%%
plt.plot(Bz)
plt.show()
#%%

plt.plot(Bs[:2000])
plt.show()



#%%
p = n * v**2/2

epsilon_1 = v * B**2 * sin_4th
epsilon_2 = v * B_T**2 * sin_4th
epsilon_3 = v * B * sin_4th

sw_e_field = v * B_T

E_KL = v * B_T * sin_squared
E_KL_sqrt = np.sqrt(E_KL)

E_KLV = v**(4/3) * B_T * sin_squared * p**(1/6)

E_WAV =  v * B_T * sin_4th
E_WAV_squared = E_WAV**2
E_WAV_sqrt = np.sqrt(E_WAV)

E_WV = v**(4/3) * B_T * sin_4th * p**(1/6)

E_SR = v * B_T * sin_4th * p**(1/2)

E_TL = n**(1/2) * v**2 * B_T * np.sin(theta_c/2)**6

#%%

print(f"Bx's shape: {Bx.shape}")

# Stack control data to the end of system measurements matrix
Theta = np.hstack((Jpar_downsampled, Bx[:, np.newaxis], By[:, np.newaxis], Bz[:, np.newaxis])) 
                   #Vx[:, np.newaxis]))

missing_index = int(missing_indices[0]/2)
#Theta = Theta[:missing_index]
print(f"Theta's shape: {Theta.shape}")

#%%

print(type(Theta))
print(np.any(np.isnan(Bz)))

plt.plot(Bz)
plt.show()
#%%

# Define SINDY model parameters
dt = 4

my_library = ps.CustomLibrary([lambda x: np.sin(x), #lambda x, y: np.sin(x + y),
                               lambda x: np.exp(x)])

# SINDyCP uses ParametrizedLibrary, to create Theta(X, U) = Theta_feat(X) x Theta_par(U) 
# Can be combined with weak formalized SINDy. Weak formulation can use WeakPDELibrary,
# Otherwise I must construct the system rows by projecting data onto weak samples.
# w_ik^v = \int_Omega_k theta(x;t) X^v(x;t) d^D x dt eq. 5 in SINDyCP paper

optimizer = ps.EnsembleOptimizer(opt=ps.STLSQ(threshold=0.010), 
                                 bagging=True, library_ensemble=True,
                                 n_models = 30) # Default aggregator is median

feature_names = None

# Finite difference amplifies noise in data.
differentiation_method = ps.FiniteDifference() 

"""
3 timesteps of the full system needs 1.92 TiB (1.1TB) RAM to model.
"""
training_start = 0
training_end = 26680

pos_index = 1130
x = Theta[training_start:training_end, pos_index] #(time, features) MUST BE (m, n), n > 0 NOT (m, )
t = np.arange(training_start * 10, training_end * 10,  10)

J_21 = Theta[training_start:training_end, 52+1]
J_01 = Theta[training_start:training_end, 52-1]
J_12 = Theta[training_start:training_end, 52+50]
J_10 = Theta[training_start:training_end, 52-50]

# Deposited inputs
# Bx[:training_end], By[:training_end],
# +50 seems to have the greatest positive effect on model
# +50 = one step "to the right", +1 = one step "down"
u = np.vstack((Bs[training_start:training_end], 
               Theta[training_start:training_end, pos_index + 50],
               #HWR[training_start:training_end]
                  )).T

# Delay the input data
u_delayed = delay_control_data(u, nr_of_delays = 1, delay_indexes = 1)


#spatio_temporal_grid = 

lib = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=3),
                    derivative_order = 0, 
                    include_interaction = True, include_bias = True)

#input_lib = ps.CustomLibrary()

combined_lib = ps.GeneralizedLibrary(libraries = [my_library, lib],
                                     tensor_array = [[1, 1]])

param_lib = ps.ParameterizedLibrary(feature_library=lib, parameter_library= lib,
                                    num_features = 3, num_parameters=2)

mod = ps.SINDy(optimizer = optimizer,
               feature_library= combined_lib,
               differentiation_method=differentiation_method)


mod.fit(x = x, t = t, u = u_delayed)

mod.print()
print(mod.score(x, t, u = u_delayed))
#%%

plt.plot(x)
plt.show()

plt.plot(u)
plt.show()
print(np.any(np.isnan(x)))
#%%

"""
model.simulate() is EXTREMELY slow. 17 minutes to predict 100 timesteps.
Only run if necessary/for a good fit.

predict() is apparently quick, however only returns x_dot.
"""

sim_length = 10
t_sim = t[:sim_length]

pred = mod.simulate(x[0, :], t, u = u[:, 0:len(t)])

print(len(t), x.shape[0])

#%%
saved_mod = mod
saved_mod.print()
#%%

mod = saved_mod
mod.print()
#%%

plt.plot(x, "b", label = "True x")
plt.plot(pred, "r--", label = "SINDy x")
plt.ylim(-1, 1)
#plt.xlim(3900, 4100)
plt.legend()
plt.show()

print(f" R² score: {mod.score(x, t, u = u[:, :len(t)])}")

#%%
print(mod.n_features_in_)
print(mod.n_output_features_)
print(mod.n_control_features_)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1st measurement
axes[0, 0].plot(t, x[:, 0], 'b', label='True x')
axes[0, 0].plot(t[:-1], pred[:, 0], 'r--', label='SINDy x')
axes[0, 0].set_ylabel('Jpar')
#axes[0, 0].set_xlim([-2.5, 2.5])
axes[0, 0].legend()
plt.show()

if x.shape[1] > 1:
    # Plot 2nd measurement
    axes[1, 0].plot(t, x[:, 1], 'b', label='True y')
    axes[1, 0].plot(t[:-1], pred[:, 1], 'r--', label='SINDy y')
    axes[1, 0].set_ylabel('Jpar')
    axes[1, 0].set_xlabel('Minutes')
    #axes[1, 0].set_ylim([-2.5, 2.5])
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
#axes[1, 1].set_ylim([-2.5, 2.5])
axes[1, 1].legend()

plt.tight_layout()
plt.show()
print("Plotted!")


#%%

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
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=central_longitude))
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())  # Northern Hemisphere
# Add features to the map
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.gridlines(draw_labels=True)
# Scatter plot of Jpar
sc = ax.scatter(longitudes, latitudes, c=Jpar[time_index, pos_index1:pos_index2], cmap='coolwarm', s=60,
                transform=ccrs.PlateCarree())
cbar = plt.colorbar(sc, orientation='vertical')
cbar.ax.tick_params(labelsize = fontsize*1)
cbar.set_label('Radial Current Density (μA/m²)', fontsize = fontsize*1.5)
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
plt.title('Average error in reconstructed data', fontsize = fontsize*1.5)
plt.show()




