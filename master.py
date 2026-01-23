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
    # Reading in Ampere data:
    year_dirs = [d for d in amp_root.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}", d.name)]
    
    #dB_Naagcm_list = []
    #dB_Eaagcm_list = []
    Jpar_list = []
    geo_cLat_list = []
    geo_lon_deg_list = []
    """
    read_ampere function breaks for day 22 in 2012. Seems to work for all of 2010-2011
    aswell as 2013+
    
    2009 is missing a lot of days
    """
    i = 0
    stop_val =  nr_days # Number of days to process
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
                        #print(f"    Extracted: {member} -> {extracted_path}")
                        #print("    Exists:", os.path.exists(extracted_path))
                        # Pass the full path as a string
                        data = read_ampere_ncdf(str(extracted_path), OutVars="J")
                        
                        Jpar_list.append(data["Jpar"])
                        geo_cLat_list.append(data["geo_cLat_deg"])
                        geo_lon_deg_list.append(data["geo_lon_deg"])
                        #dB_Naagcm_list.append(data["dB_Ngeo"])
                        #dB_Eaagcm_list.append(data["dB_Egeo"])
                        #print(i)
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
    
    # Downsample to ACE data sample frequency, dt=4 min
    Jpar = np.concatenate(Jpar_list, axis=0)[::2]
    geo_clat_deg = np.concatenate(geo_cLat_list, axis = 0)[::2]
    geo_lon_deg = np.concatenate(geo_lon_deg_list, axis = 0)[::2]               
    #dB_Naagcm_all = np.concatenate(dB_Naagcm_list, axis=0)
    #dB_Eaagcm_all = np.concatenate(dB_Eaagcm_list, axis=0)
    #print("Final shapes:", dB_Naagcm_all.shape, dB_Eaagcm_all.shape)
    print("Final shapes:", Jpar.shape, geo_clat_deg.shape, geo_lon_deg.shape)#%%
    
    return Jpar, geo_clat_deg, geo_lon_deg 

# Breaks for 2020:
def read_files(directory, year=None):
    """
    Reads .dat files from a directory. If a year is specified, only reads files for that year.
    
    Args:
        directory (str): Path to the directory containing the .dat files.
        year (int, optional): Specific year to read. If None, reads all files.
    
    Returns:
        pd.DataFrame: Combined DataFrame of all read files.
    """
    # Get all .dat files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith(".dat")])
    
    # Filter files by year if a year is specified
    if year is not None:
        files = [f for f in files if str(year) in f]  # Only include files with the specified year
    
    # List to store the dataframes
    dataframes = []
    
    # Loop through the filtered files and read them
    for filename in files:
        if filename == "mag_B_4min_2020.dat":
            break
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
        print("No files found for the specified year.")
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
#%%
Jpar, cLat_deg, lon_deg = read_Jpar(from_year_index = 1, nr_days = 2)

year_data = read_files(IMF_PATH, year=2010)
year_data_interp = year_data.interpolate(method = "linear") #dt = 4 min 
year_data = np.array(year_data_interp)

print(year_data.shape)


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

# Read in control data
Bx = np.array(year_data_interp["Bgsm_x"][:Jpar.shape[0]])
By = np.array(year_data_interp["Bgsm_y"][:Jpar.shape[0]])
Bz = np.array(year_data_interp["Bgsm_z"][:Jpar.shape[0]])

training_start = 0
training_end = 100

Vx = np.array(SW_dat_dow["VGSM_X"])[:Jpar.shape[0]]

reconnection_voltage = Milan_coupling(By, Bz, Vx)

print(f"Bx's shape: {Bx.shape}")

# Stack control data to the end of system measurements matrix
Theta = np.hstack((Jpar, Bx[:, np.newaxis], By[:, np.newaxis], Bz[:, np.newaxis], 
                   Vx[:, np.newaxis]))
print(f"Theta's shape: {Theta.shape}")


# Visualize the combined measurements and control data
plt.pcolormesh(Theta[training_start:training_end, 1195:1200])
plt.colorbar()
plt.show()

#%%

#RUN THIS SECTION EVERY TIME

# Define SINDY model parameters
dt = 4

my_library = ps.CustomLibrary([lambda x: np.sin(x), lambda x: np.cos(x), ])

# SINDyCP uses ParametrizedLibrary, to create Theta(X, U) = Theta_feat(X) x Theta_par(U) 
# Can be combined with weak formalized SINDy. Weak formulation can use WeakPDELibrary,
# Otherwise I must construct the system rows by projecting data onto weak samples.
# w_ik^v = \int_Omega_k theta(x;t) X^v(x;t) d^D x dt eq. 5 in SINDyCP paper

optimizer = ps.EnsembleOptimizer(opt=ps.STLSQ(threshold = 0.001), 
                                 bagging=True, library_ensemble=True,
                                 n_models = 10) # Default aggregator is median

feature_names = None

differentiation_method = ps.FiniteDifference() 


      
training_end = 400
x = Theta[0:training_end, 0:2] #(time, features) MUST BE (m, n), n > 0 NOT (m, )
t = np.arange(0, training_end * 4, 4)
u = np.vstack((Bx[:training_end], By[:training_end], Bz[:training_end])).T

#x = np.vstack((Theta[0:training_end, 0], Bx[0:training_end])).T


lib = ps.PolynomialLibrary()
                    #temporal_grid= t,
                    #differentiation_method=differentiation_method)

combined_lib = ps.GeneralizedLibrary(libraries = [my_library, lib])

mod = ps.SINDy(optimizer = optimizer,
               feature_library=lib,#feature_library,
               differentiation_method=differentiation_method)


mod.fit(x = x, t = t, u = u)

mod.print()

#%%
pred = mod.simulate(x[0, :], t, u = u[:, 0:len(t)])

print(len(t), x.shape[0])

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
#axes[0, 0].set_ylim([-2.5, 2.5])
axes[0, 0].legend()

# Plot 2nd measurement
axes[1, 0].plot(t, x[:, 1], 'b', label='True y')
axes[1, 0].plot(t[:-1], pred[:, 1], 'r--', label='SINDy y')
axes[1, 0].set_ylabel('Jpar')
axes[1, 0].set_xlabel('Minutes')
#axes[1, 0].set_ylim([-2.5, 2.5])
axes[1, 0].legend()

# Plot phase space
# Joinked from a pysindy test script, kept in to keep the [2, 2] looking nice
axes[0, 1].plot(x[:, 0], x[:, 1], 'b', label='True')
axes[0, 1].plot(pred[:, 0], pred[:, 1], 'r--', label='SINDy')
axes[0, 1].set_xlabel('Jpar 1')
axes[0, 1].set_ylabel('Jpar 2')
axes[0, 1].legend()

# Plot error
axes[1, 1].plot(t[:-1], x[:-1, 0] - pred[:, 0], 'b', label='Meas 1 error')
axes[1, 1].plot(t[:-1], x[:-1, 1] - pred[:, 1], 'r', label='Meas 2 error')
axes[1, 1].set_xlabel('Minutes')
axes[1, 1].set_ylabel('Error')
#axes[1, 1].set_ylim([-2.5, 2.5])
axes[1, 1].legend()

plt.tight_layout()
plt.show()
print("Plotted!")



