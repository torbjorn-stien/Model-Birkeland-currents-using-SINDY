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
from PIL import Image
from datetime import datetime, timedelta
from cartopy.feature.nightshade import Nightshade
import matplotlib.animation as animation


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
        print(f"Reading file: {filename}")  # Optional: Print the file being read
        df = read_dat(file_path)  # Read the file using your function
        dataframes.append(df)  # Append the dataframe to the list
    
    # Combine all dataframes into one (optional)
    if dataframes:
        combined_data = pd.concat(dataframes, ignore_index=True)
        print("Done reading!")
        return combined_data
    else:
        print("No files found for the specified year.")
        return pd.DataFrame()    

Jpar, cLat_deg, lon_deg = read_Jpar(from_year_index = 1, nr_days = 2)

year_data = read_files(IMF_PATH, year=2010)
year_data_interp = year_data.interpolate(method = "linear") #dt = 4 min 


def train_SINDY(input_dat, control_dat, dt, training_start, training_end,
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

    X = input_dat[training_start:training_end, :].T
    U = control_dat[training_start:training_end, :].T
    
   
    model = ps.SINDy(
        differentiation_method = differentiation_method,
        feature_library = feature_library,
        optimizer = optimizer
        )
    
    model.fit(X, t = dt, feature_names = feature_names)
    
    model.print()
    
    return model
    



