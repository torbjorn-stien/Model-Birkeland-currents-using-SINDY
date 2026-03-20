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
import glob


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
P_SW_PATH = "/nfs/revontuli/data/bjorn/ACE/P_SW"

amp_root = Path(AMPERE_PATH)
imf_root = Path(IMF_PATH)
p_sw_root = Path(P_SW_PATH)

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

def find_nans(data, print_=True):
    # Finds the lengths of nan intervals and non-nan intervals
    start_nan_indices = [] # Indices for every first nan in a given nan stretch
    end_nan_indices = []   # Indices for every last nan in a given nan stretch
    
    if np.isnan(data[0]): # Checks if first index is nan
        start_nan_indices.append(0)
        
    for i in range(1, len(data)):
        if np.isnan(data[i]) and not np.isnan(data[i - 1]):
            start_nan_indices.append(i)
           
        if not np.isnan(data[i]) and np.isnan(data[i - 1]):
            #print(i)
            end_nan_indices.append(i)
        
    if np.isnan(data[-1]): # Checks if last index is nan
        end_nan_indices.append(len(data) - 1)
    
    #print(start_nan_indices)
    #print(end_nan_indices)
    start_nan_indices = np.array(start_nan_indices)
    end_nan_indices = np.array(end_nan_indices)   
    
    
    nan_lengths = end_nan_indices - start_nan_indices # Length of any given nan stretch
    no_nan_lengths = np.zeros_like(nan_lengths)       # Length of any given clean data stretch
    
    no_nan_lengths[:-1] = start_nan_indices[1:] - end_nan_indices[:-1]
            
            
    if print_:
        print(f"This data contains {np.sum(nan_lengths)} nan indices, over {len(nan_lengths)} unique stretches")
        print(f"On average 1 nan stretch every {int(len(data)/len(nan_lengths))} data points")
    return start_nan_indices, end_nan_indices, nan_lengths, no_nan_lengths


def interpolate_nans(data, max_nan_length):

    start_nan_indices, end_nan_indices, nan_lengths, no_nan_lengths = find_nans(data, print_ = False)
    
    if len(start_nan_indices) == 0:
        print("No NaNs! You lucky bastard.")
        return data, start_nan_indices, end_nan_indices, nan_lengths, no_nan_lengths
    
    # Find NaN stretches smaller than the specified length
    small_nan_mask = nan_lengths <= max_nan_length  # Mask for stretches smaller than max_nan_length
    small_nan_indices = start_nan_indices[small_nan_mask]  # Start indices of small NaN stretches
    # Interpolate over each small NaN stretch
    for start, length in zip(small_nan_indices, nan_lengths[small_nan_mask]):
        # Ensure the stretch is not at the boundaries of the array
        if start == 0 or start + length >= len(data):
            continue
        # Interpolate linearly over the range of NaNs
        data[start:start + length] = np.linspace(
            data[start - 1],  # Value before the NaN stretch
            data[start + length],  # Value after the NaN stretch
            length + 2  # Include the boundary points
        )[1:-1]  # Exclude the boundary points (keep only interpolated values)
    # Remove these small NaN stretches from the arrays
    start_nan_indices = start_nan_indices[~small_nan_mask]
    end_nan_indices = end_nan_indices[~small_nan_mask]
    nan_lengths = nan_lengths[~small_nan_mask]
    no_nan_lengths = no_nan_lengths[~small_nan_mask]
    
    start_nan_indices, end_nan_indices, nan_lengths, no_nan_lengths = find_nans(data, print_ = False)
    
    print() 
    print(f"Interpolated over {np.sum(small_nan_mask)} NaNs within stretches smaller than {max_nan_length} indices.")
    print()
    print(f"After interpolation this data contains {np.sum(nan_lengths)} nan indices, over {len(nan_lengths)} unique stretches")
    print(f"On average 1 nan stretch every {int(len(data)/len(nan_lengths))} data points")
    
    return data, start_nan_indices, end_nan_indices, nan_lengths, no_nan_lengths


def find_overlapping_clean_data(arrays, min_length, print_=True):
    """
    Find stretches of clean data that overlap between multiple arrays.
    
    Parameters:
    -----------
    arrays : list of numpy arrays
        List of arrays to compare (all should have the same length)
    print_ : bool
        Whether to print summary information
    min_length : int
        Minimum length of clean segments to return (segments shorter than this are filtered out)
    
    Returns:
    --------
    clean_starts : numpy array
        Start indices of overlapping clean segments (meeting min_length requirement)
    clean_ends : numpy array
        End indices of overlapping clean segments (meeting min_length requirement)
    clean_lengths : numpy array
        Lengths of overlapping clean segments (meeting min_length requirement)
    """
    
    # Check that all arrays have the same length
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All arrays must have the same length")
    
    array_length = lengths[0]
    
    # Create a combined mask where True means ALL arrays have non-NaN at that position
    # Start with all True, then combine with logical AND for each array
    combined_clean_mask = np.ones(array_length, dtype=bool)
    
    if print_:
        print("\nIndividual array clean positions:")
    
    for i, arr in enumerate(arrays):
        # For each array, positions are clean if they are NOT NaN
        array_clean = ~np.isnan(arr)
        combined_clean_mask = combined_clean_mask & array_clean
        
        if print_:
            n_clean = np.sum(array_clean)
            clean_indices = np.where(array_clean)[0]
            print(f"Array {i+1}: {n_clean} clean data points ({n_clean/array_length*100:.1f}%)")
            print(f"  Clean indices: {clean_indices}")
    
    # Now find contiguous segments in the combined cl
    #ean mask
    n_combined_clean = np.sum(combined_clean_mask)
    if print_:
        print(f"\nCombined: {n_combined_clean} positions clean in ALL arrays ({n_combined_clean/array_length*100:.1f}%)")
        print(f"Combined clean mask: {combined_clean_mask}")
    
    if n_combined_clean == 0:
        if print_:
            print("No overlapping clean segments found")
        return np.array([]), np.array([]), np.array([])
    
    # Find boundaries where clean status changes
    # Add sentinel values at start and end to handle edge cases
    padded_mask = np.concatenate([[0], combined_clean_mask, [0]])
    
    # Find transitions using convolution for cleaner detection
    # diff will be 1 for start of segment (0->1), -1 for end of segment (1->0)
    transitions = np.diff(padded_mask)
    
    # Find start indices (where transition is 1)
    clean_starts = np.where(transitions == 1)[0]
    
    # Find end indices (where transition is -1)
    clean_ends = np.where(transitions == -1)[0]
    
    # Calculate all lengths
    all_clean_lengths = clean_ends - clean_starts
    
    # Filter segments based on min_length
    long_enough_mask = all_clean_lengths >= min_length
    
    clean_starts = clean_starts[long_enough_mask]
    clean_ends = clean_ends[long_enough_mask]
    clean_lengths = all_clean_lengths[long_enough_mask]
    
    if print_:
        print(f"\nFound {len(all_clean_lengths)} total overlapping clean segments:")
        for i, (start, end, length) in enumerate(zip(clean_starts_original if 'clean_starts_original' in locals() else clean_starts, 
                                                     clean_ends_original if 'clean_ends_original' in locals() else clean_ends, 
                                                     all_clean_lengths)):
            status = " (filtered out)" if length < min_length else ""
            print(f"  Segment {i+1}: indices {start}-{end-1} (length {length}){status}")
        
        print(f"\nAfter filtering (min_length={min_length}):")
        if len(clean_lengths) > 0:
            for i, (start, end, length) in enumerate(zip(clean_starts, clean_ends, clean_lengths)):
                print(f"  Segment {i+1}: indices {start}-{end-1} (length {length})")
        else:
            print(f"  No segments meet the minimum length requirement of {min_length}")
    
    return clean_starts, clean_ends, clean_lengths


def subset_data(arrays, min_length, overlap, force_overlap = False):
    """
    Parameters
    ----------
    arrays : list
        list of arrays to be split
    min_length : int
        DESCRIPTION.
    overlap : int
        DESCRIPTION.

    Returns
    -------
    clean_data_lists : list of lists
        DESCRIPTION.

    """
    
    # Find overlapping clean data stretches longer than min_len
    clean_start, clean_end, clean_len = find_overlapping_clean_data(arrays, min_length, print_ = False)
    
    # Initialize data lists
    clean_data_lists = [[] for _ in range(len(arrays))] 

    # Splits the overlapping clean segments of data so that they all have the same length (the min_length)
    # (so they can fit on the same temporal grid)
    for i, array in enumerate(arrays):
        for j in range(len(clean_len)):
            
            if clean_len[j] + overlap >= 2 * min_length: # If there is enough clean data for multiple subsets
                # Discards excess clean data
                nr_of_subsets = int((clean_len[j] - min_length)/(min_length - overlap) + 1) # Int rounds down

                subset_start = clean_start[j] # First subset start
                for k in range(nr_of_subsets):
                    # Rolling window of clean data over the total available clean data in this stretch
                    subset_end = subset_start + min_length
                    
                    data_temp = array[subset_start:subset_end]
                    # Adds data to the corresponding list
                    clean_data_lists[i].append(data_temp)
                    # Moves the start index to the next subset
                    subset_start += min_length - overlap
                    
            elif force_overlap == True: # Forces 2 subsets even if overlap is too small to allow 2 subsets
                data_temp = array[clean_start[j]:clean_start[j] + min_length]
                clean_data_lists[i].append(data_temp)
                
                data_temp = array[clean_end[j] - min_length:clean_end[j]]
                clean_data_lists[i].append(data_temp)
            
            else: # Discards excess clean data
                data_temp = array[clean_start[j]:clean_start[j] + min_length]
                clean_data_lists[i].append(data_temp)
    
    return clean_data_lists

#%%
Jpar, cLat_deg, lon_deg, missing_days, missing_indices = read_Jpar(from_year_index = 5, nr_days = 365)
#%%
Jpar_downsampled = Jpar[::2]

# If 1 column contains a nan, every column contains nan at the same place
# find_nans() is bottlenecking, vectorize it at some point
for column in range(Jpar_downsampled.shape[1]): 
    Jpar_downsampled[:, column], nan_start_Jpar, nan_end_Jpar, nan_lengths_Jpar, no_nan_lengths_Jpar = interpolate_nans(Jpar_downsampled[:, column],
                                                      max_nan_length=1)

#%%
year_data = read_files(IMF_PATH, start_year=2013, end_year=2013)
#year_data_interp = year_data.interpolate(method = "linear") #dt = 4 min 
#year_data = np.array(year_data_interp)

"""
2012, 2016 & 2020 were leap-years
"""

#%%
print(Jpar.shape)

#%%
# Reads in additional ACE parameters
# File names are nonsensical, therefore all 14 years have to be read in at once and then sorted
# Luckily the reading is fairly quick.
# 1 measurement every 64 seconds, 0.9375 samples pr. min

p_sw_file_list = glob.glob("/nfs/revontuli/data/bjorn/ACE/P_SW/*.zip")  # Reads any zip file
data_frames = []
for file in p_sw_file_list: # Reads in the zip files
    df = pd.read_csv(
        file,
        skiprows=31,
        delimiter=r"\s+",
        names=["Year", "day", "hour", "min", "sec",
               "Density_proton", "T_proton",
               "VGSE_X", "VGSE_Y", "VGSE_Z",
               "VGSM_X", "VGSM_Y", "VGSM_Z",
               "GSE_X", "GSE_Y", "GSE_Z"]
    )
    data_frames.append(df)
combined_p_sw_data = pd.concat(data_frames, ignore_index=True) # Concats the years, sorts them below
P_SW_data = combined_p_sw_data.sort_values(by=["Year", "day", "hour", "min", "sec"]).reset_index(drop=True) # 1 sample every 64 seconds

# Convert -9999 to nan
P_SW_data[P_SW_data == -9999.9] = np.nan

# add DatetimeIndex for easy resampling
P_SW_data["datetime"] = pd.to_datetime(P_SW_data["Year"].astype(str)) + pd.to_timedelta(P_SW_data["day"] - 1, unit="D") \
    + pd.to_timedelta(P_SW_data["hour"], unit="h") + pd.to_timedelta(P_SW_data["min"], unit="m") \
        + pd.to_timedelta(P_SW_data["sec"], unit="s")
     
P_SW_data = P_SW_data.drop(columns=["Year", "day", "hour", "min", "sec"]) # Drop old date columns
P_SW_data.set_index("datetime", inplace=True)

#%%

P_SW_filtered = P_SW_data.loc["2013-01-01":"2013-12-31"]

#%%

P_SW_filtered.loc[:, "Density_proton"], nan_start_n, nan_end_n, nan_lengths_n, no_nan_lengths_n = interpolate_nans(np.array(P_SW_filtered["Density_proton"]),
                                                                                        max_nan_length=1)
P_SW_filtered.loc[:, "VGSM_X"], nan_start_v, nan_end_v, nan_lengths_v, no_nan_lengths_v = interpolate_nans(np.array(P_SW_filtered["VGSM_X"]),
                                                                                        max_nan_length=1)

print(np.mean(no_nan_lengths_n))
print(np.mean(no_nan_lengths_v))

# Resample to 4min:
#P_SW_data_interp = P_SW_interp.resample("4min").mean()

"""
pandas.resample() treats NaN as 0.
e.g mean([1, NaN, 3]) = 2, where mean([NaN, NaN, NaN]) = NaN
"""

#%%

#nan_pos = np.where(np.isnan(Jpar_downsampled[:, 0]))[0]

#rows_with_nan = np.where(np.isnan(Jpar_downsampled).any(axis=1))[0]

# Read in control data
# Need complex for taking sqrt of negative values
Bx = np.array(year_data["Bgsm_x"][:Jpar_downsampled.shape[0]], dtype=complex)
By = np.array(year_data["Bgsm_y"][:Jpar_downsampled.shape[0]])
Bz = np.array(year_data["Bgsm_z"][:Jpar_downsampled.shape[0]])
Bz[Bz<-200] = 0

Bx[Bx<-100] = 0
By[By<-100] = 0

"""
Bx = np.delete(Bx, rows_with_nan)
By = np.delete(By, rows_with_nan)
Bz = np.delete(Bz, rows_with_nan)
"""

Vx = np.array(P_SW_filtered["VGSM_X"], dtype=complex)[::2][:Jpar_downsampled.shape[0]]
n = np.array(P_SW_filtered["Density_proton"], dtype=complex)[::2][:Jpar_downsampled.shape[0]]

#Vx = np.delete(Vx, nan_pos)

# Should redefine coordinates so Vx is always positive for calculating funcs like E_WAV_sqrt
v = -Vx

#Jpar_downsampled = np.delete(Jpar_downsampled, rows_with_nan, axis = 0)

"""
Fit some sort of polynomial to the smoothed n points before and after a nan interval
removes nan intervals, however can not be implemented for active periods as the 
Jpar varies too much.
"""

plt.plot(Bz)
plt.show()


#%%
reconnection_voltage = Milan_coupling(By, Bz, Vx)

theta_c = np.arctan2(By, Bz)

sin_squared = np.sin(theta_c/2)**2
sin_4th = np.sin(theta_c/2)**4

Bs = Bz
Bs[Bs>0] = 0

HWR = v * Bs
HWR[HWR == -0] = 0

Bs_delayed = delay_control_data(Bs, 7, 1)


B_T = np.sqrt(Bx.real**2 + By**2 + Bz**2)
B = Bz

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
#Theta = np.hstack((Jpar_downsampled, Bx[:, np.newaxis], By[:, np.newaxis], Bz[:, np.newaxis])).real
                   #Vx[:, np.newaxis]))

missing_index = int(missing_indices[0]/2)
#Theta = Theta[:missing_index]

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
                                 n_models = 10) # Default aggregator is median

feature_names = None

# Finite difference amplifies noise in data.
differentiation_method = ps.SmoothedFiniteDifference()

"""
3 timesteps of the full system needs 1.92 TiB (1.1TB) RAM to model.
"""
training_start = 0
training_end = 26680

pos_index = 1130 # Which position to attempt to model
pos = Jpar_downsampled[training_start:training_end, pos_index] #(time, features) MUST BE (m, n), n > 0 NOT (m, )
t = np.arange(training_start * 10, training_end * 10,  10)

# The closest measured currents to the "main" (attempted) modelled current, main current = J_11
J_21 = Jpar_downsampled[training_start:training_end, pos_index+1]
J_01 = Jpar_downsampled[training_start:training_end, pos_index-1]
J_12 = Jpar_downsampled[training_start:training_end, pos_index+50]
J_10 = Jpar_downsampled[training_start:training_end, pos_index-50]


clean_start, clean_end, clean_len = find_overlapping_clean_data([Jpar_downsampled[:, pos_index], Bs, v],
                                                                min_length = 200, print_=False)

cleaned_subset_data = subset_data([Jpar_downsampled[:, pos_index], Bs, v], 200, 10)

Jpar_clean = cleaned_subset_data[0]
Bs_clean = cleaned_subset_data[1]
v_clean = cleaned_subset_data[2]


X = Jpar_clean

# Deposited inputs
# Bx[:training_end], By[:training_end],
# +50 seems to have the greatest positive effect on model
# +50 = one step "to the right", +1 = one step "down"
u = Bs_clean# [[Bs_clean], [v_clean]]


# Delay the input data
u_delayed = u#delay_control_data(u, nr_of_delays = 1, delay_indexes = 1)

#spatio_temporal_grid = 
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

# Only temporal grid gives dx/dt = 1/2 dx/dt + 1/2 dx/dt, R² = 1
lib = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=3),
                   
                    include_interaction = True, include_bias = True,
                    implicit_terms=False)

# Various attempted libraries
#input_lib = ps.CustomLibrary()

combined_lib = ps.GeneralizedLibrary(libraries = [my_library, lib],
                                     tensor_array = [[1, 1]])

param_lib = ps.ParameterizedLibrary(feature_library=lib, parameter_library= lib,
                                    num_features = 3, num_parameters=2)

# INitialize SINDy model
mod = ps.SINDy(optimizer = optimizer,
               feature_library= combined_lib,
               differentiation_method=differentiation_method)

# Fit SINDy model
mod.fit(x = X, t = dt, u = u_delayed)

# Print the best fit approximation
mod.print()
print(mod.score(X, dt, u = u_delayed)) # R² score of model
#%%

print("Shape of x:", np.shape(X))
print("Shape of u:", np.shape(u_delayed))

#%%

# Differentiation method tests
diff = ps.SmoothedFiniteDifference(smoother_kws={"window_length" : 10})
x_dot = diff._differentiate(x, t)

plt.plot(x_dot[:2000])
plt.show()

#%%
plt.plot(x)
plt.show()
plt.plot(x_dot)
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




