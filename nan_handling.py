#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:39:33 2026

@author: tos
"""

import numpy as np

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
    """
    Currently only linear interpolation

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    max_nan_length : TYPE
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    start_nan_indices : TYPE
        DESCRIPTION.
    end_nan_indices : TYPE
        DESCRIPTION.
    nan_lengths : TYPE
        DESCRIPTION.
    no_nan_lengths : TYPE
        DESCRIPTION.

    """
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
    
    # Now find contiguous segments in the combined clean mask
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
        for i, (start, end, length) in enumerate(zip(clean_starts, clean_ends, 
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
