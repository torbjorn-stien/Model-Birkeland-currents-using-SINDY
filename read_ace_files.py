#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:07:58 2025

@author: tos
"""

import pandas as pd
import re
# Load the .dat file
#file_path = "/nfs/revontuli/data/bjorn/ACE/B_IMF/mag_B_4min_2010.dat"

def read_dat(filepath):
    with open(filepath, 'r') as file:
        header_line = file.readlines()[42]  # Assuming the 44th line contains the column names
    
    # Extract column names from the header line
    columns = header_line.strip().split()

    # Read the data using the dynamically determined column names
    file = pd.read_csv(filepath, delimiter=r'\s+', skiprows=44, names=columns, engine="python")
    
    
    # Apply the function to fix concatenated values
    #fixed_data = file.apply(fix_concatenated_values, axis=1)
    # Convert the fixed data back into a DataFrame
    #fixed_df = pd.DataFrame(fixed_data.tolist(), columns=file.columns).apply(pd.to_numeric, errors='coerce')
    
    return file


def fix_concatenated_values(row):
    fixed_row = []
    for value in row:
        if isinstance(value, str):
            # Use regex to split concatenated scientific notation numbers
            pattern = r"([-+]?\d+\.\d+e[+-]?\d+|[-+]?\d+\.?\d*)"
            matches = re.findall(pattern, value)
            fixed_row.extend(matches)
        else:
            fixed_row.append(value)
    return fixed_row


