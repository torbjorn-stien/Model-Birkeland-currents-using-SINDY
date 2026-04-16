# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:14:52 2025

@author: tbear
"""

import numpy as np
import netCDF4 as nc

import re
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from pathlib import Path

def read_ampere_ncdf(ncdfname, OutVars=None):
    """
    read_ampere_ncdf - reads Ampere netcdf-files
    
    Parameters
    ----------
    ncdfname : str
        string or char-array of net-cdf filename to be read
    OutVars : list of str, optional
        list of selected Ampere output. This makes it possible
        to select only J_||, or the B-fields in either GEO or GCM
        coordinates. The list elements should be from 'J',
        'B_GEO', 'B_GCM'.
    
    Returns
    -------
    data : dict
        dict with fields:
        npnt - number of time-steps
        year - year, int array [npnt x 1] with year (C.E.)
        doy - day of year, int array [npnt x 1] 
        time - time (h) of day int array [npnt x 1] 
        avgint - averaging time-period (s), int array [npnt x 1]
        kmax - latitude order of fit, int-array [npnt x 1 ]
        mmax - longitude order of fit, int array [npnt x 1]
        res_deg - grid latitude resolution in degrees, int array [npnt x 1]
        nLatGrid - number of latitude points in grid, int array [npnt x 1]
        nLonGrid - number of latitude points in grid, int array [npnt x 1]
        cLat_deg - co-latitude in AACGM coordinates in degrees, int array [1200 x npnt]
        mlt_hr - AACGM Magnetic Local Time (MLT) in hours, int array [1200 x npnt]
        geo_cLat_deg - co-latitude in GEO coordinates in degrees [1200 x npnt]
        geo_lon_deg - longitude in GEO coordinates in degrees [1200 x npnt]
        R - Radius from center of the Earth in kilometers, [1200 x npnt]
        Jpar - Radial current density [muA/m^2], double array [1200 x npnt]
        d_J - Radial current density residual [muA/m^2], double array [1200 x npnt]
        dB_Rgeo - magnetic field perturbation parallel to GEO radial
                  direction in units of nano-Tesla, double array [1200 x npnt] 
        dB_Ngeo - magnetic field perturbation parallel to GEO northward
                  direction in units of nano-Tesla, double array [1200 x npnt] 
        dB_Egeo - magnetic field perturbation parallel to GEO eastward
                  direction in units of nano-Tesla [1200 x npnt] 
        dB_geo - magnetic field perturbation in GEO coordinates (nT)
                 double array [3 x 1200 x npnt ]
        dB_Naagcm - Magnetic field perturbation parallel to the AACGM
                    northward direction in units of nano-Tesla, double
                    array [1200 x npnt] 
        dB_Eaagcm - Magnetic field perturbation parallel to the AACGM
                    eastward direction in units of nano-Tesla, double
                    array [1200 x npnt] [1200x720 double]
        d_dB_Rgeo - magnetic field residual parallel to GEO radial
                    direction in units of nano-Tesla, double array
                    [1200 x npnt]  
        d_dB_Ngeo - magnetic field residual parallel to GEO northward
                    direction in units of nano-Tesla, double array
                    [1200 x npnt]  
        d_dB_Egeo - magnetic field residual parallel to GEO eastward
                    direction in units of nano-Tesla, double array 
                   [1200 x npnt]  
        d_dB_Naagcm - Magnetic field residual parallel to the AACGM
                      northward direction in units of nano-Tesla, double
                      array [1200 x npnt]
        d_dB_Eaagcm - Magnetic field residual parallel to the AACGM
                      eastward direction in units of nano-Tesla, double
                      array [1200 x npnt]
    
    Example
    -------
    ncfname = 'ampere.20241223.k060_m08.north.grd.nc'
    data = read_ampere_ncdf(ncfname, ['J', 'B_GEO'])
    """
    
    # Try to open the netCDF file
    try:
        dataset = nc.Dataset(ncdfname, 'r')
    except Exception as e:
        print(f'could not open file: {ncdfname}')
        print(f'Error: {e}')
        print('returning discretely? Setting data: -1')
        return -1
    
    data = {}
    
    # Read basic variables
    data['npnt'] = dataset.variables['npnt'][:] # 2012 file(s?) do not contain npnt OR year
    data['year'] = dataset.variables['year'][:]
    data['doy'] = dataset.variables['doy'][:]
    data['time'] = dataset.variables['time'][:]
    data['avgint'] = dataset.variables['avgint'][:]
    data['kmax'] = dataset.variables['kmax'][:]
    data['mmax'] = dataset.variables['mmax'][:]
    data['res_deg'] = dataset.variables['res_deg'][:]
    data['nLatGrid'] = dataset.variables['nLatGrid'][:]
    data['nLonGrid'] = dataset.variables['nLonGrid'][:]
    data['cLat_deg'] = dataset.variables['cLat_deg'][:]
    data['mlt_hr'] = dataset.variables['mlt_hr'][:]
    data['geo_cLat_deg'] = dataset.variables['geo_cLat_deg'][:]
    data['geo_lon_deg'] = dataset.variables['geo_lon_deg'][:]
    data['R'] = dataset.variables['R'][:]
    
    # Read variables based on selection
    if OutVars is None or len(OutVars) == 0:
        # Read all variables
        data['dB_Rgeo'] = dataset.variables['db_R'][:]
        data['dB_Ngeo'] = dataset.variables['db_T'][:]
        data['dB_Egeo'] = dataset.variables['db_P'][:]
        data['dB_geo'] = dataset.variables['db_geo'][:]
        data['dB_Naagcm'] = dataset.variables['db_Th_Th'][:]
        data['dB_Eaagcm'] = dataset.variables['db_Ph_Ph'][:]
        data['Jpar'] = dataset.variables['jPar'][:]
        data['d_J'] = dataset.variables['del_jPar'][:]
        data['d_dB_Rgeo'] = dataset.variables['del_db_R'][:]
        data['d_dB_Ngeo'] = dataset.variables['del_db_T'][:]
        data['d_dB_Egeo'] = dataset.variables['del_db_P'][:]
        data['d_dB_Naagcm'] = dataset.variables['del_db_Th_Th'][:]
        data['d_dB_Eaagcm'] = dataset.variables['del_db_Ph_Ph'][:]
    else:
        # Read selected variables
        if 'J' in OutVars:
            data['Jpar'] = dataset.variables['jPar'][:]
            data['d_J'] = dataset.variables['del_jPar'][:]
        
        if 'B_GEO' in OutVars:
            data['dB_Rgeo'] = dataset.variables['db_R'][:]
            data['dB_Ngeo'] = dataset.variables['db_T'][:]
            data['dB_Egeo'] = dataset.variables['db_P'][:]
            data['dB_geo'] = dataset.variables['db_geo'][:]
            data['d_dB_Rgeo'] = dataset.variables['del_db_R'][:]
            data['d_dB_Ngeo'] = dataset.variables['del_db_T'][:]
            data['d_dB_Egeo'] = dataset.variables['del_db_P'][:]
        
        if 'B_GCM' in OutVars:
            data['dB_Naagcm'] = dataset.variables['db_Th_Th'][:]
            data['dB_Eaagcm'] = dataset.variables['db_Ph_Ph'][:]
            data['d_dB_Naagcm'] = dataset.variables['del_db_Th_Th'][:]
            data['d_dB_Eaagcm'] = dataset.variables['del_db_Ph_Ph'][:]
    
    dataset.close()
    return data



# Alternative version using xarray (often more convenient)
def read_ampere_ncdf_xarray(ncdfname, OutVars=None):
    """
    Alternative version using xarray for netCDF reading
    
    Parameters same as read_ampere_ncdf
    """
    try:
        import xarray as xr
    except ImportError:
        print("xarray not available. Please install with: pip install xarray")
        return read_ampere_ncdf(ncdfname, OutVars)
    
    try:
        ds = xr.open_dataset(ncdfname)
    except Exception as e:
        print(f'could not open file: {ncdfname}')
        print(f'Error: {e}')
        return -1
    
    data = {}
    
    # Read basic variables
    basic_vars = ['npnt', 'year', 'doy', 'time', 'avgint', 'kmax', 'mmax', 
                  'res_deg', 'nLatGrid', 'nLonGrid', 'cLat_deg', 'mlt_hr', 
                  'geo_cLat_deg', 'geo_lon_deg', 'R']
    
    for var in basic_vars:
        if var in ds.variables:
            data[var] = ds[var].values
    
    # Read variables based on selection
    if OutVars is None or len(OutVars) == 0:
        # Read all variables
        all_vars = {
            'dB_Rgeo': 'db_R', 'dB_Ngeo': 'db_T', 'dB_Egeo': 'db_P',
            'dB_geo': 'db_geo', 'dB_Naagcm': 'db_Th_Th', 'dB_Eaagcm': 'db_Ph_Ph',
            'Jpar': 'jPar', 'd_J': 'del_jPar', 'd_dB_Rgeo': 'del_db_R',
            'd_dB_Ngeo': 'del_db_T', 'd_dB_Egeo': 'del_db_P',
            'd_dB_Naagcm': 'del_db_Th_Th', 'd_dB_Eaagcm': 'del_db_Ph_Ph'
        }
        
        for data_key, nc_key in all_vars.items():
            if nc_key in ds.variables:
                data[data_key] = ds[nc_key].values
    else:
        # Read selected variables
        var_mapping = {
            'J': [('Jpar', 'jPar'), ('d_J', 'del_jPar')],
            'B_GEO': [('dB_Rgeo', 'db_R'), ('dB_Ngeo', 'db_T'), ('dB_Egeo', 'db_P'),
                     ('dB_geo', 'db_geo'), ('d_dB_Rgeo', 'del_db_R'),
                     ('d_dB_Ngeo', 'del_db_T'), ('d_dB_Egeo', 'del_db_P')],
            'B_GCM': [('dB_Naagcm', 'db_Th_Th'), ('dB_Eaagcm', 'db_Ph_Ph'),
                     ('d_dB_Naagcm', 'del_db_Th_Th'), ('d_dB_Eaagcm', 'del_db_Ph_Ph')]
        }
        
        for out_var in OutVars:
            if out_var in var_mapping:
                for data_key, nc_key in var_mapping[out_var]:
                    if nc_key in ds.variables:
                        data[data_key] = ds[nc_key].values

    
    ds.close()
    return data




def read_Jpar(from_year_index, nr_days, directory_path):
    """
    Reads Ampere data, handles missing data and returns the Birkeland current
    data with nan where there is missing data.
    
    DEPENDENT ON read_ampere_ncdf 

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
        Array of satellite co-latitude for 1200 positions each dt
    geo_lon_deg : np.array
        Array of satellite longitude for 1200 positions each dt
    missing_full_days : TYPE
        DESCRIPTION.
    missing_indices : TYPE
        DESCRIPTION.

    """
    # 2009 has incomplete data
    # Reading in Ampere data:
    amp_root = Path(directory_path)
    year_dirs = [d for d in amp_root.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}", d.name)]
    
    missing_dates = []
    missing_points = []
    missing_indices = []
    
    Jpar = np.zeros((nr_days * 720, 1200))
    geo_cLat_deg = np.zeros((nr_days * 720, 1200))
    geo_lon_deg = np.zeros((nr_days * 720, 1200))
    
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
                                geo_cLat_deg[start_indice:end_indice] = np.nan
                                geo_lon_deg[start_indice:end_indice] = np.nan
                                
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
                            
                            full_day_data_cLat = np.full((720, 1200), np.nan)
                            full_day_data_cLat[actual_indices] = np.array(data["geo_cLat_deg"])
                            
                            full_day_data_lon = np.full((720, 1200), np.nan)
                            full_day_data_lon[actual_indices] = np.array(data["geo_lon_deg"])
                        
                        else:
                            full_day_data = np.array(data["Jpar"])
                            
                            full_day_data_cLat = np.array(data["geo_cLat_deg"])
                            full_day_data_lon = np.array(data["geo_lon_deg"])

                        start_indice = i * 720
                        end_indice = start_indice + 720
                        
                        Jpar[start_indice:end_indice] = full_day_data
                        geo_cLat_deg[start_indice:end_indice] = full_day_data_cLat
                        geo_lon_deg[start_indice:end_indice] = full_day_data_lon
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
    
    missing_full_days = np.array(missing_dates)
    missing_indices = np.array(missing_indices)
    
    print("Final shapes:", Jpar.shape, geo_cLat_deg.shape, geo_lon_deg.shape)#%%
    print(f"There are {len(missing_dates)} full days missing and an additional {len(missing_indices)-(len(missing_dates) * 720)} missing indices")
    
    return Jpar, geo_cLat_deg, geo_lon_deg, missing_full_days, missing_indices

