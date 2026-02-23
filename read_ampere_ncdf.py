# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:14:52 2025

@author: tbear
"""

import numpy as np
import netCDF4 as nc

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