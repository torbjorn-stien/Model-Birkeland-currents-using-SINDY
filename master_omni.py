#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:44:41 2026

@author: tos
"""

import pysindy as ps

import numpy as np
import matplotlib.pyplot as plt

import pyomnidata as pod
from nan_handling import find_nans, interpolate_nans, find_overlapping_clean_data, subset_data

import warnings
from pysindy.utils._axes import AxesWarning


warnings.filterwarnings("ignore", category=AxesWarning)


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

omni_data = pod.GetOMNI([2010, 2022], Res = 1)


#%%

print(omni_data.dtype.names)


#%%
plt.figure(figsize=(11,6))

ax0 = pod.PlotOMNI(["Vx"], [20100101, 20100101] , fig = plt)
plt.tight_layout()
plt.show()

#%%

AE = omni_data.AE
AL = omni_data.AL
AU = omni_data.AU
Bx = omni_data.BxGSE
By = omni_data.ByGSE
Bz = omni_data.BzGSE
Vx = omni_data.Vx
Vy = omni_data.Vy
Vz = omni_data.Vz


#%%

print(np.any(np.isnan(Bx)))

#%%
start, end, nan_len, no_nan_len = find_nans(Vx)

#%%

interp_len = 15


Bx_interp = interpolate_nans(Bx, interp_len)
By_interp = interpolate_nans(By, interp_len)
Bz_interp = interpolate_nans(Bz, interp_len)
Vx_interp = interpolate_nans(Vx, interp_len)
Vy_interp = interpolate_nans(Vy, interp_len)
Vz_interp = interpolate_nans(Vz, interp_len)

#%%

plt.figure(figsize=(11,6))

ax0 = plt.plot(Vx_interp[0][:1440], linewidth=1.1)
plt.tight_layout()
plt.show()

#%%

test = subset_data([AU, Bx_interp[0]], 150, 1)


#%%

dt = 1

X = test[0]

u_delayed = test[1]

optimizer = ps.EnsembleOptimizer(opt=  ps.SR3(reg_weight_lam= 0.03,relax_coeff_nu=5,
                                             regularizer="L2"), 
                                 bagging=True, library_ensemble=True,
                                 n_models = 20) # Default aggregator is median

feature_names = None

# Finite difference amplifies noise in data.
differentiation_method = ps.SmoothedFiniteDifference()#smoother_kws={'window_length': 5})

#H_xt = np.array([1, 2])  # Adjust these values if necessary
# Only temporal grid gives dx/dt = 1/2 dx/dt + 1/2 dx/dt, R² = 1
#lib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=4),
#                    spatiotemporal_grid=spatio_temporal_grid, # spatio_temporal_grid -> "tuple" has no attribute "shape"
#                    include_interaction = True, include_bias = True,
#                    implicit_terms=True, derivative_order = 0,
#                    )

# Various attempted libraries
#input_lib = ps.CustomLibrary()

lib = ps.PDELibrary()

combined_lib = ps.GeneralizedLibrary(libraries = [ps.PolynomialLibrary(), ps.FourierLibrary()],
                                     )#tensor_array = [[1, 1]])


# Initialize SINDy model
mod = ps.SINDy(optimizer = optimizer,
               feature_library= lib,
               differentiation_method=differentiation_method)

# Fit SINDy model
mod.fit(x = X, t = dt, u = u_delayed)

# Print the best fit approximation
mod.print()
print(mod.score(X, dt, u = u_delayed)) # R² score of model


#%%

print(np.any(np.isnan(Bx)))

plt.plot(Bx)
plt.show()


