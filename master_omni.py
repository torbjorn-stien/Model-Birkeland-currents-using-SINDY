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
from pysindy.utils import compare_coefficient_plots

ps.utils.
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

def stack_data(*arrays):
    """
    Stacks and transposes arrays to be in the correct format to be used as u
    in pysindy, for multiple trajectories with multiple u's.
    
    Is designed to be used on data that has first gone through the 
    subset_data() function.
    
    Parameters:
        *arrays: Variable number of arrays (must have the same length). 
        One array for each potential control feature
    Returns:
        list: A list of arrays, where each array is the control features for
        one trajectory.
    """
    # Perform the stacking and transposing operation
    result = []
    for i in range(len(arrays[0])):
        stacked = np.vstack([arr[i] for arr in arrays]).T
        result.append(stacked)
    
    return result
#%%

omni_data = pod.GetOMNI([2010, 2022], Res = 1)


#%%

print(omni_data.dtype.names)


#%%
plt.figure(figsize=(11,6))
ax0 = pod.PlotOMNI(["AU","ByGSE", "BzGSE", ], [20100101, 20100102] , fig = plt)
plt.tight_layout()
plt.show()

#%%

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


#%%

print(np.any(np.isnan(Bx)))

#%%
start, end, nan_len, no_nan_len = find_nans(Vx)

#%%

interp_len = 15

B_interp = interpolate_nans(B, interp_len)
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

train_start = 0
train_end = len(AU)
traj_len = 50000

# AU Does not contain NaNs
AU_train = AL[train_start:train_end]
B_train = B_interp[0][train_start:train_end]
Bx_train = Bx_interp[0][train_start:train_end]
By_train = By_interp[0][train_start:train_end]
Bz_train = Bz_interp[0][train_start:train_end]


def mean_norm(data):
    norm = (data - np.nanmean(data))/np.nanmean(data)
    
    return norm

AU_train = mean_norm(AU_train)
B_train = mean_norm(B_train)
Bx_train = mean_norm(Bx_train)
By_train = mean_norm(By_train)
Bz_train = mean_norm(Bz_train)

test = subset_data([AU_train, B_train], 
                   traj_len, 1)
"""
AU_train, B_train = ps.utils.drop_nan_samples(AU_train, B_train)

Does not work

"""

#%%
# Moving average function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply moving average with a window size 
window_size = 100

AU_smoothed = []
B_smoothed = []
Bx_smoothed = []
for trajectory in range(len(test[0])):
    AU_smoothed.append(moving_average(test[0][trajectory], window_size=window_size))
    B_smoothed.append(moving_average(test[1][trajectory], window_size=window_size))
    Bx_smoothed.append(moving_average(test[2], window_size = window_size))


#%%


dt = 1

X = AU_smoothed[0]#test[0]
length = len(X)
u_delayed = B_smoothed[0]#test[1]
# Reshape to (2841, 1)
X = np.concatenate(AU_smoothed, axis=0)#[:length]  # Shape: (total_time_steps, n_features)
u_delayed = np.concatenate(B_smoothed, axis=0)#[:length]  # Shape: (total_time_steps, n_features)

optimizer = ps.EnsembleOptimizer(opt=  ps.SR3(reg_weight_lam= 0.1,relax_coeff_nu=1.0,
                                             regularizer="L2"), 
                                 bagging=True, library_ensemble=True,
                                 n_models = 1000) # Default aggregator is median

optimizer = ps.SR3(reg_weight_lam= 0.0001,relax_coeff_nu=0.5, regularizer="L2")

feature_names = None

# Finite difference amplifies noise in data.
differentiation_method = ps.SmoothedFiniteDifference()#smoother_kws={'window_length': 3})

#H_xt = np.array([1, 2])  # Adjust these values if necessary
# Only temporal grid gives dx/dt = 1/2 dx/dt + 1/2 dx/dt, R² = 1
#lib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=4),
#                    spatiotemporal_grid=spatio_temporal_grid, # spatio_temporal_grid -> "tuple" has no attribute "shape"
#                    include_interaction = True, include_bias = True,
#                    implicit_terms=True, derivative_order = 0,
#                    )

# Various attempted libraries
#input_lib = ps.CustomLibrary()

temporal_grid = np.arange(0, len(X), 1)

lib = ps.WeakPDELibrary(spatiotemporal_grid=temporal_grid)



wpde_lib = ps.WeakPDELibrary(function_library= ps.PolynomialLibrary(degree=3),
                                   spatiotemporal_grid=temporal_grid,
                                   derivative_order=2,
                                   include_bias=True, include_interaction=True)

combined_lib = ps.GeneralizedLibrary(libraries = [ps.PolynomialLibrary(degree = 3), 
                                                  ps.FourierLibrary(n_frequencies=1), 
                                                  wpde_lib],
                                     )#tensor_array = [[1, 1]])
# Initialize SINDy model
mod = ps.SINDy(optimizer = optimizer,
               feature_library= wpde_lib,
               differentiation_method=differentiation_method)

# Fit SINDy model
mod.fit(x = X, t = dt, u = u_delayed)
# Print the best fit approximation
mod.print()
print(mod.score(X, dt, u = u_delayed)) # R² score of model
#%%

sim = mod.simulate([0], t=np.arange(0, 5), u = u_delayed[:5])

#%%

fig = plt.figure(figsize = [11, 6])
axes = fig.subplots(1, 2)

fig = compare_coefficient_plots(
    mod.coef_,)

#%%
plt.plot(X)
plt.plot(u_delayed)
plt.show()

#%%
print(f"Shape of x0: {x0.shape}")
print(f"Shape of temporal_grid: {temporal_grid.shape}")
print(f"Shape of u_delayed: {u_delayed.shape}")
#%%
import numpy as np
# Ensure X[0] is correctly formatted
if X.ndim == 1:
    x0 = np.array([X[0]])  # For 1D systems
else:
    x0 = X[0]  # For multi-dimensional systems
# Check shapes of temporal_grid and u_delayed

print(f"Shape of temporal_grid: {temporal_grid.shape}")
print(f"Shape of u_delayed: {u_delayed.shape}")

# Simulate the system
sim = mod.simulate(x0, t=temporal_grid[:100], u=u_delayed[:100])
# Plot the results
plt.plot(X, label="Original Data")
plt.plot(sim, label="Simulation")
plt.legend()
plt.show()


#%%
Bx_smoothed = moving_average(Bx_train, window_size=window_size)
By_smoothed = moving_average(By_train, window_size=window_size)
Bz_smoothed = moving_average(Bz_train, window_size=window_size)

linewidth =0.9
plt.plot(AU_train, linewidth = linewidth, alpha=0.3)
plt.plot(B_train, linewidth = linewidth)
plt.plot(AU_smoothed, linewidth = linewidth, ls = "-.")
plt.plot(B_smoothed, linewidth = linewidth, ls = "--")
#plt.xlim(1000, 1600)
plt.show()

plt.plot(AU_smoothed, linewidth = linewidth, ls = "-.")
plt.plot(B_smoothed, linewidth = linewidth, ls = "--")
plt.plot(Bx_smoothed, linewidth = linewidth,)
plt.plot(By_smoothed, linewidth = linewidth, )
plt.plot(Bz_smoothed, linewidth = linewidth,)
plt.show()



