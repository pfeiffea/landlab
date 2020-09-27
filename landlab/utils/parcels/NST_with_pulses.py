# -*- coding: utf-8 -*-
"""
This script sets up and adds parcels to a network model grid using the 
landlabb.utils.parcels BedParcelInitializer and SedimentPulser classes. 

Transport of the parcels in response to precipitation is modeled using the 
network_sediment_transport component


NOTE - restart kernal before running script
"""

#%% SETUP WORKSPACE
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants
import numpy as np

from landlab.data_record import DataRecord
from landlab import NetworkModelGrid
from landlab.utils.parcels import SedimentPulser
from landlab.components import FlowDirectorSteepest, NetworkSedimentTransporter
from landlab import NetworkModelGrid
from landlab.utils.parcels import BedParcelInitializer
from landlab.plot import graph
from landlab.plot import plot_network_and_parcels


#%% TEST NETWORK MODEL GRID

y_of_node = (0, 100, 200, 200, 300, 400, 400, 125)
x_of_node = (0, 0, 100, -50, -100, 50, -150, -100)
nodes_at_link = ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5), (4, 6))

grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)

grid.at_link["channel_width"] = np.full(grid.number_of_links, 2.0)  # m
grid.at_link["channel_slope"] = np.full(grid.number_of_links, .04)  # m / m
grid.at_link["reach_length"] = np.full(grid.number_of_links, 100.0)  # m
grid.at_node["topographic__elevation"] = np.array([0.0, 10, 20, 20, 20, 40, 50, 50])/5
grid.at_node["bedrock__elevation"] = np.array([0.0, 10, 20, 20, 20, 40, 50, 50])/5
grid.at_link["drainage_area"] = [0.1, 0.015, 0.0625, 0.01, 0.04, 0.0075, 0.02]  # km2


# See plot for link ID
plt.figure(0)
graph.plot_graph(grid, at="node,link")

# %% MODEL PARAMETERS

# BedParcelInitializer parameters
BC =[0.08,-0.05]  # median grain size of initial parcels = BC[0]*ContributingArea ^ BC[1]
parcel_volume = 1 # m3 
num_starting_parcels = 10  # number of parcels in each reach 

# SedimentPulser parameters
P_d50 = 0.03 # median grain size of pulse parcels
P_parcel_volume = 1 # volume of each pulse parcel
P_links =  [1,6] # links (ID) that recieve pulse parcels
num_pulse_parcels = 50 # number of parcels in pulse to each link in P_links

# Daily average precipitation threshold [mm/hr] above which a pulse occurs
P_thresh = 130             

# %% MODEL HYDROLOGY

# Time series of daily average hourly precipitation rate (effective) [mm/hr]
Pt = np.array([10,5,5,5,5,20,50,80,150,95,80,95,30,5,5,
      5,5,5,0,1,50,135,110,20,95,60,30,2,2,2,10,8,15,125,125,125,110,100,125
      ,120,90,80,30,5,5])

# Contributing area to each link outlet [m2]
CAm = np.expand_dims((grid.at_link["drainage_area"]*1000*1000),axis=1)

# Precipitation time series [m/s]
Ptm = np.expand_dims((Pt/(3600)/1000),axis=0)

# Time series of average daily flow rate at each link
Qall = np.dot(CAm,Ptm) # daily average flow rate at all links [m3/s]

fdepth = (0.36*Qall**0.31) # flow depth, Castro and Jackson, 2001 [m]

# set initial flow depth for instantiating network model grid
grid.at_link["flow_depth"] = fdepth[:, 0]

# time step parameters for NST
timesteps = 45  # number of timesteps in model run
ts_h = 24  # hours in each time step (LandslideProbabiity is set up to run daily)
# convert time step to seconds
dt = 60 * 60 * ts_h # timestep in seconds


#%% Instantiate and run BedParcelInitializer

# set up the bed
initialize_parcels = BedParcelInitializer(grid,
                                          median_number_of_starting_parcels=num_starting_parcels)

parcels = initialize_parcels(discharge_at_link=None,user_parcel_volume=parcel_volume,
                  user_d50=BC)  #[0.03,-0.05]

## d50 based on dominate flow  tends to create large grain diameters and parcel volumes.
##  - large parcel volumes quickly go to the unactive bed layer
##  - large grain sizes are not mobilized during high flow rates

# discharge_at_link = np.full(grid.number_of_links, 1)  # m^3 / s
# parcels = initialize_parcels(discharge_at_link=Qgage) 


#%% Instantiate network sediment transporter and Sediment Pulser classes

# run flow director steepest to add flow direction field
fd = FlowDirectorSteepest(grid, "topographic__elevation")
fd.run_one_step()

nst = NetworkSedimentTransporter(
    grid,
    parcels,
    fd,
    bed_porosity = 0.3,
    g=9.81,
    fluid_density=1000,
    transport_method="WilcockCrowe",
)

v = 60*60*24 # day to second conversion
def time_to_pulse_L(time):
    Ptime = list(np.where(Pt>=P_thresh)[0]*v)
    return  time in Ptime

make_pulse = SedimentPulser(grid, parcels = parcels, time_to_pulse = time_to_pulse_L)

#%% MODEL SEDIMENT TRASNPORT WITH SEDIMENT PULSES USING NST

pulseDict = {}

for ts in range(timesteps):
    
    # update flow depth
    grid.at_link["flow_depth"] = fdepth[:, ts]
    
    # add a pulse of material to links
    make_pulse(nst._time, P_d50, num_pulse_parcels,P_links,P_parcel_volume) # run __call__ and parcels.add_item

    # record pulse attributes
    try:
        pulseDict[ts] = make_pulse._parcels.dataset 
    except:
        pulseDict[ts] = np.nan    
    
    nst.run_one_step(dt)
    
    print('ran one step')


#%%plots

# (1) parcel location in network
for i in np.arange(0,timesteps,1):
    print(i)
    fig = plot_network_and_parcels(
        grid, parcels, 
        parcel_time_index=i,  #index of time, not the time value
        parcel_color_attribute="D",
        link_attribute="sediment_total_volume", 
        parcel_size=30, 
        parcel_alpha=1.0)


# (2) volume parcels, including outlet
fig, axs = plt.subplots(1, 1,figsize=(9,5))
plt.plot(
    parcels.time_coordinates, np.nansum(parcels.dataset["volume"].values, axis=0),
    "-",linewidth = 2
)
plt.title("Total volume of parcels through time including outlet")
plt.xlabel("time")
plt.ylabel("total volume of parcels")


# (3) volume parcels upstream of outlet
InChanVol = parcels.dataset["volume"].values*1
InChanVol[np.where(parcels.dataset.element_id.values==-2)] = 0

fig, axs = plt.subplots(1, 1,figsize=(9,5))
plt.plot(
    np.array(parcels.time_coordinates)/v, np.nansum(InChanVol, axis=0), "-",
    linewidth = 2,
)
plt.title("Total volume of parcels through time in the channel network",fontsize = 14)
plt.xlabel("time [days]",fontsize = 14)
plt.ylabel("total volume of parcels $[m^3]$",fontsize = 14)
plt.ylim([0,np.nansum(parcels.dataset["volume"].values, axis=0).max()*1.2])


# (4) precipitation, volume parcels

ActInChanVol = InChanVol*1 
ActiveVol = parcels.dataset["active_layer"].values
ActInChanVol[np.where(ActiveVol!=1)] = 0

#volume active parcels upstream of outlet
fig, axs = plt.subplots(2, 1, figsize=(5,9))
axs = axs.ravel()
axs[0].plot(
    np.array(Pt), "-",
    color = 'black', linewidth = 2,label = 'total volume'
)
axs[1].plot(
    np.array(parcels.time_coordinates)/v, np.nansum(InChanVol, axis=0), "-",
    color = 'black', linewidth = 2,label = 'total volume'
)
axs[1].plot(
    np.array(parcels.time_coordinates)/v, np.nansum(ActInChanVol, axis=0), "--",
    color = 'black', alpha = 0.5, linewidth = 2,label = 'active volume'
)
axs[0].set_ylabel("daily average hourly P [mm/hr]",fontsize = 14)
axs[1].set_xlabel("time [days]",fontsize = 14)
axs[1].set_ylabel("in-network parcel volume $[m^3]$",fontsize = 14)
axs[1].set_ylim([0,np.nansum(parcels.dataset["volume"].values, axis=0).max()*1.2])
axs[0].tick_params(axis = 'both', which = 'major', labelsize = 12)
axs[1].tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.legend(fontsize = 14, loc = 'best')

