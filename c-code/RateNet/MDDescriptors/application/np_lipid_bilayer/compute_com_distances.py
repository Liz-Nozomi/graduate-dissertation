#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_com_distances.py
The purpose of this script is to compute the COM distances between the gold core
and the lipid membrane.

Written by: Alex K. Chew (03/18/2020)

"""
import os
import numpy as np
## XVG FILE
from MDDescriptors.core.read_write_tools import read_xvg
## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv

### MAIN FUNCTION
def main_compute_com_distances(path_to_sim,
                               input_prefix,
                               group_1_resname= 'DOPC',
                               group_2_resname= 'AUNP',
                               rewrite= False
                               ):
    '''
    The purpose of this script is to compute the center of mass distances between 
    two groups.
    INPUTS:
        
    '''
    ## CONVERTING TRAJECTORY
    trjconv_func = convert_with_trjconv(wd = path_to_sim)
    
    ## GETTING COM DISTANCES
    output_xvg_file, output_ndx_file = trjconv_func.compute_com_distances(input_prefix = input_prefix,
                                                                          group_1_resname = group_1_resname,
                                                                          group_2_resname = group_2_resname,
                                                                          rewrite = rewrite)
    
    ## DEFINING PATH TO XVG
    path_to_dist_xvg = os.path.join(path_to_sim,
                                    output_xvg_file)
    
    ## LOADING THE XVG FILE
    xvg = read_xvg(path_to_dist_xvg)
    
    ## DATA
    xvg_data = np.array(xvg.xvg_data).astype('float')
    
    ## GETTING TIME ARRAY
    time_array = xvg_data[:,0]
    
    ## GETTING DISTANCE
    z_dist = xvg_data[:,-1]
    
    return xvg, xvg_data, time_array, z_dist


#%% DEBUGGING
if __name__ == "__main__":
    ## DEFINING PATH TO SIMULATION
    path_to_sim = r"/home/akchew/scratch/nanoparticle_project/nplm_sims/20200205-unbiased_ROT001/NPLM_unb-1.700_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"
    
    ## DEFINING PREFIX
    input_prefix = "nplm_prod"
    
    ## RUNNING COMMAND
    xvg, xvg_data, time_array, z_dist = main_compute_com_distances(path_to_sim = path_to_sim,
                                                                   input_prefix = input_prefix,
                                                                   group_1_resname= 'DOPC',
                                                                   group_2_resname= 'AUNP',
                                                                   )
    