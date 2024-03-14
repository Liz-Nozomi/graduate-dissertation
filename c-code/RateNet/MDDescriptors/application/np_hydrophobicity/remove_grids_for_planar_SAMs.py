#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_grids_for_planar_SAMs.py

The purpose of this function is to remove grid points from the top and bottom 
SAM system. The idea is that we want to avoid issues with the top / bottom 
of the grid points to influence our systems.

Written by: Alex K. Chew (03/28/2020)
"""

import sys
import os
import numpy as np

## IMPORTING MDTRAJ
import mdtraj as md

## LOADING DAT FILE FUNCTION
from MDDescriptors.surface.core_functions import load_datafile

## IMPORTING CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## PLOTTERS
import MDDescriptors.core.plot_tools as plotter

## PICKLE TOOLS
from MDDescriptors.core.pickle_tools import pickle_results

## IMPORTING CORE FUNCTIONS
from MDDescriptors.surface.core_functions import write_datafile

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing 

## DEFINING GOLD RESIDUENAME
GOLD_RESNAME='AUI'
WATER_RESNAME='HOH'

    
### FUNCTION TO FIND NEW GRID BASED ON SOME CUTOFF
def extract_new_grid_for_planar_sams(grid,
                                     water_below_z_dim,
                                     water_above_z_dim):
    '''
    This function extracts a new set of grid points given the inputs for 
    planar SAMs
    INPUTS:
        grid: [np.array]
            grid information
        water_below_z_dim: [float]
            avg z dimension of water below
        water_above_z_dim: [float]
            avg z dimension for water above
    OUTPUTS:
        new_grid: [np.array]
            new grid array
        grid_within_range: [np.array]
            grid within range
    '''
    ## GETTING ALL GRID POINTS ABOVE AND BELOW
    grid_within_range = np.where( np.logical_and(  grid[:,-1] > water_below_z_dim,  
                                                   grid[:,-1] < water_above_z_dim ) )[0]
    
    ## GETTING ALL GRID POINTS
    new_grid = grid[grid_within_range]

    return new_grid, grid_within_range

######################
### CLASS FUNCTION ###
######################
class remove_grid_for_planar_SAMs:
    '''
    The purpose of this function is to remove grid points for planar SAMs 
    that are unreasonble and not important. This is important for SAMs 
    with a vacuum layer.
    INPUTS:
        path_gro: [str]
            path to gro file
        path_to_grid: [str]
            path to grid file
        gold_resname: [str]
            resname to gold
        water_resname: [str]
            resname to water
        traj: [obj]
            trajectory object
        grid: [np.array]
            grid array
    OUTPUTS:
        self.new_grid: [np.array]
            new grid array
    FUNCTIONS:
        load_grid_and_traj:
            function that loads grid and trajectory
        find_new_grid:
            function that finds new grid points
        print_summary:
            function that prints the summary
        plot_grid:
            function that plots the grid
            USAGE OF PLOTTING:
            updated_grid = remove_grid_for_planar_SAMs(**remove_grid_inputs)
            fig, ax = updated_grid.plot_grid()
        
    '''
    ## DEFINING INITIAL
    def __init__(self,
                 path_gro,
                 path_to_grid = None,
                 grid = None,
                 traj = None,
                 gold_resname = GOLD_RESNAME,
                 water_resname = WATER_RESNAME):
        ## STORING GRO FILE
        self.path_gro = path_gro
        self.path_to_grid = path_to_grid
        self.grid = grid
        self.traj = traj
        self.gold_resname = gold_resname
        self.water_resname = water_resname
        ## LOADING
        self.load_grid_and_traj()
        
        ## FINDING NEW GRID
        self.find_new_grid()
        
        return
    
    ### FUNCTION TO LOAD
    def load_grid_and_traj(self):
        ''' Function that loads grid and traj '''
        
        ## LOADING THE GRID
        if self.grid is None:
            self.grid = load_datafile(self.path_to_grid)
        
        ## LOADING GRO FILE
        if self.traj is None:
            self.traj = md.load(self.path_gro)
           
        return
    
    ### FUNCTION TO FIND UPPER AND LOWER INDEX
    def find_upper_lower_grid(self, grid):
        '''
        This function finds the upper and lower grid index
        INPUTS:
            self:
                class object
            grid: [np.array, shape = (N,3)]
                grid xyz values
        OUTPUTS:
            upper_grid: [np.array]
                grid values above the gold core
            lower_grid: [np.array]
                grid values below
        '''
        ## FINDING RESNAMES
        unique_resnames = calc_tools.find_unique_residue_names(traj = self.traj)
        
        ## CHECKING IF RESNAME IS DEFINED
        if GOLD_RESNAME not in unique_resnames:
            print("Error, gold is not defined")
            print("Gold resname: %s"%(GOLD_RESNAME) )
            print("Unique resnames:")
            print(unique_resnames)
            sys.exit(1)
            
        ## GETTING ATOM INDEX (ASSUMING ONE GOLD)
        gold_atom_index = calc_tools.find_residue_atom_index(traj = self.traj,
                                                             residue_name = self.gold_resname)[1][0]
        
        ## GETTING Z DIMENSION
        gold_xyz = self.traj.xyz[0,gold_atom_index,:]
        z_dim_gold = gold_xyz[:,-1]
        
        ## GETTING AVG Z DIM
        avg_z_gold = np.mean(z_dim_gold)
        
        ## GETTING ALL WATER ABOVE AND BELOW
        above_index = np.where(grid[:,2] >= avg_z_gold)[0]
        below_index = np.where(grid[:,2] < avg_z_gold)[0]
        
        ## GETTING GRIDS
        upper_grid = grid[above_index]
        lower_grid = grid[below_index]
        
        return upper_grid, lower_grid
    
    ### FUNCTION TO GET INFORMATION
    def find_new_grid(self, grid = None):
        '''
        The purpose of this function is to get new grid points. 
        INPUTS:
            self:
                class object
            grid: [logical]
                grid input. If None, then we will just use self.grid. If there 
                is a new grid you want to test (thus avoiding the loading of traj), 
                then include something here.
        OUTPUTS:
            self.new_grid: [np.array]
                grid output that omits grid points above and below and average 
                water z distance
            self.grid_within_range: [np.array]
                logical to get the grids within range
        '''
        if grid is None:
            grid = self.grid
        else:
            print("Since grid was inputted, using new grids instead of self.grid")
        ## FINDING RESNAMES
        unique_resnames = calc_tools.find_unique_residue_names(traj = self.traj)
        
        ## CHECKING IF RESNAME IS DEFINED
        if GOLD_RESNAME not in unique_resnames:
            print("Error, gold is not defined")
            print("Gold resname: %s"%(GOLD_RESNAME) )
            print("Unique resnames:")
            print(unique_resnames)
            sys.exit(1)
            
        ## GETTING ATOM INDEX (ASSUMING ONE GOLD)
        gold_atom_index = calc_tools.find_residue_atom_index(traj = self.traj,
                                                             residue_name = self.gold_resname)[1][0]
        
        ## GETTING Z DIMENSION
        self.gold_xyz = self.traj.xyz[0,gold_atom_index,:]
        z_dim_gold = self.gold_xyz[:,-1]
        ## GETTING AVG Z DIM
        self.avg_z_gold = np.mean(z_dim_gold)
        
        ## FINDING ALL WATER ATOM INDICES
        water_heavy_atom_index = calc_tools.find_heavy_atoms_index_of_residues_list(traj = self.traj,
                                                                                    residue_list = [WATER_RESNAME])
        
        ## FINDING ALL HEAVY ATOM XYZ
        z_dim_water = self.traj.xyz[0,water_heavy_atom_index,-1]
        
        ## GETTING ALL WATER ABOVE AND BELOW
        self.water_above_index = np.argwhere(z_dim_water > self.avg_z_gold)
        self.water_below_index = np.argwhere(z_dim_water < self.avg_z_gold)
        
        ## GETTING WATER Z DIM
        self.water_above_z_dim = np.mean(z_dim_water[self.water_above_index])
        self.water_below_z_dim = np.mean(z_dim_water[self.water_below_index])
    
        ## GETTING ALL GRID POINTS ABOVE AND BELOW
        self.grid_within_range = np.where( np.logical_and(  grid[:,-1] > self.water_below_z_dim,  
                                                            grid[:,-1] < self.water_above_z_dim ) )[0]
        
        ## GETTING ALL GRID POINTS
        self.new_grid = grid[self.grid_within_range]
        
        ## PRINTING SUMMARY
        self.print_summary()
        return self.new_grid
    
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function prints the summary'''            
        ## PRINTING
        print("Avg Z gold: %.3f"%(self.avg_z_gold) )
        print("----------------------------")
        print("Total waters above: %d"%(  len(self.water_above_index) ) )
        print("Avg Z water above: %.3f"%(  self.water_above_z_dim) )
        
        print("----------------------------")    
        print("Total waters below: %d"%(  len(self.water_below_index) ) )
        print("Avg Z water below: %.3f"%(  self.water_below_z_dim ) )
        
        print("----------------------------")    
        print("Number of grid originally: %d"%(len(self.grid) ))
        print("Number of grid within range: %d"%(len(self.new_grid) ))
        return
    
    ### FUNCTION TO PLOT THE POINTS
    def plot_grid(self):
        '''
        The purpose of this function is to plot the grid.
        INPUTS:
        
        OUTPUTS:
            
        '''
        
        ## CREATING 3D AXIS
        fig, ax = plotter.create_3d_axis_plot()
        
        ## ADDING SCATTER OF GOLD
        ax.scatter(self.gold_xyz[:,0],
                   self.gold_xyz[:,1],
                   self.gold_xyz[:,2],
                   s=100,
                   c='yellow',
                   edgecolors='k',
                   alpha=0.5,
                   label="gold"
                   )
        
        ## ADDING SCATTER OF GRID
        ax.scatter(self.grid[:,0],
                   self.grid[:,1],
                   self.grid[:,2],
                   s=10,
                   c='gray',
                   edgecolors='k',
                   alpha=0.5,
                   label="old_grid"
                   )
        
        ## ADDING SCATTER OF GRID
        ax.scatter(self.new_grid[:,0],
                   self.new_grid[:,1],
                   self.new_grid[:,2],
                   s=100,
                   c='red',
                   edgecolors='k',
                   label="new_grid"
                   )
        
        ## ADDING LEGEND
        ax.legend()
    
        
        return fig, ax

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## RUNNING TESTING    
    if testing == True:
    
        ## DEFINING PATH TO SIM
        path_to_sim=r"/Volumes/shared/np_hydrophobicity_project/simulations/20200326-Planar_SAMs_with_larger_z_frozen_with_vacuum/NVTspr_50_Planar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
        
        ## DEFINING GRO FILE
        gro_file="sam_prod.gro"
        
        ## DEFINING PATH TO GRO
        path_gro = os.path.join(path_to_sim,gro_file)
        
        ## DEFINING GRID PATH
        path_to_grid="/Volumes/shared/np_hydrophobicity_project/simulations/20200326-Planar_SAMs_with_larger_z_frozen_with_vacuum/NVTspr_50_Planar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps/norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-50000/grid-0_1000/out_willard_chandler.dat"
        
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## DEFINING PATH TO GRO
        parser.add_option('--path_gro', dest = 'path_gro', help = 'Path of gro file', default = '.', type=str)
        parser.add_option('--path_to_grid', dest = 'path_to_grid', help = 'Path of grid file', default = '.', type=str)
        ## DEFINING PATH TO PICKLE
        parser.add_option('--path_to_pickle', dest = 'path_to_pickle', help = 'Path to store the pickle', default = None, type=str)
        
        ## DEFINING PATH TO OUTPUT GRID
        parser.add_option('--path_output_grid', dest = 'path_output_grid', help = 'Path of output grid file', default = '.', type=str)
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## GETTING OUTPUTS
        path_gro = options.path_gro
        path_to_grid = options.path_to_grid
        path_to_pickle = options.path_to_pickle
        path_output_grid = options.path_output_grid
        
    
    #%%
    ## DEFINING GRID INPUTS
    remove_grid_inputs={
            "path_gro" : path_gro,
            "path_to_grid": path_to_grid,                        
            }
    
    ## REMOVING GRID FOR PALNAR SAMS
    updated_grid = remove_grid_for_planar_SAMs(**remove_grid_inputs)
    
    '''
    ## PLOTTING
    fig, ax = updated_grid.plot_grid()
    '''
    
    ## DEFINING PICKLE PATH
    if path_to_pickle is not None:
        ## STORING CLASS OBJECT
        pickle_results(results = [updated_grid],
                       pickle_path = os.path.join(path_to_pickle),
                       verbose = True,
                       )
    
    ## OUTPUTTING TO GRID
    if path_output_grid is not None:
        ## WRITING TO DATA FILE
        write_datafile(path_to_file=path_output_grid,
                       data = updated_grid.new_grid)