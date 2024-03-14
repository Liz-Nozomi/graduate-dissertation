# -*- coding: utf-8 -*-
"""
generate_hydration_maps.py
The purpose of this function is to generate hydration maps for a surface. 
This code assumes you already have a surface defined a set of grid points.

Written by: Alex K. Chew (12/13/2019)
Using code developed by Bradley C. Dallin


Functions/classes:
    compute_num_dist:
        computes number distribution given neighbor array
    calc_neighbors:
        class function that computes neighbors

"""
## IMPORTING NECESSARY MODULES
import os
import numpy as np

## IMPORTING TOOLS
from MDDescriptors.core.import_tools import import_traj

## IMPORTING CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## IMPORTING CORE FUNCTIONS
from MDDescriptors.surface.core_functions import load_datafile

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import R_CUTOFF, MAX_N

## CHECK PATH
from MDDescriptors.core.check_tools import check_path

## IMPORTING TIME
from MDDescriptors.core.track_time import track_time

## CRITICAL FUNCTION TO USE PERIODIC CKD TREE
from MDDescriptors.surface.periodic_kdtree import PeriodicCKDTree

### FUNCTION TO GET UNNORMALIZED DISTRIBUTION
def compute_num_dist(num_neighbors_array,
                     max_neighbors,):
    '''
    This function generates the number distribution given number of 
    neighbors array versus frame

    Parameters
    ----------
    num_neighbors_array: [np.array, shape = (num_grid, num_frames)]
        number of neighbors within the cutoff radius
    max_neighbors: [int]
        max neighbors

    Returns
    -------
    num_neighbor_dist: [np.array, shape=(num_grid_points, max_N)]
        number distributions of neighbors

    '''
    ## GENERATING DISTRIBUTION CONTAINING THE NUMBER OF GRID POINTS AND MAX NUMBER OF ITEMS
    num_neighbor_dist = np.zeros( shape = ( num_neighbors_array.shape[0], max_neighbors) )
    
    ## LOOPING THROUGH EACH FRAME
    for idx, array in enumerate(num_neighbors_array):
        ## COMPUTING NUM DIST
        num_neighbor_dist[idx, :]= np.histogram( array, 
                                                 bins = int( max_neighbors ), 
                                                 range = [ 0., max_neighbors ] )[0]

    return num_neighbor_dist

################################################   
### CLASS FUNCTION TO COMPUTE HYDRATION MAPS ###
################################################
class calc_neighbors:
    '''
    The purpose of this function is to calculate hydration maps given 
    a set of grid points. The main idea is that we count the number of water 
    molecules (denoted by the heavy atom) nearby the surface. 
    
    NOTE:
        - The assumption is that you do not have a moving box. If so, then 
        you may have issues with the grid points not being in the correct 
        positions. Therefore, you should be running NVT simulations, where 
        the volume of the box is fixed. 
        - This also assumes you want a periodic box computation. We will 
        need to readjust if no periodic boundary conditions are necessary. 
        - This class has been verified with gmx select to reproduce the same 
        neighbors list. 
    
    Parameters
    ----------
    traj: [obj]
        trajectory object. This is only used to get the unit cell lengths. 
    cutoff_radius: [float]
        cutoff of radius between grid and atom index
    grid: [np.array, shape=(num_points, 3)]
        xyz positions of grid points
    residue_list: [list]
        list of residue names you are interested in computing hydration maps. 
        By default, the list includes water and counterions. However, you 
        could only consider water, and so on. 
        
        Note: if 'all_heavy' is part of the residue list, then we will consider 
        all possible heavy atoms within the grid (water, ligands, counterions, etc.)
        
    cutoff_radius: [float]
        cutoff radius used to count the atoms
    verbose: [logical]
        True if you want to print verbosely
        
    Returns
    -------
    self.atom_index: [np.array]
        atom indices
    self.box: [np.array]
        box lengths in x, y, z dimensions
        
    Functions
    ---------
    compute: 
        function to compute neighboring array, which is critical for hydration 
        map generations
    '''
    def __init__(self,
                 traj,
                 grid,
                 residue_list = ["HOH", "CL", "NA"],
                 cutoff_radius = R_CUTOFF,
                 verbose = True
                 ):
        ## STORING RESIDUE LIST
        self.residue_list = residue_list
        self.grid = grid
        self.cutoff_radius = cutoff_radius
        self.verbose = verbose
        
        ## GETTING ALL INDEXES OF WATER
        if 'all_heavy' not in residue_list:
            self.atom_index = calc_tools.find_heavy_atoms_index_of_residues_list(traj = traj,
                                                                                 residue_list = residue_list)
        else:
            print("Since 'all_heavy' is inside residue list, we will compute all heavy atoms.")
            self.atom_index = calc_tools.find_all_heavy_atoms(traj = traj)

        ## GETTING BOX SIZE
        self.box = traj.unitcell_lengths[ 0, : ] # ASSUME BOX SIZE DOES NOT CHANGE!
        
        ## PRINTING DETAILS
        if self.verbose is True:
            print("--- Neighboring summary ---")
            print("Residue names used: %s"%( ', '.join(self.residue_list) ) )
            print("Number of atoms to look at: %d"%(len(self.atom_index) ) )
            print("Number of grid points: %d" %(len(self.grid)) )
            print("Cutoff radius (nm): %.2f"%(self.cutoff_radius) )
        
        return

    ### FUNCTION TO GET NEIGHBORS ARRAY GIVEN TRAJECTORY
    def compute(self,
                traj,
                frames = [],
                print_rate = 100,
                ):
        '''
        The purpose of this function is to compute the neighbor array given 
        a grid and an atom index of interest. This code was originally 
        developed to compute number of water atoms within the vicinity 
        of a set of grid points. Note that this takes into account periodic 
        boundary conditions via a CKDTree (periodic)
        
        Parameters
        ----------
            traj: [obj]
                trajectory object         
            frames: [list]
                list of frames to compute for trajectory
            print_rate: [int]
                print rate, default = 100 frames. 
        Returns
        -------
            num_neighbors_array: [np.array, shape = (num_grid, num_frames)]
                number of neighbors within the cutoff radius
        '''
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[tuple(frames)]
        ## GETTING TOTAL FRAMES
        total_frames = traj.time.size
        
        ## GETTING THE TIME
        time_array = traj.time
        
        ## GETTING ARRAY
        num_neighbors_array = np.zeros( shape = ( len(self.grid), total_frames ) )
        
        ## LOOPING
        for frame in range(total_frames):
    
            ## DEFINING POSITIONS
            pos = traj.xyz[frame, self.atom_index, :]
            
            ### FUNCTION TO GET TOTAL NUMBER OF GRID
            T = PeriodicCKDTree(self.box, pos)
            
            ## COMPUTING ALL NEIGHBORS
            neighbors = T.query_ball_point(self.grid, r=self.cutoff_radius)
            
            ## LOOPING THROUGH THE LIST
            for n, ind in enumerate(neighbors):
                num_neighbors_array[n][frame] += len(ind)
                
            if self.verbose is True:
                parallel_frame = int(traj.time[frame])
                if parallel_frame % print_rate == 0:
                    print("PID %d: computing neighbors for frame: %d"%(os.getpid(), parallel_frame) )
                    
        return num_neighbors_array

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ##########################
    ### LOADING TRAJECTORY ###
    ##########################
    ## DEFINING MAIN SIMULATION
    main_sim=check_path(r"S:\np_hydrophobicity_project\simulations\PLANAR")
    ## DEFINING SIM NAME
    sim_name=r"FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
    ## DEFINING WORKING DIRECTORY
    wd = os.path.join(main_sim, sim_name)
    ## DEFINING GRO AND XTC
    # gro_file = r"sam_prod-0_1000-watO_grid.gro"
    # xtc_file = r"sam_prod-0_1000-watO_grid.xtc"
    gro_file = r"sam_prod.gro"
    xtc_file = r"sam_prod_1ns.xtc"
    
    ## LOADING TRAJECTORY
    traj_data = import_traj( directory = wd,
                             structure_file = gro_file,
                             xtc_file = xtc_file,
                             )
    
    
    ## DEFINING location
    analysis_folder = r"hyd_analysis"
    grid_folder = r"grid-0_1000"
    dat_file = r"out_willard_chandler.dat"
    
    ## DETINING HYDRATION MAP LOCATION
    path_grid = os.path.join(wd,
                             analysis_folder,
                             grid_folder,
                             dat_file
                             )
    
    ## LOADING THE GRID
    grid = load_datafile(path_grid)
    
    ## DEFINING TRAJECTORY (100 for debugging)
    traj = traj_data.traj[:100]

    #%% Testing neighbor computations
    
    ## IMPORTING TIME TRACKER
    time_tracker = track_time()
    
    ## GENERATING NEIGHBORS
    neighbors = calc_neighbors(traj = traj,
                               grid = grid,
                               residue_list = ["HOH", "CL", "NA"],
                               cutoff_radius = R_CUTOFF,
                               )
    
    ## COMPUTING NEIGHBOR ARRAY
    num_neighbor_array = neighbors.compute(traj = traj,)
    
    ## PRINTING
    time_tracker.time_elasped()
    
    ## FUNCTION TO GET NUMBER NEIGHBOR DISTRIBUTIONS
    num_neighbor_dist = compute_num_dist(num_neighbors_array = num_neighbor_array,
                                         max_neighbors = MAX_N)
 