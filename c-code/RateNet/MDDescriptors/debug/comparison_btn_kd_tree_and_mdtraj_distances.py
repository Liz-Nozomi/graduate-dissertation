# -*- coding: utf-8 -*-
"""
check_ligand_to_grid_overlap.py

This script checks if the ligand heavy atoms are within the grid points of 
the self-assembled monolayer. The inputs are the ligand names, 

Written by: Alex K. Chew (01/17/2020)

"""
## IMPORTING MODULES
import os
import mdtraj as md
import numpy as np

## LOADING DATA
from MDDescriptors.surface.core_functions import load_datafile

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORT TOOLS
import MDDescriptors.core.import_tools as import_tools

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## LOADING MD LIGANDS
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import load_all_ligand_residue_names

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## CRITICAL FUNCTION TO USE PERIODIC CKD TREE
from MDDescriptors.surface.periodic_kdtree import PeriodicCKDTree

## IMPORTING TRACK TIME
from MDDescriptors.core.track_time import track_time

## FUNCTION TO COMPUTE NUMBER OF ATOMS WITHIN SPHERE
def count_atoms_in_sphere( traj, pairs, r_cutoff = 0.33, periodic = True ):
    r"""
    The purpose of this function is to comptue the number of atoms within some 
    radius of cutoff.
    INPUTS:
        traj: [obj]
            trajectory object
        pairs: [np.array, shape=(N,2)]
            pairs array that you're interested in
        r_cutoff: [float]
            cutoff that you are interested in
    OUTPUTS:
        N: [np.array]
            array that is the same size as pairs across trajectories
    """
    dist = md.compute_distances( traj, pairs, periodic = periodic )
    N = np.sum( dist <= r_cutoff, axis = 1 )
    return N

### FUNCTION TO COMPUTE NEIGHBORS
def compute_neighbor_array_with_md_traj(traj,
                                        grid,
                                        atom_index,
                                        verbose = True):
    '''
    This function computes the neighbor array using md traj functions. First, 
    an atom is moved to the grid point, then we utilize the compute_distances 
    function to estimate the number of atoms in the sphere.
    INPUTS:
        traj: [obj]
            trajectory object
        grid: [np.array]
            grid points in x, y, z positions
        atom_index: [list]
            list of atom indices
    OUTPUTS:
        num_neighbors_array: [num_grid, num_frames]
            neighbors within the grid points
    '''
    ## GETTING TOTAL FRAMES
    total_frames = traj.time.size

    ## GENERATING ZEROS 
    num_neighbors_array = np.zeros( shape = ( len(grid), total_frames ) )
    
    ## FINDING NO ATOM INDEX
    atom_not_index = np.array([ atom.index for atom in traj.topology.atoms if atom.index not in atom_index ])
    
    ## DECIDING WHICH ATOM TO CHANGE, USEFUL FOR MDTRAJ DISTANCES FUNCTION
    atom_index_to_change = atom_not_index[0] # First atom index that were not interested in

    ## GENERATING PAIRS TO THE SYSTEM
    pairs = np.array([ [ atom_index, atom_index_to_change ] for atom_index in atom_index ])
    ## LOOPING THROUGH THE GRID
    for idx in range(len(grid)):
        ## PRINTING
        if idx % 100 == 0 and verbose is True:
            print("Working on the grid: %d of %d"%(idx, len(grid)) )
        ## COPYING TRAJ
        copied_traj = traj[:]
        ## CHANGING THE POSITION OF ONE ATOM IN THE SYSTEM TO BE THE GRID ATOM
        copied_traj.xyz[:,atom_index_to_change,:] = np.tile( grid[idx,:], (total_frames, 1) )
        ## COMPUTE NUMBER OF ATOMS WITHIN A SPHERE
        N = count_atoms_in_sphere(traj = copied_traj, 
                                  pairs = pairs, 
                                  r_cutoff = cutoff_radius, 
                                  periodic = True )
        ## STORING IN NEIGHBORS
        num_neighbors_array[idx, :] = N[:]
    
    return num_neighbors_array

### FUNCTION TO COMPUTE NEIGHBOR ARRAY
def compute_neighbor_array_KD_tree(traj,
                                   grid,
                                   atom_index):
    '''
    The purpose of this function is to compute number of neighbor arrays using 
    periodic KD tree
    INPUTS:
        traj: [obj]
            trajectory object
        grid: [np.array]
            grid points in x, y, z positions
        atom_index: [list]
            list of atom indices
    OUTPUTS:
        num_neighbors_array: [num_grid, num_frames]
            neighbors within the grid points
    '''
    ## GETTING TOTAL FRAMES
    total_frames = traj.time.size
    ## GETTING ARRAY
    num_neighbors_array = np.zeros( shape = ( len(grid), total_frames ) )
    
    ## DEFINING INDEX
    index = 0

    ## GETTING BOXES
    box = traj.unitcell_lengths[ index, : ] # ASSUME BOX SIZE DOES NOT CHANGE!
    ## DEFINING POSITIONS
    pos = traj.xyz[index, atom_index, :] # ASSUME ONLY ONE FRAME IS BEING RUN
    
    ### FUNCTION TO GET TOTAL NUMBER OF GRID
    T = PeriodicCKDTree(box, pos)
    
    ## COMPUTING ALL NEIGHBORS
    neighbors = T.query_ball_point(grid, r=cutoff_radius)

    ## LOOPING THROUGH THE LIST
    for n, ind in enumerate(neighbors):
        num_neighbors_array[n][index] += len(ind)
    return num_neighbors_array


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    MAIN_SIM_FOLDER=r"S:\np_hydrophobicity_project\simulations"
    
    ## DEFINING SIMULATION NAME
    simulation_dir = "20200114-NP_HYDRO_FROZEN"
    
    ## DEFINING SIMULATION NAME
    specific_sim = "FrozenPlanar_300.00_K_C11NH2_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
    # "FrozenPlanar_300.00_K_C11COOH_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
    # "FrozenPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
    
    ## DEFINING PATHS
    path_sim = os.path.join(MAIN_SIM_FOLDER,
                            simulation_dir,
                            specific_sim)
    
    ## DEFINING GRO AND XTC
    prefix="sam_prod_0_50000-heavyatoms"
    structure_file = prefix + ".gro"
    xtc_file = prefix + ".xtc"
    
    ## DEFINING GRID PATH
    relative_grid_path = "25.6-0.24-0.1,0.1,0.1-0.33-all_heavy\grid-0_1000\out_willard_chandler.dat"
    
    ## DEFINING PATH
    path_grid = os.path.join(path_sim,
                             relative_grid_path)
    path_structure_file = os.path.join(path_sim,
                                       structure_file)
    path_xtc_file = os.path.join(path_sim,
                                 xtc_file)
    
    ## LOADING GRID
    grid = load_datafile(path_grid)
    
    ## FRAME
    frame = 0
    
    ## IMPORTING TRAJECTORIES
    traj_data = import_tools.import_traj(directory = path_sim,
                                         structure_file = structure_file,
                                         xtc_file = xtc_file,
                                         index = frame)
    
    #%% RUNNING KD TREE
    
    ## DEFINING
    traj = traj_data.traj
    
    ## DEFINING CUTOFF
    cutoff_radius = 0.33
    
    ## LOADING LIGAND RESIDUE NAMES
    ligand_residue_list = np.array(load_all_ligand_residue_names())
    
    ## FINDING RESIDUE NAME
    unique_residue_names = np.unique([ each_residue.name for each_residue in traj.topology.residues ])
    
    ## FINDING ALL LIGAND NAME
    ligand_names = np.intersect1d(unique_residue_names, ligand_residue_list)
    
    ## GETTING ALL ATOM INDICES
    atom_index = np.array(calc_tools.flatten_list_of_list([ 
                          calc_tools.find_residue_heavy_atoms(traj = traj, 
                                                       residue_name = each_ligand)
                                           for each_ligand in ligand_names]))
    

    ## DEFINING TIME TRACKER
    time_tracker = track_time()
    
    ## COMPUTING NEIGHBORS WITH KD TREE
    num_neighbors_array_1 = compute_neighbor_array_KD_tree(traj = traj,
                                                           grid = grid,
                                                           atom_index = atom_index,
                                                           )
    print("KD Tree results")
    time_tracker.time_elasped()
    ## PRINTING
    num_within_grid = np.sum(num_neighbors_array_1>0)
    print("Number of grid points with ligands: %d"%(num_within_grid))

    
    #%%
    
    ## DEFINING TIME TRACKER
    time_tracker = track_time()
    ## COMPUTING NEIGHBOR ARRAY
    num_neighbors_array_2 = compute_neighbor_array_with_md_traj(traj = traj,
                                                                grid = grid,
                                                                atom_index = atom_index,
                                                                verbose = False,
                                                                )
                      
    ## DEFINING TIME TRACKER
    print("MD traj results")
    time_tracker.time_elasped()
    
    ## PRINTING
    num_within_grid = np.sum(num_neighbors_array_2>0)
    print("Number of grid points with ligands: %d"%(num_within_grid))
    
    