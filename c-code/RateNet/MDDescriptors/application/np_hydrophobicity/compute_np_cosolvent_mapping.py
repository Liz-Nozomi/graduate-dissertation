#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_np_cosolvent_mapping.py

This code extracts cosolvent mapping information and uses it to generate probability 
distributions of cosolvents around the NP. The main idea is to extract locations 
where cosolvents like to bind to, which could inform us on potential protein binding 
sites. 


Written by: Alex K. Chew (04/20/2020)
"""
## IMPORTING MODULES
import os
import numpy as np

## MODULE TO LOAD GRO
import mdtraj as md

## TRAJECTORY DETAILS
import MDDescriptors.core.calc_tools as calc_tools # Loading trajectory details

### IMPORTING LIGAND REISDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj, get_atom_indices_of_ligands_in_traj

### IMPORTING GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, NP_SIM_PATH

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools

## CRITICAL FUNCTION TO USE PERIODIC CKD TREE
from MDDescriptors.surface.periodic_kdtree import PeriodicCKDTree

## FUNCTION TO LOAD GRID
from MDDescriptors.surface.core_functions import load_datafile

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

### GLOBAL VARIABLES
## DEFINING GOLD ATOM NAME
GOLD_ATOM_ELEMENT=["Au",'BAu']

### FUNCTION TO GET GOLD COM
def compute_gold_com_for_traj( traj,
                               gold_element_symbol = GOLD_ATOM_ELEMENT):
    '''
    This function computes the gold center of mass for a trajectory.
    INPUTS:
        traj: [obj]
            md trajectory
        gold_element_symbol: [list]
            element symbol for gold
    OUTPUTS:
        gold_com: [np.array, shape = (frame, 3)]
            gold center of mass location
    '''

    ## FINDING ALL GOLD ATOMS
    gold_atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.element.symbol in gold_element_symbol  ]
    
    ## GETTING ALL DISTANCES
    gold_xyz = traj.xyz[:,gold_atom_index,:]
    
    ## AVERAGE COM
    gold_com = np.mean(gold_xyz, axis = 1)
    return gold_com

### FUNCTION TO GET ORIGINAL NP NAME
def find_original_np_hydrophobicity_name(path_to_sim):
    '''
    The purpose of this function is to get the hydrophobicity nomenclature 
    given the path to simulation. This is simply a series of name manipulation 
    to match the hydrophobicity name, e.g.
    
    Current name:
        comap_MET,FMA_1_1_50-EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1_likelyindex_1
    Original name:
        MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1_likelyindex_1    
    
    INPUTS:
        path_to_sim: [str]
            path to simulation directory
    OUTPUTS:
        orig_np_name: [str]
            original nanoparticle input name
    '''

    ## GETTING BASENAME
    np_basename = os.path.basename(path_to_sim)
    if "Planar" not in np_basename:
    
        ## SPLITTING
        np_split = np_basename.split('-')
        
        ## GETTING SPRING
        spring_constant = np_split[0].split('_')[-1]
        
        ## GETTING NEW NAME
        orig_np_name = "MostlikelynpNVTspr_" + spring_constant + "-" +  '-'.join(np_split[1:])
    else:
        
        ## SPLITTING
        split_name = np_basename.split('_')
        # comap_PRO,FMA_1_5_NVTspr_50_Planar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps
        
        orig_np_name = '_'.join(split_name[4:])
    
    return orig_np_name

### FUNCTION TO COMPUTE NEIGHBOR ARRAY
def compute_neighbor_array_KD_tree_npt(traj,
                                       grid,
                                       atom_index,
                                       cutoff_radius):
    '''
    The purpose of this function is to compute number of neighbor arrays using 
    periodic KD tree. This code has been adjusted to take into account fluctuations 
    in the box size
    INPUTS:
        traj: [obj]
            trajectory object
        grid: [np.array]
            grid points in x, y, z positions
        atom_index: [list]
            list of atom indices
        cutoff_radius: [float]
            cutoff radius to check
    OUTPUTS:
        num_neighbors_array: [num_grid, num_frames]
            neighbors within the grid points
            
            
    Note that this function outputs the same results as the mdtraj distance protocol:
    ## IMPORT
    from MDDescriptors.application.np_hydrophobicity.check_ligand_to_grid_overlap import compute_neighbor_array_with_md_traj
    
    ## COMPUTING NUMBER OF NEIGHBORS ARRAY
    num_neighbors_array = compute_neighbor_array_with_md_traj(traj = traj,
                                                              grid = new_grid,
                                                              atom_index = mapping.solvent_atom_index_dict['HOH'],
                                                              cutoff_radius = 0.33)
    ## COMPUTING WITH KD TREE
    num_neighbors_array_kdtree = compute_neighbor_array_KD_tree_npt(traj = traj,
                                                                    grid = new_grid,
                                                                    atom_index = mapping.solvent_atom_index_dict['HOH'],
                                                                    cutoff_radius = 0.33)
    '''
    ## GETTING TOTAL FRAMES
    total_frames = traj.time.size
    ## GETTING ARRAY
    num_neighbors_array = np.zeros( shape = ( len(grid), total_frames ) )
    
    ## LOOPING THROUGH EACH FRAME
    for each_frame in range(total_frames):

        ## GETTING BOXES
        box = traj.unitcell_lengths[ each_frame, : ] # ASSUME BOX SIZE DOES NOT CHANGE!
        ## DEFINING POSITIONS
        pos = traj.xyz[each_frame, atom_index, :] # ASSUME ONLY ONE FRAME IS BEING RUN
        
        ### FUNCTION TO GET TOTAL NUMBER OF GRID
        T = PeriodicCKDTree(box, pos)
        
        ## COMPUTING ALL NEIGHBORS
        neighbors = T.query_ball_point(grid, r=cutoff_radius)
    
        ## LOOPING THROUGH THE LIST
        for n, ind in enumerate(neighbors):
            num_neighbors_array[n][each_frame] += len(ind)
    return num_neighbors_array

### CLASS FUNCTION TO COMPUTE COSOLVENT MAPPING
class compute_np_cosolvent_mapping:
    '''
    The purpose of this function is to count the number of occurances that 
    cosolvents are being mapped to the surface. 
    INPUTS:
        traj_data: [obj]
            trajectory data
        grid: [np.array]
            wc grid arrays
        wc_gold_com: [np.array]
            gold center of mass for WC interface
        cutoff: [float]
            radius cutoff to count the cosolvents in nanometers
        print_freq: [int]
            frequency of time to print for every N ps
    OUTPUTS:
        self.cosolvent_map_gold_com: [np.array]
            cosolvent mapping gold center of mass across the trajectory
    FUNCTIONS:
        find_all_solvent_atom_index:
            computes all solvent atom indices
        print_summary:
            prints the summary of calculation
        compute:
            main function to compute the number of solvents within the grid points
    CHECKING EXAMPLES:
        This can check whether the WC interface is correctly centered each frame
        fig = plot_wc_interface_with_ligands(traj_data = traj_data,
                                             mapping = mapping,
                                             frame=-1,
                                             )
    '''
    def __init__(self,
                 traj_data,
                 grid,
                 wc_gold_com,
                 cutoff = 0.33,
                 print_freq = 100,
                 ):
        ## STORING INPUTS
        self.orig_grid = grid
        self.wc_gold_com = wc_gold_com
        self.cutoff = cutoff
        self.print_freq = print_freq
        
        ## DEFINING TRAJ
        traj = traj_data.traj
        
        ## STORING TIME
        self.traj_time = traj.time
        
        ## GETTING ALL LIGANDS
        self.ligand_names = get_ligand_names_within_traj(traj = traj)
        
        ## FINDING ALL SOLVENT MOLECULES
        self.solvent_list = [each_residue for each_residue in traj_data.residues if each_residue not in self.ligand_names]
        
        ## FINDING ALL SOLVENT HEAVY ATOM INDEX
        self.solvent_atom_index_dict = self.find_all_solvent_atom_index(traj = traj)
        
        ## GETTING COM FOR TRAJ
        self.cosolvent_map_gold_com = compute_gold_com_for_traj(traj = traj)
        
        ## GETTING TRANSLATION ARRAY
        self.translation_array = self.cosolvent_map_gold_com - self.wc_gold_com
        
        ## COMPUTING NUMBER OF SOLVENTS
        self.n_solvents = len(self.solvent_list)
        self.n_grid_pts = len(self.orig_grid)
        
        ## SUMMARY
        self.print_summary()
        
        return
    
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        '''
        This function prints the summary for the analysis
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            void
        '''
        print("------- SUMMARY --------")
        print("Total grid points: %d"%(self.n_grid_pts) )
        print("Total solvents: %d"%(self.n_solvents) )
        print("List of solvent names: %s"%(', '.join(self.solvent_list)) )
        print("Ligand names: %s"%(', '.join(self.ligand_names)) )
        print("Total trajectory time: %d"%(len(self.traj_time)))
        print("Initial time: %d"%(self.traj_time[0]))
        print("Final time: %d"%(self.traj_time[-1]))
        
        return
    
    ### FUNCTION TO GET NEW GRID FOR SPECIFIC TIME INDEX
    def get_new_grid_given_time_index(self,
                                      time_idx,):
        '''
        This function gets the new grid for a given time index. It does so by 
        taking the translation vector and adding to original grid array.
        INPUTS:
            self: [obj]
                class object
            time_idx: [int]
                index for the time
        OUTPUTS:
            new_grid: [np.array]
                new grid based on translations
        '''
        ## GETTING NEW GRID
        translation_array = self.translation_array[time_idx]
        
        ## FINDING NEW GRID
        new_grid = self.orig_grid + translation_array
        
        return new_grid
    
    ### FUNCTION TO GET ALL SOLVENT HEAVY ATOMS
    def find_all_solvent_atom_index(self, traj):
        '''
        The purpose of this function is to find all solvent heavy atom index.
        INPUTS:
            self: [obj]
                class object
            traj: [obj]
                trajectory object
        OUTPUTS:
            solvent_atom_index_dict: [dict]
        '''
        solvent_atom_index_dict = {}
        
        ## LOOPING THROGUH EACH SOLVENT
        for solvent_resname in self.solvent_list:
            heavy_atom_index = calc_tools.find_residue_heavy_atoms(traj = traj, residue_name = solvent_resname)
            ## STORING
            solvent_atom_index_dict[solvent_resname] = heavy_atom_index[:]
            
        return solvent_atom_index_dict
    
    ### FUNCTION TO COMPUTE PAIRS WITHIN A FRAME
    def compute(self,
                traj,
                frames = []):
        '''
        The purpose of this function is to compute the number of solvent residues 
        within list of grid points.
        '''
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[frames]
        
        
        ## GETTING TOTAL FRAMES
        total_frames = traj.time.size
        
        ## GETTING ARRAY
        num_neighbors_array = np.zeros( shape = ( total_frames, self.n_solvents, self.n_grid_pts ) )
        
        ## LOOPING
        for idx, time in enumerate(traj.time):
            if time % self.print_freq == 0:
                print("Computing cosolvent mapping for time: %d "%(time))
            ## GETTING TIME INDEX
            time_idx = np.where(self.traj_time == time)[0][0]
            
            ## FINDING NEW GRID
            new_grid = self.get_new_grid_given_time_index(time_idx = time_idx)
            
            ## LOOPING THROUGH EACH SOLVENT
            for solvent_idx, solvent_resname in enumerate(self.solvent_list):
                ## FINDING ATOM INDEX
                atom_index = self.solvent_atom_index_dict[solvent_resname]
                
                ## RUNNING NEIGHBORS FUNCTION
                num_neighbors_array_kdtree = compute_neighbor_array_KD_tree_npt(traj = traj[idx],
                                                                                grid = new_grid,
                                                                                atom_index = atom_index,
                                                                                cutoff_radius = self.cutoff)
                
                ## STORING
                num_neighbors_array[idx,solvent_idx,:] = num_neighbors_array_kdtree.T[0]
                
        return num_neighbors_array

### FUNCTION TO PLOT WC INTERFACE
def plot_wc_interface_with_ligands(traj_data,
                                   mapping,
                                   frame=0,
                                   ):
    '''
    Function that debugs the wc interface to make sure that we are capturing the 
    correct locations.
    INPUTS:
        traj_data; [obj]
            trajectory object
        mapping: [obj]
            mapping object
        frame: [int]
            frame index
    OUTPUTS:
        fig:
            mayavi figure
    '''

    ## PLOTTING FUNCTIONS
    import MDDescriptors.core.plot_tools as plot_funcs
    
    ## GETTING GRID
    new_grid = mapping.get_new_grid_given_time_index(time_idx = frame)
    
    ## GETTING GOLD INDEX    
    au_index = [atom.index for atom in traj_data.topology.atoms if atom.name == 'Au' or atom.name == 'BAu']
    
    ## GETTING ATOM INDICES AND LIGAND NAME
    ligand_names, atom_index = get_atom_indices_of_ligands_in_traj( traj = traj_data.traj )
    
    ## IMPORTING MLAB
    from mayavi import mlab
    
    grid = new_grid[:]
    ## CLOSING ALL MLAB FIGURES
    mlab.clf()
    # PLOTTING WITH MAYAVI
    figure = mlab.figure('Scatter plot',
                         bgcolor = (1, 1, 1))
    
    points = mlab.points3d(grid[:,0],
                           grid[:,1],
                           grid[:,2],
                           figure = figure,
                           color=(.5,.5,.5),
                           opacity=0.02,
                           transparent=True,
                           )
    ## FIGURE FROM 
    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                              atom_index = atom_index,
                              frame = frame,
                              figure = figure,
                              dict_atoms = plot_funcs.ATOM_DICT,
                              dict_colors = plot_funcs.COLOR_CODE_DICT)
    
    ## PLOTTING GOLD FIGURE
    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                       atom_index = au_index,
                                       frame = frame,
                                       figure = fig,
                                       dict_atoms = plot_funcs.ATOM_DICT,
                                       dict_colors = plot_funcs.COLOR_CODE_DICT)
    
    return fig

### MAIN FUNCTION THAT COMPUTES EVERYTHING
def main_compute_np_cosolvent_mapping(path_to_sim,
                                      input_prefix,
                                      func_inputs,
                                      path_to_wc_folder,
                                      n_procs = 1,
                                      initial_frame = 2000,
                                      final_frame = 12000,
                                      rewrite = False):
    '''
    This function is the main one that runs cosolvent mapping protocols
    INPUTS:
        func_inputs: [dict]
            dictionary for the main function
        n_procs: [int] 
            number of processors
        rewrite: [logical]
            True if you want to rewrite
        path_to_wc_folder: [str]
            path to WC folder
        initial_frame: [float]
            initial frame in ps
        final_frame: [float]
            final frame in ps
    '''
    ## CONVERTING TRAJECTORY
    trjconv_func = convert_with_trjconv(wd = path_to_sim)

    ## DEFINING TRJCONV INPUTS
    trjconv_inputs={
            'input_prefix': input_prefix,
            'first_frame': initial_frame,
            'last_frame': final_frame,
            'gro_output_time_ps': initial_frame,
            'rewrite': rewrite,
            }
    
    ## GENERATING HEAVY ATOMS ONLY
    output_gro_file, output_xtc_file, output_ndx_file = trjconv_func.generate_heavy_atoms_only(**trjconv_inputs)
    
    ## LOADING TRAJECTORY
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = output_gro_file,
                                         xtc_file = output_xtc_file,
                                         discard_overlapping_frames = True,
                                         )

    ## GETTING ORIGINAL NP NAME
    orig_np_name = find_original_np_hydrophobicity_name(path_to_sim = path_to_sim)

    ## LOADING WC GRID
    relative_path_to_wc_grid=os.path.join("26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",
                                          "grid-45000_50000",
                                          "out_willard_chandler.dat"
                                          )
    
    ## DEFINING RELATIVE PATH TO GOR
    path_to_wc_gro=os.path.join(path_to_wc_folder,
                                orig_np_name,
                                "sam_prod.gro")
    
    ## DEFINING PATH TO WC FILE
    path_to_wc_data = os.path.join(path_to_wc_folder,
                                   orig_np_name,
                                   relative_path_to_wc_grid)
    
    ## LOADING THE GRID
    grid = load_datafile(path_to_wc_data)
    ''' Outputs:
        array([[0.889, 3.461, 3.669],
               [0.895, 3.423, 3.669],
               [0.895, 3.461, 3.606],
               ...,
               [6.174, 4.746, 3.768],
               [6.173, 4.746, 3.867],
               [6.169, 4.746, 3.966]])
    '''
    
    ## LOADING GRO FILE
    wc_gro = md.load(path_to_wc_gro) # , top=path_to_wc_gro

    ## COMPUTING CENTER OF MASS
    wc_gold_com = compute_gold_com_for_traj( traj = wc_gro  )
    
    ## DEFINING INPUTS
    np_inputs={
            'traj_data' : traj_data,
            'grid': grid,
            'wc_gold_com': wc_gold_com,
            **func_inputs
            }
    
    
    ## RUNNING FUNCTION
    mapping = compute_np_cosolvent_mapping(**np_inputs)
    
    ## COMPUTING OUTPUT BY SPLITTING TRAJECTORY
    num_neighbors_array = parallel_analysis_by_splitting_traj(traj = traj_data.traj, 
                                                              class_function = mapping.compute, 
                                                              n_procs = n_procs,
                                                              combine_type="concatenate_axis_0",
                                                              want_embarrassingly_parallel = True)

    ## COMPUTING
    # num_neighbors_array = mapping.compute(traj = traj_data.traj, frames = np.array([0,1, 2]))
    
    return num_neighbors_array, mapping


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    ## PATH TO SIM
    np_parent = "20200416-cosolvent_mapping"
    np_specific_dir= "comap_MET,FMA_1_1_50-EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1_likelyindex_1"
    path_to_sim=os.path.join(NP_SIM_PATH,
                             np_parent,
                             np_specific_dir
                             )
    
    ## DEFINING PREFIX
    input_prefix="sam_prod"
    
    ## DEFINING PARENT DIR
    parent_wc_folder = "20200401-Renewed_GNP_sims_with_equil"
    
    ## DEFINING WC INTERFACE LOCATION
    path_to_wc_folder = os.path.join(PARENT_SIM_PATH,
                                     parent_wc_folder)
    
    np_mapping_inputs = {
            'cutoff': 0.33,
            }
    
    main_np_cosolvent_mapping_inputs={
            'path_to_sim': path_to_sim,
            'func_inputs': np_mapping_inputs,
            'input_prefix': 'sam_prod',
            'n_procs': 1,
            'path_to_wc_folder': path_to_wc_folder,
            'initial_frame': 2000,
            'final_frame': 12000,
            'rewrite': False
            }
    
    ## PERFORMING COSOLVENT MAPPING
    num_neighbors_array, mapping = main_compute_np_cosolvent_mapping(**main_np_cosolvent_mapping_inputs)
    
    
    '''
    fig = plot_wc_interface_with_ligands(traj_data = traj_data,
                                         mapping = mapping,
                                         frame=-1,
                                         )
    '''


