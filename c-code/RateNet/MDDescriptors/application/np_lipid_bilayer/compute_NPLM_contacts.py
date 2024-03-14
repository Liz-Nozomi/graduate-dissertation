# -*- coding: utf-8 -*-
"""
compute_NPLM_contacts.py
The purpose of this function is to compute the number of contacts between 
lipid membrane and the nanoparticle. We will need to define some cutoff radius 
for when the nanoparticle and lipid membrane are defined as contacted. 

Author: Alex K. Chew (01/23/2020)

INPUTS:
    - gro and xtc file
OUTPUTS:
    - pickle file with the stored number of contacts for each group over time
ALGORITHM:
    - Load gro and xtc file
    - Generate heavy atom lists for lipid membrane and nanoparticle
    - Use mdtraj's distance formulation to compute distances

EXTRACTION WITH GROMACS:
    # Generating XTC file
    gmx trjconv -f nplm_prod.xtc -s nplm_prod.tpr -o nplm_prod_center_last_5ns.xtc -b 45000 -pbc mol
    Selection: non-Water
    
    # Generating GRO file
    gmx trjconv -f nplm_prod.xtc -s nplm_prod.tpr -o nplm_prod_center_last_5ns.gro -dump 45000 -pbc mol
    Selection: non-Water
"""

## IMPORTING TOOLS
import os
import MDDescriptors.core.import_tools as import_tools
import MDDescriptors.core.calc_tools as calc_tools
import numpy as np
import mdtraj as md

## IMPORTING LAST FRAME TOOL
from MDDescriptors.core.traj_itertools import get_traj_from_last_frame

## IMPORTING FUNCTION FOR LIGAND NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv
## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

### FUNCTION TO COMPUTE TOTAL CONTACTS
def compute_total_contacts(dist_storage):
    '''
    The purpose of this function is to compute total contacts given a 
    distance storage. We will simply get the length of each array per time 
    frame. 
    INPUTS:
        dist_storage: [list]
            list of distances stored
    OUTPUTS:
        total_contacts_array: [np.array]
            total contacts array
    '''
    total_contacts_array = np.array([ len(each_array) for each_array in dist_storage ] )
    
    return total_contacts_array


### FUCNTION TO GET NPLM NAMES AND INDEXES
def get_nplm_heavy_atom_details(traj,
                                lm_res_name = 'DOPC',
                                atom_detail_type='all'):
    '''
    The purpose of this function is to get the nplm np heavy atom index 
    and lipid membrane heavy atom index.
    INPUTS:
        traj: [obj]
            trajectory object
        lm_res_name: [str]
            lipid membrane residue name
        atom_detail_type: [logical]
            type of detail that you want
                'all': means both lm and np
                'lm': means lm only
                'np': means nanoparticle only
    OUTPUTS:
        ligand_names: [list]
            list of ligand names
        np_heavy_atom_index: [np.array]
            nanoparticle heavy atom index
        np_heavy_atom_names: [np.array]
            numpy array of the atom names
        lm_heavy_atom_index: [np.array]
            lipid membrane heavy atom index
        lm_heavy_atom_names: [np.array]
            lipid membrane atom names
    '''
    if atom_detail_type == "np" or atom_detail_type == "all":
        ## FINDING LIGAND NAMES
        ligand_names = get_ligand_names_within_traj(traj=traj)
        
        ## GETTING NANOPARTICLE HEAVY ATOMS
        np_heavy_atom_index = np.array([ calc_tools.find_residue_heavy_atoms(traj = traj,
                                              residue_name = each_ligand) for each_ligand in ligand_names ]).flatten()
    
    
        ## GETTING NP HEAVY ATOM NAMES
        np_heavy_atom_names = np.array([ traj.topology.atom(each_atom).name  for each_atom in np_heavy_atom_index])

    ## LM ONLY
    if atom_detail_type == "lm" or atom_detail_type == "all" :

        ## GETTING HEAVY ATOM INDEX OF MEMBRANE
        lm_heavy_atom_index = calc_tools.find_residue_heavy_atoms(traj = traj,
                                                                       residue_name = lm_res_name)
        
        ## GETTING LM HEAVY ATOM NAMES
        lm_heavy_atom_names = np.array([ traj.topology.atom(each_atom).name for each_atom in lm_heavy_atom_index])

    if atom_detail_type == "all":
        return ligand_names, np_heavy_atom_index, np_heavy_atom_names, lm_heavy_atom_index, lm_heavy_atom_names
    elif atom_detail_type == "lm":
        return lm_heavy_atom_index, lm_heavy_atom_names
    elif atom_detail_type == "np":
        return ligand_names, np_heavy_atom_index, np_heavy_atom_names

##########################################
### CLASS FUNCTION TO COMPUTE CONTACTS ###
##########################################
class compute_NPLM_contacts:
    '''
    The purpose of this function is to compute the contacts between nanoparticle 
    and lipid membranes.
    INPUTS:
        traj: [obj]
            traj data object
        lm_res_name: [str]
            lipid membrane residue name
        cutoff: [float, default=0.5]
            cutoff in nm to be considered a cutoff
        print_freq: [int]
            frequency to print output
    OUTPUTS:
        self.ligand_names: [list]
            list of ligand names inside your trajectory
    '''
    def __init__(self, 
                 traj,
                 cutoff = 0.5,
                 lm_res_name = "DOPC",
                 print_freq=100):
        
        ## STORING SELF
        self.cutoff = cutoff
        self.lm_res_name = lm_res_name
        self.print_freq = print_freq

        ## STORING TIME
        self.traj_time = traj.time
        
        ## GETTING INFORMATION FOR LIGAND DETAILS
        self.ligand_names, \
        self.np_heavy_atom_index, \
        self.np_heavy_atom_names, \
        self.lm_heavy_atom_index, \
        self.lm_heavy_atom_names = get_nplm_heavy_atom_details(traj = traj,
                                                               lm_res_name = self.lm_res_name)
        
        ''' Depreciated, added into above function
        ## FINDING LIGAND NAMES
        self.ligand_names = get_ligand_names_within_traj(traj=traj)
        
        ## GETTING NANOPARTICLE HEAVY ATOMS
        self.np_heavy_atom_index = np.array([ calc_tools.find_residue_heavy_atoms(traj = traj,
                                              residue_name = each_ligand) for each_ligand in self.ligand_names ]).flatten()

    
        ## GETTING NP HEAVY ATOM NAMES
        self.np_heavy_atom_names = np.array([ traj.topology.atom(each_atom).name  for each_atom in self.np_heavy_atom_index])
    
        ## GETTING HEAVY ATOM INDEX OF MEMBRANE
        self.lm_heavy_atom_index = calc_tools.find_residue_heavy_atoms(traj = traj,
                                                                       residue_name = self.lm_res_name)
        
        ## GETTING LM HEAVY ATOM NAMES
        self.lm_heavy_atom_names = np.array([ traj.topology.atom(each_atom).name for each_atom in self.lm_heavy_atom_index])
        '''
        
        
        ## COMPUTING TOTAL NUMBER OF ATOMS
        self.total_atom_1 = len(self.np_heavy_atom_index)
        self.total_atom_2 = len(self.lm_heavy_atom_index)
        

        ## GENERATING ATOM PAIRS
        self.atom_pairs = calc_tools.create_atom_pairs_list(atom_1_index_list = self.lm_heavy_atom_index, 
                                                            atom_2_index_list = self.np_heavy_atom_index)
        
        return
    
    ### FUNCTION TO COMPUTE CONTACTS
    def compute(self, traj, frames = [], verbose = True, want_distance_only = False):
        '''
        The purpose of this function is to compute the contacts for a 
        given trajectory. 
        INPUTS:
            traj: [obj]
                trajectory object to run on
            frames: [list]
                list of frames to compute for trajectory
            want_distance_only: [logical]
                True if you onnly want the distance instead of the contacts
        OUTPUTS:
            nearest_dist_array: [list]
                nearest distance array, e.g.
                    [array([[1428, 2541],
                            [1428, 2544],
                This gives you the nearest distance between NP heavy atoms and 
                lipid membrane per frame basis. 
        '''
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[frames] 
            # traj[tuple(frames)]

        ## PRINTING
        print("Computing contacts for trajectory: %d of %d"%(traj.time[0],traj.time[-1]))
        ## GETTING TOTAL FRAMES
        total_frames = len(traj)
        ## CALCULATING DISTANCES
        distances = md.compute_distances(
                                        traj = traj,
                                        atom_pairs = self.atom_pairs,
                                        periodic = True
                ) ## RETURNS TIMEFRAME X (NUM_ATOM_1 X NUM_GOLD) NUMPY ARRAY
    
        ## RESHAPING THE DISTANCES
        distances = distances.reshape(total_frames, 
                                      self.total_atom_1, 
                                      self.total_atom_2)
        if want_distance_only is True:
            return distances
        ## GETTING LOCATIONS
        locations = np.argwhere(distances <= self.cutoff)
        
        ## GETTING FRAME INDEXES
        frame_index = np.arange(len(traj.time))
        
        ## CREATING EMPTY ARRAY
        nearest_dist_array = []
        
        ## LOOPING THROUGH TIME INDEXES
        for each_frame in frame_index:
            ## FINDING ALL LOCATIONS
            idx_match = np.where(locations[:,0] == each_frame)[0]
            
            ## ATOM INDICES
            atom_1_index = locations[idx_match, 1][:,np.newaxis]
            atom_2_index = locations[idx_match, 2][:,np.newaxis]
            
            ## STORING
            nearest_dist_array.append( np.concatenate( (atom_1_index,
                                                        atom_2_index),
                                                        axis = 1 ) )
        
        return nearest_dist_array
        # RETURNS LIST OF ATOM_INDEX_1, ATOM_INDEX_2 THAT ARE IN CONTACT            

### MAIN FUNCTION TO COMPUTE CONTACTS
def main_compute_contacts(path_to_sim,
                          input_prefix,
                          func_inputs,
                          last_time_ps = 50000,
                          selection = 'non-Water',
                          n_procs = 1,
                          gro_output_time_ps = 0,
                          rewrite = False):
    '''
    The purpose of this function is to compute the number of contacts for the 
    nanoparticle lipid membrane systems.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        selection: [str]
            selection to run the contacts with
        func_inputs: [dict]
            dictionary for the main contacts function
        input_prefix: [str]
            input prefix for tpr and gro files
        rewrite: [logical]
            True if you want to rewrite
        n_procs: [int] 
            number of processors used to compute number of contacts
        frame_rate: [int]
            frame rate for number of contacts script
        gro_output_time_ps: [float]
            time to output in picoseconds
    OUTPUTS:
        class_object: [obj]
            object storing contacts dictionary
        results: [list]
            list with the number of contacts
    '''
    ## CONVERTING TRAJECTORY
    trjconv_func = convert_with_trjconv(wd = path_to_sim)
    ## GETTING ONLY SPECIFIC SELECTION
    gro_file, xtc_file, ndx_file = trjconv_func.generate_gro_xtc_specific_selection(input_prefix = input_prefix,
                                                                                    selection = selection,
                                                                                    rewrite = rewrite,
                                                                                    gro_output_time_ps = gro_output_time_ps)
    ## LOADING FILES
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = gro_file,
                                         xtc_file = xtc_file,
                                         )
    
    ## UPDATING TRAJECTORY BASED ON TIME
    traj_data.traj = get_traj_from_last_frame(traj = traj_data.traj,
                                              last_time_ps = last_time_ps)
 
    ## INITIALIZING THE FUNCTION
    class_object = compute_NPLM_contacts(traj = traj_data.traj,
                                         **func_inputs)
    
    ## COMPUTING OUTPUT BY SPLITTING TRAJECTORY
    results = parallel_analysis_by_splitting_traj(traj = traj_data.traj, 
                                                  class_function = class_object.compute, 
                                                  n_procs = n_procs,
                                                  combine_type="append_list",
                                                  want_embarrassingly_parallel = True)
        
    ## CLEANING DATA
    class_object.atom_pairs = []
    
    return class_object, results


#%%
if __name__ == "__main__":
    ## DEFINING PATH TO SIMULATION
    path_sim_parent = r"R:/scratch/nanoparticle_project/simulations"
    sim_parent = r"20200120-US-sims_NPLM_rerun_stampede"
    sim_folder= r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    ## DEFINING RELATIVE SIM PATH
    relative_sim_path = r"4_simulations/1.300"
    relative_sim_path = r"4_simulations/4.700"
    
    ## DEFINING GRO AND XTC
    gro_file = "nplm_prod_center_last_5ns.gro"
    xtc_file = "nplm_prod_center_last_5ns.xtc"
    
    
    ### PULLING SIMULATIONS
    sim_parent = r"20200113-NPLM_PULLING"
    sim_folder= r"NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    relative_sim_path = r""
    
    ## DEFINING GRO AND XTC
    gro_file = "nplm_pull_skip_1_non-Water_0_-1.gro"
    xtc_file = "nplm_pull_skip_1_non-Water_0_-1.xtc"
    
    ## DEFINING PATH TO SIMULATION
    path_sim = os.path.join(path_sim_parent, 
                            sim_parent,
                            sim_folder,
                            relative_sim_path)
    

    
    ## LOADING FILES
    traj_data = import_tools.import_traj(directory = path_sim,
                                         structure_file = gro_file,
                                         xtc_file = xtc_file,
                                         )
    
    #%%
    ## DEFINING RESIDUE NAME OF LIPID MEMBRANE
    lm_res_name = "DOPC"

    ## DEFINING CUTOFF
    cutoff = 0.5

    ## GENERATING INPUT SCRIPT
    input_dict = { "traj": traj_data.traj,
                   "cutoff" : cutoff,
                   "lm_res_name" : lm_res_name,}

    ## INITIATING SCRIPT
    contacts = compute_NPLM_contacts(**input_dict)


    #%%
    
    ## DEFINING TOTAL FRAMES
    total_frames = 2
    
    ## DEFINING TRAJ
    traj = traj_data.traj[0:total_frames]
    # traj = traj_data.traj[-total_frames-1:-1]
    
    ## RUNNING CODE
    nearest_dist_array_2 = contacts.compute( traj = traj)
    
    #%%
    
    ## GETTING LOCATIONS
    locations = np.argwhere(distances <= 0.5)
    
    ## GETTING FRAME INDEXES
    frame_index = np.arange(len(traj.time))
    
    ## CREATING EMPTY ARRAY
    nearest_dist_array = []
    
    ## LOOPING THROUGH TIME INDEXES
    for each_frame in frame_index:
        ## FINDING ALL LOCATIONS
        idx_match = np.where(locations[:,0] == each_frame)[0]
        
        ## ATOM INDICES
        atom_1_index = locations[idx_match, 1][:,np.newaxis]
        atom_2_index = locations[idx_match, 2][:,np.newaxis]
        
        ## STORING
        nearest_dist_array.append( np.concatenate( (atom_1_index,
                                                    atom_2_index),
                                                    axis = 1 ) )
    
    
    
    #%%
        
    ## OUTPUTTING TOTAL CONTACTS
#    total_contacts_array = compute_total_contacts(dist_storage = nearest_dist_array)

    #%%
    ## GETTING TOTAL MEMORY AVAILABLE
    import psutil
    ## GETTING MEMORY
    vm = psutil.virtual_memory()
    
    #%%
    import time
    ## CREATING TRAJECTORIES
    traj = traj_data.traj
    traj_list = []
    # [None for x in  range(len(traj)) ]
    ## LOOPING
    for frame in range(len(traj)):
        print(frame)
        traj_list.append([traj, frame])
        # time.sleep(1)
        
        
        
    
    
    

