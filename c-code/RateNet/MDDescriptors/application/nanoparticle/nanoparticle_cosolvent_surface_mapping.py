# -*- coding: utf-8 -*-
"""
nanoparticle_cosolvent_surface_mapping.py
The purpose of this script is to generate cosolvent mapping on the surface

CREATED ON: 11/2/2019

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""
import os
import mdtraj as md
import numpy as np
import multiprocessing

## TRAJECTORY DETAILS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading trajectory details

### IMPORTING LIGAND REISDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import ligand_residue_list as ligand_names
from MDDescriptors.application.nanoparticle.stored_parallel_scripts import compute_cosolvent_mapping

## IMPORTING PARALLEL TOOL
from MDDescriptors.application.nanoparticle.parallel import parallel

## CHECKING PATH
from MDDescriptors.core.check_tools import check_path, check_testing
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

## PICKLE TOOLS
import MDDescriptors.core.pickle_tools as pickle_tools

### FUNCTION TO COMPUTE HYDRATION MAPS
def calc_surface_mapping( input_details, n_procs = 28 ):
    r"""
    This function computes the hydration maps given the trajectory and data file of 
    WC-interface.
    INPUTS:
        traj: [obj]
            trajectory object
        out_path: [str]
            output path that stores the WC interface
        wcdatfilename: [str]
            datafile for the willard-chandler file name
        r_cutoff: [float]
            cutoff for the hydration maps
        max_N: [int]
            max 
    OUTPUTS:
        hydration map results
    """
    
    ## COMPUTING NUBMER OF PROCESSORS
    if n_procs < 0:
        n_procs = multiprocessing.cpu_count()

    ## DEFINING TRAJ
    traj = input_details['traj_data'].traj

    ## INITIATING THE MAPPING PROTOCOL
    mapping = compute_cosolvent_mapping(**input_details)
    
    ## COMPUTING 
    unnorm_p_N = parallel_analysis_by_splitting_traj(traj = traj, 
                                                     class_function = mapping.compute, 
                                                     input_details = input_details, 
                                                     n_procs = n_procs,
                                                     combine_type="sum") # combining sum
    ## RETURNS: NUM SOLVENTS, NUM GRID POINTS, MAX_N
    return mapping, unnorm_p_N

### MAIN FUNCTION
def cosolvent_mapping_main(path_analysis,
                           gro_file,
                           xtc_file,
                           path_pdb,
                           max_N,
                           cutoff,
                           path_pickle,
                           n_procs = 1,
                           verbose = True
                           ):
    '''
    Main function to perform cosolvent mapping
    INPUTS:
        path_analysis: [str]
            path to analysis
        gro_file: [str]
            file name for gro file, assumed with be in the path_analysis
        xtc_file: [str]
            file name for XTC file
        path_pdb: [str]
            path to the pdb file
        max_N: [int]
            maximum number of atoms within the sphere, typically is 10 but could be larger
        cutoff: [float]
            cutoff for searching protocol
        n_procs: [int]
            number of processors
        path_pickle: [str]
            path to where the pickle is stored
    OUTPUTS:
        void, output will be a pickle
    '''
    mapping = None
    unnorm_p_N = None
    ## CHECKING IF PICKLE PATH DOES NOT EXIST
    if os.path.exists(path_pickle) is False:
        
        ### LOADING TRAJECTORY
        traj_data = import_tools.import_traj( directory = path_analysis, # Directory to analysis
                     structure_file = gro_file, # structure file
                      xtc_file = xtc_file, # trajectories
                      )
        
        ## DEBUGGING WITH THE FIRST 10 FRAMES
        # traj_data.traj = traj_data.traj[0:10]
        
        ## LOADING THE GRID PDB FILE
        grid_pdb = md.load_pdb(path_pdb)
        
        ## LOADING GRIDDING XYZ
        grid_xyz = grid_pdb.xyz[0]

        ### DEFINING INPUT DATA
        input_details = {   'traj_data'     : traj_data,                      # Trajectory information
                            'ligand_names'  : ligand_names,
                            'grid_xyz'      : grid_xyz,   # Name of the ligands of interest
                            'verbose'       : True,                      # ITP FILE
                            'max_N'         : max_N,
                            'cutoff'        : cutoff, # nm
                            }
        ### CALCULATING DISTRIBUTION
        mapping, unnorm_p_N = calc_surface_mapping( input_details, n_procs = n_procs )
        
        ## STORING PICKLE
        pickle_tools.pickle_results(results = [mapping, unnorm_p_N],
                                    pickle_path = path_pickle,
                                    verbose = verbose,
                                    )
    else:
        print("Pickle path exists: %s"%(path_pickle))
        print("Not rerunning analysis")

    return mapping, unnorm_p_N


    
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## RUNNING TESTING    
    if testing == True:
    
        ## DEFINING PATH TO ANALYSIS
        path_analysis=r"R:\scratch\nanoparticle_project\simulations\191017-mixed_sams_most_likely_sims\MostNP-EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1-lidx_1-cosf_10000-aceticacid_formate_methylammonium_propane_1_molfrac_300"
        ## CHECKING PATH
        path_analysis = check_path(path_analysis)
        
        ## GRO AND XTC
        gro_file="sam_prod.gro"
        xtc_file="sam_prod.xtc"
        
        ## DEFINING NUMBER CORES
        n_procs = 2
        
        ## DEFINING PDB FILE
        grid_pdb_filename="wc_aligned.pdb"
        
        ## DEFINING OUTPUT PICKLING PATH
        pickle_name = "cosolvent_map_2.pickle"
        path_pickle = os.path.join(path_analysis, pickle_name)
        
        ## DEFINING PICKLES AND MAX N
        max_N = 10
        cutoff = 0.33
        
        ## DEFINING PATH TO PDB
        path_pdb = os.path.join( path_analysis,
                                 grid_pdb_filename )
        
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## ANALYSIS PATH
        parser.add_option('--path_sim', dest = 'path_analysis', help = 'Path of simulation', default = '.', type=str)
        
        ## DEFINING GRO AND XTC FILE
        parser.add_option('--gro', dest = 'gro_file', help = 'Name of gro file', default = 'sam.gro', type=str)
        parser.add_option('--xtc', dest = 'xtc_file', help = 'Name of xtc file', default = 'sam.xtc', type=str)
        parser.add_option('--path_pdb', dest = 'path_pdb', help = 'Path for PDB grid', default = 'wc_aligned.pdb', type=str)
        
        ## DEFINING PATH PICKLE
        parser.add_option('--path_pickle', dest = 'path_pickle', help = 'Path of pickle', default = '.', type=str)

        ## DEFINING NUMBER OF PROCESSORS        
        parser.add_option('--n_procs', dest = 'n_procs', help = 'Number of processors', default = 20, type=int)
        
        ## DEFINING DETAILS OF HISTOGRAM
        parser.add_option('--max_N', dest = 'max_N', help = 'Maximum number for a histogram', default = 10, type=int)
        parser.add_option('--cutoff', dest = 'cutoff', help = 'Cutoff for search grid', default = 0.33, type=float)
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## ANALYSIS PATH
        path_analysis = options.path_analysis
        
        # GRO/XTC
        gro_file = options.gro_file
        xtc_file = options.xtc_file
        
        # GRIDDING PDB FILE
        path_pdb = options.path_pdb

        ## DEFINING NUMBER OF PROCESSORS        
        n_procs = options.n_procs
        
        ## DEFINING PATH TO PICKLE
        path_pickle = options.path_pickle
        
        ## DETAILS FOR ANALYSIS
        max_N = options.max_N
        cutoff = options.cutoff
    
    ##########################################################################
    ### MAIN CODE
    ##########################################################################
    
    ## DEFINING INPUTS
    mapping_inputs = {
            'path_analysis'     : path_analysis,
            'gro_file'          : gro_file,
            'xtc_file'          : xtc_file,
            'path_pdb'          : path_pdb,
            'max_N'             : max_N,
            'cutoff'            : cutoff,
            'path_pickle'       : path_pickle,
            'n_procs'           : n_procs,
            'verbose'           : True,            
            }
    
    ## RUNNING MAPPING PROTOCOL
    mapping, unnorm_p_N = cosolvent_mapping_main(**mapping_inputs)
    
    
    #%%
    
    '''
    ## RESTORE PICKLE INFORMATION
    results = pickle_tools.load_pickle_results(path_pickle)[0]
    
    ## EXTRACTING DETAILS
    mapping, unnorm_p_N = results[0], results[1]
    '''
    
    
    
    