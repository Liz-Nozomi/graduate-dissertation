# -*- coding: utf-8 -*-
"""
run_nplm_contacts_parallel.py
The purpose of this script is to run the NPLM contacts code in parallel. 

Written by: Alex K. Chew (01/22/2020)


"""
## IMPORTING MODULES
import os
import mdtraj as md
import numpy as np

## IMPORTING MEMORY
from MDDescriptors.core.memory import get_frame_rate_based_on_memory 

## IMPORTING CONTACT TOOL
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import compute_NPLM_contacts

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing, check_dir

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import par_compute_class_function

## GETTING ITERTOOLS
from MDDescriptors.core.traj_itertools import get_pickle_name_for_itertool_traj, update_neighbor_log



#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## RUNNING TESTING    
    if testing == True:
        
        ## NUMBER OF CORES
        n_procs = 2
        # 2
        
        ## DEFINING PATH TO SIMULATION
        path_sim_parent = r"R:/scratch/nanoparticle_project/simulations/20200120-US-sims_NPLM_rerun_stampede"
        sim_folder= r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
        
        ## DEFINING RELATIVE SIM PATH
        relative_sim_path = r"4_simulations/1.300"
        relative_sim_path = r"4_simulations/4.700"
        
        ## DEFINING PATH TO SIMULATION
        path_sim = os.path.join(path_sim_parent, 
                                sim_folder,
                                relative_sim_path)
        
        ## DEFINING PATH PICKLE
        path_pickle = os.path.join(path_sim,
                                   "compute_contacts")
        
        ## DEFINING GRO AND XTC
        gro_file = "nplm_prod_center_last_5ns.gro"
        xtc_file = "nplm_prod_center_last_5ns.xtc"
        

        
        ## DEFINING RESIDUE NAME OF LIPID MEMBRANE
        lm_res_name = "DOPC"
    
        ## DEFINING CUTOFF
        cutoff = 0.5
        
        ## DEFINING FRAME RATE
        frame_rate = None
        
        ## DEFINING LOG FILE
        log_file = "log.log"
        
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## REPRESENTATION TYPE
        parser.add_option('--path', dest = 'simulation_path', help = 'Path of simulation', default = '.', type=str)
        
        ## DEFINING GRO AND XTC FILE
        parser.add_option('--gro', dest = 'gro_file', help = 'Name of gro file', default = 'sam.gro', type=str)
        parser.add_option('--xtc', dest = 'xtc_file', help = 'Name of xtc file', default = 'sam.xtc', type=str)
        
        ## DEFINING OUTPUT PICKLE PATH
        parser.add_option('--path_pickle', dest = 'path_pickle', help = 'Path of pickle', default = '.', type=str)
        
        ## LOG FILE
        parser.add_option('--log_file', dest = 'log_file', help = 'Log file of the pickle', default = '.', type=str)
        
        ## LIPID MEMBRANE
        parser.add_option('--lm_name', dest = 'lm_name', help = 'Name of lipid mambrane', default = 'DOPC', type=str)
        
        ### DEFINING CUTOFF
        parser.add_option('--cutoff_radius', 
                          dest = 'cutoff_radius', 
                          help = 'Cutoff for grid counting', 
                          default = 0.5, type=float)
        
        ### DEFINING FRAME RATE
        parser.add_option('--frame_rate', dest = 'frame_rate', help = 'frame rate', default = None, type=str)
        
        ## GETTING NUMBER OF PROCESSORS
        parser.add_option('--n_procs', dest = 'n_procs', help = 'Number of processors', default = 20, type=int)
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## GETTING OUTPUTS
        path_sim = options.simulation_path
        path_pickle = options.path_pickle
        log_file = options.log_file
        
        ## LIPID MEMBRANE RESNAME
        lm_res_name = options.lm_name
        
        ## GRO/XTC
        gro_file = options.gro_file
        xtc_file = options.xtc_file
        
        ## RADIUS OF CUTOFF
        cutoff = options.cutoff_radius
        
        ## GETTING FRAME RAET
        frame_rate = options.frame_rate
        
        ## CHECKING FRAME RATE
        if frame_rate == 'None' or frame_rate == None:
            frame_rate = None
        else:
            frame_rate = int(frame_rate)
        
        ## TECHNICAL DETAILS
        n_procs = options.n_procs
        
    
    ## PRINTING
    print("Simulation path: %s"%( path_sim ))
    print("Pickle path: %s"%( path_pickle ))
    print("Gro file: %s"%(gro_file) )
    print("Xtc file: %s"%(xtc_file) )
    print("Cutoff: %.3f"%(cutoff))
    print("Number of processors: %d"%(n_procs) )
    if frame_rate != None:
        print("Frame rate: %d"%(frame_rate) )
    
    ## DEFINING XTC PATH
    xtc_path = os.path.join( path_sim, xtc_file )
    gro_path = os.path.join( path_sim, gro_file )
    
    ## GENERATING INPUT SCRIPT
    class_inputs = {"cutoff" : cutoff,
                    "lm_res_name" : lm_res_name,}
        
    ## DEFINING INPUTS FOR PARALLEL FUNCTION
    input_vars={'xtc_path' : xtc_path,
                'gro_path' : gro_path,
                'class_function': compute_NPLM_contacts,
                'frame_rate': frame_rate,
                'class_inputs': class_inputs,
                'n_procs'  : n_procs,
                'path_pickle': path_pickle,
                'combine_type': "append_list",
                'want_embarrassingly_parallel' :  True, #True
                'log_file': log_file,
                }
    
    ## RUNNING FUNCTION
    results_series = par_compute_class_function(**input_vars)
    
    #%% TESTING IF YOU GET THE SAME RESULT
    
    '''
    ## DEFINING INPUTS
    input_vars={'xtc_path' : xtc_path,
                'gro_path' : gro_path,
                'class_function': compute_NPLM_contacts,
                'frame_rate': frame_rate,
                'class_inputs': class_inputs,
                'n_procs'  : 1,
                'path_pickle': path_pickle,
                'combine_type': "append_list",
                'want_embarrassingly_parallel' : True,
                }
    
    ## RUNNING FUNCTION
    results_series = compute_class_function(**input_vars)
    
    ## DEFINING INPUTS
    input_vars={'xtc_path' : xtc_path,
                'gro_path' : gro_path,
                'class_function': compute_NPLM_contacts,
                'frame_rate': frame_rate,
                'class_inputs': class_inputs,
                'n_procs'  : n_procs,
                'path_pickle': path_pickle,
                'combine_type': "append_list",
                'want_embarrassingly_parallel' : True,
                }
    
    ## RUNNING FUNCTION
    results_parallel = compute_class_function(**input_vars)
    
    ## TESTING
    #%%
    
    for idx in range(len(results_series)):
        diff = np.sum(results_series[idx]-results_parallel[idx])
        print(diff)
    '''
    