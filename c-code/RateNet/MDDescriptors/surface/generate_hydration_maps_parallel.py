# -*- coding: utf-8 -*-
"""
generate_hydration_maps_parallel.py
The purpose of this function is to generate hydration maps in parallel. This 
function should also generate pickles for each frames and then repickle all 
files once completed. 

Written by: Alex K. Chew (01/03/2020)

The main idea is to loop through the trajectory, perhaps using iterloads, and 
generate pickles for each frame instance. We will have one file that will 
generate the different jobs. Afterwards, we will combine all the 
pickles into one pickle file to save space. There will be print commands to 
clearly delineate what is going on. 
"""

## IMPORTING NECESSARY MODULES
import os
import mdtraj as md

## CHECK PATH
from MDDescriptors.core.check_tools import check_path, check_dir

## PICKLE TOOLS
from MDDescriptors.core.pickle_tools import pickle_results

## IMPORTING CORE FUNCTIONS
from MDDescriptors.surface.core_functions import load_datafile

## IMPORTING SURFACE
from MDDescriptors.surface.generate_hydration_maps import calc_neighbors

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import R_CUTOFF

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

## IMPORTING TIME
from MDDescriptors.core.track_time import track_time

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing 


## IMPORTING MEMORY
from MDDescriptors.core.memory import get_frame_rate_based_on_memory 

## USING CORE FUNCTIONS FROM SURFACE
from MDDescriptors.surface.core_functions import get_list_args

## GETTING ITERTOOLS
from MDDescriptors.core.traj_itertools import get_pickle_name_for_itertool_traj, update_neighbor_log

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import par_compute_class_function

#### FUNCTION TO GET NEAREST NEIGHBOR ARRAYS
#def compute_nearest_neighbor_arrays(xtc_path,
#                                    gro_path,
#                                    grid,
#                                    path_pickle,
#                                    pickle_log = "neighbors.log",
#                                    frame_rate = None,
#                                    residue_list = ["HOH", "CL", "NA"],
#                                    cutoff_radius = R_CUTOFF,
#                                    n_procs = 1,
#                                    verbose = True,
#                                    ):
#    '''
#    The purpose of this function is to compute the nearest neighbor array. 
#    INPUTS:
#        xtc_path: [str]
#            path to xtc file
#        gro_path: [str]
#            path to gro file
#        grid: [np.array, shape = (num_grid, 3)]
#            x,y,z grid points to look for neighbors
#        path_pickle: [str]
#            location to store your pickles in
#        pickle_log: [str]
#            location to store all your pickle information
#        frame_rate: [int]
#            frame rate for loading trajectory. If None, we will look for 
#            the optimal trajectory rate based on your available memory. 
#            Note that you should have a lower frame rate if you have a 
#            lack of memory available. 
#        residue_list: [list]
#            list of residues to look for
#        cutoff_radius: [float]
#            cutoff radius used to count the atoms
#        n_procs: [int]
#            number of processor to run the parallel code, default = 1 (no parallel)
#        verbose: [logical]
#            True if you want to print out every detail
#    OUTPUTS:
#        num_neighbor_array: [np.array, shape=(num_grid, frames)]
#            number of neighbor array, used primarily for debugging
#    '''
#    ## DEFINING PATH PICKLE LOG
#    path_pickle_log = os.path.join(path_pickle,
#                                   pickle_log)
#    
#    ## GETTING FRAME_RATE IF NOT AVAILABLE
#    if frame_rate is None:
#        frame_rate =  get_frame_rate_based_on_memory(gro_path = gro_path, 
#                                                     xtc_path = xtc_path)        
#    ## IMPORTING TIME TRACKER
#    overall_time = track_time()
#    
#    ## DEFINING FIRST INSTANCE
#    first_fail = False
#    
#    ## WHILE LOOP TO ENSURE PARALLELIZATION CORRECTLY PARALLELIZES. 
#    # The issue here is that we have large memory errors when running the optimal 
#    # neighboring array option. If we split the trajectory in a more slow for-loop 
#    # fashion, we should be able to parallelize. 
#    while True:
#        
#        try:
#            ## PRINTING
#            if verbose is True:
#                print("Loading trajectory with frame rate: %d"%(frame_rate))
#        
#            ## LOOPING THROUGH TRAJECTORY
#            for idx, traj in enumerate(md.iterload(xtc_path, top=gro_path, chunk = frame_rate)):
#                ## PRINTING
#                if verbose is True:
#                    print("*** Loading trajectory, %d frames ***"%(len(traj)))
#                
#                ## FREEZING FOR ONE INDEX <- - remove when complete
#                # if idx in [0, 1, 2, 3, 4, 5, 6]:
#                ## IMPORTING TIME TRACKER
#                time_tracker = track_time()
#                ## DEFINING NEIGHBORS
#                if idx == 0:
#                    neighbors = calc_neighbors(traj = traj,
#                                               grid = grid,
#                                               residue_list = residue_list,
#                                               cutoff_radius = cutoff_radius,
#                                               )
#                    
#                
#                ## SERIAL CODE
#                if n_procs == 1:
#                    ## COMPUTING NEIGHBOR ARRAY
#                    num_neighbor_array = neighbors.compute_neighbor_array(traj = traj,)
#                    
#                else: ## PARALLEL CODE
#                    num_neighbor_array = parallel_analysis_by_splitting_traj(traj = traj, 
#                                                                             class_function = neighbors.compute_neighbor_array, 
#                                                                             n_procs = n_procs,
#                                                                             combine_type="concatenate_axis_1",
#                                                                             want_embarrassingly_parallel = True)
#                    
#                ## GETTING PICKLE NAME
#                pickle_name = get_pickle_name_for_itertool_traj(traj)
#                
#                ## STORING PICKLE RESULTS
#                pickle_results(results = [num_neighbor_array],
#                               pickle_path = os.path.join(path_pickle, pickle_name),
#                               verbose = True,
#                               )
#                
#                ## UPDATING PICKLE LOG
#                update_neighbor_log(path_pickle_log = path_pickle_log,
#                                    index = idx,
#                                    pickle_name = pickle_name)
#                
#                ## PRINTING TIME
#                time_tracker.time_elasped()
#        except (OSError, MemoryError): # Due to memory issues, decrease frame rate
#            print("Since OSError found for frame rate %d, decreasing the frame rate!"%(frame_rate) )
#            if first_fail is False:
#                if frame_rate > 10000:
#                    frame_rate = 10000  
#                else:
#                    frame_rate = int(frame_rate * 3/4)
#                first_fail = True
#            else:
#                frame_rate = int(frame_rate * 3/4)
#            
#            ## CHECKING IF FRAME RATE GOES TO ZER0. IF SO, WE NEED TO CHANGE FRAME RATE
#            # AND ALSO, WE MAY NEED TO CHANGE PARALLILIZATION BACK TO 1
#            if frame_rate <= 0:
#                frame_rate = 1 # Changing back to 1
#                n_procs = 1
#                print("--------------------------------")
#                print("Since frame_rate has decreased below 0, changing parallel code to serial")
#                print("This may be due to low available RAM space.")
#                print("--------------------------------")
#            print("New frame rate: %d"%(frame_rate) )
#        else:
#            print("Code is completed!")
#            break
#    
#    ## PRINTING
#    overall_time.time_elasped(prefix_string="Overall")
#    
#    return num_neighbor_array


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    
    ## SEE IF TESTING IS ON
    testing = check_testing()

    #######################
    ### DEFINING INPUTS ###
    #######################

    ## RUNNING TESTING    
    if testing == True:
        
        ## NUMBER OF CORES
        n_procs = 2
        
        ## DEFINING MAIN SIMULATION
        main_sim=check_path(r"S:\np_hydrophobicity_project\simulations\PLANAR")
        ## DEFINING SIM NAME
        sim_name=r"FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
        ## DEFINING WORKING DIRECTORY
        simulation_path = os.path.join(main_sim, sim_name)
        ## DEFINING GRO AND XTC
        gro_file = r"sam_prod.gro"
        xtc_file = r"sam_prod_1ns.xtc"
    
        ## DEFINING ANALYSIS LOCATION
        analysis_folder = r"16-0.24-0.1,0.1,0.1-0.33-HOH_CL_NA"
        grid_folder = r"grid-0_1000"
        dat_file = r"out_willard_chandler.dat"
        
        ## DEFINING LOCATION TO STORE PICKLES
        pickle_folder = "compute_neighbors"
        pickle_log = "neighbors.log"
        
        ## DEFINING PATH TO PICKLE FOLDER
        path_pickle = os.path.join(simulation_path,
                                   analysis_folder,
                                   pickle_folder,
                                   )
            
        ## DETINING HYDRATION MAP LOCATION
        path_grid = os.path.join(simulation_path,
                                 analysis_folder,
                                 grid_folder,
                                 dat_file
                                 )
        
        ## DEFINING RESIDUE LIST
        residue_list = ["HOH", "CL", "NA"]
        
        ## DEFINING CUTOFF
        cutoff_radius = R_CUTOFF
        
        ## DEFINING FRAME RATE
        frame_rate = None
        
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
        
        ## DEFINING RESIDUE LIST
        parser.add_option("--residue_list", dest="residue_list", action="callback", type="string", callback=get_list_args,
                  help="residue list separated by comma (no whitespace)", default = [ "all_heavy" ] )
        
        ## DEFINING OUTPUT PICKLE PATH
        parser.add_option('--path_pickle', dest = 'path_pickle', help = 'Path of pickle', default = '.', type=str)
        
        ## LOG FILE
        parser.add_option('--pickle_log', dest = 'pickle_log', help = 'Log file of the pickle', default = '.', type=str)
        
        ### DEFINING GRID PATH
        parser.add_option('--path_grid', dest = 'path_grid', help = 'Path of grid', default = '.', type=str)
        
        ### DEFINING CUTOFF
        parser.add_option('--cutoff_radius', 
                          dest = 'cutoff_radius', 
                          help = 'Cutoff for grid counting', 
                          default = R_CUTOFF, type=float)
        
        ### DEFINING FRAME RATE
        parser.add_option('--frame_rate', dest = 'frame_rate', help = 'frame rate', default = "10000", type=str)
        
        ## GETTING NUMBER OF PROCESSORS
        parser.add_option('--n_procs', dest = 'n_procs', help = 'Number of processors', default = 20, type=int)
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## GETTING OUTPUTS
        simulation_path = options.simulation_path
        path_pickle = options.path_pickle
        path_grid = options.path_grid
        pickle_log = options.pickle_log
        
        ## GRO/XTC
        gro_file = options.gro_file
        xtc_file = options.xtc_file
        
        ## RESIDUE LIST
        residue_list = options.residue_list
        
        ## RADIUS OF CUTOFF
        cutoff_radius = options.cutoff_radius
        
        ## GETTING FRAME RAET
        frame_rate = options.frame_rate
        
        ## CHECKING FRAME RATE
        if frame_rate == 'None' or frame_rate == None:
            frame_rate = None
        else:
            frame_rate = int(frame_rate)
        
        ## TECHNICAL DETAILS
        n_procs = options.n_procs
        
    ###################
    ### MAIN SCRIPT ###
    ###################

    ## DEFINING XTC PATH
    gro_path = os.path.join(simulation_path, gro_file)
    xtc_path = os.path.join(simulation_path, xtc_file)

    ## LOADING THE GRID
    grid = load_datafile(path_grid)

    ## GENERATING INPUT SCRIPT
    class_inputs = {"grid" : grid,
                    "cutoff_radius" : cutoff_radius,
                    "residue_list" : residue_list,}
        
    ## DEFINING INPUTS FOR PARALLEL FUNCTION
    input_vars={'xtc_path' : xtc_path,
                'gro_path' : gro_path,
                'class_function': calc_neighbors,
                'frame_rate': frame_rate,
                'class_inputs': class_inputs,
                'n_procs'  : n_procs,
                'path_pickle': path_pickle,
                'combine_type': "concatenate_axis_1", # "append_list",
                'want_embarrassingly_parallel' :  True,
                'log_file': pickle_log,
                }

    ## RUNNING FUNCTION
    num_neighbor_array = par_compute_class_function(**input_vars)
    

#    ### RUNNING NEIGHBOR SEARCH
#    num_neighbor_array = compute_nearest_neighbor_arrays(xtc_path = xtc_path,
#                                                         gro_path = gro_path,
#                                                         grid = grid,
#                                                         path_pickle = path_pickle,
#                                                         pickle_log = pickle_log,
#                                                         frame_rate = frame_rate,
#                                                         residue_list = residue_list,
#                                                         cutoff_radius = cutoff_radius,
#                                                         n_procs = n_procs,
#                                                         )
#
