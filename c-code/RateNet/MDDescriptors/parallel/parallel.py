#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel.py
This code contains all scripts necessary to parallel jobs. 

FUNCTIONS:
    debug_par_function: 
        function used to debug parallel code
    parallel_analysis_by_splitting_traj:
        Analysis is done using splittin trajectory functions
    par_compute_class_function:
        parallel for loop for classes
CREATED ON: 11/2/2019



"""
import multiprocessing
import datetime
import mdtraj as md
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # Loading trajectory details
import sys
import os

## IMPORTING MEMORY
from MDDescriptors.core.memory import get_frame_rate_based_on_memory 

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing, check_dir, stop_if_does_not_exist

## GETTING ITERTOOLS
from MDDescriptors.core.traj_itertools import get_pickle_name_for_itertool_traj, update_neighbor_log

## PICKLE TOOLS
from MDDescriptors.core.pickle_tools import pickle_results

## IMPORTING TIME
from MDDescriptors.core.track_time import track_time


### TESTING FUNCTION
def debug_par_function(x):
    return x*x

### GENERALIZED PARALLEL FUNCTION
def parallel_analysis_by_splitting_traj(traj, 
                                        class_function, 
                                        input_details = None, 
                                        n_procs = 1,
                                        combine_type="sum",
                                        want_embarrassingly_parallel = False):
    '''
    The purpose of this script is to generalize the code for splitting trajectories. 
    We assume that you want to initialize the class function, then run a class function for it.
    ASSUMPTION:
        - output is a numpy array that could be merged
    INPUTS:
        traj: [traj]
            trajectory details
        class_function: [func]
            function that you want to parallelize
        input_details: [dict] ** DEPRECIATED ** This does not work
            input details that will be inserted into the function
        n_procs: [int]
            number of processors. This will also be the extent of splitting trajectories.
        combine_type: [str]
            combination type, default is sum
                This will just add all the details into a single sum file
                Alternative:
                    "None" -- No summing, just output everything and you'll figure it out later
                    "concatenate_axis_1" -- concatenate the results together along the axis 1 (e.g. num_grid vs. frame)
                    "append_list" -- appends the list of values
        want_embarrassingly_parallel: [logical]
            True if you want to input trajectory as a per frame basis. By default, 
            this is False meaning that we split the trajectory based on the number 
            of processors. If True, we will split the trajectory based on the total 
            number of trajectories; then, we will input each frame into the 
            function. 
    OUTPUT:
        results_output: [tuple]
            output from your function
    '''
    ## COMPUTING NUBMER OF PROCESSORS
    if n_procs < 0:
        n_procs = multiprocessing.cpu_count()
    
    
    ## TRACKING THE TIME
    start = datetime.datetime.now()
    
    ## RUNNING MULTIPROCESSING DETAILS USING COSOLVENT MAPPING PROTOCOL
    if n_procs != 1:
        ## PRINTING
        print( "\n--- Running %s on %d cores ---\n" %( class_function.__name__, n_procs ) )
        
        ## PRINTING FOR MASSIVELY PARALLEL
        if want_embarrassingly_parallel is False:
            ## DEFINING TRAJ SPLIT
            list_splitted = calc_tools.split_list(alist = traj, wanted_parts = n_procs)
            print("SPLITTED TRAJECTORY SUCCESSFUL! %d trajectories containing %d frames"%( len(list_splitted), len(traj) ) )
            ## RUNNING MULTIPROCESSING
            with multiprocessing.Pool( processes = n_procs ) as pool:
                ## RUNNING FUNCTION
                results = pool.map(class_function, list_splitted)
            
        else:
            print("Since embarrassingly parallel is turned on, we split the trajectory based on total number of frames")
            # ## RUNNING MULTIPROCESSING
            # with multiprocessing.Pool( processes = n_procs ) as pool:
            ## SPLITTING LIST
            frame_indexes = np.arange(len(traj))
            ## CREATING LIST SPLITTED
            list_splitted = [ (traj, [each_index]) for each_index in frame_indexes]
            ## PRINTING
            print("SPLITTED TRAJECTORY SUCCESSFUL ACROSS %d frames"%( len(traj) ) )
            print("GENERATING POOL FOR %d PROCESSORS"%(n_procs))
            ## RUNNING STARMAP FOR MULTIPROCESSING
            with multiprocessing.Pool( processes = n_procs ) as pool:
                ## RUNNING FUNCTION
                results = pool.starmap(class_function, list_splitted)
        
        try:
            ## SUMMARY
            if combine_type == "sum":
                results_output = np.sum(results,axis = 0)
            elif combine_type == "concatenate_axis_0":
                results_output = np.concatenate( results,axis = 0 )
            elif combine_type == "concatenate_axis_1":
                results_output = np.concatenate(results,axis =1)
            elif combine_type == "append_list":
                results_output = calc_tools.flatten_list_of_list(results)
            elif combine_type == "None":
                results_output = results
        except Exception:
            print("Error in parallel combining outputs!")
            print("Here is the current combination type: %s"%(combine_type)  )
            print("First two results as an example:")
            print(results[0])
            if len(results) > 2:
                print(results[1])
            ## EXITING
            print("Stopping here to prevent errors!")
            sys.exit(1)
            
        ## DECOMPOSING LIST -- error, leaving alone for now
        # results_output = np.array(calc_tools.flatten_list_of_list( results ))
    
    else:
        print("\n--- Running code serially on one core! ---\n")
        ## SERIAL APPROACH
        results_output = class_function(traj = traj)
    
    ## PRINTING TIME ELAPSED
    time_elapsed = datetime.datetime.now() - start
    print( 'Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed) )
    
    return results_output

### FUNCTION TO COMPUTE PARALLEL CODE
def par_compute_class_function(xtc_path,
                               gro_path,
                               class_function,
                               class_inputs = {},
                               path_pickle = None,
                               log_file = "log_file.log",
                               class_obj_pickle = "class_obj.pickle",
                               frame_rate = None,
                               n_procs = 1,
                               verbose = True,
                               combine_type = "concatenate_axis_0",
                               want_embarrassingly_parallel = False):
    '''
    The purpose of this function is to compute the contacts in parallel. 
    INPUTS:
        xtc_path: [str]
            path to xtc file
        gro_path: [str]
            path to gro file
        class_function: [obj]
            class function to run
        class_inputs: [dict]
            inputs for the class
        frame_rate: [int]
            frame rate for loading trajectory. If None, we will look for 
            the optimal trajectory rate based on your available memory. 
            Note that you should have a lower frame rate if you have a 
            lack of memory available. 
        log_file: [str]
            log file
        n_procs: [int]
            number of processor to run the parallel code, default = 1 (no parallel)
        verbose: [logical]
            True if you want to print out every detail
        combine_type: [str]
            way to combine the results afterwards
            "concatenate_axis_0" is the default
        want_embarrassingly_parallel: [logical]
            True if you want to input trajectory as a per frame basis
    OUTPUTS:
        results: [list]
            list of results
    '''
    ## DEFINING PATH PICKLE LOG
    path_pickle_log = os.path.join(path_pickle,
                                   log_file)
    
    ## GETTING FRAME_RATE IF NOT AVAILABLE
    if frame_rate is None:
        frame_rate =  get_frame_rate_based_on_memory(gro_path = gro_path, 
                                                     xtc_path = xtc_path)     
        
    ## IMPORTING TIME TRACKER
    overall_time = track_time()
    
    ## DEFINING FIRST INSTANCE
    first_fail = False
    
    ## CHECKING FILES EXISTENCE
    stop_if_does_not_exist(gro_path)
    stop_if_does_not_exist(xtc_path)
    
    ## WHILE LOOP TO ENSURE PARALLELIZATION CORRECTLY PARALLELIZES. 
    # The issue here is that we have large memory errors when running the optimal 
    # neighboring array option. If we split the trajectory in a more slow for-loop 
    # fashion, we should be able to parallelize. 
    while True:
        try:
            ## PRINTING
            if verbose is True:
                print("Loading trajectory with frame rate: %d"%(frame_rate))
            
            ## LOOPING THROUGH TRAJECTORY
            for idx, traj in enumerate(md.iterload(xtc_path, top=gro_path, chunk = frame_rate)):
                ## GETTING TOTAL FRAMES
                total_frames = len(traj)
                ## PRINTING
                if verbose is True:
                    print("*** Loading trajectory, %d frames ***"%(total_frames))
                ## DEFINING NEIGHBORS
                if idx == 0:
                    ## CHECKING DIRECTORY
                    check_dir(path_pickle)
                    ## ADDING TRAJECTORY
                    class_inputs['traj'] = traj
                    ## INITIATION
                    class_object = class_function(**class_inputs)
                    if first_fail is False:
                        ## STORING CLASS OBJECT
                        pickle_results(results = [class_object],
                                       pickle_path = os.path.join(path_pickle, class_obj_pickle),
                                       verbose = True,
                                       )
                
                ## IMPORTING TIME TRACKER
                time_tracker = track_time()
                
                #########################
                ### RUNNING MAIN CODE ###
                #########################
                
                ## SERIAL CODE
                if n_procs == 1:
                    if want_embarrassingly_parallel is False:
                        ## COMPUTING OUTPUT
                        results = class_object.compute(traj = traj)
                    else:
                        print("Since embarrassingly parallel is turned on, we will run for each frame")
                        results = []
                        ## LOOPING THROUGH EACH FRAME
                        for frame in range(len(traj)):
                            results.extend(class_object.compute(traj = traj[frame]))
                    
                else:
                    ## COMPUTING OUTPUT BY SPLITTING TRAJECTORY
                    results = parallel_analysis_by_splitting_traj(traj = traj, 
                                                                  class_function = class_object.compute, 
                                                                  n_procs = n_procs,
                                                                  combine_type=combine_type,
                                                                  want_embarrassingly_parallel = want_embarrassingly_parallel)
                ## PRINTING TIME
                time_tracker.time_elasped()
                
                ## GETTING PICKLE NAME
                pickle_name = get_pickle_name_for_itertool_traj(traj)
                
                ## STORING PICKLE RESULTS
                pickle_results(results = [results],
                               pickle_path = os.path.join(path_pickle, pickle_name),
                               verbose = True,
                               )
                
                ## UPDATING PICKLE LOG
                update_neighbor_log(path_pickle_log = path_pickle_log,
                                    index = idx,
                                    pickle_name = pickle_name)
                
        except (OSError, MemoryError): # Due to memory issues, decrease frame rate
            print("Since OSError found for frame rate %d, decreasing the frame rate!"%(frame_rate) )
            if first_fail is False:
                if frame_rate > 10000:
                    frame_rate = 10000  
                else:
                    frame_rate = int(frame_rate * 3/4)
                first_fail = True
            else:
                frame_rate = int(frame_rate * 3/4)
            
            ## STOPPING IF FRAME RATE IS 1
            if frame_rate == 1 and n_procs == 1:
                print("Error in code! Frame rate is 1 and number of processors is 1!")
                print("Check if the code is going overboard, we cannot load a single frame!")
                print("Stopping here!")
                sys.exit(1)
                
            ## CHECKING IF FRAME RATE GOES TO ZER0. IF SO, WE NEED TO CHANGE FRAME RATE
            # AND ALSO, WE MAY NEED TO CHANGE PARALLILIZATION BACK TO 1
            if frame_rate <= 0:
                frame_rate = 1 # Changing back to 1
                n_procs = 1
                print("--------------------------------")
                print("Since frame_rate has decreased below 0, changing parallel code to serial")
                print("This may be due to low available RAM space.")
                print("--------------------------------")            

            print("New frame rate: %d"%(frame_rate) )
        else:
            print("Code is completed!")
            break
    ## PRINTING
    overall_time.time_elasped(prefix_string="Overall")
    return results
