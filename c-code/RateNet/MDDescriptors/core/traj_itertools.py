# -*- coding: utf-8 -*-
"""
traj_itertools.py
This function contains pickling tools for itertool trajectories
INPUTS:
    get_pickle_name_for_itertool_traj:
        function that gets the pickle names
    update_neighbor_log:
        function that updates the neighbor logs
    get_traj_from_last_frame:
        function that could get the last frame of the trajectory

Written by: Alex K. Chew (01/24/2020)

"""
import numpy as np

### FUNCTION TO GET PICKLE FILE NAME
def get_pickle_name_for_itertool_traj(traj = None,
                                      traj_initial = None,
                                      traj_final = None,
                                      ):
    '''
    The purpose of this function is to get the traj time, then output a pickle 
    name.
    INPUTS:
        traj: [obj]
            trajectory information
        traj_initial: [int]
            initial trajectory (optional)
        traj_final: [int]
            final trajectory (optional)
        
    OUTPUTS:
        pickle_name: [str]
            pickle name to save for itertools
    '''
    ## GETTING TRAJ TIMES
    if traj_initial is None:
        traj_initial = traj.time[0]
    if traj_final is None:
        traj_final   = traj.time[-1]
    
    ## OUTPUT PICKLE NAME
    pickle_name = "%d-%d.pickle"%(int(traj_initial),int(traj_final) )
    
    return pickle_name


### FUNCTION TO UPDATE NEIGHBOR LOG
def update_neighbor_log(path_pickle_log,
                        index, 
                        pickle_name):
    '''
    The purpose of this function is to update neighbor log.
    INPUTS:
        path_pickle_log: [str]
            path of the pickle log
        index: [int]
            index
        pickle_name: [str]
            pickle name string
    OUTPUTS:
        void, it will create a file and update the log file
    '''
    print("Current index:")
    print(index)
    ## IF INITIAL INDEX, THEN WRITE FILE
    if index == 0:
        file_write = "w"
    else:
        file_write = "a"
    ## OPENING
    with open(path_pickle_log, file_write) as f:
        f.write("%d, %s\n"%(index, pickle_name))
    return

### FUNCTION TO GET THE LAST OF THE TRAJECTOY
def get_traj_from_last_frame(traj,
                             last_time_ps = 0):
    '''
    The purpose of this function is to get the last N ps of the trajectory.
    INPUTS:
        traj: [obj]
            trajectory object used to get the last frame
        last_time_ps: [float]
            last time in picosecond
    OUTPUTS:
        traj_shorten: [obj]
            new trajectory with updated times
    '''
    ## GETTING THE TIME
    time_array = traj.time    
    ## ONLY SELECT INDICES THAT ARE WITHIN THE LAST TIME
    if last_time_ps > 0:
        ## NORMALIZE BY FIRST VALUE
        time_array = time_array - time_array[0]
        ## GETTING DIFF
        diff = np.abs(time_array - time_array[-1])
        indices = np.where(diff <= last_time_ps)[0]
    else:
        indices = np.arange(len(time_array))
        
    ## SHORTEN INDICES
    traj_shorten = traj[indices]
    
    return traj_shorten