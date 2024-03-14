# -*- coding: utf-8 -*-
"""
memory.py
This file stores all memory functions

Written by: Alex K. Chew (01/03/2020)

FUNCTIONS:
    get_size:
        gets size of any object
    get_total_available_memory:
        gets total available memory for python
    get_frame_rate_based_on_memory:
        gets frame rate based on memory
"""

## IMPORTING MODULES
import sys
import mdtraj as md

### FUNCTION TO GET THE SIZE OF OBJECTS (DEPRECIATED)
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

### FUNCTION TO FIND TOTAL MEMORY
def get_total_available_memory(verbose = True):
    '''
    This function computes the total memory available
    '''
    from psutil import virtual_memory
    mem = virtual_memory()
    if verbose is True:
        print("Total memory available is %.1f GB"%( mem.available / 10**9 ) )
    return mem.available

### FUNCTION TO GET FRAME RATE
def get_frame_rate_based_on_memory(gro_path, 
                                   xtc_path,
                                   verbose = True
                                   ):
    '''
    The purpose of this function is to get the frame rate based on available 
    memory. This will first load the first frame of the trajectory, then 
    compute the size of the positions. Afterwards, we will estimate how 
    much memory we could load in one instance. 
    
    INPUTS:
        gro_path: [str]
            path to gro file
        xtc_path: [str]
            path to xtc file
        verbose: [logical]
            True if you want to print details
    OUTPUTS:
        frame_rate: [int]
            frame rate that you could load in a single instance
    '''
    ## LOADING SINGLE FRAME
    traj = md.load_frame(xtc_path, top=gro_path, index = 0)
    ## GETTING SIZE OF SINGLE TRAJECTORY
    traj_size = get_size(traj.xyz)
    ## GETTING AVAILABLE MEMORY
    available_memory = get_total_available_memory(verbose=verbose)
    ## GETTING FRAME RATE
    frame_rate = int( available_memory / traj_size )
    ## PRINTING 
    if verbose is True:
        print("Total frame rate is: %d"%(frame_rate) ) 
    
    return frame_rate