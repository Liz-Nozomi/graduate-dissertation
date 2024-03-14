# -*- coding: utf-8 -*-
"""
root_mean_sq.py
this script contains the rmsd and rmsf functions from gromacs

CREATED ON: 02/24/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os
import copy
import numpy as np
import mdtraj as md
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path
## IMPORT TRAJECTORY FUNCTION
from MDDescriptors.application.sams.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl

##############################################################################
## CLASSES AND FUNCTIONS
##############################################################################
## FUNCTION TO COMPUTE RMSD
def compute_rmsd( sim_working_dir = "None",
                  input_prefix    = "None",
                  target_atoms    = "BACKBONE",
                  ndx_file        = "peptide.ndx",
                  rewrite         = False,
                  **kwargs ):
    r'''
    Function loads gromacs trajectory and computes root-mean-sq distribution
    '''
    if sim_working_dir  == "None" \
        or input_prefix == "None":
        print( "ERROR: missing inputs. Check inputs and try again" )
        return 1
    else:
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_rmsd.pkl" )
        if rewrite is not True and os.path.exists( path_pkl ):
            rmsd = load_pkl( path_pkl )
        else:
            ## LOAD MD TRAJECTORY
            traj = load_md_traj( path_traj    = sim_working_dir,
                                 input_prefix = input_prefix,
                                 **kwargs )
            ## PRECENTER TRAJECTORY
            traj.center_coordinates()
            ## GET REFERENCE TRAJ            
            ref_traj = get_reference_trajectory( traj, reference = "mean" )
            ## READ NDX FILE
            ndx_data = read_ndx( os.path.join( sim_working_dir, ndx_file ) )
            ## EXTRACT TARGET ATOMS
            target_indices = ndx_data[target_atoms]
            ## COMPUTE RMSD
            rmsd = md.rmsd( traj,
                            ref_traj,
                            frame = 0,
                            atom_indices = target_indices )
            ## SAVE RESULTS AS PKL FILE
            save_pkl( rmsd, path_pkl )
        return rmsd            

## FUNCTION TO COMPUTE RMSF
def compute_rmsf( sim_working_dir = "None",
                  input_prefix    = "None",
                  target_atoms    = "BACKBONE",
                  ndx_file        = "peptide.ndx",
                  rewrite         = False,
                  **kwargs ):
    r'''
    Function loads gromacs trajectory and computes root-mean-sq fluctuation
    '''
    if sim_working_dir  == "None" \
        or input_prefix == "None":
        print( "ERROR: missing inputs. Check inputs and try again" )
        return 1
    else:
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_rmsf.pkl" )
        if rewrite is not True and os.path.exists( path_pkl ):
            rmsf = load_pkl( path_pkl )
        else:
            ## LOAD MD TRAJECTORY
            traj = load_md_traj( path_traj    = sim_working_dir,
                                 input_prefix = input_prefix,
                                 **kwargs )
            ## PRECENTER TRAJECTORY
            traj.center_coordinates()
            ## GET REFERENCE TRAJ            
            ref_traj = get_reference_trajectory( traj, reference = "mean" )
            ## READ NDX FILE
            ndx_data = read_ndx( os.path.join( sim_working_dir, ndx_file ) )
            ## EXTRACT TARGET ATOMS
            target_indices = ndx_data[target_atoms]
            ## COMPUTE RMSD
            rmsf = md.rmsf( traj,
                            ref_traj,
                            frame = 0,
                            atom_indices = target_indices )
            ## SAVE RESULTS AS PKL FILE
            save_pkl( rmsf, path_pkl )
        return rmsf 

## FUNCTION TO GET REFERENCE TRAJECTORY
def get_reference_trajectory( traj, 
                              reference = "mean" 
                             ):
    r'''
    Function to compute reference trajectory
    '''
    copied_traj = copy.deepcopy( traj )
    ## GET MEAN COORDINATES
    xyz_mean = np.mean( copied_traj.xyz, axis = 0 )
    ## SET TRAJ XYZ TO MEAN VALUES
    copied_traj.xyz = xyz_mean[np.newaxis,...]
    ## RETURN REFERENCE TRAJ
    return copied_traj

## FUNCTION TO READ NDX FILE
def read_ndx( filename ):
    r'''
    '''
    with open( filename, 'r' ) as outputfile: # Opens gro file and make it writable
        file_data = outputfile.readlines()
    
    # SPLIT FILE INTO LINES
    groups = {}
    lines = [ n.strip( '\n' ) for n in file_data ]
    indices = []
    for nn, line in enumerate(lines):
        if "[" in line and len(indices) < 1:
            ## ADD FIRST GROUP TO DICT
            key = line.strip('[ ')
            key = key.strip(' ]')
        elif "[" in line and len(indices) > 1:
            ## ADD GROUP TO DICT
            groups[key] = np.array([ int(ndx)-1 for ndx in indices ])
            indices = []
            key = line.strip('[ ')
            key = key.strip(' ]')
        else:
            ## ADD INDICES TO GROUP
            indices += line.split()

    ## GET LAST GROUP
    groups[key] = np.array([ int(ndx)-1 for ndx in indices ])
    
    return groups
        