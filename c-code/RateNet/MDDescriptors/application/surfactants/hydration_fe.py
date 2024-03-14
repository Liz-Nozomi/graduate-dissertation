# -*- coding: utf-8 -*-
"""
hydration_fe.py
The purpose of this script is to compute the hydration fe

CREATED ON: 03/18/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os, sys
import pickle
import numpy as np  # Used to do math functions
import mdtraj as md # Used for distance calculations
import MDDescriptors.core.import_tools as import_tools          # Loading trajectory details
from MDDescriptors.core.calc_tools import compute_displacements # Loading calculation tools

## IMPORTING TRACKING TIME
from MDDescriptors.core.track_time import track_time
from MDDescriptors.core.initialize import checkPath2Server

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

##############################################################################
## CLASSES & FUNCTION
##############################################################################
### FUNCTION TO COMPUTE HBONDS
def compute_hydration_fe( traj,
                          path_pkl,
                          ref_coords = [ 0, 0, 0 ],
                          r_cutoff = 0.33,
                          periodic = True,
                          residue_list = ['HOH'],
                          verbose = True,
                          print_freq = 100,
                          rewrite = False,
                          **kwargs ):
    R'''
    The purpose of this function is to compute the hydration fe of a spherical cavity. 
    '''
    if os.path.exists( path_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
    else:
        ## TRACKING TIME
        timer = track_time()
        
        ## EXTRACT HEAVY ATOM INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in traj.topology.atoms 
                                  if atom.element.symbol != "H"
                                  and atom.element.symbol != "VS" ])
        ## COMPUTE RDF BETWEEN GROUP AND SOLVENT MOLECULES
        dist_vector = compute_displacements( traj, 
                                             atom_indices,
                                             box_dimensions = traj.unitcell_lengths[0,:],
                                             ref_coords = ref_coords, 
                                             periodic = periodic )
        dist = np.sqrt( np.sum( dist_vector**2, axis = -1 ) )
    
        ## COUNT ATOMS IN CAVITY
        dist = r_cutoff - dist
        N = np.sum( dist >= 0., axis = 1 )
        N_range = ( 0, N.max() )
        histo, N = np.histogram( N, range = N_range, bins = N.max() )
        N = N[:-1]
        p_N = histo / np.trapz( histo, x = N )
        log_p = -np.log( p_N )
        p = np.polyfit( N, log_p, 2 )
        fit = np.poly1d(p)
        log_p_fit = fit(N)
        
        ## OUTPUTTING TIME
        timer.time_elasped()
        ## SAVE PICKLE FILE
        save_pkl( [ N, p_N, log_p, log_p_fit ], path_pkl )
        ## SAVE LOG(p)
        path_log_pkl = path_pkl.replace( "hydration_fe", "hydration_fe_log" )
        save_pkl( [ N, log_p ], path_log_pkl )
        ## SAVE FIT
        path_fit_pkl =  path_pkl.replace( "hydration_fe", "hydration_fe_fit" )
        save_pkl( [ N, log_p_fit ], path_fit_pkl )
    
def load_pkl( path_pkl ):
    r'''
    Function to load data from pickle file
    '''
    print( "LOADING PICKLE FROM %s" % ( path_pkl ) )
    with open( path_pkl, 'rb' ) as input:
        data = pickle.load( input )
        
    return data

def save_pkl( data, path_pkl ):
    r'''
    Function to save data as pickle
    '''
    print( "PICKLE FILE SAVED TO %s" % ( path_pkl ) )
    with open( path_pkl, 'wb' ) as output:
        pickle.dump( data, output, pickle.HIGHEST_PROTOCOL )
