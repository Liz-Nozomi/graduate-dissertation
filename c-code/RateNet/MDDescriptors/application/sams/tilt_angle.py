# -*- coding: utf-8 -*-
##############################################################################
# sam_analysis: A Python Library for the characterization of SAMs
#
# Author: Bradley C. Dallin
# email: bdallin@wisc.edu
# 
##############################################################################


##############################################################################
# Imports
##############################################################################

from __future__ import print_function, division
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
from MDDescriptors.core.initialize import checkPath2Server
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

__all__ = [ 'tilt_angle' ]


##############################################################################
# SAM tilt angle function
##############################################################################

def tilt_angle( traj, ligand_indices, periodic = True ):
    R'''
    Function to calculate the thinkness of SAM from provided indices
    '''
    pairs = []
    for indices in ligand_indices:
        ## GET LIGAND COM COORDS AND ASSIGN TO A DUMMY ATOM
        ligand_com = calculate_com( traj, indices )
        traj.xyz[ :, indices[1], : ] = ligand_com
        pairs.append([ indices[1], indices[0] ])
        
    ## DETERMINE LIGAND VECTORS
    pairs = np.array( pairs )
    vectors = md.compute_displacements( traj, pairs, periodic = periodic )
    dist = np.sqrt( np.sum( vectors**2., axis = 2 ) )
    tilt_angles = np.arccos( np.abs(vectors[:,:,2] / dist ) ) * 180 / np.pi
    
    return tilt_angles

def calculate_com( traj, atom_indices ):
    R'''
    '''
    group_masses = np.array([ traj.topology.atom(ii).element.mass for ii in atom_indices ]) # assumes all water
    group_mass = group_masses.sum()
    coords = traj.xyz[:,atom_indices,:]
    return np.sum( coords * group_masses[np.newaxis,:,np.newaxis], axis=1 ) / group_mass
    

##############################################################################
# Stand alone script (TESTING)
##############################################################################

if __name__ == "__main__":
    ## TESTING
    solvents = [ "HOH" ]
    counterions = [ "CL" ]
    wd = r'R:\simulations\polar_sams\indus\sam_single_12x12_300K_C13NH3_tip4p_nvt_CHARMM36_2x2x0.3nm\equil'
    path2AnalysisDir = checkPath2Server( wd ) # PC Side
    top_file = r"sam_equil_whole.gro"
    xtc_file = r"sam_equil_whole.xtc"
    ### LOADING TRAJECTORY  
    traj_data = import_tools.import_traj( directory      = path2AnalysisDir, # Directory to analysis
                                          structure_file = top_file,         # structure file
                                          xtc_file       = xtc_file,         # trajectories
                                        )
    
    ligand_indices = [ [ atom.index for atom in residue.atoms ] for residue in traj_data.traj.topology.residues if residue.name not in solvents + counterions ]
        
    tilt_angles = tilt_angle( traj_data.traj, ligand_indices, periodic = False )
    
    ## PLOTTING
    plt.figure()
    t = traj_data.traj.time
    for ii in range(tilt_angles.shape[1]):
        plt.plot( t, tilt_angles[:,ii], linestyle = "--", linewidth = 2 )
        
    plt.xlabel( "sim. time" )
    plt.ylabel( "tilt angle (degrees)" )
    
    plt.figure()
    t = traj_data.traj.time
    y = tilt_angles.mean(axis=1)
    plt.plot( t, y, linestyle = "--", linewidth = 2 )
        
    plt.xlabel( "sim. time" )
    plt.ylabel( "tilt angle (degrees)" )