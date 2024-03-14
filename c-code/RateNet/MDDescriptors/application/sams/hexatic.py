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
import mdtraj as md
import numpy as np

__all__ = [ 'hexatic' ]


##############################################################################
# Hexatic order function
##############################################################################

def hexatic( traj, indices, cutoff = 0.7, periodic = True ):
    R'''
    Function to calculate the in-plane global bond-orientational (hexatic) order parameter
    '''
    ## LOOP THROUGH INPUT LIGAND BY LIGAND
    results = np.zeros( shape = (traj.time.size,len(indices)) )
    
    for ndx, tail in enumerate(indices):
        results[:,ndx] = calc_psi( tail, indices, cutoff, periodic )
        
    return results

## FUNCTION TO CALCULATE PSI
def calc_psi( point, all_points, cutoff, periodic ):
    R'''
    Calculates hexatic order of a given point    
    '''
    psi = np.nan * np.ones( shape = traj.time.size, dtype = 'complex' )
    ## DEFINE REFEREN
    ref_vector = np.array([ 1, 0 ])
    ref_mag = np.sqrt( ref_vector[0]**2 + ref_vector[1]**2 )
    
    ## MAKE LIST OF POTENTIAL NEIGHBOR ATOMS
    potential_neighbors = np.array([ [ ii, point ] for ii in all_points if ii != point ])
    vector = md.compute_displacements( traj, potential_neighbors, periodic = periodic ) # only care about x and y displacements
    dist = np.sqrt( np.sum( vector**2., axis = 2 ) )
    ## DETERMINE ATOMS IN CUTOFF
    dist[abs( dist ) > cutoff] = 0.
    mask = abs( dist ) > 0.
    n_neighbors = mask.sum( axis = 1 )
    ## CALCULATE ANGLE USING DOT PRODUCT AND DETERMINANT
    theta = np.zeros( shape = dist.shape )
    exp_theta = np.zeros( shape = dist.shape, dtype = 'complex' )
    dot_vec = vector[:,:,0] * ref_vector[0] + vector[:,:,1] * ref_vector[1]
    det_vec = vector[:,:,0] * ref_vector[1] - vector[:,:,1] * ref_vector[0]
    theta[mask] = np.arccos( dot_vec[mask] / ( dist[mask] * ref_mag ) )
    theta[det_vec < 0.] = 2. * np.pi - theta[det_vec < 0.]
    exp_theta[mask] = np.exp( 6j * theta[mask] )
    psi[n_neighbors > 0.] = np.sum( exp_theta[n_neighbors > 0.], axis = 1 ) / n_neighbors[n_neighbors > 0.]
    
    return np.abs( psi )**2.

##############################################################################
# Stand alone script (TESTING)
##############################################################################

if __name__ == "__main__":    
    from core import import_traj
    ## TESTING
    wd = r'R:/simulations/physical_heterogeneity/unbiased/sam_single_8x8_300K_butanethiol_tip3p_nvt_CHARMM36/'
    args = {  
              'in_file': 'sam_prod_whole',
              'in_path':   wd,
              'solvents':  [ 'HOH', 'MET' ],
           }
    
    out_path = wd + 'output_files/'
        
    traj_import = import_traj()
    traj = traj_import.load( **args )
    ligand_ndx = [ [ atom.index for atom in residue.atoms if 'H' not in atom.name ] for residue in traj.topology.residues if residue.name not in args['solvents'] ]
    end_group_ndx = np.array([ ligand[-1] for ligand in ligand_ndx ])
    
    data = hexatic( traj, indices = end_group_ndx )