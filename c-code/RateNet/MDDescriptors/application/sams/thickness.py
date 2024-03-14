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

__all__ = [ 'thickness' ]


##############################################################################
# SAM thickness function
##############################################################################

def thickness( traj, pairs, periodic = True ):
    R'''
    Function to calculate the thinkness of SAM from provided indices
    '''
    ## DETERMINE LIGAND VECTORS
    vectors = md.compute_displacements( traj, pairs, periodic = periodic )
    
    ## DETERMINE SAM THICKNESS AND LIGAND LENGTHS
    thickness = vectors[:,:,2]
    length = np.sqrt( np.sum( vectors**2., axis = 2 ) )
    
    return thickness, length

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
    pairs = np.array([ [ ligand[0], ligand[-1] ] for ligand in ligand_ndx ])
        
    thickness, length = thickness( traj, pairs )