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

__all__ = [ 'rmsd' ]


##############################################################################
# Hexatic order class
##############################################################################

def rmsd( traj, ligand_indices, periodic = True ):
    R'''
    function which computes the rmsd of input indices 
    ***make sure your trajectory makes molecules whole***
    '''
    ref_traj = traj[0]
    ref_traj.xyz[0,...] = traj.xyz[:].mean(axis=0)
    ## LOOP THROUGH INPUT LIGAND BY LIGAND
    results = np.zeros( shape = (traj.time.size,len(ligand_indices)) )
    for ndx, lig in enumerate( ligand_indices ):
        results[:,ndx] = md.rmsd( traj, ref_traj, frame = 0, atom_indices = lig )
        
    return results
   
    
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
    
    data = rmsd( traj, ligand_ndx )