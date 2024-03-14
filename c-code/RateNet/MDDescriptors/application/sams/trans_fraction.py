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

__all__ = [ 'trans_fraction' ]

##############################################################################
# Hexatic order class
##############################################################################

def trans_fraction( traj, ligand_indices, periodic = True ):
    R'''
    Function to calculate the fraction of trans dihedrals       
    '''
    ## GET DIHEDRAL LIST
    dihedral_list = find_dihedral_list( ligand_indices )
    
    ## CALCULATE DIHEDRALS
    dihedrals = calc_dihedral_angles( traj, dihedral_list, len(ligand_indices), periodic = periodic )
    
    return dihedrals
        
### FUNCTION TO FIND DIHEDRAL LIST
def find_dihedral_list( indices ):
    R'''
    The purpose of this function is to find all dihedral indices. 
    '''
    ## CREATING A BLANK LIST
    dihedral_list = []
    ## GENERATING DIHEDRAL LIST BASED ON HEAVY ATOMS
    for ligand in indices:
        ## ONLY CHECKS LIGINDS WITH THE TAIL GROUP INSIDE THE CONTACT AREA
        ## LOOPING THROUGH TO GET A DIHEDRAL LIST (Subtract owing to the fact you know dihedrals are 4 atoms)
        for each_iteration in range(len(ligand)-3):
            ## APPENDING DIHEDRAL LIST
            dihedral_list.append(ligand[each_iteration:each_iteration+4])
                
    return dihedral_list

### FUNCTION TO CALCULATE THE DIHEDRALS
def calc_dihedral_angles( traj, dihedral_list, n_ligands, periodic = True ):
    '''
    The purpose of this function is to calculate all the dihedral angles given a list
    INPUTS:
        self: class object
        traj: trajectory from md.traj
        dihedral_list:[list] list of list of all the possible dihedrals
        periodic: [logical] True if you want periodic boundaries
    OUTPUTS:
       dihedrals: [np.array, shape=(time_frame, dihedral_index)]  dihedral angles in degrees from 0 to 360 degrees
    '''
    ## CALCULATING DIHEDRALS FROM MDTRAJ
    dihedrals = md.compute_dihedrals(traj, dihedral_list, periodic = periodic )
    # RETURNS NUMPY ARRAY AS A SHAPE OF: (TIME STEP, DIHEDRAL) IN RADIANS
    dihedrals = np.rad2deg(dihedrals) # np.array( dihedrals ) * 180 / np.pi # 
    dihedrals[ dihedrals < 0 ] = dihedrals[ dihedrals < 0 ] + 360  # to ensure every dihedral is between 0-360
    dihedrals = dihedrals.reshape((dihedrals.shape[0],n_ligands,dihedrals.shape[1]//n_ligands))
    dihedrals_per_frame = np.mean((dihedrals < 240) & (dihedrals > 120), axis=2)
    
    return dihedrals_per_frame

    
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
    
    data = trans_fraction( traj, ligand_ndx )    
    