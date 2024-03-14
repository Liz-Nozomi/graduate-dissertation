# -*- coding: utf-8 -*-
"""
fnxl_grp_area.py
script to calculate area occupied by functional groups on SAMs

"""
##############################################################################
# Imports
##############################################################################
import sys, os
if "linux" in sys.platform and "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg') # turn off interactive plotting

if r'R:/analysis_scripts' not in sys.path:
    sys.path.append( r'R:/analysis_scripts' )

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import analysis_tools as at
from datetime import datetime
import mdtraj as md
import numpy as np

#%%
##############################################################################
# Load analysis inputs and trajectory
##############################################################################   
if __name__ == "__main__":
    # --- TESTING ---
    wd = r'R:/simulations/polar_sams/spatial_dependence/n11_amide_0/'
#    wd = r'/mnt/r/simulations/physical_heterogeneity/autocorrelation/sam_single_8x8_300K_dodecanethiol_tip3p_nvt_CHARMM36_0.1ps_2/'
#    wd = sys.argv[1]
    args = {  
#              'in_file':   sys.argv[2],
              'in_file': 'sam_whole',
              'in_path':   wd + 'equil/',
              'coords_file': 'sam_slab_coordinates.csv',
              'dimensions_file': 'sam_slab_dimensions.csv',
              'solvents':  [ 'HOH', 'MET' ],
           }
    
    out_path = wd + 'output_files/'
        
    traj_import = at.import_traj()
    traj = traj_import.load( **args )
    traj = traj[len(traj)//4:]
    print( traj )
    
##############################################################################
# Execute/test the script
##############################################################################
#%%
    ## FIND ALL CARBON ATOMS IN LIGANDS
    monolayer_atoms = [ [ atom.index for atom in residue.atoms if 'C' in atom.name ] for residue in traj.topology.residues if residue.name not in args['solvents'] ]
    
    # determine direction of ligand vectors
    monolayers = []
    headAtoms = np.array( [ ligand[0] for ligand in monolayer_atoms ] )
    tailAtoms = np.array( [ ligand[-1] for ligand in monolayer_atoms ] )
    normalComponent = np.mean( traj.xyz[ :, headAtoms, 2 ], axis = 0 ) - \
                      np.mean( traj.xyz[ :, tailAtoms, 2 ], axis = 0 ) # may need to rethink when doing mixed height SAMs

    monolayers.append( [ ligand for ligand, normal in zip( monolayer_atoms, normalComponent ) if normal > 0. ] ) # monolayer is pointing up
    monolayers.append( [ ligand for ligand, normal in zip( monolayer_atoms, normalComponent ) if normal < 0. ] ) # monolayer is pointing down

    # calculate the surface position of the monolayers
    for ndxMonolayer, monolayer in enumerate( monolayers ):
        if monolayer: # if there is a monolayer continue
            tailGroups = np.array( [ ligand[-1] for ligand in monolayer ] )
            monolayerPosition = traj.xyz[ :, tailGroups, : ]
            xMono = np.mean( monolayerPosition[:,:,0] )
            yMono = np.mean( monolayerPosition[:,:,1] )

    ref = xMono - 0.5*np.mean(traj.unitcell_lengths[:,0])
    ligand_ndx = [ [ atom.index for atom in residue.atoms ] for residue in traj.topology.residues if residue.name not in args['solvents'] ]
    atom_names = [ [ atom.name for atom in residue.atoms ] for residue in traj.topology.residues if residue.name not in args['solvents'] ]

    tail_groups = [ [ "C35", "H36", "H37", "H38" ], [ "C38", "O39", "N40", "H41", "H42" ] ]

    atoms = [ 'C35', 'C38', 'O39', 'N40' ]
    c = [ 'black', 'grey', 'red', 'blue' ]
    histos = []
    for jj, atom in enumerate( atoms ):
        atom_ndx = []
        for ll, tail_group in enumerate( tail_groups ):
            for ndx in range(len(atom_names)):
                if all( elem in atom_names[ndx] for elem in tail_group ):
                    group_ndx = [ ligand_ndx[ndx][ii] for ii, name in enumerate( atom_names[ndx] ) if name in tail_group ]
                    for ii in group_ndx:
                        if atom == traj.topology.atom(ii).name:
                            atom_ndx.append( ii )
                
        histo = at.spatial_distribution( traj, atom_ndx, axis = 0, reference = ref, bin_width = 0.02 )
        histos.append(histo)
        
        plt.plot( histo[:,0], histo[:,1], linestyle = '-', linewidth = '1.5', color = c[jj] )

    