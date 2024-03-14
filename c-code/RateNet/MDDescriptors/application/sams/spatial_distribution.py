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
import numpy as np

__all__ = [ 'spatial_distribution' ]


##############################################################################
# spatial distribution function
##############################################################################
def spatial_distribution( traj, indices, axis = 0, reference = 0., bin_width = 0.05 ):
    R'''
    '''
    axis_max = np.mean( traj.unitcell_lengths[:,axis] )
    
    ## GET ATOM POSITIONS ALONG DESIRED AXIS
    positions = traj.xyz[ :, indices, axis ] - reference
    positions[positions<0.0] += axis_max
    positions[positions>axis_max] -= axis_max
    
    ## HISTOGRAM POSITIONS
    n_bins = np.ceil( axis_max / bin_width ).astype('int')
    bins = np.arange( 0.5*bin_width, bin_width * ( 0.5 + n_bins ), bin_width )
    histo = np.histogram( positions, bins = n_bins, range = [ 0., axis_max ] )[0]
    histo = histo / histo.sum()
    if len(bins) > len(histo):
        return np.vstack(( bins[:len(histo)], histo )).transpose()
    elif len(bins) < len(histo):
        return np.vstack(( bins, histo[:len(bins)] )).transpose()
    else:
        return np.vstack(( bins, histo )).transpose()

 
##############################################################################
# Stand alone script (TESTING)
##############################################################################

if __name__ == "__main__":    
    from core import import_traj
    import matplotlib.pyplot as plt
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
    
    data = spatial_distribution( traj, end_group_ndx )
    plt.figure()
    plt.plot( data[:,0], data[:,1] )