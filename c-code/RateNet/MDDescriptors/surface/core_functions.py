# -*- coding: utf-8 -*-
R"""
core_functions.py
This script contains functions frequently used. 

FUNCTIONS:
    get_list_args: get list arguments
    flatten_list_of_list: flattens list of list
    split_list: splits list evenly
"""

##############################################################################
# Imports
##############################################################################

import os
import numpy as np
import mdtraj as md

## IMPORTING GRIDDING TOOL
from MDDescriptors.surface.willard_chandler_parallel import compute_wc_grid

## PICKLE TOOL
import MDDescriptors.core.pickle_tools as pickle_tools

__all__ = [ 'count_atoms_in_sphere', 'calc_unnorm_dist', 'calc_mu',
            'create_grid' ]

##############################################################################
# Functions
##############################################################################
## CALL FUNCTION TO CONVERT TO LIST
def get_list_args(option, opt_str, value, parser):
    setattr(parser.values, option.dest, value.split(','))

## FUNCTION TO COMPUTE NUMBER OF ATOMS WITHIN SPHERE
def count_atoms_in_sphere( traj, pairs, r_cutoff = 0.33, d = -1, periodic = True ):
    r"""
    The purpose of this function is to comptue the number of atoms within some 
    radius of cutoff.
    INPUTS:
        traj: [obj]
            trajectory object
        pairs: [np.array, shape=(N,2)]
            pairs array that you're interested in
        r_cutoff: [float]
            cutoff that you are interested in
    OUTPUTS:
        N: [np.array]
            array that is the same size as pairs across trajectories
    """
    dist = r_cutoff - md.compute_distances( traj, pairs, periodic = periodic )
    N = np.sum( dist >= 0., axis = 1 )
    return N

def calc_unnorm_dist( N, d = -1 ):
    r"""
    This function computes the un-normalized histogram distribution. This defines 
    d, which is some large range of value.
    INPUTS:
        N: [np.array]
            numpy array containing your distribution
        d: [int]
            number of bins used for the histogram
    OUTPUTS:
        histogram: [np.array]
            histogram data for the un-normalized distribution
    """
    if d < 0.:
        d = np.max( N )
        
    return np.histogram( N, bins = int( d ), range = [ 0., d ] )[0]

### FUNCTION TO NORMALIZE DISTRIBUTION
def get_x_y_from_p_N(p_N, d = 10):
    '''
    The purpose of this function is to compute the x and y values used for 
    fitting the probability distribution. The assumption is that we have a 
    Gaussian distribution.
    INPUTS:
        p_N: [np.array]
            probability array
        d: [int]
            maximum number of occurances for probability arrya
    OUTPUTS:
        x: [np.array]
            occurances where the probability > 0
        y: [np.array]
            normalized -log of the probability
        p: [polyfit]
            poly fit result (coefficients)
        p_func: [func]
            p_func function that could output poly fit
    '''
    ## NORMALIZING P(N) DISTRIBUTION SO THAT THE SUM IS 1
    norm_p = p_N / p_N.sum()
    ## GETTING X VALUES THAT ARE GREATER THAN 0
    x = np.arange( 0, d, 1 )[ norm_p > 0. ]
    ## GETTING UNNORM Y
    unnorm_y = -np.log( norm_p[ norm_p > 0. ] )
    ## SUBTRACTING BY THE MINIMA
    y = unnorm_y - unnorm_y.min()
    ## FITTING POLYNOMIAL TO GAUSSIAN
    p = np.polyfit( x, y, 2 )
    ## GETTING P FUNCTION
    p_func = np.poly1d(p)
    return x, y, p, p_func

### FUNCTION TO GET MU FROM UNNORM P_N
def compute_mu_from_num_dist_for_one_grid_point(unnorm_p_N,
                                                grid_index,
                                                max_N,):
    '''
    The purpose of this function is to compute mu for a single index.
    INPUTS:
        unnorm_p_N: [np.array, shape = (num_points, max_d)]
            unnormalized histogram
        grid_index: [int]
            grid index
        max_N: [int]
            maximum N value
    OUTPUTS:
        mu: [float]
            mu value of the grid point given the unnormalized probability distribution.
    '''
    ## DEFINING PROBABILITY
    p_N = unnorm_p_N[grid_index]

    ## GETTING VALUES
    x, y, p, p_func = get_x_y_from_p_N(p_N = p_N,
                                       d = max_N)

    ## DEFINING MU VALUE
    mu = p_func(0)
    
    return mu

### FUNCTION TO COMPUTE MU
def calc_mu( p_N_matrix, d = -1 ):
    r"""
    This function computes mu
    INPUTS:
        p_N_matrix: [np.array]
            matrix
        d: [int]
            Number of waters that fit into volume. Typically 8 waters found.
            If it's negative, we take the maximum of the p_N_matrix
    """
    if d < 0.:
        d = np.max( p_N_matrix )
    
    ## GETTING MU VALUES
    mu = np.zeros( len(p_N_matrix) )
    for ii, p_N in enumerate( p_N_matrix ):
        
        ## GETTING MU PER GRID POINT
        mu[ii] = compute_mu_from_num_dist_for_one_grid_point(unnorm_p_N = p_N_matrix,
                                                             grid_index = ii,
                                                             max_N = d)
#   Same as above, just simplified
#        ## GETTING VALUES
#        x, y, p, p_func = get_x_y_from_p_N(p_N = p_N,
#                                           d = d)
#
#        ## FINDING WHEN P FUNCTION IS 0
#        mu[ii] = p_func(0)
        
    return mu

### FUNCTION TO LOAD WC DATAFILE
def load_datafile(path_to_file):
    '''
    The purpose of this function is to load the WC data file
    INPUTS:
        path_to_file: [str]
            path to the data file
    OUTPUTS:
        data: [np.array, shape = (n_points, (x,y,z,value))]
            data from the wc interface
    '''
    ## OPENING DATA FILE
    with open( path_to_file ) as raw_data:
        data = raw_data.readlines()
    
    ## TRANSFORMING DATA TO NUMPY ARRAY
    data = np.array( [ [ float(el) for el in line.split(',') ] for line in data[2:] ] )
    
    return data

### FUNCTION TO WRITE THE PDB
def create_pdb(data, 
               path_pdb_file, 
               traj = None,
               b_factors = None,
               box_dims = []):
    R'''
    The purpose of this function is to create a PDB file.
    INPUTS:
        data: [np.array]
            x, y, z coordinates
        path_pdb_file: [str]
            path to the pdb file
        traj: [obj]
            trajectory object. If None, we assume box_dims is supplied
        b_factors: [np.array]
            array of b-factors
        box_dims: [np.array, shape=3]
            box dimensions in nanometers
    OUTPUTS:
        pdb file
    '''
    ## CHECKING IF TRAJ IS DEFINED 
    if traj is not None:
        ## DEFINING BOX DIMENSIONS
        box_dims = np.zeros(3)
        box_dims[0] = traj.unitcell_lengths[0,0]
        box_dims[1] = traj.unitcell_lengths[0,1]
        box_dims[2] = traj.unitcell_lengths[0,2]
        
    ## CHECKING B FACTORS
    if b_factors is None:
        b_factors = np.ones(len(data))
    else:
        ## CHECK IF B FACTORS MATCH COORD
        if len(data) != len(b_factors):
            print("Warning! The length of your data does not match that of the b-factors")
            print("Data length: %d"%(len(data) ) )
            print("B-faactor length: %d"%(len(data) ) )
            print("Perhaps you missed a b-factor somewhere?")
    
    ## WRITING PDB FILE
    with open( path_pdb_file, 'w+' ) as pdbfile:
        pdbfile.write( 'TITLE     frame t=1.000 in water\n' )
        pdbfile.write( 'REMARK    THIS IS A SIMULATION BOX\n' )
        pdbfile.write( 'CRYST1{:9.3f}{:9.3f}{:9.3f}{:>7s}{:>7s}{:>7s} P 1           1\n'.format( box_dims[0]*10, 
                                                                                                 box_dims[1]*10,
                                                                                                 box_dims[2]*10, 
                                                                                                 '90.00', '90.00', '90.00' ) )
        pdbfile.write( 'MODEL        1\n' )
        for ndx, coord in enumerate( data ):
            line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format( \
                    'ATOM', ndx+1, 'C', '', 'SUR', '', 1, '', coord[0]*10, coord[1]*10, coord[2]*10, 1.00, b_factors[ndx], '', '' )
            pdbfile.write( line )
            
        pdbfile.write( 'TER\n' )
        pdbfile.write( 'ENDMDL\n' )

    return

### FUNCTION TO WRITE DAT FILE
def write_datafile(path_to_file,
                   data):
    '''
    The purpose of this function is to write to the data file
    INPUTS:
        path_to_file: [str]
            path to the file
        data: [np.array, shape = (N,3)]
            data points in x, y, z positions
    OUTPUTS:
        output of a data file
    '''
    ## OPENING AND WRITING TO FILE
    with open( path_to_file, 'w+' ) as outfile:
        outfile.write( '# x y z\n\n' )                      
        for line in data:
            outfile.write( '{:0.3f},{:0.3f},{:0.3f}\n'.format( line[0], line[1], line[2] ) ) # Outputting in x y z format
    return

### FUNCTION TO CREATE WC GRID
def create_grid(traj, 
                out_path, 
                wcdatfilename, 
                wcpdbfilename, 
                alpha = 0.24, 
                mesh = [ 0.1, 0.1, 0.1 ], 
                contour = 16., 
                write_pdb = True, 
                n_procs = 28,
                want_rewrite = False,
                last_frame = 1000,
                verbose = False, 
                want_debug = False,
                want_normalize_c = False):
    r"""
    This script creates the grid used for willard chandler interface. 
    INPUTS:
        traj: [traj object]
            trajectory information
        out_path: [str]
            path to output
        wcdatfilename: [str]
            willard chandler data file name, which stores the xyz positions of the interface
        wcpdbfilename: [str]
            willard chandler PDB file, visualized with PDB, etc. on VMD.
        alpha: [float]
            alpha value for WC interface
        contour: [float]
            contour value for WC interface, default=16 for half the bulk number density of water
        write_pdb: [logical]
            True if you want to write a PDB file
        n_procs: [int]
            number of processors desired
        want_rewrite: [logical]
            True if you want to rewrite
        last_frame: [int]
            Last frame to use
        verbose: [logical]
            True if you want to print out details
        want_debug: [logical]
            True if you want to debug the WC grid. This will print out additional 
            information, such as the interface_points, interface, avg density field, and so forth.
        want_normalize_c: [logical]
            True if you want to normalize contour from 0 to 1
    OUTPUTS:
        data: [np.array, shape=(N,3)]
            data for the WC interface contained in x, y, z positions in space
    """
    ## DEFINING PATH TO WC FILE
    path_wc_dat_file = os.path.join(out_path, wcdatfilename)
    path_pdb_file = os.path.join( out_path, wcpdbfilename)
    ## SEEING IF YOU WANT TO RUN THE WC GRIDDING
    if os.path.isfile(path_wc_dat_file) is False or want_rewrite is True:
            
        ## COMPUTTING GRID
        if want_debug is False:
            data = compute_wc_grid( traj = traj, 
                                    sigma = alpha, 
                                    mesh = mesh, 
                                    contour = contour, 
                                    n_procs = n_procs,
                                    print_freq = 100,
                                    want_normalize_c = want_normalize_c)[0]
        else:
            ## STORING ALL POSSIBLE DATA
            data, interface, avg_density_field = compute_wc_grid( traj = traj, 
                                                                  sigma = alpha, 
                                                                  mesh = mesh, 
                                                                  contour = contour, 
                                                                  n_procs = n_procs,
                                                                  print_freq = 100,
                                                                  want_normalize_c = want_normalize_c)
            
        ## PRINTING TO FILE
        if verbose is True:
            print( '--- Outputs written to %s' %(out_path) )
            
        ## WRITING TO DATA FILE
        write_datafile(path_to_file=path_wc_dat_file,
                       data = data)
        
        ## ALSO WRITING THE PICKLE FILE
        if want_debug is True:
            
            path_debug_pickle = os.path.join(out_path, "debug.pickle")
            ## STORING
            pickle_tools.pickle_results(results = [data, interface, avg_density_field],
                                        pickle_path = path_debug_pickle,
                                        verbose = True)
        
    else:
        ## LOADING THE DATA
        data = load_datafile(path_to_file = path_wc_dat_file)

    ## FUNCTION TO WRITE THE PDB FILE OF PROBES
    if write_pdb is True or os.path.isfile(path_pdb_file) is False or want_rewrite is True:
        # WRITE PDB FILE WITH PROBES   
        if verbose is True:         
            print( '--- Grid PDB file written to: %s' %( path_pdb_file) )
        ## WRITING PDB
        create_pdb(traj = traj, 
                   data = data, 
                   path_pdb_file = path_pdb_file)
        
    return data
    

### FUNCTION TO FLATTEN LIST OF LIST
def flatten_list_of_list( my_list ):
    ''' This flattens list of list '''
    return [item for sublist in my_list for item in sublist]
