# -*- coding: utf-8 -*-
"""
generate_wc_grid.py
This is the main code to run analysis on nanoparticles coated with SAMs.

INSTALLATION NOTES:
    - May need ski-kit image
        pip install scikit-image

Updated by Alex K. Chew (10/24/2019)

Possible error in WC code includes:
    - By default, WC code accounts for only the heavy atoms in the system. It means 
    that if you have more than one heavy atom, you may be overcounting the number 
    of atoms in the system. Future work will focus on incorporating either center 
    of mass in the system or some other metric for more complex solvents. 
    - It is recommended that you use gmx trjconv and omit all the non-heavy atoms in the 
    system. Using 1 ns of the simulation, it should not take much space, and it is a 
    good method to debug the system. 


"""
##############################################################################
# Imports
##############################################################################
import os, sys
import pickle
if "linux" in sys.platform and "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg') # turn off interactive plotting

import numpy as np
import mdtraj as md

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## IMPORTING GRIDDING TOOL
from MDDescriptors.surface.willard_chandler_parallel import compute_wc_grid

## IMPORTING TRACKING TIME
from MDDescriptors.core.initialize import checkPath2Server

### FUNCTIONS
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

def compute_sam_height( traj ):
    r'''
    '''
    sam_ligand_heavy_indices = [ [ atom.index for atom in residue.atoms if atom.element.symbol != "H" ] \
                                   for residue in traj.topology.residues if residue.name not in [ "HOH" ] ]
    
    min_z_values = []
    max_z_values = []
    for ligand in sam_ligand_heavy_indices:
        z_time_avg = traj.xyz[:,ligand,2]
        min_z_values.append( np.min(z_time_avg) )
        max_z_values.append( np.max(z_time_avg) )
    
    sam_height = np.mean( max_z_values )
    
    return sam_height

#%%
##############################################################################
# Load analysis inputs and trajectory
##############################################################################   
if __name__ == "__main__":
    # --- TESTING ---
          
    ## DEFINING MAIN SIMULATION
    main_sim = r"R:\simulations\np_hydrophobicity"
    ## DEFINING SIM NAME
    sim_name=r"np_planar_300K_dodecanethiol_CHARMM36"
    ## DEFINING WORKING DIRECTORY
    wd = checkPath2Server( os.path.join(main_sim, sim_name) )
    
    ## DEFINING GRO AND XTC
    height_gro_file = r"sam_prod_1ns.gro"
    height_xtc_file = r"sam_prod_1ns.xtc"
    wc_gro_file = r"sam_prod_1ns_water_oxygens.gro"
    wc_xtc_file = r"sam_prod_1ns_water_oxygens.xtc"
            
    ## DEFINING NUMBER OF PROCESSORS
    n_procs = 20
    
    ## DEFINING MESH
    mesh = WC_DEFAULTS['mesh']
    
    ## DEFINING OUTPUT FILE
    output_file = "output_files"
    
    ## WILLARD-CHANDLER VARIABLES
    alpha = WC_DEFAULTS['alpha']
    contour = WC_DEFAULTS['contour']
            
    ## DEFINING PATHS
    path_height_gro = os.path.join( wd, height_gro_file )
    path_height_xtc = os.path.join( wd, height_xtc_file )
    path_wc_gro = os.path.join( wd, wc_gro_file )
    path_wc_xtc = os.path.join( wd, wc_xtc_file )
    
    ## OUTPUT PATHS
    out_path = os.path.join( wd, output_file )
    
    ##############################################################################
    # Execute/test the script
    ##############################################################################
    #%%
    ## PRINTING
    print("Loading trajectory for SAM height calculation")
    print(" --> XTC file: %s"%(path_height_gro) )
    print(" --> GRO file: %s"%(path_height_xtc) )
    ## LOADING TRAJECTORY
    traj = md.load(path_height_xtc, top = path_height_gro)
    
    height = compute_sam_height( traj )
    del traj
    
    ## PRINTING
    print("Loading trajectory for SAM height calculation")
    print(" --> XTC file: %s"%(path_height_gro) )
    print(" --> GRO file: %s"%(path_height_xtc) )
    ## LOADING TRAJECTORY
    traj = md.load(path_wc_xtc, top = path_wc_gro)
    
    ## PRINTING
    print("Generating grid")
    print(" --> Output path: %s"%(out_path) )
    print(" --> N_Procs: %s"%(n_procs) )
    print(" --> Mesh: %s"%(', '.join([str(x) for x in mesh]) )  )
    print(" --> Sigma/alpha value: %.3f"%(alpha) )
    print(" --> Contour level: %.3f"%(contour) )    
    print(" --> Total trajectory frames: %s"%(len(traj)) )
        
    ## COMPUTTING GRID
    points, _, avg_density_field = compute_wc_grid( traj = traj, 
                                                    sigma = alpha, 
                                                    mesh = mesh, 
                                                    contour = 30, #contour, 
                                                    n_procs = n_procs,
                                                    print_freq = 100)
    mean_z = points[:,2].mean()
    wc_height = points[points[:,2]<mean_z,2].mean()
    print( "SAM height: %0.2f" % height )
    print( "Cavity z position: %0.2f" % ( height + 0.33 ) )
    print( "WC z position: %0.2f" % ( wc_height ) )
    