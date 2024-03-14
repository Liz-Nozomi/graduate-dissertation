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
import sys, os
if "linux" in sys.platform and "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg') # turn off interactive plotting

import os
import numpy as np
import mdtraj as md

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## CREATING GRID FUCNTIONS
from MDDescriptors.surface.core_functions import create_grid, get_list_args

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing 

#%%
##############################################################################
# Load analysis inputs and trajectory
##############################################################################   
if __name__ == "__main__":
    # --- TESTING ---
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## RUNNING TESTING    
    if testing == True:      
        ## DEFINING MAIN SIMULATION
        main_sim=r"/home/bdallin/simulations/polar_sams/indus/2x2x0.3nm/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36_2x2x0.3nm"
        ## DEFINING SIM NAME
        sim_name=r"equil"
        ## DEFINING WORKING DIRECTORY
        wd = os.path.join(main_sim, sim_name)
        ## DEFINING GRO AND XTC
        gro_file = r"sam_wc.gro"
        # r"sam_prod.gro"
        xtc_file = r"sam_wc.xtc"
        # r"sam_prod.xtc"
        
        ## DEFINING PREFIX
        output_prefix = "convergence"
        
        ## DEFINING NUMBER OF PROCESSORS
        n_procs = 20
        
        ## DEFINING MESH
        mesh = WC_DEFAULTS['mesh']
        
        ## DEFINING OUTPUT FILE
        output_file = "output_files"
        
        ## WILLARD-CHANDLER VARIABLES
        alpha = WC_DEFAULTS['alpha']
        contour = WC_DEFAULTS['contour']
        
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## REPRESENTATION TYPE
        parser.add_option('--path', dest = 'simulation_path', help = 'Path of simulation', default = '.', type=str)
        
        ## DEFINING GRO AND XTC FILE
        parser.add_option('--gro', dest = 'gro_file', help = 'Name of gro file', default = 'sam.gro', type=str)
        parser.add_option('--xtc', dest = 'xtc_file', help = 'Name of xtc file', default = 'sam.xtc', type=str)
        
        ## OUTPUT PREFIX
        parser.add_option('--output_file', dest = 'output_file', help = 'Output directory name', default = 'output', type=str)
        parser.add_option('--output_prefix', dest = 'output_prefix', help = 'Output prefix', default = 'output', type=str)

        parser.add_option('--n_procs', dest = 'n_procs', help = 'Number of processors', default = 20, type=int)
        
        ## DEFINING MESH SIZE
        # parser.add_option('--mesh', dest = 'mesh', help = 'Mesh size', default = 0.1, type=float)
        parser.add_option("--mesh", dest="mesh", action="callback", type="string", callback=get_list_args,
                  help="Mesh separated by comma (no whitespace)", default = [ 0.1, 0.1, 0.1 ] )
        
        ## ADDING OTHER WC VARIABLES
        parser.add_option('--alpha', dest = 'alpha', help = 'Alpha (or sigma) value for the distribution', default = "None", type=str)
        parser.add_option('--contour', dest = 'contour', help = 'Desired contour level', default = "None", type=str)
        
        ## ADDING DEBUGGING OPTIONS
        parser.add_option('--debug', dest = 'want_debug', help = 'Turn on if you want to store pickles for wc interface', default = False, 
                          action = "store_true")
        
        ## ADDING NORMALIZING OPTIONS
        parser.add_option('--want_normalize_c', dest = 'want_normalize_c', help = 'true if you want normalize c values', 
                          default = "false", type=str)
        
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## DEFINING OUTPUTS
        # WORKING DIRECTORY
        wd = options.simulation_path
        # GRO/XTC
        gro_file = options.gro_file
        xtc_file = options.xtc_file
        
        ## OUTPUT PREFIX
        output_prefix = options.output_prefix

        ## TECHNICAL DETAILS
        n_procs = options.n_procs
        
        ## DEFINING MESH SIZE
        mesh = options.mesh
        
        ## CONVERTING ALL MESH TO POINT
        mesh = [float(i) for i in mesh]
        
        ## DEFINING OPTION DIRECTORY
        output_file = options.output_file
        
        ## DEBUGGING
        want_debug = options.want_debug
        
        ## NORMALIZED C
        want_normalize_c = options.want_normalize_c
        if want_normalize_c == "true":
            want_normalize_c = True
        else:
            want_normalize_c = False

        ## CHECKING IF ALPHA OR CONTOUR IS SPECIFIED
        if options.alpha == "None":
            alpha = WC_DEFAULTS['alpha']
        else:
            alpha = float(options.alpha)
        
        ## CONTOUR LEVEL
        if options.contour == "None":
            contour = None
            # WC_DEFAULTS['contour']
        else:
            contour = float(options.contour)
        
    ## LOGICALS (DEFAULT)
    write_pdb = True # Writes PDB file
    
    ## DEFINING PATHS
    path_gro = os.path.join( wd, gro_file )
    path_xtc = os.path.join( wd, xtc_file )
    
    ## OUTPUT PATHS
    out_path = os.path.join( wd, output_file )
    wcdatfilename = output_prefix + '_willard_chandler.dat'
    wcpdbfilename = output_prefix + '_willard_chandler.pdb'
    ##############################################################################
    # Execute/test the script
    ##############################################################################
    #%%
    ## PRINTING
    print("Loading trajectory")
    print(" --> XTC file: %s"%(path_xtc) )
    print(" --> GRO file: %s"%(path_gro) )
    ## LOADING TRAJECTORY
    traj = md.load(path_xtc, top = path_gro)
    
    ## PRINTING
    print("Generating grid")
    print(" --> Output path: %s"%(out_path) )
    print(" --> WC file: %s"%(wcdatfilename) )
    print(" --> N_Procs: %s"%(n_procs) )
    print(" --> Mesh: %s"%(', '.join([str(x) for x in mesh]) )  )
    print(" --> Sigma/alpha value: %.3f"%(alpha) )
    if type(contour) is float:
        print(" --> Contour level: %.3f"%(contour) )    
    else:
        print(" --> Contour level: %s"%(contour) )
    print(" --> Total trajectory frames: %s"%(len(traj)) )
    if want_debug is True:
        print(" --> Note: Debug has been turned on with --debug. This will output a pickle of the density field.")
    if want_normalize_c is True:
        print(" --> Note: Since normalized c values is turned on, inteface selected will be based on normalized densities")
        
    ## GENERATING GRID
    grid = create_grid( traj, 
                        out_path, 
                        wcdatfilename, 
                        wcpdbfilename, 
                        alpha = alpha, 
                        mesh = mesh, 
                        contour = contour, 
                        write_pdb = write_pdb, 
                        n_procs = n_procs,
                        verbose = True,
                        want_rewrite = True,
                        want_debug = want_debug,
                        want_normalize_c = want_normalize_c)
