"""
pickle_functions.py
contains function to load and save pickle files

CREATED ON: 04/07/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT MDTRAJ
import mdtraj as md
## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv
## CREATING GRID FUCNTIONS
from MDDescriptors.surface.core_functions import create_grid
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
## FUNCTION TO COMPUTE Z HEIGHT OF WC INTERFACE
def compute_willard_chandler_grid( path_wd,
                                   input_prefix,
                                   want_rewrite = False,
                                   **kwargs ):
    r'''
    Compute WC interface and calculates average height
    '''
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv ( wd = path_wd )
    ## GRID ARGUEMENTS
    func_inputs = {
                    'input_prefix': input_prefix,
                    'center_residue_name': "None",
                    'only_last_ns': True,
                    'rewrite': want_rewrite
                   }    
    ## CONVERTING TRJ TO HEAVY WATER ATOMS FOR WC ANALYSIS
    wc_gro_file, wc_xtc_file = trjconv_conversion.generate_water_heavy_atoms(**func_inputs)
    ## LOADING TRAJECTORY
    traj = md.load( os.path.join( path_wd, wc_xtc_file ),
                    top = os.path.join( path_wd, wc_gro_file ) )
    ## GENERATING GRID
    grid = create_grid( traj = traj, 
                        out_path = os.path.join( path_wd, r"output_files" ),
                        wcdatfilename = "willard_chandler_grid.dat", 
                        wcpdbfilename = "willard_chandler_grid.pdb", 
                        write_pdb = True, 
                        n_procs = 20,
                        verbose = True,
                        want_rewrite = want_rewrite,
                        alpha = WC_DEFAULTS["alpha"],
                        contour = 16., # use 0.5 bulk contour
                        mesh = WC_DEFAULTS["mesh"],
                        )
    ## RETURN RESULTS
    return grid

## FUNCTION TO COMPUTE Z HEIGHT OF WC INTERFACE
def compute_willad_chandler_height( path_wd,
                                    input_prefix,
                                    want_rewrite = False,
                                    **kwargs ):
    r'''
    '''
    ## COMPUTE WILLARD-CHANDLER GRID
    grid = compute_willard_chandler_grid( path_wd, 
                                          input_prefix,
                                          want_rewrite,
                                          **kwargs )    
    ## REMOVE AIR-WATER INTERFACE
    grid = grid[grid[:,2] < grid[:,2].mean()]
    ## Z_REF = avg(grid)
    height = grid[:,2].mean()
    
    return height

## FUNCTION TO COMPUTE WC SURFACE STATISTICS
def compute_wc_statistic( sim_working_dir,
                          input_prefix,
                          want_rewrite = False,
                          **kwargs ):
    r'''
    '''
    path_out = os.path.join( sim_working_dir, r"output_files" )
    ## COMPUTE WILLARD-CHANDLER GRID
    grid = compute_willard_chandler_grid( sim_working_dir, 
                                          input_prefix,
                                          want_rewrite,
                                          **kwargs )    
    ## REMOVE AIR-WATER INTERFACE
    grid = grid[grid[:,2] < grid[:,2].mean()]
    ## COMPUTE DIFFERENCE BETWEEN MIN AND MAX HEIGHT
    min_max_diff = grid[:,2].max() - grid[:,2].min()
    save_pkl( min_max_diff, os.path.join( path_out, input_prefix + r"_height_difference.pkl" ) )
    ## COMPUTE AVERAGE HEIGHT
    avg_height = grid[:,2].mean()
    save_pkl( avg_height, os.path.join( path_out, input_prefix + r"_mean_height.pkl" ) )
    ## COMPUTE HEIGHT VARIANCE
    var_height = grid[:,2].var()
    save_pkl( var_height, os.path.join( path_out, input_prefix + r"_height_variance.pkl" ) )    
    