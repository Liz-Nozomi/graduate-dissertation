"""
sam_analysis.py
this is the main script for SAM analysis

CREATED ON: 02/18/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path
## IMPORTING ANALYSIS FUNCTION
#from MDDescriptors.application.sams.rdf import compute_interfacial_rdf
from MDDescriptors.application.sams.density import compute_density_profile
from MDDescriptors.application.sams.triplet_angle import compute_triplet_angle_distribution
from MDDescriptors.application.sams.h_bond_properties import compute_num_hbonds
from MDDescriptors.application.sams.willard_chandler import compute_wc_statistic
from MDDescriptors.application.sams.triplet_entropy import compute_triplet_entropy
from MDDescriptors.application.sams.hydration_dynamics import hydration_residence_time_profile
## IMPORT WILLARD-CHANDER HEIGHT FUNCTION
from MDDescriptors.application.sams.willard_chandler import compute_willad_chandler_height
## IMPORT TRAJECTORY FUNCTION
from MDDescriptors.application.sams.trajectory import load_md_traj

##############################################################################
## FUNCTIONS
##############################################################################
### FUNCTION TO CHECK IF YOU ARE RUNNING ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## USER SPECIFIED INPUTS
    z_cutoff     = 0.3       # nm, z distance cutoff, only includes water below this value
    r_cutoff     = 0.33      # nm, radius cutoff for small cavities
    use_com      = True      # use com of water, false oxygen atom used as ref
    n_procs      = 28        # number of processors to use in parallel
    periodic     = True      # use pbc in calculations
    residue_list = [ "SOL", "HOH" ] # list of residues to include in calculations
    verbose      = True      # print out to cmd line
    print_freq   = 100       # print after N steps
    want_rewrite = False     # rewrite files
    ## GROMACS OUTPUT
    input_prefix = "sam_prod" # sys.argv[1]
    ## MAIN DIRECTORY
    main_dir = r"R:\simulations\polar_sams\unbiased\sample3"    
    ## SUB DIRECTORIES
    sub_directories = [ 
                        "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",

                       ]    
    ## LIST OF ANALYSIS FUNCTIONS
    analysis_list = { 
                      "residence_time" : hydration_residence_time_profile,
                    }
    ## LOOP THROUGH DIRECTORIES
    for sd in sub_directories:
        ## PREPARE DIRECTORY FOR ANALYSIS
        path_to_sim = check_server_path( os.path.join( main_dir, sd ) )
        ## COMPUTE ANALYSIS IN ANALYSIS_LIST
        for analysis_key, function_obj in analysis_list.items():
            print( "\n--- executing %s function ---" % analysis_key )            
            ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
            analysis_kwargs = { 
                                'sim_working_dir' : path_to_sim,
                                'input_prefix'    : input_prefix,
                                'rewrite'         : want_rewrite,
                                'z_cutoff'        : z_cutoff,
                                'r_cutoff'        : r_cutoff,
                                'use_com'         : use_com,
                                'n_procs'         : n_procs,
                                'periodic'        : periodic,
                                'residue_list'    : residue_list,
                                'verbose'         : verbose, 
                                'print_freq'      : print_freq }
            ## LOAD MD TRAJECTORY
            traj = load_md_traj( path_traj    = path_to_sim,
                                 input_prefix = input_prefix )
            ## GET Z_REF FROM WC GRID
            z_ref = compute_willad_chandler_height( path_wd      = path_to_sim, 
                                                    input_prefix = input_prefix ) 
            ## EXECUTE FUNCTIONS
            C, tau = function_obj( traj,
                              z_ref        = z_ref,
                              z_cutoff     = 0.3,  # Shell ACS Nano used 0.5, INDUS uses 0.3 nm cavity
                              n_procs      = 20,
                              use_com      = True,
                              periodic     = True,
                              residue_list = [ 'SOL', 'HOH' ],
                              verbose      = True,
                              print_freq   = 100,
                              rewrite      = False)

          